"""Profile peak memory and iteration time with and without optimizer state sharding.

默认配置：单节点、2 GPU、xl、FP32、AdamW；全局 batch=4，每卡 batch=2，序列长 512，词表 10000。
脚本在同一实例上依次运行普通 AdamW 和分片 AdamW，两组均使用 OverLapDDP。

AutoDL 租卡建议（2026-10-26，供默认配置选型参考）：
- 推荐同一实例租用 2 张相同的 80GB GPU，例如 A800 80GB 或 A100 80GB。
  如需更多显存余量，可考虑同机 2 张 PRO 6000 96GB；价格和同机双卡库存以租用页面为准。
- 租卡应满足未分片 AdamW 基线的需求，以便两组实验在同一硬件上完成。
  不要仅按分片后的状态大小选卡；两组的模型参数和梯度仍在每张 GPU 上保留完整副本。
- 每卡约 34.1 亿参数；FP32 参数约 12.7 GiB，未分片时参数、梯度及 AdamW 的 m/v 合计约 50.8 GiB。
  这只是主要张量的估算，还需容纳激活、通信缓冲和优化器临时张量；完整 xl 双卡峰值尚未实测。
  24GB、32GB、48GB 单卡不适合默认未分片基线；两张卡的显存不能直接合并。
- 确认租用的是同一台机器上的两张卡。比较时保持卡型和互联拓扑一致，记录 nvidia-smi topo -m。
  如果要研究高速互联下的通信开销，选择并确认实际具有 NVLink/NVSwitch 连接的实例。
  80GB 或 96GB 的显存容量本身不能说明两卡之间采用什么互联。
- 当前脚本先在 CPU 创建模型，建议主机内存至少 64GB；同时确认实例实际分配的内存限额。
  预留足够磁盘空间安装项目依赖，并使用兼容项目 PyTorch/CUDA 版本的驱动。
- 建议先按量租用并完成环境、小模型流程检查，再执行默认 xl 实验；本脚本测内存和时间无需安装 nsys。
  两组都在同一实例运行，结果保存后按平台方式停止实例计费。

租用页面：https://www.autodl.com/home
单机多卡说明：https://www.autodl.com/docs/distributed_training/
GPU 与主机内存说明：https://www.autodl.com/docs/gpu/
"""

import csv
import os
from pathlib import Path

from torch.multiprocessing.spawn import spawn
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp import OverLapDDP, ShardedOptimizer
import torch
import timeit
import statistics
import torch.distributed as dist


path = Path(__file__).with_name("benchmark_dist.csv")
FILEDS = [
    "device_name",
    "optim_type",
    "rank",
    "init_peak_mem_GiB",
    "before_optim_peak_mem_mean_GiB",
    "after_optim_peak_mem_mean_GiB",
    "world_size",
    "step_time_mean_ms",
    "step_comm_time_mean_ms",
    "comm_ratio",
]

model_zoo = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}
model_size = "xl"
vocab_size = 10000
batch_size = 4
context_length = 512


def save_results(results):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FILEDS)
        writer.writerows(results)


def synchronize(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def setup(rank, world_size, device):
    if device == "cuda":
        torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl" if device == "cuda" else "gloo", rank=rank, world_size=world_size)


def run_dist(rank, optim_type, world_size, device, x_all, y_all, warmup_iter=5, repeate_iter=10):
    try:
        setup(rank, world_size, device)
        step = batch_size // world_size
        x = x_all[rank * step : (rank + 1) * step, :].to(device)
        y = y_all[rank * step : (rank + 1) * step, :].to(device)
        model = BasicsTransformerLM(vocab_size, context_length, **model_zoo[model_size]).to(device)
        ddp_model = OverLapDDP(model)
        if optim_type == "naive":
            ddp_optimizer = AdamW(ddp_model.parameters())
        else:
            ddp_optimizer = ShardedOptimizer(ddp_model.parameters(), AdamW)

        init_peak_mem_GiB = torch.cuda.max_memory_allocated() / 1024**3

        for _ in range(warmup_iter):
            synchronize(device)
            ddp_optimizer.zero_grad()
            logits = ddp_model(x)
            loss = cross_entropy(logits, y)
            loss.backward()
            ddp_model.finish_gradient_synchronization()
            ddp_optimizer.step()

        step_times = []
        step_comm_times = []
        before_optim_peak_mems = []
        after_optim_peak_mems = []

        for _ in range(repeate_iter):
            ddp_optimizer.zero_grad()

            synchronize(device)
            torch.cuda.reset_peak_memory_stats()
            start_time = timeit.default_timer()

            logits = ddp_model(x)
            loss = cross_entropy(logits, y)
            loss.backward()

            synchronize(device)
            comm_start_time = timeit.default_timer()

            ddp_model.finish_gradient_synchronization()

            synchronize(device)
            before_optim_peak_mems.append(torch.cuda.max_memory_allocated())
            comm_end_time = timeit.default_timer() - comm_start_time
            step_comm_times.append(comm_end_time)
            torch.cuda.reset_peak_memory_stats()

            ddp_optimizer.step()

            synchronize(device)
            after_optim_peak_mems.append(torch.cuda.max_memory_allocated())
            end_time = timeit.default_timer()
            step_times.append(end_time - start_time)

        step_times_mean = statistics.mean(step_times)
        step_comm_times_mean = statistics.mean(step_comm_times)
        before_optim_peak_mems_mean = statistics.mean(before_optim_peak_mems)
        after_optim_peak_mems_mean = statistics.mean(after_optim_peak_mems)

        result_item = {
            "device_name": torch.cuda.get_device_name() if device == "cuda" else "cpu",
            "optim_type": optim_type,
            "rank": rank,
            "world_size": world_size,
            "init_peak_mem_GiB": init_peak_mem_GiB,
            "before_optim_peak_mem_mean_GiB": before_optim_peak_mems_mean / 1024**3,
            "after_optim_peak_mem_mean_GiB": after_optim_peak_mems_mean / 1024**3,
            "step_time_mean_ms": step_times_mean * 1000,
            "step_comm_time_mean_ms": step_comm_times_mean * 1000,
            "comm_ratio": step_comm_times_mean / step_times_mean * 100,
        }

        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(result_item, gathered, dst=0)

        if rank == 0:
            save_results(gathered)

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def benchmark(optim_type, device, world_size=2):
    assert batch_size % world_size == 0, "Batch size must be divisible by world size"
    if device == "cuda":
        assert world_size <= torch.cuda.device_count(), "Not enough GPUs"
    x = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long)
    spawn(run_dist, args=(optim_type, world_size, device, x, y), nprocs=world_size, join=True)


if __name__ == "__main__":
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_zoo ", "Batch Size", "Context Length", "Vocab Size"])
        writer.writerow([model_size, batch_size, context_length, vocab_size])
        writer = csv.DictWriter(f, fieldnames=FILEDS)
        writer.writeheader()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for optim_type in ["naive", "sharded"]:
        benchmark(optim_type, device, world_size=2)
