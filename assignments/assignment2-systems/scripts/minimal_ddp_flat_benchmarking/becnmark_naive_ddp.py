"""Benchmark Flat DDP training and gradient synchronization on one node with two GPUs.

AutoDL 租卡建议（默认 xl、FP32、AdamW、全局 batch=4、每卡 batch=2、序列长 512）：
- 预算优先：同一实例租 2 张 A800 80GB，官网参考价约 5.59 元/卡时，合计 11.18 元/时。
- 显存余量优先：同一实例租 2 张 PRO 6000 96GB，约 6.98 元/卡时，合计 13.96 元/时。
- 每卡约 34.1 亿参数：FP32 参数约 12.7 GiB；参数、梯度、AdamW 的 m/v 合计约 50.8 GiB。
  还需容纳激活、通信缓冲及优化器临时张量；上述数值不是完整训练峰值，完整 xl 双卡尚未实测。
- DDP 每卡保存完整模型和优化器状态，显存不会相加；24GB、32GB、48GB 单卡不适合当前配置。
- 必须选择同一台机器上的两张卡；比较 DDP 方案时保持卡型、互联拓扑和精度一致，并记录拓扑。
- 当前脚本先在 CPU 创建模型，建议实例配备至少 64GB 主机内存。
参考价于 2026-10-24 检索，实际价格及同机双卡库存以租用页面为准：
https://www.autodl.com/home
https://www.autodl.com/docs/distributed_training/
"""

import csv
import os
from pathlib import Path

from torch.multiprocessing.spawn import spawn
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp import FlatDDP
import torch
import timeit
import statistics
import torch.distributed as dist


path = Path(__file__).with_name("benchmark_dist.csv")
FILEDS = ["device_name", "rank", "world_size", "step_time_mean_ms", "step_comm_time_mean_ms", "comm_ratio"]

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
        writer.writeheader()
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


def run_dist(rank, world_size, device, x_all, y_all, warmup_iter=5, repeate_iter=10):
    try:
        setup(rank, world_size, device)
        step = batch_size // world_size
        x = x_all[rank * step : (rank + 1) * step, :].to(device)
        y = y_all[rank * step : (rank + 1) * step, :].to(device)
        model = BasicsTransformerLM(vocab_size, context_length, **model_zoo[model_size]).to(device)
        ddp_model = FlatDDP(model)
        ddp_optimizer = AdamW(ddp_model.parameters())

        for _ in range(warmup_iter):
            synchronize(device)
            ddp_optimizer.zero_grad()
            logits = ddp_model(x)
            loss = cross_entropy(logits, y)
            loss.backward()
            ddp_model.finish_gradient_synchronization(ddp_optimizer)
            ddp_optimizer.step()

        step_times = []
        step_comm_times = []

        for _ in range(repeate_iter):
            ddp_optimizer.zero_grad()
            synchronize(device)
            start_time = timeit.default_timer()
            logits = ddp_model(x)
            loss = cross_entropy(logits, y)
            loss.backward()
            synchronize(device)
            comm_start_time = timeit.default_timer()
            ddp_model.finish_gradient_synchronization(ddp_optimizer)
            synchronize(device)
            comm_end_time = timeit.default_timer() - comm_start_time
            step_comm_times.append(comm_end_time)
            ddp_optimizer.step()
            synchronize(device)
            end_time = timeit.default_timer()
            step_times.append(end_time - start_time)

        step_times_mean = statistics.mean(step_times)
        step_comm_times_mean = statistics.mean(step_comm_times)

        result_item = {
            "device_name": torch.cuda.get_device_name() if device == "cuda" else "cpu",
            "rank": rank,
            "world_size": world_size,
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


def benchmark(device, world_size=2):
    assert batch_size % world_size == 0, "Batch size must be divisible by world size"
    if device == "cuda":
        assert world_size <= torch.cuda.device_count(), "Not enough GPUs"
    x = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long)
    spawn(run_dist, args=(world_size, device, x, y), nprocs=world_size, join=True)


if __name__ == "__main__":
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_zoo ", "Batch Size", "Context Length", "Vocab Size"])
        writer.writerow([model_size, batch_size, context_length, vocab_size])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    benchmark(device, world_size=2)
