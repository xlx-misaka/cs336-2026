r"""Benchmark OverLap DDP training and gradient synchronization on one node with two GPUs.

默认配置：单节点、2 GPU、xl、FP32、AdamW；全局 batch=4，每卡 batch=2，序列长 512，词表 10000。
每种实现先预热 5 步，再运行 10 个正式迭代。

AutoDL 租卡建议：
- 在同一实例中选择 2 张相同 GPU：可选 2 x A800 80GB；需要更多显存余量时可选 2 x PRO 6000 96GB。
  价格、库存及分配的主机内存以租用页面为准：https://www.autodl.com/home
- 每卡约 34.1 亿参数；FP32 参数约 12.7 GiB，参数、梯度和 AdamW 的 m/v 合计约 50.8 GiB。
  激活、通信缓冲和优化器临时张量还会增加显存占用；80/96GB 是选型建议，完整 xl 双卡尚未实测。
- DDP 每卡保存完整模型和优化器状态，两张卡的显存不能合并；24GB、32GB、48GB 单卡不适合默认配置。
- 模型先在 CPU 创建，建议实例配备至少 64GB 主机内存，并为报告预留磁盘空间。
- 比较三种实现时使用相同的卡型、互联拓扑、精度和 batch 配置，记录 nvidia-smi topo -m 的结果。
  单机多卡说明：https://www.autodl.com/docs/distributed_training/

环境准备（以下命令均在仓库 assignment2-systems 根目录执行）：
    uv sync
    nvidia-smi
    nvidia-smi topo -m

直接运行（脚本内部通过 spawn 创建两个进程，无需再用 torchrun）：
    CUDA_VISIBLE_DEVICES=0,1 uv run python -m scripts.ddp_overlap_individual_parameters_benchmarking.benchmark_overlap_ddp

Nsight Systems 采集（服务器需有 nsys；可从 NVIDIA 官方安装 CLI）：
    nsys --version
    mkdir -p profiles
    CUDA_VISIBLE_DEVICES=0,1 uv run nsys profile \
        --trace=cuda,nvtx \
        --pytorch=autograd-nvtx \
        --sample=none --cpuctxsw=none \
        --output=profiles/ddp_compare \
        python -m scripts.ddp_overlap_individual_parameters_benchmarking.benchmark_overlap_ddp

当前代码没有调用 profiler start/stop，使用全程采集，不添加 --capture-range=cudaProfilerApi。
采集后将 profiles/ddp_compare.nsys-rep 下载到本机，用 Nsight Systems GUI 的 File > Open 打开。
GUI 版本应与采集端相同或更新；profiler 会引入开销，对比时保持相同采集设置。
安装与使用：https://developer.nvidia.com/nsight-systems/get-started
https://docs.nvidia.com/nsight-systems/UserGuide/index.html

脚本依次运行 naive、flat、overlap；同一报告中的 NVTX 标签包含 rank、实现类型和训练阶段。
此脚本的 benchmark_dist.csv 当前只写配置，不写迭代耗时；性能和重叠情况从 Nsight 报告查看。
NVTX train 是 CPU 范围，当前边界未做 GPU 同步；不要直接将其长度当作 GPU 完整迭代耗时。
截图应选取预热后的训练迭代，对比同一 GPU 上反向计算 kernel 和 NCCL kernel 的执行区间。
"""

import csv
import os
from pathlib import Path

from torch.multiprocessing.spawn import spawn
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp import NaiveDDP, OverLapDDP, FlatDDP
import torch
import torch.distributed as dist


path = Path(__file__).with_name("benchmark_dist.csv")
FILEDS = ["ddp_type", "device_name", "rank", "world_size", "step_time_mean_ms", "step_comm_time_mean_ms", "comm_ratio"]

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


def setup(rank, world_size, device):
    if device == "cuda":
        torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl" if device == "cuda" else "gloo", rank=rank, world_size=world_size)


def run_dist(rank, ddp_type, world_size, device, x_all, y_all, warmup_iter=5, repeate_iter=10):
    try:
        setup(rank, world_size, device)
        step = batch_size // world_size
        x = x_all[rank * step : (rank + 1) * step, :].to(device)
        y = y_all[rank * step : (rank + 1) * step, :].to(device)
        model = BasicsTransformerLM(vocab_size, context_length, **model_zoo[model_size]).to(device)
        if ddp_type == "naive":
            ddp_model = NaiveDDP(model)
        elif ddp_type == "flat":
            ddp_model = FlatDDP(model)
        else:
            ddp_model = OverLapDDP(model)
        ddp_optimizer = AdamW(ddp_model.parameters())

        for _ in range(warmup_iter):
            ddp_optimizer.zero_grad()
            logits = ddp_model(x)
            loss = cross_entropy(logits, y)
            loss.backward()
            ddp_model.finish_gradient_synchronization()
            ddp_optimizer.step()

        for _ in range(repeate_iter):
            ddp_optimizer.zero_grad()
            with torch.cuda.nvtx.range(f"rank{rank}/{ddp_type}/train"):
                with torch.cuda.nvtx.range(f"rank{rank}/{ddp_type}/forward"):
                    logits = ddp_model(x)
                loss = cross_entropy(logits, y)

                with torch.cuda.nvtx.range(f"rank{rank}/{ddp_type}/backward"):
                    loss.backward()

                with torch.cuda.nvtx.range(f"rank{rank}/{ddp_type}/wait_gradients"):
                    ddp_model.finish_gradient_synchronization()

                ddp_optimizer.step()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def benchmark(ddp_type, device, world_size=2):
    assert batch_size % world_size == 0, "Batch size must be divisible by world size"
    if device == "cuda":
        assert world_size <= torch.cuda.device_count(), "Not enough GPUs"
    x = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long)
    spawn(run_dist, args=(ddp_type, world_size, device, x, y), nprocs=world_size, join=True)


if __name__ == "__main__":
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_zoo ", "Batch Size", "Context Length", "Vocab Size"])
        writer.writerow([model_size, batch_size, context_length, vocab_size])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for ddp_type in ["naive", "flat", "overlap"]:
        benchmark(ddp_type, device, world_size=2)
