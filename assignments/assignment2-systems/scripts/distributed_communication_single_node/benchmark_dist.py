"""Single-node all-reduce benchmark.
AutoDL 租卡建议（2026-10-24 查询的官网参考价）：
首选同一实例内 6 × RTX 3090 24GB：单卡约 1.32 元/时，6 卡约 7.92 元/时。
备选同一实例内 6 × RTX 4090 24GB：单卡约 1.88 元/时，6 卡约 11.28 元/时。
同一实例的 GPU 应位于同一物理主机；分别使用其中 2、4、6 张卡完成对比。
每卡最大输入约 1GB，另需 CUDA / NCCL 缓冲；24GB 显存足够用于本实验。
优先比较同机库存、租价和卡间互连条件，并在报告中记录 GPU 型号及 PCIe / NVLink 拓扑。
先在本机用 CPU 调通流程，租机后先测 2 卡、1MB，并确认主机驱动兼容项目的 PyTorch / CUDA 环境。
参考：https://www.autodl.com/ ；实际价格、折扣及库存以算力市场为准，保存结果后及时关机。
"""

import csv
import itertools
import os
from pathlib import Path
import statistics
import timeit
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn

path = Path(__file__).with_name("benchmark_dist.csv")
FILEDS = ["device_name", "world_size", "data_size_MB", "rank", "time_ms"]


def synchronize(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def setup(rank, device, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl" if device == "cuda" else "gloo", rank=rank, world_size=world_size)
    if device == "cuda":
        torch.cuda.set_device(rank)


def save_results(results):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FILEDS)
        writer.writerows(results)


def distributed_demo(rank, device, world_size, data_size_MB, warmup_iter, repeate_iter):
    try:
        setup(rank, device, world_size)
        data = torch.randn(data_size_MB * 10**6 // 4, dtype=torch.float32, device=device)

        for _ in range(warmup_iter):
            dist.all_reduce(data, async_op=False)
            synchronize(device)

        cost_time_list = []
        for _ in range(repeate_iter):
            start_time = timeit.default_timer()
            dist.all_reduce(data, async_op=False)
            synchronize(device)
            end_time = timeit.default_timer()
            cost_time_list.append(end_time - start_time)

        cost_time_mean = statistics.mean(cost_time_list)
        result_item = {
            "device_name": torch.cuda.get_device_name() if device == "cuda" else "cpu",
            "world_size": world_size,
            "data_size_MB": data_size_MB,
            "rank": rank,
            "time_ms": cost_time_mean * 1000,
        }

        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(result_item, gathered, dst=0)

        if rank == 0:
            save_results(gathered)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def benchmark_dist():
    data_sizes_MB = [1, 10, 100, 1000]
    num_processes = [2, 4, 6]
    device = "cuda"
    warmup_iter = 5
    repeate_iter = 10

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FILEDS)
        writer.writeheader()

    for data_size_MB, num_process in itertools.product(data_sizes_MB, num_processes):
        if device == "cuda" and torch.cuda.device_count() < num_process:
            continue
        spawn(fn=distributed_demo, args=(device, num_process, data_size_MB, warmup_iter, repeate_iter), nprocs=num_process, join=True)

    print("Benchmark completed.")


if __name__ == "__main__":
    benchmark_dist()
