"""Benchmark eager and compiled attention in one run, with 100 passes each.

显存估算：默认未编译 FP32 最大配置约需 49 GiB（由中等规模实测外推），编译版峰值另计。
AutoDL 推荐：单张 A800 80GB（官网参考价约 5.59 元/时），给默认矩阵及运行时留出余量。
省钱试跑可选 RTX 4090 24GB（约 1.88 元/时），较大配置 OOM 会正常记录；48GB 不保证跑通最大配置。
参考价于 2026-10-21 检索自 https://www.autodl.com/home ，库存和成交价以算力市场为准。
"""

import csv
import gc
import statistics
import time
from itertools import product
from pathlib import Path

import torch
from cs336_basics.model import scaled_dot_product_attention

D_MODELS = [16, 32, 64, 128]
CONTEXT_LENGTHS = [256, 1024, 4096, 8192, 16384]
COMPILE_MODES = [False, True]
WARMUP = 5
STEPS = 100


def benchmark(d_model, context_length, compiled=False):
    torch.manual_seed(0)
    inputs = [torch.randn(8, context_length, d_model, device="cuda", requires_grad=True) for _ in range(3)]
    upstream = torch.randn_like(inputs[0])
    positions = torch.arange(context_length, device="cuda")
    mask = positions[:, None] >= positions[None, :]
    attention = torch.compile(scaled_dot_product_attention) if compiled else scaled_dot_product_attention

    for _ in range(WARMUP):
        output = attention(inputs[0], inputs[1], inputs[2], mask)
        output.backward(upstream)
        for tensor in inputs:
            tensor.grad = None
        del output
    torch.cuda.synchronize()

    forward_times, backward_times, before_backward_memory = [], [], []
    for _ in range(STEPS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = attention(inputs[0], inputs[1], inputs[2], mask)
        torch.cuda.synchronize()
        forward_times.append(time.perf_counter() - start)
        before_backward_memory.append(torch.cuda.memory_allocated())

        start = time.perf_counter()
        output.backward(upstream)
        torch.cuda.synchronize()
        backward_times.append(time.perf_counter() - start)
        for tensor in inputs:
            tensor.grad = None
        del output

    return {
        "forward_ms": statistics.mean(forward_times) * 1000,
        "backward_ms": statistics.mean(backward_times) * 1000,
        "before_backward_MiB": statistics.mean(before_backward_memory) / 1024**2,
        "status": "ok",
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    gpu = torch.cuda.get_device_name()
    path = Path(__file__).with_name(f"{gpu}_benchmark_attn.csv")
    fields = ["gpu", "compiled", "d_model", "context_length", "forward_ms", "backward_ms", "before_backward_MiB", "status"]
    print(f"GPU: {gpu}\nResults: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for compiled, dim, length in product(COMPILE_MODES, D_MODELS, CONTEXT_LENGTHS):
            row = {"gpu": gpu, "compiled": compiled, "d_model": dim, "context_length": length}
            try:
                row.update(benchmark(dim, length, compiled))
            except torch.cuda.OutOfMemoryError:
                row.update(forward_ms="OOM", backward_ms="OOM", before_backward_MiB="OOM", status="oom")
            gc.collect()
            torch.cuda.empty_cache()
            writer.writerow(row)
            file.flush()
            print(f"compiled={compiled} D={dim} N={length}: {row['status']}", flush=True)


if __name__ == "__main__":
    main()
