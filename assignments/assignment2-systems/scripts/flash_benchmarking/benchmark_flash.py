"""Run the course FlashAttention benchmark matrix once."""

import csv
import gc
from functools import partial
from itertools import product
from pathlib import Path
from typing import cast

import torch
import triton.testing as tt
from cs336_basics.model import scaled_dot_product_attention

from cs336_systems.flash_attn import TritonFlashAttention

MODES = ["pytorch", "triton"]
SEQUENCE_LENGTHS = [2**i for i in range(7, 17)]
D_MODELS = [16, 32, 64, 128]
DTYPES = [torch.bfloat16, torch.float32]
PHASES = ["fwd", "bwd", "fwd_bwd"]


def triton_attention(q, k, v):
    return TritonFlashAttention.apply(q, k, v, True)


def forward_backward(attention, inputs, upstream):
    attention(*inputs).backward(upstream)


def measure(attention, inputs, upstream, phase) -> float:
    output = run = None
    try:
        if phase == "fwd":
            run = partial(attention, *inputs)
        elif phase == "bwd":
            output = attention(*inputs)
            run = partial(output.backward, upstream, retain_graph=True)
        else:
            run = partial(forward_backward, attention, inputs, upstream)
        return cast(float, tt.do_bench(run, grad_to_none=inputs, return_mode="mean"))
    finally:
        # Only backward-only timing keeps a forward graph between repetitions.
        output = run = None
        for tensor in inputs:
            tensor.grad = None


def benchmark(mode, sequence_length, d_model, dtype):
    torch.manual_seed(0)
    inputs = [torch.randn(1, sequence_length, d_model, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
    upstream = torch.randn_like(inputs[0])
    if mode == "pytorch":
        positions = torch.arange(sequence_length, device="cuda")
        mask = positions[:, None] >= positions[None, :]
        attention = partial(scaled_dot_product_attention, mask=mask)
    else:
        attention = triton_attention

    results: dict[str, str | float] = {"status": "ok"}
    for phase in PHASES:
        try:
            results[f"{phase}_ms"] = measure(attention, inputs, upstream, phase)
        except torch.cuda.OutOfMemoryError:
            results[f"{phase}_ms"] = "OOM"
            results["status"] = "oom"
        gc.collect()
        torch.cuda.empty_cache()
    return results


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    gpu = torch.cuda.get_device_name()
    path = Path(__file__).with_name(f"{gpu}_benchmark_flash.csv")
    fields = ["gpu", "mode", "sequence_length", "d_model", "dtype", "fwd_ms", "bwd_ms", "fwd_bwd_ms", "status"]
    print(f"GPU: {gpu}\nResults: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for mode, length, dim, dtype in product(MODES, SEQUENCE_LENGTHS, D_MODELS, DTYPES):
            row = {"gpu": gpu, "mode": mode, "sequence_length": length, "d_model": dim, "dtype": str(dtype)}
            try:
                row.update(benchmark(mode, length, dim, dtype))
            except torch.cuda.OutOfMemoryError:
                row.update(fwd_ms="OOM", bwd_ms="OOM", fwd_bwd_ms="OOM", status="oom")
            gc.collect()
            torch.cuda.empty_cache()
            writer.writerow(row)
            file.flush()
            print(f"{mode} N={length} D={dim} {dtype}: {row['status']}", flush=True)


if __name__ == "__main__":
    main()
