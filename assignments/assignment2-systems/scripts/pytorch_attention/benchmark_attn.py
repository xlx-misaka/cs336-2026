"""Time eager or compiled causal PyTorch attention and measure CUDA memory."""

import argparse
import csv
import gc
import statistics
import time
from pathlib import Path

import torch
from cs336_basics.model import scaled_dot_product_attention

_OUTPUT_DIR = Path(__file__).resolve().parent
_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
_FIELDS = [
    "gpu",
    "compiled",
    "dtype",
    "causal",
    "d_model",
    "context_length",
    "batch_size",
    "warmup",
    "steps",
    "forward_samples",
    "backward_samples",
    "forward_ms",
    "backward_ms",
    "baseline_MiB",
    "before_backward_MiB",
    "forward_live_increment_MiB",
    "forward_peak_MiB",
    "backward_phase_peak_MiB",
    "status",
    "oom_stage",
]


def _mean(values, scale: float = 1.0):
    return statistics.mean(values) * scale if values else ""


def benchmark(d_model, context_length, batch_size=8, torch_compile=False, *, dtype=torch.float32, warmup=5, steps=100):
    if min(d_model, context_length, batch_size, steps) <= 0 or warmup < 0:
        raise ValueError("Sizes and steps must be positive; warmup must be nonnegative")
    if dtype not in _DTYPES.values():
        raise ValueError("dtype must be float32 or bfloat16")
    if torch_compile and warmup == 0:
        raise ValueError("Compiled benchmarks need at least one warmup iteration")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    result = {
        "gpu": torch.cuda.get_device_name(),
        "compiled": torch_compile,
        "causal": True,
        "dtype": str(dtype).removeprefix("torch."),
        "d_model": d_model,
        "context_length": context_length,
        "batch_size": batch_size,
        "warmup": warmup,
        "steps": steps,
        "status": "ok",
        "oom_stage": "",
    }
    inputs = []
    upstream = positions = mask = output = attention = None
    forward_times, backward_times = [], []
    baselines, live_memory, live_increments, forward_peaks, backward_peaks = [], [], [], [], []
    stage = "inputs"
    try:
        generator = torch.Generator(device="cuda").manual_seed(0)
        inputs = [torch.randn(batch_size, context_length, d_model, device="cuda", dtype=dtype, generator=generator).requires_grad_() for _ in range(3)]
        upstream = torch.randn(inputs[0].shape, device="cuda", dtype=dtype, generator=generator)
        positions = torch.arange(context_length, device="cuda")
        mask = positions[:, None] >= positions[None, :]
        attention = torch.compile(scaled_dot_product_attention) if torch_compile else scaled_dot_product_attention

        for _ in range(warmup):
            stage = "warmup_forward"
            output = attention(inputs[0], inputs[1], inputs[2], mask)
            stage = "warmup_backward"
            output.backward(upstream)
            for tensor in inputs:
                tensor.grad = None
            output = None
        torch.cuda.synchronize()

        for _ in range(steps):
            # Synchronize before starting the host timer as well as after each pass.
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            stage = "forward"
            start = time.perf_counter()
            output = attention(inputs[0], inputs[1], inputs[2], mask)
            torch.cuda.synchronize()
            forward_times.append(time.perf_counter() - start)
            baselines.append(baseline)
            live_memory.append(torch.cuda.memory_allocated())
            live_increments.append(live_memory[-1] - baseline)
            forward_peaks.append(torch.cuda.max_memory_allocated())

            torch.cuda.reset_peak_memory_stats()
            stage = "backward"
            start = time.perf_counter()
            output.backward(upstream)
            torch.cuda.synchronize()
            backward_times.append(time.perf_counter() - start)
            backward_peaks.append(torch.cuda.max_memory_allocated())
            for tensor in inputs:
                tensor.grad = None
            output = None
    except torch.cuda.OutOfMemoryError:
        result.update(status="oom", oom_stage=stage)
    finally:
        output = upstream = positions = mask = attention = None
        inputs.clear()
        tensor = None
    # An OOM in backward does not discard successfully measured forward passes.
    result.update(
        forward_samples=len(forward_times),
        backward_samples=len(backward_times),
        forward_ms=_mean(forward_times, 1000),
        backward_ms=_mean(backward_times, 1000),
        baseline_MiB=_mean(baselines, 1 / 1024**2),
        before_backward_MiB=_mean(live_memory, 1 / 1024**2),
        forward_live_increment_MiB=_mean(live_increments, 1 / 1024**2),
        forward_peak_MiB=_mean(forward_peaks, 1 / 1024**2),
        backward_phase_peak_MiB=_mean(backward_peaks, 1 / 1024**2),
    )
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for forward and backward")
    parser.add_argument("--d-models", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[256, 1024, 4096, 8192, 16384])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=_DTYPES, default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(*args.d_models, *args.context_lengths, args.batch_size, args.steps) <= 0 or args.warmup < 0:
        parser.error("Sizes and steps must be positive; warmup must be nonnegative")
    if args.compile and args.warmup == 0:
        parser.error("Compiled benchmarks need at least one warmup iteration to exclude compilation")
    return args


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    gpu = torch.cuda.get_device_name()
    suffix = "_compile" if args.compile else ""
    path = args.output or _OUTPUT_DIR / f"{gpu}_benchmark_attn{suffix}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {gpu}\nResults: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_FIELDS)
        writer.writeheader()
        file.flush()
        for dim in args.d_models:
            for length in args.context_lengths:
                result = benchmark(
                    dim,
                    length,
                    args.batch_size,
                    args.compile,
                    dtype=_DTYPES[args.dtype],
                    warmup=args.warmup,
                    steps=args.steps,
                )
                writer.writerow(result)
                file.flush()
                if result["status"] == "ok":
                    print(
                        f"D={dim} N={length}: fwd={result['forward_ms']:.3f}ms bwd={result['backward_ms']:.3f}ms before_bwd={result['before_backward_MiB']:.2f}MiB",
                        flush=True,
                    )
                else:
                    print(f"D={dim} N={length}: OOM during {result['oom_stage']}", flush=True)


if __name__ == "__main__":
    main()
