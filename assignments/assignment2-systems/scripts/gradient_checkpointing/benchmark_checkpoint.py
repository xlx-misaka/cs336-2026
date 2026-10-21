"""Measure total peak CUDA memory for nonnested checkpoint granularities."""

import argparse
import csv
import gc
from pathlib import Path

import torch
from cs336_basics.model import RotaryEmbedding, TransformerBlock
from torch.utils.checkpoint import checkpoint

_OUTPUT_DIR = Path(__file__).resolve().parent
_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
_FIELDS = [
    "gpu",
    "d_model",
    "d_ff",
    "num_heads",
    "num_layers",
    "context_length",
    "batch_size",
    "dtype",
    "k",
    "warmup",
    "steps",
    "baseline_MiB",
    "peak_mem_MiB",
    "peak_increment_MiB",
    "status",
    "oom_stage",
]


def k_blocks(blocks, x, i, k):
    for j in range(i, i + k):
        x = blocks[j](x)
    return x


def half_block(blocks, x, i):
    def attn_only(block, x):
        return x + block.attn(block.ln1(x))

    def ffn_only(block, x):
        return x + block.ffn(block.ln2(x))

    block = blocks[i]
    x = checkpoint(attn_only, block, x, use_reentrant=False)
    return checkpoint(ffn_only, block, x, use_reentrant=False)


def _valid_k(k):
    return k == 0.5 or (k >= 1 and float(k).is_integer())


def run_model_checkpoint(blocks, x, n, k):
    if not _valid_k(k):
        raise ValueError("k must be 0.5 or a positive integer")
    if not 0 < n <= len(blocks):
        raise ValueError("n must be between 1 and the number of blocks")
    if k == 0.5:
        for i in range(n):
            x = half_block(blocks, x, i)
    else:
        k = int(k)
        for i in range(0, n, k):
            # The final segment may contain fewer than k blocks.
            x = checkpoint(k_blocks, blocks, x, i, min(k, n - i), use_reentrant=False)
    return x


def benchmark(
    n,
    k,
    *,
    d_model=2560,
    d_ff=10240,
    num_heads=32,
    context_length=2048,
    batch_size=4,
    dtype=torch.float32,
    warmup=5,
    steps=1,
):
    if not _valid_k(k):
        raise ValueError("k must be 0.5 or a positive integer")
    if min(n, d_model, d_ff, num_heads, context_length, batch_size, steps) <= 0 or warmup < 0:
        raise ValueError("Sizes and steps must be positive; warmup must be nonnegative")
    if d_model % num_heads or (d_model // num_heads) % 2:
        raise ValueError("d_model must be divisible by num_heads with an even head dimension for RoPE")
    if dtype not in _DTYPES.values():
        raise ValueError("dtype must be float32 or bfloat16")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    result = {
        "gpu": torch.cuda.get_device_name(),
        "d_model": d_model,
        "d_ff": d_ff,
        "num_heads": num_heads,
        "num_layers": n,
        "context_length": context_length,
        "batch_size": batch_size,
        "dtype": str(dtype).removeprefix("torch."),
        "k": k,
        "warmup": warmup,
        "steps": steps,
        "baseline_MiB": "",
        "peak_mem_MiB": "",
        "peak_increment_MiB": "",
        "status": "ok",
        "oom_stage": "",
    }
    blocks = x = output = None
    stage = "model"
    try:
        # Reuse the same initialization and input across checkpoint strategies.
        torch.manual_seed(0)
        blocks = torch.nn.ModuleList()
        for _ in range(n):
            blocks.append(
                TransformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    positional_encoder=RotaryEmbedding(dim=d_model // num_heads, context_length=context_length),
                ).to(device="cuda", dtype=dtype)
            )
        stage = "inputs"
        x = torch.randn(batch_size, context_length, d_model, device="cuda", dtype=dtype, requires_grad=True)
        stage = "warmup"
        for _ in range(warmup):
            output = run_model_checkpoint(blocks, x, n, k)
            output.mean().backward()
            blocks.zero_grad(set_to_none=True)
            x.grad = None
            output = None
        torch.cuda.synchronize()

        baseline = torch.cuda.memory_allocated()
        result["baseline_MiB"] = baseline / 1024**2
        peak = baseline
        stage = "measurement"
        for _ in range(steps):
            torch.cuda.reset_peak_memory_stats()
            output = run_model_checkpoint(blocks, x, n, k)
            output.mean().backward()
            torch.cuda.synchronize()
            peak = max(peak, torch.cuda.max_memory_allocated())
            output = None
            blocks.zero_grad(set_to_none=True)
            x.grad = None
            torch.cuda.synchronize()
        result["peak_mem_MiB"] = peak / 1024**2
        # This includes gradients and temporary buffers, not just activations.
        result["peak_increment_MiB"] = (peak - baseline) / 1024**2
    except torch.cuda.OutOfMemoryError:
        result.update(status="oom", oom_stage=stage)
    finally:
        blocks = x = output = None
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=2560)
    parser.add_argument("--d-ff", type=int, default=10240)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=_DTYPES, default="float32")
    parser.add_argument("--k-values", nargs="+", type=float, default=[0.5, 1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.num_layers, args.d_model, args.d_ff, args.num_heads, args.context_length, args.batch_size, args.steps) <= 0 or args.warmup < 0:
        parser.error("Sizes and steps must be positive; warmup must be nonnegative")
    if args.d_model % args.num_heads or (args.d_model // args.num_heads) % 2:
        parser.error("d-model must be divisible by num-heads with an even head dimension for RoPE")
    if not all(_valid_k(k) for k in args.k_values):
        parser.error("k-values must be 0.5 or positive integers")
    return args


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    gpu = torch.cuda.get_device_name()
    path = args.output or _OUTPUT_DIR / f"{gpu}_benchmark_checkpoint.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {gpu}\nResults: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_FIELDS)
        writer.writeheader()
        file.flush()
        for k in args.k_values:
            result = benchmark(
                args.num_layers,
                k,
                d_model=args.d_model,
                d_ff=args.d_ff,
                num_heads=args.num_heads,
                context_length=args.context_length,
                batch_size=args.batch_size,
                dtype=_DTYPES[args.dtype],
                warmup=args.warmup,
                steps=args.steps,
            )
            writer.writerow(result)
            file.flush()
            if result["status"] == "ok":
                print(f"k={k:g}: peak={result['peak_mem_MiB']:.2f} MiB, baseline={result['baseline_MiB']:.2f} MiB", flush=True)
            else:
                print(f"k={k:g}: OOM during {result['oom_stage']}", flush=True)


if __name__ == "__main__":
    main()
