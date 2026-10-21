"""Compare causal PyTorch and Triton attention on one CUDA device."""

import argparse
import csv
import gc
from functools import partial
from pathlib import Path

import torch
import triton.testing as tt
from cs336_basics.model import scaled_dot_product_attention

from cs336_systems import flash_attn

_OUTPUT_DIR = Path(__file__).resolve().parent
_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
_PHASES = ("fwd", "bwd", "fwd_bwd")
_FIELDS = [
    "gpu",
    "mode",
    "batch_size",
    "causal",
    "sequence_length",
    "d_model",
    "dtype",
    "q_tile_size",
    "k_tile_size",
    "warmup_ms",
    "rep_ms",
    "fwd_mean_ms",
    "bwd_mean_ms",
    "fwd_bwd_mean_ms",
    "fwd_status",
    "bwd_status",
    "fwd_bwd_status",
    "status",
    "oom_stage",
]


def _release_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()


def _triton_attention(q, k, v):
    return flash_attn.TritonFlashAttention.apply(q, k, v, True)


def _forward_backward(attention, inputs, upstream):
    attention(*inputs).backward(upstream)


def _measure(attention, inputs, upstream, phase, warmup_ms, rep_ms):
    """Keep a saved graph alive only for the backward-only measurement."""
    output = run = None
    try:
        if phase == "bwd":
            output = attention(*inputs)
            run = partial(output.backward, upstream, retain_graph=True)
        elif phase == "fwd":
            run = partial(attention, *inputs)
        else:
            run = partial(_forward_backward, attention, inputs, upstream)

        return tt.do_bench(
            run,
            warmup=warmup_ms,
            rep=rep_ms,
            grad_to_none=inputs,
            return_mode="mean",
        )
    finally:
        output = run = None
        for tensor in inputs:
            tensor.grad = None


def benchmark(mode, sequence_length, d_model, dtype, *, warmup_ms=25, rep_ms=100):
    if mode not in ("pytorch", "triton"):
        raise ValueError("mode must be pytorch or triton")
    if sequence_length <= 0 or d_model <= 0 or warmup_ms < 0 or rep_ms <= 0:
        raise ValueError("Dimensions and rep_ms must be positive; warmup_ms must be nonnegative")
    if dtype not in _DTYPES.values():
        raise ValueError("dtype must be bfloat16 or float32")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    result = {
        "gpu": torch.cuda.get_device_name(),
        "mode": mode,
        "batch_size": 1,
        "causal": True,
        "sequence_length": sequence_length,
        "d_model": d_model,
        "dtype": str(dtype).removeprefix("torch."),
        "q_tile_size": flash_attn.B_q if mode == "triton" else "",
        "k_tile_size": flash_attn.B_k if mode == "triton" else "",
        "warmup_ms": warmup_ms,
        "rep_ms": rep_ms,
        "status": "ok",
        "oom_stage": "",
    }
    for phase in _PHASES:
        result[f"{phase}_mean_ms"] = ""
        result[f"{phase}_status"] = "not_run"

    inputs = []
    upstream = positions = mask = attention = None
    try:
        # Identical random inputs for both implementations of a configuration.
        generator = torch.Generator(device="cuda").manual_seed(0)
        inputs = [torch.randn(1, sequence_length, d_model, dtype=dtype, device="cuda", generator=generator).requires_grad_() for _ in range(3)]
        upstream = torch.randn(inputs[0].shape, dtype=dtype, device="cuda", generator=generator)
        if mode == "pytorch":
            positions = torch.arange(sequence_length, device="cuda")
            mask = positions[:, None] >= positions[None, :]

            attention = partial(scaled_dot_product_attention, mask=mask)
        else:
            attention = _triton_attention
    except torch.cuda.OutOfMemoryError:
        result.update(status="oom", oom_stage="inputs")

    if result["status"] == "ok":
        for phase in _PHASES:
            try:
                result[f"{phase}_mean_ms"] = _measure(attention, inputs, upstream, phase, warmup_ms, rep_ms)
                result[f"{phase}_status"] = "ok"
            except torch.cuda.OutOfMemoryError:
                result[f"{phase}_status"] = "oom"
                result["status"] = "oom"
                result["oom_stage"] = ",".join(filter(None, (result["oom_stage"], phase)))
            _release_cuda_memory()

    inputs.clear()
    upstream = positions = mask = attention = None
    _release_cuda_memory()
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=("pytorch", "triton"), default=["pytorch", "triton"])
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[2**i for i in range(7, 17)])
    parser.add_argument("--d-models", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--dtypes", nargs="+", choices=_DTYPES, default=list(_DTYPES))
    parser.add_argument("--q-tile-size", type=int, default=flash_attn.B_q)
    parser.add_argument("--k-tile-size", type=int, default=flash_attn.B_k)
    parser.add_argument("--warmup-ms", type=float, default=25)
    parser.add_argument("--rep-ms", type=float, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(value <= 0 for value in args.sequence_lengths + args.d_models) or args.warmup_ms < 0 or args.rep_ms <= 0:
        parser.error("Dimensions and rep-ms must be positive; warmup-ms must be nonnegative")
    for tile in (args.q_tile_size, args.k_tile_size):
        if tile < 16 or tile & (tile - 1):
            parser.error("Tile sizes must be powers of two and at least 16")
    if "triton" in args.modes and min(args.sequence_lengths) < max(args.q_tile_size, args.k_tile_size):
        parser.error("Triton sequence lengths must be at least both tile sizes")
    if "triton" in args.modes and any(dim < 16 or dim & (dim - 1) for dim in args.d_models):
        parser.error("Triton dimensions must be powers of two and at least 16")
    return args


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    flash_attn.B_q, flash_attn.B_k = args.q_tile_size, args.k_tile_size
    gpu = torch.cuda.get_device_name()
    path = args.output or _OUTPUT_DIR / f"{gpu}_benchmark_flash.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {gpu}\nResults: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_FIELDS)
        writer.writeheader()
        file.flush()
        for mode in args.modes:
            for length in args.sequence_lengths:
                for dim in args.d_models:
                    for dtype in args.dtypes:
                        result = benchmark(mode, length, dim, _DTYPES[dtype], warmup_ms=args.warmup_ms, rep_ms=args.rep_ms)
                        writer.writerow(result)
                        file.flush()
                        times = []
                        for phase in _PHASES:
                            value = result[f"{phase}_mean_ms"]
                            times.append(f"{phase}={value:.3f}ms" if isinstance(value, (int, float)) else f"{phase}={result[f'{phase}_status']}")
                        note = f" OOM during {result['oom_stage']}" if result["status"] == "oom" else ""
                        print(f"{mode} N={length} D={dim} {dtype}: {' '.join(times)}{note}", flush=True)


if __name__ == "__main__":
    main()
