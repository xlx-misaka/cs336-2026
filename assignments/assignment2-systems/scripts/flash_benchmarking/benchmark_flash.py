import csv
from pathlib import Path

import torch
import triton.testing as tt
from cs336_basics.model import scaled_dot_product_attention

from cs336_systems.flash_attn import TritonFlashAttention

_OUTPUT_DIR = Path(__file__).resolve().parent


def benchmark(mode, sequence_length, d_model, dtype):

    Q = torch.randn(1, sequence_length, d_model, dtype=dtype, requires_grad=True, device="cuda")
    K = torch.randn(1, sequence_length, d_model, dtype=dtype, requires_grad=True, device="cuda")
    V = torch.randn(1, sequence_length, d_model, dtype=dtype, requires_grad=True, device="cuda")

    attn = None
    if mode == "pytorch":
        i = torch.arange(sequence_length, device="cuda")
        j = torch.arange(sequence_length, device="cuda")
        mask = i[:, None] >= j[None, :]
        attn = lambda Q, K, V: scaled_dot_product_attention(Q, K, V, mask)  # noqa: E731
    else:
        attn = lambda Q, K, V: TritonFlashAttention.apply(Q, K, V, True)  # noqa: E731

    O = attn(Q, K, V)  # noqa: E741

    def run_fwd():
        attn(Q, K, V)

    def run_bwd():
        O.sum().backward(retain_graph=True)

    def run_fwd_bwd():
        attn(Q, K, V).sum().backward()

    fwd_mean_ms = tt.do_bench(run_fwd)
    bwd_mean_ms = tt.do_bench(run_bwd, grad_to_none=[Q, K, V])
    fwd_bwd_mean_ms = tt.do_bench(run_fwd_bwd, grad_to_none=[Q, K, V])

    return {
        "mode": mode,
        "sequence_length": sequence_length,
        "d_model": d_model,
        "dtype": dtype,
        "fwd_mean_ms": fwd_mean_ms,
        "bwd_mean_ms": bwd_mean_ms,
        "fwd_bwd_mean_ms": fwd_bwd_mean_ms,
        "status": "ok",
    }


def _save_results_csv(results):
    res_path = _OUTPUT_DIR / f"{torch.cuda.get_device_name()}_benchmark_flash.csv"

    filednames = [
        "mode",
        "sequence_length",
        "d_model",
        "dtype",
        "fwd_mean_ms",
        "bwd_mean_ms",
        "fwd_bwd_mean_ms",
        "status",
    ]

    with open(file=res_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=filednames)
        writer.writeheader()
        writer.writerows(results)

    print("=" * 80)
    print(f"Saved results to {res_path}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    results = []

    for mode in ["pytorch", "triton"]:
        for sequence_length in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
            for d_model in [16, 32, 64, 128]:
                for dtype in [torch.bfloat16, torch.float32]:
                    try:
                        result = benchmark(mode, sequence_length, d_model, dtype)
                    except torch.cuda.OutOfMemoryError:
                        result = {
                            "mode": mode,
                            "sequence_length": sequence_length,
                            "d_model": d_model,
                            "dtype": dtype,
                            "fwd_mean_ms": "",
                            "bwd_mean_ms": "",
                            "fwd_bwd_mean_ms": "",
                            "status": "oom",
                        }
                    results.append(result)

    _save_results_csv(results)
