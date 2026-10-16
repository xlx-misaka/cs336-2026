import csv
from pathlib import Path

import torch

from cs336_basics.model import RotaryEmbedding, TransformerBlock
from torch.utils.checkpoint import checkpoint

_OUTPUT_DIR = Path(__file__).resolve().parent

d_model = 768
d_ff = 3072
num_heads = 12
num_layers = 12

context_length = 512
batch_size = 4

device = torch.device("cuda")


def k_blocks(blocks, x, i, k):
    for j in range(i, i + k):
        x = blocks[j](x)
    return x


def half_block(blocks, x, i):

    def attn_only(block, x):
        return x + block.attn(block.ln1(x))

    def ffn_only(block, x):
        return x + block.ffn(block.ln2(x))

    b = blocks[i]
    x = checkpoint(attn_only, b, x, use_reentrant=False)
    x = checkpoint(ffn_only, b, x, use_reentrant=False)
    return x


def run_model_checkpoint(blocks, x, n, k):

    if k == 0.5:
        for i in range(n):
            x = half_block(blocks, x, i)
    else:
        i = 0
        while i < n:
            x = checkpoint(k_blocks, blocks, x, i, k, use_reentrant=False)
            i += k
    return x


def benchmark(n, k):
    """Measure forward/backward peak allocation without saved-tensor hooks."""

    assert k == 0.5 or n % k == 0, "n 必须是 k 的倍数"

    blocks = torch.nn.ModuleList(
        TransformerBlock(
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            positional_encoder=RotaryEmbedding(
                dim=d_model // num_heads,
                context_length=context_length,
            ),
        ).to(device)
        for _ in range(n)
    )

    x = torch.randn(
        (batch_size, context_length, d_model),
        device=device,
        requires_grad=True,
    )

    optim = torch.optim.AdamW(blocks.parameters())

    warmup_iter = 5
    for _ in range(warmup_iter):
        logits = run_model_checkpoint(blocks, x, n, k)
        assert logits is not None
        logits.mean().backward()
        optim.zero_grad(set_to_none=True)
        x.grad = None
        del logits

    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)

    logits = run_model_checkpoint(blocks, x, n, k)
    assert logits is not None
    logits.mean().backward()

    torch.cuda.synchronize(device)

    peak_mem_mib = torch.cuda.max_memory_allocated(device) / 1024**2

    del logits
    optim.zero_grad(set_to_none=True)
    x.grad = None

    print("=" * 80)
    print(f"n = {n}, k = {k}: {peak_mem_mib:.2f} MiB")

    return {"num_layers": n, "k": k, "peak_mem_MiB": peak_mem_mib}


def _save_results_csv(results):
    res_path = _OUTPUT_DIR / "benchmark_checkpoint.csv"

    with open(res_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "d_model",
                "d_ff",
                "num_heads",
                "num_layers",
                "context_length",
                "batch_size",
            ]
        )
        writer.writerow(
            [
                d_model,
                d_ff,
                num_heads,
                num_layers,
                context_length,
                batch_size,
            ]
        )

        writer.writerow([])

        writer.writerow(
            [
                "k",
                "peak_mem_MiB",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result["k"],
                    result["peak_mem_MiB"],
                ]
            )

    print(f"Saved results to {res_path}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能运行")

    results = []

    for k in [0.5, 1, 2, 3, 4, 6]:
        results.append(benchmark(num_layers, k))

    _save_results_csv(results)

    print("=" * 80)


# TODO: 还需租显卡完成xl配置，要求48GB
