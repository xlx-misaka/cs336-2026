import torch

from cs336_basics.model import RotaryEmbedding, TransformerBlock
from torch.utils.checkpoint import checkpoint


def benchmark(n=32, k=1):
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能运行")

    device = torch.device("cuda")

    assert n % k == 0, "n 必须是 k 的倍数"

    d_model = 2560
    d_ff = 10240
    num_heads = 32

    context_length = 2048
    batch_size = 4

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

    def k_blocks(x, i):
        for j in range(i, i + k):
            x = blocks[j](x)
        return x

    def blocks_checkpoint(x):
        i = 0
        while i < n:
            x = checkpoint(k_blocks, x, i, use_reentrant=False)
            i += k
        return x

    optim = torch.optim.AdamW(blocks.parameters())

    warmup_iter = 5
    for _ in range(warmup_iter):
        logits = blocks_checkpoint(x)
        assert logits is not None
        logits.mean().backward()
        optim.zero_grad()
        del logits

    x.grad = None
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    logits = blocks_checkpoint(x)
    assert logits is not None
    logits.mean().backward()

    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated()
    print(f"n = {n}, k = {k} :{mem / 1024**3:.2f} GiB")


if __name__ == "__main__":
    for k in range(1, 16):
        if 32 % k != 0:
            continue
        benchmark(n=32, k=k)
