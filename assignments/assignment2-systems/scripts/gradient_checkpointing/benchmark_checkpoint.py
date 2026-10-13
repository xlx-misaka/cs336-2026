import torch

from cs336_basics.model import RotaryEmbedding, TransformerBlock
from torch.utils.checkpoint import checkpoint

n = 12

d_model = 768
d_ff = 3072
num_heads = 12

context_length = 512
batch_size = 4

device = torch.device("cuda")


def k_block(n, k):

    assert n % k == 0, "n 必须是 k 的倍数"

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

    def model_checkpoint(x):
        i = 0
        while i < n:
            x = checkpoint(k_blocks, x, i, use_reentrant=False)
            i += k
        return x

    optim = torch.optim.AdamW(blocks.parameters())

    warmup_iter = 5
    for _ in range(warmup_iter):
        logits = model_checkpoint(x)
        assert logits is not None
        logits.mean().backward()
        optim.zero_grad()
        del logits

    x.grad = None
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    logits = model_checkpoint(x)
    assert logits is not None
    logits.mean().backward()

    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated()
    print(f"n = {n}, k = {k} :{mem / 1024**2:.2f} MiB")


def half_block(n):

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

    def attn_only(block, x):
        return x + block.attn(block.ln1(x))

    def ffn_only(block, x):
        return x + block.ffn(block.ln2(x))

    def half_block(x, i):
        b = blocks[i]
        x = checkpoint(attn_only, b, x, use_reentrant=False)
        x = checkpoint(ffn_only, b, x, use_reentrant=False)
        return x

    def model_checkpoint(x):
        for i in range(n):
            x = half_block(x, i)
        return x

    optim = torch.optim.AdamW(blocks.parameters())

    warmup_iter = 5
    for _ in range(warmup_iter):
        logits = model_checkpoint(x)
        assert logits is not None
        logits.mean().backward()
        optim.zero_grad()
        del logits

    x.grad = None
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    logits = model_checkpoint(x)
    assert logits is not None
    logits.mean().backward()

    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated()
    print(f"n = {n}, k = {0.5} :{mem / 1024**2:.2f} MiB")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能运行")

    half_block(n)

    for k in [1, 2, 3]:
        k_block(n, k=k)
