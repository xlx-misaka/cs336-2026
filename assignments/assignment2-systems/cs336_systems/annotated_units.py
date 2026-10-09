import math

from einops import einsum
import torch
import torch.cuda.nvtx as nvtx
from cs336_basics.nn_utils import softmax


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)

    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx.range("final matmul"):
        out = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

    return out
