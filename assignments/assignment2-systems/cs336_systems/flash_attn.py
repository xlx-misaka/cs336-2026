from math import ceil, sqrt

import torch

B_q = 16
B_k = 16


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        # 0. prepare variables
        d = Q.shape[-1]
        N_q = Q.shape[-2]
        N_k = K.shape[-2]

        num_query_row_tiles = ceil(N_q / B_q)
        num_key_row_tiles = ceil(N_k / B_k)

        O = torch.zeros_like(Q)  # ..., N_q, d
        L = torch.zeros(Q.shape[:-1], device=Q.device, dtype=Q.dtype)  # ..., N_q

        S = torch.zeros(Q.shape[:-1] + (N_k,), device=Q.device, dtype=Q.dtype)  # ..., N_q, N_k

        for i in range(num_query_row_tiles):
            m = torch.Tensor(Q.shape[:-2] + (B_q, num_key_row_tiles)).fill_(float("-inf"))  # ..., B_q, num_key_row_tiles

            for j in range(num_key_row_tiles):
                # 1. load Q, K, V from memory.
                Q_block = Q[..., i * B_q : min((i + 1) * B_q, N_q), :]  # ..., B_q, d
                K_block = K[..., j * B_k : min((j + 1) * B_k, N_k), :]  # ..., B_k, d
                V_block = V[..., j * B_k : min((j + 1) * B_k, N_k), :]  # ..., B_k, d

                # 2. S = Q @ K^T / sqrt(d)
                S[..., i * B_q : min((i + 1) * B_q, N_q), j * B_k : min((j + 1) * B_k, N_k)] = Q_block @ K_block.transpose(-2, -1) / sqrt(d)
                S_block = S[..., i * B_q : min((i + 1) * B_q, N_q), j * B_k : min((j + 1) * B_k, N_k)]  # ..., B_q, B_k

                # 3. m_j = max(rowmax(S), m_{j-1})
                m[..., :, j] = S_block.max(-1).values  # ..., B_q
                if j > 0:
                    m[..., :, j] = torch.maximum(m[..., :, j], m[..., :, j - 1])

                # 4. P = exp(S - m_j)
                P_block = (S_block - m[..., :, j, None]).exp()  # ..., B_q, B_k

                # 5. L_j = rowsum(P) + exp(m_{j-1} - m_j) * L_{j-1}
                if j > 0:
                    L[..., i * B_q : min((i + 1) * B_q, N_q)] *= (m[..., :, j - 1] - m[..., :, j]).exp()
                L[..., i * B_q : min((i + 1) * B_q, N_q)] += P_block.sum(-1)  # ..., B_q

                # 6. O_j = diag(exp(m_{j-1} - m_j)) * O_{j-1} + P @ V
                if j > 0:
                    O[..., i * B_q : min((i + 1) * B_q, N_q), :] *= (m[..., :, j - 1] - m[..., :, j])[..., None].exp()
                O[..., i * B_q : min((i + 1) * B_q, N_q), :] += P_block @ V_block  # ..., B_q, d

            # 7. O = O_Tk / diag(L_Tk)
            O[..., i * B_q : min((i + 1) * B_q, N_q), :] = O[..., i * B_q : min((i + 1) * B_q, N_q), :] / L[..., i * B_q : min((i + 1) * B_q, N_q), None]  # ..., B_q, d

            # 8. L = m_Tk + log(L_Tk)
            L[..., i * B_q : min((i + 1) * B_q, N_q)] = m[..., :, -1] + L[..., i * B_q : min((i + 1) * B_q, N_q)].log()  # ..., B_q

        # 9. saved for backward
        ctx.save_for_backward(L, Q, K, V, O)

        return O

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError


class FlashAttnTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        # 0. prepare variables
        d = Q.shape[-1]
        N_q = Q.shape[-2]
        N_k = K.shape[-2]

        num_query_row_tiles = ceil(N_q / B_q)
        num_key_row_tiles = ceil(N_k / B_k)

        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError
