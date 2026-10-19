from math import ceil, sqrt
import torch
import triton
from triton import language as tl

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
    is_causul: tl.constexpr,
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

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # load Q, O, L
    Q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    O_block = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
    L_block = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")

    # get m
    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    m_new = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # load K, V
        K_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # S = Q @ K.T / sqrt(d)
        S_block = tl.dot(Q_block, tl.trans(K_block)) * scale

        # mask S
        if is_causul:
            q_start_pos = query_tile_index * Q_TILE_SIZE
            k_start_pos = key_tile_index * K_TILE_SIZE
            q_pos = q_start_pos + tl.arange(0, Q_TILE_SIZE)
            k_pos = k_start_pos + tl.arange(0, K_TILE_SIZE)
            mask = q_pos[:, None] >= k_pos[None, :]
            S_block = tl.where(mask, S_block, float("-inf"))

        # m_j = max(m_{j-1}, rowmax(S_block)) if j > 0
        m_new = tl.maximum(m, tl.max(S_block, axis=-1))
        if key_tile_index == 0:
            m = m_new

        # P = exp(S - m_j)
        P_block = tl.exp(S_block - m_new[:, None])

        # O_j = O_{j-1} * exp(m_{j-1} - m_j) + P @ V
        O_block = tl.dot(P_block.to(V_block.dtype), V_block, acc=O_block * tl.exp(m - m_new)[:, None])

        # L_j = L_{j-1} * exp(m_{j-1} - m_j) + rowsum(P)
        L_block = L_block * tl.exp(m - m_new) + tl.sum(P_block, axis=-1)

        # Update block pointers, m for next iteration
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        m = m_new

    tl.store(O_block_ptr, (O_block / L_block[:, None]).to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, (tl.log(L_block) + m).to(L_block_ptr.type.element_ty))


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        # 0. prepare variables
        d = Q.shape[-1]
        N_q = Q.shape[-2]
        N_k = K.shape[-2]

        assert N_q >= B_q
        assert N_k >= B_k

        num_query_row_tiles = ceil(N_q / B_q)
        num_key_row_tiles = ceil(N_k / B_k)

        O = torch.zeros_like(Q, dtype=torch.float32)  # ..., N_q, d  # noqa: E741
        L = torch.zeros(Q.shape[:-1], device=Q.device, dtype=torch.float32)  # ..., N_q

        S = torch.zeros(Q.shape[:-1] + (N_k,), device=Q.device, dtype=torch.float32)  # ..., N_q, N_k

        ctx.d = d
        ctx.B_q = B_q
        ctx.B_k = B_k
        ctx.is_causal = is_causal

        for i in range(num_query_row_tiles):
            m = torch.Tensor(Q.shape[:-2] + (B_q, num_key_row_tiles)).fill_(float("-inf")).to(device=Q.device, dtype=torch.float32)  # ..., B_q, num_key_row_tiles

            for j in range(num_key_row_tiles):
                # 1. load Q, K, V from memory.
                Q_block = Q[..., i * B_q : min((i + 1) * B_q, N_q), :]  # ..., B_q, d
                K_block = K[..., j * B_k : min((j + 1) * B_k, N_k), :]  # ..., B_k, d
                V_block = V[..., j * B_k : min((j + 1) * B_k, N_k), :]  # ..., B_k, d

                # 2. S = Q @ K^T / sqrt(d)
                S[..., i * B_q : min((i + 1) * B_q, N_q), j * B_k : min((j + 1) * B_k, N_k)] = Q_block @ K_block.transpose(-2, -1) / sqrt(d)

                # mask S
                if is_causal:
                    q_pos = num_query_row_tiles * B_q + torch.arange(0, B_q, device=Q.device)
                    k_pos = num_key_row_tiles * B_k + torch.arange(0, B_k, device=Q.device)
                    mask = q_pos[:, None] >= k_pos[None, :]
                    S[..., i * B_q : min((i + 1) * B_q, N_q), j * B_k : min((j + 1) * B_k, N_k)] = torch.where(
                        mask, S[..., i * B_q : min((i + 1) * B_q, N_q), j * B_k : min((j + 1) * B_k, N_k)], float("-inf")
                    )

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

        dO = grad_outputs[0]

        L, Q, K, V, O = ctx.saved_tensors

        raise NotImplementedError


class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        assert Q.is_cuda and K.is_cuda and V.is_cuda
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()

        BATCH_SIZE = Q.shape[0]
        N_QURIES = Q.shape[-2]
        N_KEYS = K.shape[-2]
        D = Q.shape[-1]
        scale = 1.0 / sqrt(D)
        Q_TILE_SIZE = B_q
        K_TILE_SIZE = B_k

        assert N_QURIES >= Q_TILE_SIZE
        assert N_KEYS >= K_TILE_SIZE

        O = torch.zeros_like(Q, dtype=torch.float32)  # noqa: E741
        L = torch.zeros(Q.shape[:-1], device=Q.device, dtype=torch.float32)

        ctx.D = D
        ctx.Q_TILE_SIZE = Q_TILE_SIZE
        ctx.K_TILE_SIZE = K_TILE_SIZE
        ctx.is_causal = is_causal

        flash_fwd_kernel[triton.cdiv(N_QURIES, Q_TILE_SIZE), BATCH_SIZE](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            N_QURIES,
            N_KEYS,
            scale,
            ctx.D,
            ctx.Q_TILE_SIZE,  # pyright: ignore[reportArgumentType]
            ctx.K_TILE_SIZE,  # pyright: ignore[reportArgumentType]
            ctx.is_causal,  # pyright: ignore[reportArgumentType]
        )

        ctx.save_for_backward(L, Q, K, V, O)

        return O

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError
