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
    N_QUERIES: tl.constexpr,
    N_KEYS: tl.constexpr,
    scale,
    DIM: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, DIM),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, DIM),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, DIM),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, DIM),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM),
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
    L_block = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    O_block = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero").to(L_block.dtype)

    # get m
    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    m_new = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # load K, V
        K_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # S = Q @ K.T / sqrt(d)
        S_block = tl.dot(Q_block, tl.trans(K_block)) * scale

        # mask S for causal and padding
        q_start_pos = query_tile_index * Q_TILE_SIZE
        k_start_pos = key_tile_index * K_TILE_SIZE
        q_pos = q_start_pos + tl.arange(0, Q_TILE_SIZE)
        k_pos = k_start_pos + tl.arange(0, K_TILE_SIZE)
        q_mask = q_pos < q_start_pos + (min(N_QUERIES, (query_tile_index + 1) * Q_TILE_SIZE) - query_tile_index * Q_TILE_SIZE)
        k_mask = k_pos < k_start_pos + (min(N_KEYS, (key_tile_index + 1) * K_TILE_SIZE) - key_tile_index * K_TILE_SIZE)
        mask = q_mask[:, None] & k_mask[None, :]
        if is_causal:
            mask &= q_pos[:, None] >= k_pos[None, :]
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

    tl.store(O_block_ptr, (O_block / L_block[:, None]).to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, (tl.log(L_block) + m).to(L_block_ptr.type.element_ty), boundary_check=(0,))


@triton.jit
def flash_bwd_kernel_D(
    O_ptr,
    dO_ptr,
    D_ptr,
    stride_ob,
    stride_oq,
    stride_od,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_db,
    stride_dq,
    N_QURIES: tl.constexpr,
    DIM: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
):

    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QURIES, DIM),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QURIES, DIM),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QURIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    O_block = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dO_block = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # D =  rowsum(O * dO)
    D_block = tl.sum(O_block * dO_block, axis=-1, dtype=tl.float32)

    tl.store(D_block_ptr, D_block.to(D_block_ptr.type.element_ty), boundary_check=(0,))


@triton.jit
def flash_bwd_kernel_KV(
    Q_ptr,
    K_ptr,
    V_ptr,
    L_ptr,
    dO_ptr,
    D_ptr,
    dK_ptr,
    dV_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_lb,
    stride_lq,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_db,
    stride_dq,
    stride_dkb,
    stride_dkk,
    stride_dkd,
    stride_dvb,
    stride_dvk,
    stride_dvd,
    scale,
    N_QUERIES: tl.constexpr,
    N_KEYS: tl.constexpr,
    DIM: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):

    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, DIM),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, DIM),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, DIM),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, DIM),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, DIM),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, DIM),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0),
    )

    K_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    V_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dK_block = tl.zeros((K_TILE_SIZE, DIM), dtype=tl.float32)
    dV_block = tl.zeros((K_TILE_SIZE, DIM), dtype=tl.float32)

    for query_tile_index in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dO_block = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        L_block = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        D_block = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

        S_block = tl.dot(Q_block, K_block.T) * scale

        # mask S for causal and padding
        q_start_pos = query_tile_index * Q_TILE_SIZE
        k_start_pos = key_tile_index * K_TILE_SIZE
        q_pos = q_start_pos + tl.arange(0, Q_TILE_SIZE)
        k_pos = k_start_pos + tl.arange(0, K_TILE_SIZE)
        q_mask = q_pos < q_start_pos + (min(N_QUERIES, (query_tile_index + 1) * Q_TILE_SIZE) - query_tile_index * Q_TILE_SIZE)
        k_mask = k_pos < k_start_pos + (min(N_KEYS, (key_tile_index + 1) * K_TILE_SIZE) - key_tile_index * K_TILE_SIZE)
        mask = q_mask[:, None] & k_mask[None, :]
        if is_causal:
            mask &= q_pos[:, None] >= k_pos[None, :]
        S_block = tl.where(mask, S_block, float("-inf"))

        P_block = tl.exp(S_block - L_block[:None])

        dV_block = tl.dot(P_block.T, dO_block, acc=dV_block)

        dP_block = tl.dot(dO_block, V_block.T)

        dS_block = P_block * (dP_block - D_block[:None])

        dK_block = tl.dot(dS_block.T, Q_block * scale, acc=dK_block)

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

    tl.store(dK_block_ptr, dK_block.to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV_block.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))


@triton.jit
def flash_bwd_kernel_Q(
    Q_ptr,
    K_ptr,
    V_ptr,
    L_ptr,
    dO_ptr,
    D_ptr,
    dQ_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_lb,
    stride_lq,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_db,
    stride_dq,
    stride_dqb,
    stride_dqk,
    stride_dqd,
    scale,
    N_QUERIES: tl.constexpr,
    N_KEYS: tl.constexpr,
    DIM: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):

    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, DIM),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, DIM),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, DIM),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, DIM),
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

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, DIM),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_KEYS, DIM),
        strides=(stride_dqk, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )

    Q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dO_block = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    L_block = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    D_block = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

    dQ_block = tl.zeros((Q_TILE_SIZE, DIM), dtype=tl.float32)

    for key_tile_index in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        K_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        S_block = tl.dot(Q_block, K_block.T) * scale

        # mask S for causal and padding
        q_start_pos = query_tile_index * Q_TILE_SIZE
        k_start_pos = key_tile_index * K_TILE_SIZE
        q_pos = q_start_pos + tl.arange(0, Q_TILE_SIZE)
        k_pos = k_start_pos + tl.arange(0, K_TILE_SIZE)
        q_mask = q_pos < q_start_pos + (min(N_QUERIES, (query_tile_index + 1) * Q_TILE_SIZE) - query_tile_index * Q_TILE_SIZE)
        k_mask = k_pos < k_start_pos + (min(N_KEYS, (key_tile_index + 1) * K_TILE_SIZE) - key_tile_index * K_TILE_SIZE)
        mask = q_mask[:, None] & k_mask[None, :]
        if is_causal:
            mask &= q_pos[:, None] >= k_pos[None, :]
        S_block = tl.where(mask, S_block, float("-inf"))

        P_block = tl.exp(S_block - L_block[:None])

        dP_block = tl.dot(dO_block, V_block.T)

        dS_block = P_block * (dP_block - D_block[:None])

        dQ_block = tl.dot(dS_block, K_block * scale, acc=dQ_block)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr, dQ_block.to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))


def flash_bwd_torch(Q, K, V, O, dO, L, is_causal=False):
    N_q = Q.shape[-2]
    N_k = K.shape[-2]
    d = Q.shape[-1]

    dQ = torch.zeros_like(Q, dtype=torch.float32)
    dV = torch.zeros_like(V, dtype=torch.float32)
    dK = torch.zeros_like(K, dtype=torch.float32)

    D = (O * dO).sum(dim=-1)

    assert N_q >= B_q and N_k >= B_k

    for query_block_index in range(ceil(N_q / B_q)):
        st_qb = query_block_index * B_q
        ed_qb = min((query_block_index + 1) * B_q, N_q)

        for key_block_index in range(ceil(N_k / B_k)):
            st_kb = key_block_index * B_k
            ed_kb = min((key_block_index + 1) * B_k, N_k)

            # S = Q @ K^T / sqrt(d)
            S_block = Q[..., st_qb:ed_qb, :] @ K[..., st_kb:ed_kb, :].transpose(-2, -1) * d**-0.5

            # mask S
            if is_causal:
                q_tile_indices = torch.arange(st_qb, ed_qb, device=Q.device)
                k_tile_indices = torch.arange(st_kb, ed_kb, device=Q.device)
                mask = q_tile_indices[:, None] >= k_tile_indices[None, :]
                S_block = torch.where(mask, S_block, float("-inf"))

            # P = exp(S - L)
            P_block = (S_block - L[..., st_qb:ed_qb, None]).exp()

            # dV = P^T @ dO
            dV[..., st_kb:ed_kb, :] += P_block.transpose(-2, -1).to(dO.dtype) @ dO[..., st_qb:ed_qb, :]

            # dP = dO @ V^T
            dP_block = dO[..., st_qb:ed_qb, :] @ V[..., st_kb:ed_kb, :].transpose(-2, -1).to(dO.dtype)

            # dS = P * (dP - D)
            dS_block = P_block * (dP_block - D[..., st_qb:ed_qb, None])

            # dQ = dS @ K / sqrt(d)
            dQ[..., st_qb:ed_qb, :] += dS_block @ K[..., st_kb:ed_kb, :].to(dS_block.dtype) * d**-0.5

            # dK = dS^T @ Q / sqrt(d)
            dK[..., st_kb:ed_kb, :] += dS_block.transpose(-2, -1) @ Q[..., st_qb:ed_qb, :].to(dS_block.dtype) * d**-0.5

    return dQ.to(dO.dtype), dK.to(dO.dtype), dV.to(dO.dtype), None


class TorchFlashAttention(torch.autograd.Function):
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

        O = torch.zeros_like(Q, dtype=torch.float32)  # ..., N_q, d
        L = torch.zeros(Q.shape[:-1], device=Q.device, dtype=torch.float32)  # ..., N_q

        ctx.d = d
        ctx.B_q = B_q
        ctx.B_k = B_k
        ctx.is_causal = is_causal

        for i in range(num_query_row_tiles):
            _B_q = min((i + 1) * B_q, N_q) - i * B_q

            m = torch.Tensor(Q.shape[:-2] + (_B_q, num_key_row_tiles)).fill_(float("-inf")).to(device=Q.device, dtype=torch.float32)  # ..., B_q, num_key_row_tiles

            for j in range(num_key_row_tiles):
                _B_k = min((j + 1) * B_k, N_k) - j * B_k
                # 1. load Q, K, V from memory.
                Q_block = Q[..., i * B_q : min((i + 1) * B_q, N_q), :]  # ..., B_q, d
                K_block = K[..., j * B_k : min((j + 1) * B_k, N_k), :]  # ..., B_k, d
                V_block = V[..., j * B_k : min((j + 1) * B_k, N_k), :]  # ..., B_k, d

                # 2. S = Q @ K^T / sqrt(d)
                S_block = Q_block @ K_block.transpose(-2, -1) / sqrt(d)

                # mask S
                if is_causal:
                    q_pos = i * B_q + torch.arange(0, _B_q, device=Q.device)
                    k_pos = j * B_k + torch.arange(0, _B_k, device=Q.device)
                    mask = q_pos[:, None] >= k_pos[None, :]
                    S_block = torch.where(mask, S_block, float("-inf"))

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
                O[..., i * B_q : min((i + 1) * B_q, N_q), :] += P_block @ V_block.to(P_block.dtype)  # ..., B_q, d

            # 7. O = O_Tk / diag(L_Tk)
            O[..., i * B_q : min((i + 1) * B_q, N_q), :] = O[..., i * B_q : min((i + 1) * B_q, N_q), :] / L[..., i * B_q : min((i + 1) * B_q, N_q), None]  # ..., B_q, d

            # 8. L = m_Tk + log(L_Tk)
            L[..., i * B_q : min((i + 1) * B_q, N_q)] = m[..., :, -1] + L[..., i * B_q : min((i + 1) * B_q, N_q)].log()  # ..., B_q

        # 9. saved for backward
        ctx.save_for_backward(L, Q, K, V, O)

        return O.to(Q.dtype)

    @staticmethod
    def backward(ctx, *grad_outputs):
        dO = grad_outputs[0]

        L, Q, K, V, O = ctx.saved_tensors

        return flash_bwd_torch(Q, K, V, O, dO, L, ctx.is_causal)


class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        assert Q.is_cuda and K.is_cuda and V.is_cuda
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()

        BATCH_SIZE = Q.shape[0]
        N_QURIES = Q.shape[-2]
        N_KEYS = K.shape[-2]
        DIM = Q.shape[-1]
        scale = 1.0 / sqrt(DIM)
        Q_TILE_SIZE = B_q
        K_TILE_SIZE = B_k

        assert N_QURIES >= Q_TILE_SIZE
        assert N_KEYS >= K_TILE_SIZE

        O = torch.zeros_like(Q)
        L = torch.zeros(Q.shape[:-1], device=Q.device, dtype=torch.float32)

        ctx.DIM = DIM
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
            DIM,
            Q_TILE_SIZE,  # pyright: ignore[reportArgumentType]
            K_TILE_SIZE,  # pyright: ignore[reportArgumentType]
            is_causal,  # pyright: ignore[reportArgumentType]
        )

        ctx.save_for_backward(L, Q, K, V, O)

        return O

    @staticmethod
    def backward(ctx, *grad_outputs):
        dO = grad_outputs[0]

        L, Q, K, V, O = ctx.saved_tensors

        BATCH_SIZE = Q.shape[0]
        N_QURIES = Q.shape[1]
        N_KEYS = K.shape[1]
        DIM = ctx.DIM

        Q_TILE_SIZE = ctx.Q_TILE_SIZE
        K_TILE_SIZE = ctx.K_TILE_SIZE

        T_q = triton.cdiv(N_QURIES, Q_TILE_SIZE)
        T_k = triton.cdiv(N_KEYS, K_TILE_SIZE)

        scale = ctx.scale
        is_causal = ctx.is_causal

        D = torch.zeros(N_QURIES, device=Q.device, dtype=dO.dtype)

        flash_bwd_kernel_D[T_q, BATCH_SIZE](
            O,
            dO,
            D,
            O.stride(0),
            O.stride(1),
            O.stride(2),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            D.stride(0),
            D.stride(1),
            N_QURIES,
            DIM,
            Q_TILE_SIZE,
        )

        dK = torch.zeros((BATCH_SIZE, N_KEYS, DIM), device=Q.device, dtype=dO.dtype)
        dV = torch.zeros((BATCH_SIZE, N_KEYS, DIM), device=Q.device, dtype=dO.dtype)

        flash_bwd_kernel_KV[T_k, BATCH_SIZE](
            Q,
            K,
            V,
            L,
            dO,
            D,
            dK,
            dV,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            L.stride(0),
            L.stride(1),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            D.stride(0),
            D.stride(1),
            dK.stride(0),
            dK.stride(1),
            dK.stride(2),
            dV.stride(0),
            dV.stride(1),
            dV.stride(2),
            scale,
            N_QURIES,
            N_KEYS,
            DIM,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal,
        )

        dQ = torch.zeros((BATCH_SIZE, N_QURIES, DIM), device=Q.device, dtype=dO.dtype)

        flash_bwd_kernel_Q[T_q, BATCH_SIZE](
            Q,
            K,
            V,
            L,
            dO,
            D,
            dQ,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            L.stride(0),
            L.stride(1),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            D.stride(0),
            D.stride(1),
            dQ.stride(0),
            dQ.stride(1),
            dQ.stride(2),
            scale,
            N_QURIES,
            N_KEYS,
            DIM,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal,
        )

        return dQ, dK, dV, None
