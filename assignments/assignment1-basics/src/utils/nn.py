"""Transformer 基础模块的作业级实现。

本模块故意保持自包含，不直接调用 PyTorch 的融合实现，以便数值定义可以
直接阅读和验证。模块类均兼容 ``torch.nn.Module``，所有张量重排都使用
显式的 ``einops`` 表达式表达。
"""

import torch
from einops import einsum, rearrange, reduce
from torch import nn


def si_lu(x: torch.Tensor) -> torch.Tensor:
    """逐元素应用 SiLU/Swish 激活函数。

    参数
    --------
    x:
        任意形状的输入张量。

    返回
    --------
    torch.Tensor
        与 ``x`` 形状相同的张量，计算公式为 ``x * sigmoid(x)``。
    """
    return x * torch.sigmoid(x)


def swi_glu(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    """应用带门控的 SwiGLU 前馈变换。

    输入先经过 ``w1`` 和 ``w3`` 线性投影；前者的 SiLU 结果作为门控去缩放
    后者的投影结果，最后用 ``w2`` 映射回模型维度。

    参数
    --------
    x:
        输入张量，形状为 ``(..., d_model)``。
    w1:
        门控投影权重，形状为 ``(d_ff, d_model)``。
    w2:
        下投影权重，形状为 ``(d_model, d_ff)``。
    w3:
        值投影权重，形状为 ``(d_ff, d_model)``。

    返回
    --------
    torch.Tensor
        输出张量，形状为 ``(..., d_model)``。
    """
    w1 = einsum(w1, x, "d_ff d_model, ... d_model -> ... d_ff")
    w3 = einsum(w3, x, "d_ff d_model, ... d_model -> ... d_ff")

    output = si_lu(w1) * w3

    output = einsum(w2, output, "d_model d_ff, ... d_ff -> ... d_model")

    return output


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """沿指定维度计算数值稳定的 softmax。

    参数
    --------
    x:
        任意形状的输入张量。
    dim:
        进行概率归一化的维度。

    返回
    --------
    torch.Tensor
        与 ``x`` 形状相同的张量；沿 ``dim`` 求和为 1，且所有元素非负。
    """
    e = (x - x.max(dim=dim, keepdim=True).values).exp()

    output = e / e.sum(dim=dim, keepdim=True)

    return output


def scaled_dot_product_att(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """计算缩放点积注意力。

    参数
    --------
    Q:
        Query 张量，形状为 ``(..., n, d_k)``。
    K:
        Key 张量，形状为 ``(..., m, d_k)``。
    V:
        Value 张量，形状为 ``(..., m, d_v)``。
    mask:
        可选布尔张量，可广播到 ``(..., n, m)``。
        ``True`` 的位置允许注意力；``False`` 的位置会填入负无穷。

    返回
    --------
    torch.Tensor
        注意力输出，形状为 ``(..., n, d_v)``。
    """
    scores: torch.Tensor = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / Q.shape[-1] ** 0.5

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    return einsum(softmax(scores), V, "... n m, ... m d_v -> ... n d_v")


class Linear(nn.Module):
    """无偏置线性变换，使用截断正态分布初始化。

    属性
    --------
    weight:
        可学习权重，形状为 ``(out_features, in_features)``。
    """

    weight: torch.Tensor

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """初始化线性权重。

        参数
        --------
        in_features:
            输入特征维度。
        out_features:
            输出特征维度。
        device:
            参数所在的设备。
        dtype:
            参数使用的浮点 dtype。
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))

        std = (2.0 / (in_features + out_features)) ** 0.5

        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对最后一维为 ``in_features`` 的输入做线性变换。

        参数
        --------
        x:
            输入张量，形状为 ``(..., in_features)``。

        返回
        --------
        torch.Tensor
            输出张量，形状为 ``(..., out_features)``。
        """
        return einsum(x, self.weight, "... input, output input -> ... output")


class Embedding(nn.Module):
    """将整数 token ID 映射为可学习稠密向量。

    属性
    --------
    weight:
        嵌入表，形状为 ``(num_embeddings, embedding_dim)``。
    """

    weight: torch.Tensor

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """初始化嵌入表。

        参数
        --------
        num_embeddings:
            嵌入表条目数。
        embedding_dim:
            每个嵌入向量的维度。
        device:
            参数所在的设备。
        dtype:
            参数使用的浮点 dtype。
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))

        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """根据整数 token ID 查询对应的嵌入向量。

        参数
        --------
        token_ids:
            任意形状的整数张量，每个元素都是嵌入表索引。

        返回
        --------
        torch.Tensor
            形状为 ``token_ids.shape + (embedding_dim,)`` 的张量。
        """
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """不做中心化的均方根归一化。

    属性
    --------
    eps:
        数值稳定性常数。
    weight:
        可学习仿射权重，形状为 ``(d_model,)``。
    """

    eps: float
    weight: torch.Tensor

    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """初始化仿射权重与 epsilon。

        参数
        --------
        d_model:
            被归一化的特征维度。
        eps:
            在求平方根倒数前加入的小常数，用于数值稳定。
        device:
            权重所在的设备。
        dtype:
            权重使用的浮点 dtype。
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.eps = eps

        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """沿最后一维归一化，同时保持输入 dtype 不变。

        参数
        --------
        x:
            输入张量，形状为 ``(..., d_model)``。

        返回
        --------
        torch.Tensor
            形状和 dtype 均与 ``x`` 相同的张量。
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.rsqrt(reduce(x.pow(2), "... d -> ... 1", "mean") + self.eps)

        x = einsum(x * rms, self.weight, "... d, d -> ... d")

        return x.to(in_dtype)


class Swi_GLU_FFN(nn.Module):
    """包含三个线性投影的 SwiGLU 前馈子层。

    属性
    --------
    w1:
        门控投影，形状为 ``(d_ff, d_model)``。
    w2:
        下投影，形状为 ``(d_model, d_ff)``。
    w3:
        值投影，形状为 ``(d_ff, d_model)``。
    """

    w1: Linear
    w2: Linear
    w3: Linear

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """初始化 SwiGLU 投影矩阵。

        参数
        --------
        d_model:
            输入与输出特征维度。
        d_ff:
            隐藏前馈特征维度。
        device:
            参数所在的设备。
        dtype:
            参数使用的浮点 dtype。
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.w1 = Linear(d_model, d_ff, **factory_kwargs)
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)
        self.w3 = Linear(d_model, d_ff, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用 SwiGLU 变换。

        参数
        --------
        x:
            输入张量，形状为 ``(..., d_model)``。

        返回
        --------
        torch.Tensor
            输出张量，形状为 ``(..., d_model)``。
        """
        return swi_glu(x, self.w1.weight, self.w2.weight, self.w3.weight)


class SiLU_FFN(nn.Module):
    w1: Linear
    w2: Linear

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.w1 = Linear(d_model, d_ff, **factory_kwargs)
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)

    def forward(self, x: torch.Tensor):
        return self.w2(si_lu(self.w1(x)))


class RotaryPositionalEmbedding(nn.Module):
    """用非复数方式实现旋转位置编码。

    属性
    --------
    cos:
        预计算余弦表，形状为 ``(max_seq_len, d_k // 2)``。
    sin:
        预计算正弦表，形状为 ``(max_seq_len, d_k // 2)``。
    """

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        """预计算绝对 token 位置所需的余弦和正弦表。

        参数
        --------
        theta:
            控制每对旋转角频率的基数。
        d_k:
            query 或 key head 的维度，应为偶数以便两两配对。
        max_seq_len:
            缓存位置角度的数量。
        device:
            角度表所在的设备。
        """
        super().__init__()

        assert d_k % 2 == 0, "每个 head 的维度必须是偶数，才能两两配对旋转"

        pair_index = torch.arange(0, d_k, 2, device=device)

        freqs = theta ** -(pair_index / d_k)

        positions = torch.arange(max_seq_len, device=device)

        angles = einsum(positions, freqs, "max_seq_len, d_k_half -> max_seq_len d_k_half")

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """按 token 位置旋转 ``x`` 的最后一维。

        参数
        --------
        x:
            输入张量，形状为 ``(..., sequence_length, d_k)``。
        token_positions:
            整数张量，其值用于从预计算角度表中选择行。
            它与 ``x`` 广播，常见形状为 ``(sequence_length,)`` 或
            ``(batch_size, sequence_length)``。

        返回
        --------
        torch.Tensor
            与 ``x`` 形状相同的旋转后张量。
        """
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        output_even = einsum(cos, x_even, "... d_k_half, ... d_k_half -> ... d_k_half") - einsum(
            sin, x_odd, "... d_k_half, ... d_k_half -> ... d_k_half"
        )
        output_odd = einsum(sin, x_even, "... d_k_half, ... d_k_half -> ... d_k_half") + einsum(
            cos, x_odd, "... d_k_half, ... d_k_half -> ... d_k_half"
        )

        output = rearrange(torch.stack((output_even, output_odd), dim=-1), "... d_k_half p -> ... (d_k_half p)")

        return output


class MultiHeadSelfAttention(nn.Module):
    """多头因果自注意力，可选加旋转位置编码。

    属性
    --------
    num_heads:
        注意力头数。
    q_proj:
        Query 投影，形状为 ``(d_model, d_model)``。
    k_proj:
        Key 投影，形状为 ``(d_model, d_model)``。
    v_proj:
        Value 投影，形状为 ``(d_model, d_model)``。
    output_proj:
        输出投影，形状为 ``(d_model, d_model)``。
    _causal_mask:
        缓存的下三角因果掩码，形状为 ``(seq_len, seq_len)``。
    """

    num_heads: int
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    output_proj: Linear
    _causal_mask: torch.Tensor | None

    def __init__(
        self, d_model: int, num_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """初始化融合的 query、key、value 和输出投影。

        参数
        --------
        d_model:
            模型维度，必须能被 ``num_heads`` 整除。
        num_heads:
            注意力头数。
        device:
            参数所在的设备。
        dtype:
            参数使用的浮点 dtype。

        异常
        --------
        AssertionError
            当 ``d_model`` 不能被 ``num_heads`` 整除时抛出。
        """
        super().__init__()

        assert d_model % num_heads == 0

        factory_kwargs = {"device": device, "dtype": dtype}

        self.num_heads = num_heads
        self._causal_mask: torch.Tensor | None = None

        self.q_proj = Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **factory_kwargs)
        self.output_proj = Linear(d_model, d_model, **factory_kwargs)

    def forward(
        self, x: torch.Tensor, position_embedding: nn.Module | None = None, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """应用因果多头自注意力。

        参数
        --------
        x:
            输入张量，形状为 ``(..., sequence_length, d_model)``。
        position_embedding:
            可选 callable，用于旋转每个 head 的 query 与 key。
        token_positions:
            传给 ``position_embedding`` 的位置张量。
            当 ``position_embedding`` 非空时必须提供。

        返回
        --------
        torch.Tensor
            输出张量，形状为 ``(..., sequence_length, d_model)``。

        异常
        --------
        AssertionError
            当提供 ``position_embedding`` 但未提供
            ``token_positions`` 时抛出。
        """
        Q: torch.Tensor = self.q_proj(x)
        K: torch.Tensor = self.k_proj(x)
        V: torch.Tensor = self.v_proj(x)

        Q = rearrange(Q, "... d_n (h d_h) -> ... h d_n d_h", h=self.num_heads)
        K = rearrange(K, "... d_n (h d_h) -> ... h d_n d_h", h=self.num_heads)
        V = rearrange(V, "... d_n (h d_h) -> ... h d_n d_h", h=self.num_heads)

        if position_embedding is not None:
            assert token_positions is not None
            Q = position_embedding(Q, token_positions)
            K = position_embedding(K, token_positions)

        seq_len = Q.shape[-2]

        # 缓存截至目前所见的最大下三角掩码。掩码不属于参数状态，可在设备
        # 或序列长度变化后重新构建。
        if self._causal_mask is None or self._causal_mask.device != Q.device or self._causal_mask.shape[-1] < seq_len:
            self._causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool))

        mask = self._causal_mask[:seq_len, :seq_len]

        output = scaled_dot_product_att(Q, K, V, mask)
        output = rearrange(output, "... h d_n d_h -> ... d_n (h d_h)")
        output = self.output_proj(output)

        return output


class TransformerBlock(nn.Module):
    """含因果注意力与 SwiGLU FFN 的 pre-norm Transformer 块。

    属性
    --------
    max_seq_len:
        最大可接受序列长度。
    token_positions:
        缓存位置索引，形状为 ``(max_seq_len,)``。
    ln1:
        注意力前的 RMSNorm。
    ln2:
        FFN 前的 RMSNorm。
    rope:
        旋转位置编码。
    attn:
        多头因果自注意力。
    ffn:
        SwiGLU 前馈子层。
    """

    max_seq_len: int
    token_positions: torch.Tensor
    ln1: RMSNorm
    ln2: RMSNorm
    rope: RotaryPositionalEmbedding | None
    attn: MultiHeadSelfAttention
    ffn: Swi_GLU_FFN | SiLU_FFN

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_rms_norm: bool = True,
        norm_position: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
    ):
        """初始化块内子模块与缓存位置索引。

        参数
        --------
        d_model:
            模型特征维度。
        num_heads:
            注意力头数。
        d_ff:
            隐藏前馈维度。
        max_seq_len:
            最大可接受序列长度，同时也是缓存位置数。
        theta:
            旋转位置编码的频率基数。
        device:
            可训练参数与缓存位置所在的设备。
        dtype:
            可训练参数使用的浮点 dtype。
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        assert norm_position in ("pre", "post")
        assert ffn_type in ("swiglu", "silu")

        self.use_rms_norm = use_rms_norm
        self.norm_position = norm_position

        self.max_seq_len = max_seq_len

        self.ln1 = RMSNorm(d_model, **factory_kwargs)
        self.ln2 = RMSNorm(d_model, **factory_kwargs)
        self.rope = (
            RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device) if use_rope else None
        )
        self.attn = MultiHeadSelfAttention(d_model, num_heads, **factory_kwargs)
        self.ffn = (
            Swi_GLU_FFN(d_model, d_ff, **factory_kwargs)
            if ffn_type == "swiglu"
            else SiLU_FFN(d_model, d_ff, **factory_kwargs)
        )

        self.register_buffer(
            "token_positions",
            torch.arange(max_seq_len, device=device),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行一次 pre-norm 注意力与 FFN，并使用残差连接。

        参数
        --------
        x:
            输入张量，形状为 ``(..., sequence_length, d_model)``，其中
            ``sequence_length <= max_seq_len``。

        返回
        --------
        torch.Tensor
            与 ``x`` 形状相同的输出张量。

        异常
        --------
        AssertionError
            当 ``sequence_length`` 超过 ``max_seq_len`` 时抛出。
        """
        seq_len = x.shape[-2]
        assert seq_len <= self.max_seq_len

        token_positions = self.token_positions[:seq_len].to(device=x.device)

        if self.use_rms_norm and self.norm_position == "pre":
            x = x + self.attn(self.ln1(x), self.rope, token_positions)
            x = x + self.ffn(self.ln2(x))
        elif self.norm_position == "post":
            x = self.ln1(x + self.attn(x, self.rope, token_positions))
            x = self.ln2(x + self.ffn(x))
        else:
            x = x + self.attn(x, self.rope, token_positions)
            x = x + self.ffn(x)

        return x


class TransformerLM(nn.Module):
    """仅包含解码器的 Transformer 语言模型。

    属性
    --------
    token_embeddings:
        token 嵌入表，形状为 ``(vocab_size, d_model)``。
    layers:
        ``num_layers`` 个 TransformerBlock。
    ln_final:
        最终输出的 RMSNorm。
    lm_head:
        输出投影，形状为 ``(vocab_size, d_model)``。
    """

    token_embeddings: Embedding
    layers: nn.ModuleList
    ln_final: RMSNorm
    lm_head: Linear

    def __init__(
        self,
        vocab_size: int,
        context_length: int = 256,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 16,
        d_ff: int = 1344,
        rope_theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_rms_norm: bool = True,
        norm_position: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
    ):
        """初始化 token 嵌入、Transformer 层、最终归一化与输出头。

        参数
        --------
        vocab_size:
            输出词表大小。
        context_length:
            每个块接受的最大序列长度。
        d_model:
            隐藏维度与嵌入维度。
        num_layers:
            Transformer 块数量。
        num_heads:
            每个块中的注意力头数。
        d_ff:
            每个块中的隐藏前馈维度。
        rope_theta:
            旋转位置编码频率基数。
        device:
            参数与 buffer 所在的设备。
        dtype:
            参数使用的浮点 dtype。
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.use_rms_norm = use_rms_norm

        self.token_embeddings = Embedding(vocab_size, d_model, **factory_kwargs)

        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                context_length,
                rope_theta,
                use_rms_norm=use_rms_norm,
                norm_position=norm_position,
                use_rope=use_rope,
                ffn_type=ffn_type,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )

        self.ln_final = RMSNorm(d_model, **factory_kwargs)

        self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """从 token ID 计算下一 token 的 logits。

        参数
        --------
        token_ids:
            整数张量，形状为 ``(..., sequence_length)``，其中
            ``sequence_length <= context_length``。

        返回
        --------
        torch.Tensor
            未归一化 logits，形状为
            ``token_ids.shape + (vocab_size,)``。
        """
        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x) if self.use_rms_norm else x

        x = self.lm_head(x)

        return x
