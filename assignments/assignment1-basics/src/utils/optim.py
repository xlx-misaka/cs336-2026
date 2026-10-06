from collections.abc import Callable, Iterable
from math import cos, pi, sqrt
from typing import Any, overload

import torch
from torch import Tensor


def cross_entropy(x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """计算数值稳定的 softmax 交叉熵，并返回批内平均值。

    参数
    --------
    x:
        模型输出的 logits，形状为 ``(..., vocab_size)``。
    targets:
        正确 token 的整数索引，形状与 ``x`` 去掉最后一维后相同。

    返回
    --------
    torch.Tensor
        标量交叉熵损失，表示批内所有位置损失的均值。
    """
    # 减去每行最大值，避免 exp 溢出；后续只影响数值稳定性，不改变 softmax 结果。
    shifted = x - x.max(dim=-1, keepdim=True).values

    # log(sum(exp(shifted)))：softmax 分母的对数。
    log_sum_exp = shifted.exp().sum(dim=-1).log()
    # 取目标位置对应的 logit。
    target_logits = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    return (log_sum_exp - target_logits).mean()


def lr_cosine_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    """根据训练步数返回线性 warmup 后 cosine 衰减的学习率。

    参数
    --------
    it(0-based):
        当前训练步数。
    max_learning_rate:
        warmup 结束时的峰值学习率。
    min_learning_rate:
        cosine 衰减结束时达到的最低学习率。
    warmup_iters:
        线性 warmup 的步数。
    cosine_cycle_iters:
        cosine 衰减结束的步数。

    返回
    --------
    float
        当前步数对应的学习率。
    """

    if it < warmup_iters:
        # warmup 阶段：从 0 线性增加到峰值学习率。
        return it / warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        # cosine 阶段：从峰值平滑衰减到最低学习率。
        cosine_span = cosine_cycle_iters - warmup_iters
        if cosine_span <= 0:
            return min_learning_rate
        return min_learning_rate + 0.5 * (1 + cos((it - warmup_iters) / cosine_span * pi)) * (
            max_learning_rate - min_learning_rate
        )
    else:
        # 超过计划步数后保持最低学习率。
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """对所有参数的梯度做全局 L2 范数裁剪。

    参数
    --------
    parameters:
        待检查梯度的模型参数。
    max_l2_norm:
        梯度的目标最大 L2 范数。总范数超过该值时会被等比缩小。
    eps:
        防止除零的小常数。
    """
    grads: list[Tensor] = [param.grad for param in parameters if param.grad is not None]

    if not grads:
        return

    # 计算所有梯度拼接后的全局 L2 范数。
    total_norm = torch.sqrt(
        sum(
            (grad.detach().square().sum() for grad in grads),
            start=torch.zeros((), dtype=grads[0].dtype, device=grads[0].device),
        )
    )
    if total_norm > max_l2_norm:
        # 将所有梯度按同一系数缩放，保持方向不变。
        clip_coef = max_l2_norm / (total_norm + eps)
        for grad in grads:
            grad.mul_(clip_coef)


class AdamW(torch.optim.Optimizer):
    """使用解耦权重衰减的 Adam 优化器。

    与原始 Adam 不同，权重衰减直接作用在参数 ``theta`` 上，而不是混入梯度。
    更新公式如下：

    ``theta = theta * (1 - lr * weight_decay)``
    ``m = beta1 * m + (1 - beta1) * grad``
    ``v = beta2 * v + (1 - beta2) * grad ** 2``
    ``theta = theta - lr_t * m / (sqrt(v) + eps)``

    其中 ``lr_t`` 是经过偏差校正后的学习率：

    ``lr_t = lr * sqrt(1 - beta2 ** t) / (1 - beta1 ** t)``
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        """初始化 AdamW 超参数并交给 ``torch.optim.Optimizer`` 管理。

        参数
        --------
        params:
            待优化的参数或参数组。
        lr:
            初始学习率。
        betas:
            一阶矩 ``m`` 和二阶矩 ``v`` 的指数衰减系数 ``(beta1, beta2)``。
        eps:
            防止二阶矩开方后除以零的小常数。
        weight_decay:
            解耦权重衰减系数。
        """

        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas 必须满足 0.0 <= beta < 1.0")

        defaults: dict[str, float | tuple[float, float]] = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], float] | None = None,
    ) -> float | None:
        """执行一次 AdamW 参数更新。

        参数
        --------
        closure:
            可选闭包，调用后会重新计算并返回 loss。仅在需要复用
            ``torch.optim.Optimizer`` 的闭包接口时提供。

        返回
        --------
        float | None
            如果提供了 ``closure``，返回闭包计算得到的 loss；否则返回 ``None``。
        """

        loss: float | None = None if closure is None else closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1: float
            beta2: float
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]
            theta: Tensor

            for theta in group["params"]:
                if theta.grad is None:
                    continue

                grad: Tensor = theta.grad
                state: dict[str, Any] = self.state[theta]

                # 首次更新时为当前参数分配时间步、一阶矩和二阶矩。
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(theta)
                    state["v"] = torch.zeros_like(theta)

                state["t"] += 1
                t: int = state["t"]
                m: Tensor = state["m"]
                v: Tensor = state["v"]

                # AdamW 的解耦权重衰减：先直接缩小参数，再按梯度更新。
                theta.mul_(1 - lr * weight_decay)

                # 对一阶/二阶矩做偏差校正，避免训练初期更新量过小。
                lr_t = lr * sqrt(1 - beta2**t) / (1 - beta1**t)

                # 指数移动平均更新一阶矩和二阶矩。
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 使用 Adam 更新规则，注意 eps 在开方后加入。
                theta.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)

        return loss
