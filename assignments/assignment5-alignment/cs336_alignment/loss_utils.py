from typing import Literal

import torch


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rewards = raw_rewards_or_advantages.view(-1, 1).contiguous()

    per_token_policy_gradient_loss = -policy_log_probs * rewards

    return (per_token_policy_gradient_loss, {})


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    per_sequence_policy_gradient_loss = per_token_policy_gradient_loss.where(
        mask == True,
        torch.tensor(
            0.0,
            device=per_token_policy_gradient_loss.device,
            dtype=per_token_policy_gradient_loss.dtype,
        ),
    ).sum(dim=-1)

    if loss_normalization == "sequence":
        if not mask.any(dim=-1).all(dim=-1):
            raise ValueError("Mask must have at least one token per sequence.")

        per_sequence_policy_gradient_loss = (
            per_sequence_policy_gradient_loss / mask.sum(dim=-1)
        )

        total_policy_gradient_loss = (
            per_sequence_policy_gradient_loss.sum(dim=-1)
            / per_sequence_policy_gradient_loss.shape[-1]
        )

    else:
        if normalization_constant is None:
            raise ValueError(
                "Normalization constant must be provided when using 'constant' normalization."
            )

        total_policy_gradient_loss = (
            per_sequence_policy_gradient_loss.sum(dim=-1) / normalization_constant
        )

    return total_policy_gradient_loss
