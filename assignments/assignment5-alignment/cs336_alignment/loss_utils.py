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

    metadata: dict[str, torch.Tensor] = {}

    if importance_reweighting_method == "none":
        per_token_policy_gradient_loss = -policy_log_probs * rewards
    elif importance_reweighting_method == "noclip":
        assert old_log_probs is not None
        per_token_policy_gradient_loss = (
            -rewards * (policy_log_probs - old_log_probs).exp()
        )
    elif importance_reweighting_method == "grpo":
        assert old_log_probs is not None and cliprange is not None
        w = torch.exp(policy_log_probs - old_log_probs)
        per_token_policy_gradient_loss = -rewards * w
        w = torch.minimum(
            torch.maximum(w, torch.full_like(w, 1 - cliprange)),
            torch.full_like(w, 1 + cliprange),
        )
        clipped_loss = -rewards * w

        if response_mask is not None:
            clipped_tokens = (
                clipped_loss > per_token_policy_gradient_loss
            ) & response_mask
            metadata["clip_count"] = clipped_tokens.sum()
            metadata["clip_total"] = response_mask.sum()

        per_token_policy_gradient_loss = torch.maximum(
            clipped_loss, per_token_policy_gradient_loss
        )

    else:
        raise NotImplementedError

    return (per_token_policy_gradient_loss, metadata)


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
        per_sequence_policy_gradient_loss = (
            per_sequence_policy_gradient_loss / mask.sum(dim=-1).clamp_min(1)
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
        if normalization_constant <= 0:
            raise ValueError(
                "Normalization constant must be greater than 0 when using 'constant' normalization."
            )

        total_policy_gradient_loss = (
            per_sequence_policy_gradient_loss.sum(dim=-1) / normalization_constant
        )

    return total_policy_gradient_loss
