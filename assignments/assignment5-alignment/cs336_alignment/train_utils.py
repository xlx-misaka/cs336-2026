from collections.abc import Callable
from typing import Literal

import torch
from transformers import PreTrainedTokenizerBase

from cs336_alignment.data_utils import tokenize_prompt_and_output
from cs336_alignment.loss_utils import (
    aggregate_loss_across_microbatch,
    compute_policy_gradient_loss,
)
from cs336_alignment.model_utils import get_response_log_probs
from cs336_alignment.reward_utils import (
    compute_group_normalized_rewards,
    compute_rollout_rewards,
)


def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be greater than 0.")

    device = next(model.parameters()).device
    tokenized = tokenize_prompt_and_output(
        repeated_prompts, rollout_responses, tokenizer
    )
    input_ids = tokenized["input_ids"].to(device=device)
    labels = tokenized["labels"].to(device=device)
    response_mask = tokenized["response_mask"].to(device=device)

    rewards, raw_rewards_metadata = compute_rollout_rewards(
        reward_fn,
        rollout_responses,
        repeated_ground_truths,
    )
    rewards = rewards.to(device=device)

    group_normalized_rewards, _ = compute_group_normalized_rewards(
        rewards,
        group_size,
        baseline,
        advantage_eps,
        advantage_normalizer,
    )

    microbatch_size = (len(input_ids) - 1) // gradient_accumulation_steps + 1
    loss = torch.zeros((), device=device)
    token_entropy = torch.zeros((), device=device)

    for i in range(0, len(input_ids), microbatch_size):
        inputs_microbatch = input_ids[i : min(len(input_ids), i + microbatch_size)]
        labels_microbatch = labels[i : min(len(input_ids), i + microbatch_size)]
        log_probs, token_entropy_microbatch = get_response_log_probs(
            model,
            inputs_microbatch,
            labels_microbatch,
            return_token_entropy=True,
        ).values()
        per_token_policy_gradient_loss, _ = compute_policy_gradient_loss(
            group_normalized_rewards[i : min(len(input_ids), i + microbatch_size)],
            log_probs,
            importance_reweighting_method,
            old_log_probs,
            cliprange,
            response_mask[i : min(len(input_ids), i + microbatch_size)],
        )
        loss_microbatch = (
            aggregate_loss_across_microbatch(
                per_token_policy_gradient_loss,
                response_mask[i : min(len(input_ids), i + microbatch_size)],
                loss_normalization,
                normalization_constant,
            )
            * (len(inputs_microbatch))
            / len(input_ids)
        )

        loss_microbatch.backward()
        loss += loss_microbatch.detach()

        token_entropy_microbatch = token_entropy_microbatch.detach().where(
            response_mask[i : min(len(input_ids), i + microbatch_size)],
            0.0,
        )
        token_entropy_microbatch = token_entropy_microbatch.sum(-1) / response_mask[
            i : min(len(input_ids), i + microbatch_size)
        ].sum(-1)
        token_entropy += token_entropy_microbatch.sum(-1) / len(input_ids)

    if max_grad_norm is not None:
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )
    else:
        grads = [param.grad for param in model.parameters() if param.grad is not None]
        gradient_norm = torch.nn.utils.get_total_norm(grads)

    optimizer.step()
    optimizer.zero_grad()

    return (
        loss,
        {
            **raw_rewards_metadata,
            "token_entropy": token_entropy,
            "gradient_norm": gradient_norm,
        },
    )
