from collections.abc import Callable
from typing import Literal

import torch


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rollout rewards for a list of responses and ground truths."""
    if not rollout_responses:
        raise ValueError("rollout_responses must not be empty")

    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError(
            "rollout_responses and repeated_ground_truths must have the same length"
        )

    rollout_batch_size = len(rollout_responses)
    raw_rewards = torch.empty((rollout_batch_size,))
    results = []
    for i, (response, answer) in enumerate(
        zip(rollout_responses, repeated_ground_truths)
    ):
        result = reward_fn(response, answer)
        raw_rewards[i] = result["reward"]
        results.append(result)

    return (
        raw_rewards,
        {
            "mean_total_rewards": sum(r["reward"] for r in results) / len(results),
            "mean_format_rewards": sum(r["format_reward"] for r in results)
            / len(results),
        },
    )


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    if group_size <= 1:
        raise ValueError("group_size must be greater than 1")

    rewards = raw_rewards.view(-1, group_size).contiguous()
    rewards_mean = rewards.mean(dim=-1, keepdim=True)
    rewards_std = rewards.std(dim=-1, keepdim=True)

    if baseline == "mean":
        rewards = rewards - rewards_mean
    if advantage_normalizer == "std":
        rewards = rewards / (rewards_std + advantage_eps)
    elif advantage_normalizer == "mean":
        rewards = rewards / (rewards_mean + advantage_eps)

    advantages = rewards.view(-1).contiguous()

    return (advantages, {})
