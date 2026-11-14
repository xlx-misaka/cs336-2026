from collections.abc import Callable

import torch


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rollout rewards for a list of responses and ground truths."""
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
            "mean total rewards": sum(r["reward"] for r in results) / len(results),
            "mean format rewards": sum(r["format_reward"] for r in results)
            / len(results),
        },
    )
