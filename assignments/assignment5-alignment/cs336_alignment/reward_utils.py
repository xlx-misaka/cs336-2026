from collections.abc import Callable

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
