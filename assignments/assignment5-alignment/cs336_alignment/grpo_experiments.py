from datetime import datetime
from typing import Any

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.train_utils import train_grpo


def _run_grpo_standard_on_policy_experiments(
    seed: int = 0,
    training_hyperparameters_overrides: dict[str, Any] | None = None,
    experiment_name: str = "standard-on-policy",
    group_name: str | None = None,
):
    model_id = "models/OLMo-2-0425-1B"

    with open("cs336_alignment/prompts/r1_zero.prompt") as f:
        prompt = f.read()

    reward_fn = r1_zero_reward_fn
    train_set_path = "data/gsm8k/train.jsonl"
    valid_set_path = "data/gsm8k/test.jsonl"
    sampling_hyperparameters = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "seed": seed,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True,
    }
    training_hyperparameters = {
        "learning_rate": 1e-5,
        "betas": (0.9, 0.95),
        "weight_decay": 0.0,
        "rollout_batch_size": 256,
        "train_batch_size": 256,
        "group_size": 8,
        "gradient_accumulation_steps": 32,
        "max_grad_norm": 1,
        "num_rollout_steps": 200,
        "n_train_examples": 6400,
        "n_val_examples": 1024,
        "eval_interval": 10,
        "log_rollouts_interval": 40,
        **(training_hyperparameters_overrides or {}),
    }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    train_grpo(
        model_id,
        prompt,
        reward_fn,
        train_set_path,
        valid_set_path,
        sampling_hyperparameters,
        training_hyperparameters,
        vllm_gpu=1,
        experiment_name=f"{experiment_name}-{timestamp}",
        group_name=group_name,
    )


def _run_smoke(seed: int = 0) -> None:
    print("***** Running smoke test *****", flush=True)

    _run_grpo_standard_on_policy_experiments(
        seed,
        training_hyperparameters_overrides={
            "num_rollout_steps": 2,
            "n_train_examples": 64,
            "n_val_examples": 32,
            "eval_interval": 1,
            "log_rollouts_interval": 1,
        },
        experiment_name="standard-on-policy-smoke",
    )


def _run_correctness_check(seed: int = 0, steps: int = 50) -> None:
    print(
        "***** Running correctness check *****",
        flush=True,
    )

    _run_grpo_standard_on_policy_experiments(
        seed,
        training_hyperparameters_overrides={
            "num_rollout_steps": steps,
            "eval_interval": max(1, steps // 5),
            "log_rollouts_interval": max(1, steps // 5),
        },
        experiment_name="standard-on-policy-correctness",
    )


def _run_4_seed_experiments():
    print("***** Running full run (4 seeds x 200 steps) *****", flush=True)

    for seed in range(4):
        _run_grpo_standard_on_policy_experiments(
            seed,
            experiment_name=f"standard-on-policy-seed-{seed}",
            group_name="4 seed experiments",
        )


def _run_learning_rate_ablation():
    print(
        "***** Running learning rate experiments *****",
        flush=True,
    )

    for learning_rate in [3e-6, 1e-5, 3e-5, 1e-4]:
        _run_grpo_standard_on_policy_experiments(
            training_hyperparameters_overrides={
                "learning_rate": learning_rate,
            },
            experiment_name=f"standard-on-policy-learning-rate-{learning_rate:.0e}",
            group_name="Learning rate ablation experiments",
        )


if __name__ == "__main__":
    ...
