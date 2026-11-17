import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import wandb
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.train_utils import grpo_train_step
from cs336_alignment.vllm_utils import VLLMServer


class _RecordDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _get_data_loader(
    data_path: str | Path,
    batch_size: int,
    n_examples: int,
    seed: int,
    shuffle: bool = False,
):
    if n_examples <= 0:
        raise ValueError("n_examples must be non-negative.")

    with open(data_path, encoding="utf-8") as f:
        rng = random.Random(seed)
        dataset = _RecordDataset([json.loads(line) for line in f])
        if len(dataset) < n_examples:
            raise ValueError(
                f"n_examples ({n_examples}) is greater than the number of examples in the dataset ({len(dataset)})"
            )
        subset = Subset(
            dataset,
            rng.sample(range(len(dataset)), k=n_examples),
        )
        data_loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        return data_loader


def _get_rollout_params(
    server: VLLMServer,
    batch: dict,
    training_hyperparameters: dict,
    sampling_hyperparameters: dict,
    prompt: str,
):
    prompts = [prompt.format(question=question) for question in batch["question"]]
    completions = server.generate_completions(
        prompts,
        {
            **sampling_hyperparameters,
            "n": training_hyperparameters["group_size"],
        },
    )
    repeated_prompts = [
        prompt
        for prompt in prompts
        for _ in range(training_hyperparameters["group_size"])
    ]
    rollout_responses = [completion.text for completion in completions]
    repeated_ground_truths = [
        answer.rsplit("####", 1)[-1].strip()
        for answer in batch["answer"]
        for _ in range(training_hyperparameters["group_size"])
    ]
    return repeated_prompts, rollout_responses, repeated_ground_truths


def grpo_experiments_standard_on_policy(
    model_id: str,
    prompt: str,
    reward_fn: Callable[[str, str], dict[str, float]],
    train_set_path: Path | str,
    valid_set_path: Path | str,
    sampling_hyperparameters: dict[str, Any],
    training_hyperparameters: dict[str, Any],
    device: str = "cuda",
):
    """运行标准 on-policy GRPO 实验。

    Args:
        model_id (str): Hugging Face 模型 ID 或本地模型目录。
        prompt (str): 包含问题占位符的提示模板。
        train_set_path (Path | str): 训练集 JSONL 文件路径。
        valid_set_path (Path | str): 验证集 JSONL 文件路径。
        sampling_hyperparameters (dict[str, Any]): 回答生成配置，包含以下字段：
            temperature (float): 采样温度，参考值为 1.0。
            max_tokens (int): 每条回答的最大生成 token 数，参考值为 512。
            n (int): 每个输入 prompt 生成的回答数。问题只输入一次时可设为
                group_size；prompt 已重复 group_size 次时应设为 1。
            seed (int): 当前实验使用的生成随机种子。
            stop (list[str]): 停止字符串，需匹配提示模板。
                r1_zero 使用 ["</answer>"]。
            include_stop_str_in_output (bool): 是否保留停止字符串。
                r1_zero 使用 True。
        training_hyperparameters (dict[str, Any]): 训练及实验流程配置，包含以下字段：
            learning_rate (float): 学习率，参考值为 1e-5。
            betas (tuple[float, float]): AdamW 的 beta 参数，参考值为 (0.9, 0.95)。
            weight_decay (float): 权重衰减，参考值为 0.0。
            rollout_batch_size (int): 每轮生成的回答总数，参考值为 256。
            train_batch_size (int): 每次完整训练更新的回答数，参考值为 256。
            group_size (int): 每个问题对应的回答数，参考值为 8。
            gradient_accumulation_steps (int): 梯度累积步数，参考值为 32。
            max_grad_norm (float | None): 梯度裁剪阈值，参考值为 1.0。
                None 表示不裁剪。
            num_rollout_steps (int): rollout 循环次数，参考值为 200。
            n_train_examples (int): 使用的训练题目数，参考值为 6400。
            n_val_examples (int): 验证题目数，参考值为 1024。
            num_rollout_steps (int): 训练的总步数，参考值为 200。
            eval_interval (int): 评测间隔，以 rollout batch 数计，参考值为 10。
            log_rollouts_interval (int): 保存生成文本的间隔，以 rollout batch
                数计，参考值为 40。
            baseline (str): 组内奖励基线，标准 GRPO 使用 "mean"。
            advantage_normalizer (str): advantage 归一化方式，标准 GRPO 使用 "std"。
            advantage_eps (float): 归一化时防止除零的小常数，参考值为 1e-6。
            importance_reweighting_method (str): 重要性加权方式，on-policy
                实验使用 "none"。
            loss_normalization (str): loss 聚合方式，标准 GRPO 使用 "sequence"。
            old_log_probs (torch.Tensor): 旧的 log_probs，用于 on-policy
                实验。
            cliprange (float): clip range，用于 PPO 等算法。
            normalization_constant (int): 归一化常数，用于 sequence 归一化。


    Notes:
        参考值不会自动填充为默认参数，需要在相应组件中显式读取配置。
        记录到 W&B config 不会自动应用参数。
        batch size 按回答数计：256 条回答、group_size 为 8，对应 32 个问题。
    """
    if (
        training_hyperparameters["rollout_batch_size"]
        % training_hyperparameters["group_size"]
        != 0
        or training_hyperparameters["rollout_batch_size"]
        < training_hyperparameters["group_size"]
    ):
        raise ValueError(
            "rollout_batch_size must be divisible by group_size and greater than group_size."
        )
    if (
        training_hyperparameters["train_batch_size"]
        < training_hyperparameters["group_size"]
        or training_hyperparameters["train_batch_size"]
        % training_hyperparameters["group_size"]
        != 0
    ):
        raise ValueError(
            "train_batch_size must be greater than or equal to group_size and divisible by group_size."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device=device)  # type: ignore
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_hyperparameters["learning_rate"],
        betas=training_hyperparameters["betas"],
        weight_decay=training_hyperparameters["weight_decay"],
    )
    server = VLLMServer(model_id)
    train_loader = _get_data_loader(
        train_set_path,
        training_hyperparameters["rollout_batch_size"]
        // training_hyperparameters["group_size"],
        training_hyperparameters["n_train_examples"],
        sampling_hyperparameters["seed"],
    )
    valid_loader = _get_data_loader(
        valid_set_path,
        training_hyperparameters["rollout_batch_size"]
        // training_hyperparameters["group_size"],
        training_hyperparameters["n_val_examples"],
        sampling_hyperparameters["seed"],
    )
    run = wandb.init(
        project="cs336-grpo",
        name="standard-on-policy",
        config={
            "model_id": model_id,
            "sampling": sampling_hyperparameters,
            "training": training_hyperparameters,
        },
    )

    server.start()
    server.init_weight_sync(device)

    train_iter = iter(train_loader)
    for step in range(1, training_hyperparameters["num_rollout_steps"] + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        repeated_prompts, rollout_responses, repeated_ground_truths = (
            _get_rollout_params(
                server,
                batch,
                training_hyperparameters,
                sampling_hyperparameters,
                prompt,
            )
        )

        loss, train_log = grpo_train_step(
            model,
            tokenizer,
            optimizer,
            gradient_accumulation_steps=training_hyperparameters[
                "gradient_accumulation_steps"
            ],
            max_grad_norm=training_hyperparameters["max_grad_norm"],
            reward_fn=reward_fn,
            repeated_prompts=repeated_prompts,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=training_hyperparameters["group_size"],
            baseline=training_hyperparameters["baseline"],
            advantage_eps=training_hyperparameters["advantage_eps"],
            advantage_normalizer=training_hyperparameters["advantage_normalizer"],
            importance_reweighting_method=training_hyperparameters[
                "importance_reweighting_method"
            ],
            old_log_probs=training_hyperparameters["old_log_probs"],
            cliprange=training_hyperparameters["cliprange"],
            loss_normalization=training_hyperparameters["loss_normalization"],
            normalization_constant=training_hyperparameters["normalization_constant"],
        )

        print(f"step: {step}, loss: {loss}, mean_reward: {train_log['mean_reward']}")
        log_data: dict[str, Any] = {
            "train/" + key: val for key, val in train_log.items()
        }

        server.sync_policy_weights(model)

        if step % training_hyperparameters["log_rollouts_interval"] == 0:
            rollout_rewards = [
                reward_fn(response, ground_truth)["reward"]
                for response, ground_truth in zip(
                    rollout_responses, repeated_ground_truths
                )
            ]
            log_data["rollouts"] = wandb.Table(
                columns=["prompt", "response", "reward"],
                data=[
                    [prompt, response, reward]
                    for prompt, response, reward in zip(
                        repeated_prompts, rollout_responses, rollout_rewards
                    )
                ],
            )

        if step % training_hyperparameters["eval_interval"] == 0:
            mean_reward = 0
            accuracy = 0
            pass_k = [0] * training_hyperparameters["group_size"]

            for batch in valid_loader:
                repeated_prompts, rollout_responses, repeated_ground_truths = (
                    _get_rollout_params(
                        server,
                        batch,
                        training_hyperparameters,
                        sampling_hyperparameters,
                        prompt,
                    )
                )
                for i in range(
                    0, len(rollout_responses), training_hyperparameters["group_size"]
                ):
                    flag = False
                    for j in range(training_hyperparameters["group_size"]):
                        reward_metadata = reward_fn(
                            rollout_responses[i + j], repeated_ground_truths[i + j]
                        )
                        if reward_metadata["answer_reward"] == 1:
                            flag = True
                            accuracy += 1
                        pass_k[j] += flag
                        mean_reward += reward_metadata["reward"]
            mean_reward /= (
                training_hyperparameters["n_val_examples"]
                * training_hyperparameters["group_size"]
            )
            accuracy /= (
                training_hyperparameters["n_val_examples"]
                * training_hyperparameters["group_size"]
            )
            pass_k = {
                f"eval/pass@{i + 1}": x / training_hyperparameters["n_val_examples"]
                for i, x in enumerate(pass_k)
            }

            print(f"eval, mean_reward: {mean_reward}, accuracy: {accuracy}")
            log_data.update(
                {
                    "eval/mean_reward": mean_reward,
                    "eval/accuracy": accuracy,
                }
                | pass_k
            )

        run.log(log_data, step=step)

    server.stop()
    run.finish()
