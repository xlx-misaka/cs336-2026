import json
import random
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
import wandb
from torch.utils.data import DataLoader, Dataset, Subset

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
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


def grpo_experiments(
    model_id: str,
    prompt: str,
    reward_fn: Callable[[str, str], dict[str, float]],
    train_set_path: Path | str,
    valid_set_path: Path | str,
    sampling_hyperparameters: dict[str, Any],
    training_hyperparameters: dict[str, Any],
    policy_device: str = "cuda:0",
    vllm_gpu: int = 1,
    vllm_port: int = 8000,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    normalization_constant: int | None = None,
    experiment_name: str = "standard-on-policy",
):
    """运行标准 on-policy GRPO 实验。

    Args:
        model_id (str): Hugging Face 模型 ID 或本地模型目录。
        prompt (str): 包含问题占位符的提示模板。
        reward_fn (Callable[[str, str], dict[str, float]]): 奖励函数，返回
            "reward"、"format_reward"、"answer_reward" 三个字段。r1_zero
            使用 drgrpo_grader.r1_zero_reward_fn，question_only 使用
            drgrpo_grader.question_only_reward_fn。
        train_set_path (Path | str): 训练集 JSONL 文件路径。
        valid_set_path (Path | str): 验证集 JSONL 文件路径。
        sampling_hyperparameters (dict[str, Any]): 回答生成配置，包含以下字段：
            temperature (float): 采样温度，参考值为 1.0。
            max_tokens (int): 每条回答的最大生成 token 数，参考值为 512。
            n (int): 每个输入 prompt 生成的回答数。当前实现会忽略该字段，
                统一设为 group_size；每个问题只向 vLLM 发送一次。
            seed (int): 训练集/验证集采样与生成共用的随机种子。
            stop (list[str]): 停止字符串，需匹配提示模板。
                r1_zero 使用 ["</answer>"]。
            include_stop_str_in_output (bool): 是否保留停止字符串。
                r1_zero 使用 True。
        training_hyperparameters (dict[str, Any]): 训练及实验流程配置。
            必填字段：
            learning_rate (float): 学习率，参考值为 1e-5。
            betas (tuple[float, float]): AdamW 的 beta 参数，参考值为 (0.9, 0.95)。
            weight_decay (float): 权重衰减，参考值为 0.0。
            rollout_batch_size (int): 每轮生成的回答总数，参考值为 256。
            train_batch_size (int): 每次完整训练更新的回答数，参考值为 256。
                注意：当前实现只做合法性校验，实际每次更新使用整个 rollout
                batch（rollout_batch_size 条回答）。
            group_size (int): 每个问题对应的回答数，参考值为 8。
            gradient_accumulation_steps (int): 梯度累积步数，参考值为 32。
            max_grad_norm (float | None): 梯度裁剪阈值，参考值为 1.0。
                None 表示不裁剪。
            num_rollout_steps (int): rollout 循环次数，参考值为 200。
            n_train_examples (int): 使用的训练题目数，参考值为 6400。
            n_val_examples (int): 验证题目数，参考值为 1024。
            eval_interval (int): 评测间隔，以 rollout batch 数计，参考值为 10。
            log_rollouts_interval (int): 保存生成文本的间隔，以 rollout batch
                数计，参考值为 40。
        policy_device (str): policy 模型所在设备，默认 "cuda:0"。必须带 GPU
            编号，不能只写 "cuda"。
        vllm_gpu (int): vLLM 服务使用的 GPU 编号，默认 1。需要与
            policy_device 使用不同的 GPU。
        vllm_port (int): vLLM HTTP 服务端口，默认 8000。
        baseline (Literal["mean", "none"]): 组内奖励基线，默认 "mean"。
        advantage_eps (float): 归一化时防止除零的小常数，默认 1e-6。
        advantage_normalizer (Literal["std", "none", "mean"]): advantage
            归一化方式，默认 "std"。
        importance_reweighting_method (Literal["none", "noclip", "grpo", "gspo"]):
            重要性加权方式，默认 "none"。
        loss_normalization (Literal["sequence", "constant"]): loss 聚合方式，
            默认 "sequence"。
        old_log_probs (torch.Tensor | None): 旧的 log_probs，默认 None；
            on-policy 实验不需要。
        cliprange (float | None): clip range，默认 None；
            importance_reweighting_method 为 "grpo" 或 "gspo" 时使用。
        normalization_constant (int | None): 归一化常数，默认 None；
            用于 "constant" 聚合方式。
        experiment_name (str): 实验名称，默认为"standard-on-policy"。

    Notes:
        算法选项通过函数参数传入，未显式传参时使用上述默认值。
        写入 W&B config 的是 model_id、sampling_hyperparameters、
        training_hyperparameters 以及算法参数；policy_device、vllm_gpu、
        vllm_port、old_log_probs 不会写入。
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

    print("***** Running training *****", flush=True)
    print(f"run = {experiment_name}, model = {model_id}", flush=True)
    print(
        f"devices = policy={policy_device}, vllm=gpu{vllm_gpu}:{vllm_port}",
        flush=True,
    )
    print(
        f"schedule = steps={training_hyperparameters['num_rollout_steps']}, "
        f"rollout={training_hyperparameters['rollout_batch_size']}, "
        f"group={training_hyperparameters['group_size']}, "
        f"accum={training_hyperparameters['gradient_accumulation_steps']}",
        flush=True,
    )
    print(
        f"data = n_train={training_hyperparameters['n_train_examples']}, "
        f"n_val={training_hyperparameters['n_val_examples']}, "
        f"seed={sampling_hyperparameters['seed']}, "
        f"lr={training_hyperparameters['learning_rate']}",
        flush=True,
    )
    model, tokenizer = get_model_and_tokenizer(model_id, policy_device)
    policy_param = next(model.parameters())
    print(f"policy = {policy_param.dtype} on {policy_param.device}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_hyperparameters["learning_rate"],
        betas=training_hyperparameters["betas"],
        weight_decay=training_hyperparameters["weight_decay"],
    )
    server = VLLMServer(
        model_id,
        gpu=vllm_gpu,
        port=vllm_port,
    )
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
        name=f"{experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "model_id": model_id,
            "sampling": sampling_hyperparameters,
            "training": training_hyperparameters,
            "algorithm": {
                "baseline": baseline,
                "advantage_eps": advantage_eps,
                "advantage_normalizer": advantage_normalizer,
                "importance_reweighting_method": importance_reweighting_method,
                "loss_normalization": loss_normalization,
                "cliprange": cliprange,
                "normalization_constant": normalization_constant,
            },
        },
    )

    print(f"***** Starting vLLM (gpu={vllm_gpu}, port={vllm_port}) *****", flush=True)
    server.start()
    print("***** vLLM ready *****", flush=True)
    print("***** Initializing weight sync *****", flush=True)
    server.init_weight_sync(policy_device)
    print("***** Weight sync ready *****", flush=True)

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
            baseline=baseline,
            advantage_eps=advantage_eps,
            advantage_normalizer=advantage_normalizer,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )

        loss_value = loss.item()
        log_data: dict[str, Any] = {
            "train/" + key: float(val) if isinstance(val, torch.Tensor) else val
            for key, val in train_log.items()
        } | {"train/loss": loss_value}

        server.sync_policy_weights(model)
        total_steps = training_hyperparameters["num_rollout_steps"]
        metrics = {
            "loss": round(loss_value, 4),
            "grad_norm": round(float(train_log["gradient_norm"]), 3),
            "entropy": round(float(train_log["token_entropy"]), 3),
            "reward": round(float(train_log["mean_reward"]), 3),
            "format_reward": round(float(train_log["mean_format_reward"]), 3),
            "learning_rate": training_hyperparameters["learning_rate"],
        }
        print(f"step {step}/{total_steps}: {metrics}", flush=True)

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
            eval_start = time.monotonic()
            group_size = training_hyperparameters["group_size"]
            n_val = training_hyperparameters["n_val_examples"]
            print("***** Running evaluation *****", flush=True)
            print(f"Num examples = {n_val}", flush=True)
            print(f"Num rollouts = {n_val * group_size}", flush=True)
            mean_reward = 0
            mean_format_reward = 0
            mean_response_length = 0
            accuracy = 0
            pass_k = [0] * group_size

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
                        mean_format_reward += reward_metadata["format_reward"]
                        mean_response_length += len(rollout_responses[i + j])
            mean_reward /= (
                training_hyperparameters["n_val_examples"]
                * training_hyperparameters["group_size"]
            )
            mean_format_reward /= (
                training_hyperparameters["n_val_examples"]
                * training_hyperparameters["group_size"]
            )
            mean_response_length /= (
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

            eval_metrics = {
                "eval_reward": round(mean_reward, 3),
                "eval_format_reward": round(mean_format_reward, 3),
                "eval_response_length_chars": round(mean_response_length, 1),
                "eval_accuracy": round(accuracy, 3),
                "eval_pass@1": round(pass_k["eval/pass@1"], 3),
                f"eval_pass@{group_size}": round(pass_k[f"eval/pass@{group_size}"], 3),
                "eval_runtime": round(time.monotonic() - eval_start, 1),
            }
            print(f"step {step} eval: {eval_metrics}", flush=True)
            log_data.update(
                {
                    "eval/mean_reward": mean_reward,
                    "eval/mean_format_reward": mean_format_reward,
                    "eval/mean_response_length": mean_response_length,
                    "eval/accuracy": accuracy,
                }
                | pass_k
            )

        run.log(log_data, step=step)

    server.stop()
    run.finish()


def _run_grpo_standard_on_policy_experiments(
    seed: int = 0,
    training_hyperparameters_overrides: dict[str, Any] | None = None,
    experiment_name: str = "standard-on-policy",
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

    grpo_experiments(
        model_id,
        prompt,
        reward_fn,
        train_set_path,
        valid_set_path,
        sampling_hyperparameters,
        training_hyperparameters,
        vllm_gpu=1,
        experiment_name=experiment_name,
    )


def _run_smoke(seed: int = 0) -> None:
    print(f"***** Running smoke test (seed={seed}) *****", flush=True)
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


def _run_full():
    print("***** Running full run (4 seeds x 200 steps) *****", flush=True)
    for seed in range(4):
        print(f"***** Running seed {seed} *****", flush=True)
        _run_grpo_standard_on_policy_experiments(
            seed,
            experiment_name=f"standard-on-policy-seed-{seed}",
        )


if __name__ == "__main__":
    ...
