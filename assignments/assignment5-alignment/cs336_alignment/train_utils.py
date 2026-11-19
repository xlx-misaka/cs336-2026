import json
import math
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import torch
import wandb
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import PreTrainedTokenizerBase

from cs336_alignment.checkpoint import get_model_and_tokenizer
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
from cs336_alignment.vllm_utils import VLLMServer


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

    advantages, _ = compute_group_normalized_rewards(
        rewards,
        group_size,
        baseline,
        advantage_eps,
        advantage_normalizer,
    )

    none_zero_mask = advantages != 0
    input_ids = input_ids[none_zero_mask]
    labels = labels[none_zero_mask]
    advantages = advantages[none_zero_mask]
    response_mask = response_mask[none_zero_mask]

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
            advantages[i : min(len(input_ids), i + microbatch_size)],
            log_probs,
            importance_reweighting_method,
            old_log_probs,
            cliprange,
            response_mask[i : min(len(input_ids), i + microbatch_size)],
        )

        loss_microbatch = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss,
            response_mask[i : min(len(input_ids), i + microbatch_size)],
            loss_normalization,
            normalization_constant,
        )
        if loss_normalization == "sequence":
            loss_microbatch = (
                loss_microbatch * (len(inputs_microbatch)) / len(input_ids)
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


_REQUIRED_TRAINING_KEYS = (
    "learning_rate",
    "betas",
    "weight_decay",
    "rollout_batch_size",
    "train_batch_size",
    "group_size",
    "gradient_accumulation_steps",
    "max_grad_norm",
    "num_rollout_steps",
    "n_train_examples",
    "n_val_examples",
    "eval_interval",
    "log_rollouts_interval",
)

_REQUIRED_SAMPLING_KEYS = ("temperature", "max_tokens", "seed")

_ALLOWED_SAMPLING_KEYS = {
    "temperature",
    "top_p",
    "max_tokens",
    "n",
    "seed",
    "stop",
    "include_stop_str_in_output",
}


def _check_hyperparameters(
    sampling_hyperparameters: dict[str, Any], training_hyperparameters: dict[str, Any]
) -> None:
    """Validate sampling/training hyperparameters before any model/GPU work."""
    if not isinstance(training_hyperparameters, dict):
        raise TypeError("training_hyperparameters must be a dict")
    if not isinstance(sampling_hyperparameters, dict):
        raise TypeError("sampling_hyperparameters must be a dict")

    missing = [
        key for key in _REQUIRED_TRAINING_KEYS if key not in training_hyperparameters
    ]
    if missing:
        raise ValueError(
            f"training_hyperparameters is missing required keys: {missing}"
        )
    unknown_training = sorted(
        set(training_hyperparameters) - set(_REQUIRED_TRAINING_KEYS)
    )
    if unknown_training:
        raise ValueError(
            f"training_hyperparameters has unknown keys: {unknown_training}"
        )

    missing_sampling = [
        key for key in _REQUIRED_SAMPLING_KEYS if key not in sampling_hyperparameters
    ]
    if missing_sampling:
        raise ValueError(
            f"sampling_hyperparameters is missing required keys: {missing_sampling}"
        )
    unknown_sampling = sorted(set(sampling_hyperparameters) - _ALLOWED_SAMPLING_KEYS)
    if unknown_sampling:
        raise ValueError(
            f"sampling_hyperparameters has unknown keys: {unknown_sampling}"
        )

    def _positive_int(key: str) -> int:
        value = training_hyperparameters[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"training_hyperparameters[{key!r}] must be an integer, got {value!r}"
            )
        if value <= 0:
            raise ValueError(
                f"training_hyperparameters[{key!r}] must be a positive integer, "
                f"got {value!r}"
            )
        return value

    def _finite_number(key: str, *, allow_zero: bool) -> float:
        value = training_hyperparameters[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"training_hyperparameters[{key!r}] must be a number, got {value!r}"
            )
        number = float(value)
        invalid = (
            not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero)
        )
        if invalid:
            kind = "non-negative" if allow_zero else "positive"
            raise ValueError(
                f"training_hyperparameters[{key!r}] must be a {kind} finite "
                f"number, got {value!r}"
            )
        return number

    group_size = _positive_int("group_size")
    if group_size < 2:
        raise ValueError(
            "training_hyperparameters['group_size'] must be at least 2, "
            f"got {group_size}"
        )

    for key in ("rollout_batch_size", "train_batch_size"):
        batch_size = _positive_int(key)
        if batch_size % group_size != 0 or batch_size < group_size:
            raise ValueError(
                f"training_hyperparameters[{key!r}] must be a multiple of "
                f"group_size ({group_size}) and at least group_size, "
                f"got {batch_size}"
            )

    for key in (
        "gradient_accumulation_steps",
        "num_rollout_steps",
        "n_train_examples",
        "n_val_examples",
        "eval_interval",
        "log_rollouts_interval",
    ):
        _positive_int(key)

    _finite_number("learning_rate", allow_zero=False)
    _finite_number("weight_decay", allow_zero=True)

    if training_hyperparameters["max_grad_norm"] is not None:
        _finite_number("max_grad_norm", allow_zero=False)

    betas = training_hyperparameters["betas"]
    if not isinstance(betas, (tuple, list)):
        raise TypeError(
            f"training_hyperparameters['betas'] must be a pair of floats, got {betas!r}"
        )
    if len(betas) != 2:
        raise ValueError(
            f"training_hyperparameters['betas'] must be a pair of floats, got {betas!r}"
        )
    for beta in betas:
        if isinstance(beta, bool) or not isinstance(beta, (int, float)):
            raise TypeError(
                "training_hyperparameters['betas'] entries must be numbers, "
                f"got {betas!r}"
            )
        if not 0.0 <= float(beta) < 1.0:
            raise ValueError(
                "training_hyperparameters['betas'] entries must be numbers "
                f"in [0, 1), got {betas!r}"
            )

    temperature = sampling_hyperparameters["temperature"]
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError(
            f"sampling_hyperparameters['temperature'] must be a number, "
            f"got {temperature!r}"
        )
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError(
            "sampling_hyperparameters['temperature'] must be a positive finite "
            f"number; greedy sampling is incompatible with group_size={group_size}"
        )

    max_tokens = sampling_hyperparameters["max_tokens"]
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise TypeError(
            f"sampling_hyperparameters['max_tokens'] must be an integer, "
            f"got {max_tokens!r}"
        )
    if max_tokens <= 0:
        raise ValueError(
            f"sampling_hyperparameters['max_tokens'] must be positive, "
            f"got {max_tokens!r}"
        )

    seed = sampling_hyperparameters["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            f"sampling_hyperparameters['seed'] must be an integer, got {seed!r}"
        )

    if "n" in sampling_hyperparameters:
        n = sampling_hyperparameters["n"]
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(
                f"sampling_hyperparameters['n'] must be an integer, got {n!r}"
            )
        if n <= 0:
            raise ValueError(
                f"sampling_hyperparameters['n'] must be positive, got {n!r}"
            )
        if n != group_size:
            raise ValueError(
                "sampling_hyperparameters['n'] is overridden by group_size "
                f"({group_size}); omit it or set n={group_size}, got {n}"
            )

    if "top_p" in sampling_hyperparameters:
        top_p = sampling_hyperparameters["top_p"]
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
            raise TypeError(
                f"sampling_hyperparameters['top_p'] must be a number, got {top_p!r}"
            )
        if not math.isfinite(float(top_p)) or not 0.0 < float(top_p) <= 1.0:
            raise ValueError(
                f"sampling_hyperparameters['top_p'] must be in (0, 1], got {top_p!r}"
            )
        if float(top_p) != 1.0:
            raise ValueError(
                "sampling_hyperparameters['top_p'] is not forwarded to vLLM; "
                "only top_p=1.0 is currently supported"
            )

    stop = sampling_hyperparameters.get("stop")
    if stop is not None and (
        not isinstance(stop, (list, tuple))
        or not all(isinstance(item, str) and item for item in stop)
    ):
        raise ValueError(
            "sampling_hyperparameters['stop'] must be a list of non-empty "
            f"strings or None, got {stop!r}"
        )

    include_stop = sampling_hyperparameters.get("include_stop_str_in_output")
    if include_stop is not None and not isinstance(include_stop, bool):
        raise TypeError(
            "sampling_hyperparameters['include_stop_str_in_output'] must be a "
            f"bool, got {include_stop!r}"
        )


def train_grpo(
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
    group_name: str | None = None,
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
            top_p (float): 采样 top-p，参考值为 1.0。vllm_utils 目前不转发
                该字段，只支持 1.0。
            max_tokens (int): 每条回答的最大生成 token 数，参考值为 512。
            n (int): 每个输入 prompt 生成的回答数。当前实现统一使用
                group_size；如果提供，必须等于 group_size，否则会报错。
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
        group_name (str | None): 实验组名称，默认 None。

    Notes:
        算法选项通过函数参数传入，未显式传参时使用上述默认值。
        写入 W&B config 的是 model_id、sampling_hyperparameters、
        training_hyperparameters 以及算法参数；policy_device、vllm_gpu、
        vllm_port、old_log_probs 不会写入。
        batch size 按回答数计：256 条回答、group_size 为 8，对应 32 个问题。
    """
    _check_hyperparameters(sampling_hyperparameters, training_hyperparameters)

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
        name=f"{experiment_name}",
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
        group=group_name,
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

        server.sync_policy_weights(model)

        metrics = {
            "loss": round(loss.item(), 4),
            "grad_norm": round(float(train_log["gradient_norm"]), 3),
            "entropy": round(float(train_log["token_entropy"]), 3),
            "reward": round(float(train_log["mean_reward"]), 3),
            "format_reward": round(float(train_log["mean_format_reward"]), 3),
            "learning_rate": training_hyperparameters["learning_rate"],
        }
        print(
            f"step {step}/{training_hyperparameters['num_rollout_steps']}: {metrics}",
            flush=True,
        )

        log_data: dict[str, Any] = {
            "train/" + key: float(val) if isinstance(val, torch.Tensor) else val
            for key, val in train_log.items()
        } | {"train/loss": loss.item()}

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
            print("***** Running evaluation *****", flush=True)

            eval_start = time.monotonic()
            mean_reward = 0
            mean_format_reward = 0
            mean_response_length = 0
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
                f"eval_pass@{training_hyperparameters['group_size']}": round(
                    pass_k[f"eval/pass@{training_hyperparameters['group_size']}"], 3
                ),
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
