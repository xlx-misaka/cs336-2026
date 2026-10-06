"""序列数据批处理与训练检查点持久化工具。

本模块保持小而明确的实现：``get_batch`` 从一维整数 token 数据集中采样连续
窗口，构造语言模型的输入/标签对；``save_checkpoint`` 与 ``load_checkpoint``
使用固定的字典结构封装 ``torch.save``/``torch.load``，以便同时恢复模型与优化器状态。
"""

import csv
import json
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any, BinaryIO

import numpy as np
import numpy.typing as npt
import torch

from src.utils import nn
from src.utils.optim import cross_entropy


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """从一维 token 数据集中随机采样语言模型输入与标签。

    对每个样本，随机起始索引 ``s`` 从有效区间
    ``[0, len(dataset) - context_length)`` 均匀采样。输入序列为
    ``dataset[s:s + context_length]``，标签序列为
    ``dataset[s + 1:s + context_length + 1]``。

    参数
    --------
    dataset:
        一维 NumPy 整数数组，元素为 token ID；长度至少为
        ``context_length + 1``。
    batch_size:
        采样样本数。必须大于 0。
    context_length:
        每个样本的输入 token 数。必须大于 0，且不超过
        ``len(dataset) - 1``。
    device:
        返回张量所在的 PyTorch 设备，例如 ``"cpu"``、``"cuda"`` 或
        ``"cuda:0"``。

    返回
    --------
    tuple[torch.Tensor, torch.Tensor]
        返回 ``(x, y)`` 二元组。
        ``x`` 形状为 ``(batch_size, context_length)``、dtype 为
        ``torch.int64``，保存输入 token ID。
        ``y`` 形状与 dtype 与 ``x`` 相同，保存向右平移一位后的目标 token ID。

    异常
    --------
    AssertionError
        当 ``len(dataset)`` 不大于 ``context_length`` 时抛出。
    """
    num_possible_starting_indices = len(dataset) - context_length

    assert num_possible_starting_indices > 0, "dataset 长度必须大于 context_length"

    possible_starting_indices = np.random.randint(0, num_possible_starting_indices, size=batch_size)

    x_indices = possible_starting_indices[:, None] + np.arange(0, context_length)
    y_indices = possible_starting_indices[:, None] + np.arange(1, context_length + 1)

    x = torch.from_numpy(dataset[x_indices]).to(dtype=torch.int64, device=device)
    y = torch.from_numpy(dataset[y_indices]).to(dtype=torch.int64, device=device)

    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """将模型状态、优化器状态与迭代数序列化为检查点。

    检查点会被写成一个字典，保留键 ``"model_state_dict"``、
    ``"optimizer_state_dict"`` 与 ``"iteration"``。

    参数
    --------
    model:
        需要持久化 ``state_dict()`` 的模型。
    optimizer:
        需要持久化 ``state_dict()`` 的优化器。
    iteration:
        序列化时已经完成的训练迭代数。
    out:
        目标路径或 ``torch.save`` 接受的二进制文件对象。

    返回
    --------
    None
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: str | torch.device | None = None,
) -> int:
    """从检查点恢复模型与优化器状态。

    参数
    --------
    src:
        检查点路径或 ``torch.load`` 接受的二进制文件对象。
    model:
        加载序列化状态的目标模型；状态会被原地写入。
    optimizer:
        加载序列化状态的目标优化器；状态会被原地写入。
    map_location:
        传入 ``torch.load`` 的设备映射，用于在不同设备间迁移检查点，
        例如在 GPU 上保存、之后在 CPU 上恢复。默认 ``None`` 表示保持
        检查点原来的设备。

    返回
    --------
    int
        检查点保存时记录的迭代数。

    异常
    --------
    KeyError
        当检查点字典缺失预期键时抛出。
    """
    checkpoint: dict[str, Any] = torch.load(src, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["iteration"]


def format_duration(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}时{minutes:02d}分{seconds:02d}秒"


def load_texts_iterable(input_path: str, boundaries: list[str]) -> Iterable[str]:

    with open(file=input_path, encoding="utf-8") as f:
        text = []
        while True:
            line = f.readline()
            if line == "":
                break
            text.append(line)
            if line.strip() in boundaries:
                yield "".join(text)
                text = []
        if text:
            yield "".join(text)


@torch.no_grad()
def estimate_loss(
    model: nn.TransformerLM,
    dataset_valid: np.ndarray,
    batch_size: int,
    eval_iters: int,
    context_length: int,
    device: str,
) -> float:

    model_state = model.training

    model.eval()

    losses = torch.zeros(eval_iters, device=torch.device(device))

    for k in range(eval_iters):
        x, y = get_batch(dataset_valid, batch_size=batch_size, context_length=context_length, device=device)
        logits = model(x)
        losses[k] = cross_entropy(logits, y).item()

    model.train(model_state)

    return float(losses.mean().item())


@torch.no_grad()
def decoding(
    model: nn.TransformerLM,
    input_ids: torch.Tensor,
    end_tokens: list[int],
    max_generation_length: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    context_length: int = 256,
) -> torch.Tensor:

    assert input_ids.dim() == 1, "input_ids 必须是 1D 张量"

    assert temperature > 0, "temperature 必须大于 0"

    if top_p is not None:
        assert 0.0 < top_p <= 1.0, "top_p 必须在 (0, 1] 范围内"

    model_state = model.training

    model.eval()

    end_tokens_set = set(end_tokens)

    for _ in range(max_generation_length):
        x = input_ids[-context_length:]

        logits = model(x)

        next_logits = logits[..., -1, :]

        next_distribution = nn.softmax(next_logits / temperature, dim=-1)

        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(next_distribution, dim=-1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)

            keep = (cumsum - sorted_probs) < top_p
            keep[..., 0] = True

            sorted_probs = sorted_probs.masked_fill(~keep, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            sampled = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(dim=-1, index=sampled)

        else:
            next_token = torch.multinomial(next_distribution, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if int(next_token.item()) in end_tokens_set:
            break

    model.train(model_state)

    return input_ids


class ExperimentLogger:
    def __init__(
        self,
        experiment_name: str,
        config: dict[str, Any],
        base_dir: str | Path = "experiments",
    ) -> None:
        self.start_time = time.perf_counter()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / experiment_name / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # 保存本次实验完整配置
        config_path = self.run_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        self.metrics_path = self.run_dir / "metrics.csv"
        self.metrics_file = self.metrics_path.open("w", newline="", encoding="utf-8")

        self.fieldnames = [
            "step",
            "elapsed_seconds",
            "train_loss",
            "val_loss",
            "lr",
            "tokens_seen",
        ]
        self.writer = csv.DictWriter(self.metrics_file, fieldnames=self.fieldnames)
        self.writer.writeheader()

    def log(
        self,
        step: int,
        train_loss: float | None = None,
        val_loss: float | None = None,
        lr: float | None = None,
        tokens_seen: int | None = None,
    ) -> None:
        row = {
            "step": step,
            "elapsed_seconds": int(time.perf_counter() - self.start_time),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "tokens_seen": tokens_seen,
        }

        self.writer.writerow(row)
        self.metrics_file.flush()

    def close(self) -> None:
        self.metrics_file.close()
