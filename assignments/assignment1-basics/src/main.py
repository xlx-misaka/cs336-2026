import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils import nn
from src.utils.optim import AdamW, cross_entropy, gradient_clipping, lr_cosine_schedule
from src.utils.Tokenizer import BpeTokenizer, serialize_bpe_data, train_bpe
from src.utils.tools import (
    ExperimentLogger,
    decoding,
    estimate_loss,
    format_duration,
    get_batch,
    load_checkpoint,
    load_texts_iterable,
    save_checkpoint,
)

CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the JSON configuration file used by all training and generation tasks."""
    with Path(config_path).open(encoding="utf-8") as f:
        return json.load(f)


def resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_tokenizer(config: dict[str, Any], tokenizer_path: str | None = None) -> BpeTokenizer:
    tokenizer_path = tokenizer_path or config["encoding"]["tokenizer_path"]
    assert tokenizer_path is not None
    return BpeTokenizer.from_files(tokenizer_path, config["tokenizer"]["special_tokens"])


def build_model(config: dict[str, Any], device: str | None = None) -> nn.TransformerLM:
    model_config = config["model"]
    resolved_device = torch.device(resolve_device(device))
    return nn.TransformerLM(
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        rope_theta=model_config["rope_theta"],
        use_rms_norm=model_config.get("use_rms_norm", True),
        norm_position=model_config.get("norm_position", "pre"),
        use_rope=model_config.get("use_rope", True),
        ffn_type=model_config.get("ffn_type", "swiglu"),
        device=resolved_device,
    )


def build_optimizer(model: nn.TransformerLM, config: dict[str, Any]) -> AdamW:
    optimizer_config = config["training"]["optimizer"]
    return AdamW(
        model.parameters(),
        lr=config["training"]["max_learning_rate"],
        betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["eps"],
        weight_decay=optimizer_config["weight_decay"],
    )


def train_bpe_on_TinyStories(config: dict[str, Any]) -> None:
    bpe_config = config["bpe"]
    special_tokens = config["tokenizer"]["special_tokens"]
    start_time = time.perf_counter()

    vocab, merges = train_bpe(bpe_config["train_text_path"], bpe_config["vocab_size"], special_tokens)

    end_time = time.perf_counter()

    serialize_bpe_data(
        vocab,
        merges,
        bpe_config["output_path"],
    )

    print(f"训练用时:{format_duration(end_time - start_time)}")


def encode_tokens(config: dict[str, Any], split: str) -> None:
    encoding_config = config["encoding"]
    split_config = encoding_config[split]
    special_tokens = config["tokenizer"]["special_tokens"]
    buff_size = encoding_config["buff_size"]

    tokenizer = load_tokenizer(config)
    texts_iterable = load_texts_iterable(split_config["text_path"], special_tokens)
    total_len = 0

    with open(split_config["output_path"], mode="wb") as f:
        buff = []
        for id in tokenizer.encode_iterable(texts_iterable):
            buff.append(id)
            if len(buff) == buff_size:
                x = np.ascontiguousarray(buff, dtype=np.uint16)
                x.tofile(f)
                total_len += buff_size
                buff.clear()
        if buff:
            x = np.ascontiguousarray(buff, dtype=np.uint16)
            x.tofile(f)
            total_len += len(buff)

    print(f"总token:{total_len}")


def training_together_on_TinyStories(config: dict[str, Any]) -> None:
    model_config = config["model"]
    training_config = config["training"]

    vocab_size = model_config["vocab_size"]
    context_length = model_config["context_length"]
    max_l2_norm = training_config["max_l2_norm"]
    min_learning_rate = training_config["min_learning_rate"]
    max_learning_rate = training_config["max_learning_rate"]
    warmup_iters = training_config["warmup_iters"]
    eval_iters = training_config["eval_iters"]
    eval_batch_size = training_config["eval_batch_size"]
    num_steps = training_config["num_steps"]
    batch_size = training_config["batch_size"]
    log_interval = training_config["log_interval"]
    eval_interval = training_config["eval_interval"]
    checkpoint_interval = training_config["checkpoint_interval"]
    experiment_name = training_config["experiment_name"]
    train_path = training_config["train_path"]
    valid_path = training_config["valid_path"]
    resume_path = training_config["resume_path"]
    device = resolve_device(training_config["device"])

    dataset_train = np.memmap(train_path, mode="r", dtype=np.uint16)
    assert dataset_train.ndim == 1, "训练数据必须是一维 token id 数组"
    assert dataset_train.min() >= 0 and dataset_train.max() < vocab_size, "训练数据包含超出词表范围的 token id"

    if valid_path and Path(valid_path).exists():
        dataset_valid = np.memmap(valid_path, mode="r", dtype=np.uint16)
        assert dataset_valid.min() >= 0 and dataset_valid.max() < vocab_size, "验证数据包含超出词表范围的 token id"
    else:
        dataset_valid = None

    model = build_model(config, device)
    optimizer = build_optimizer(model, config)

    start_step = 1
    if resume_path:
        start_step = load_checkpoint(resume_path, model, optimizer, map_location=device) + 1
        print(f"从 checkpoint 恢复，从 step {start_step} 继续训练")

    config = {
        "experiment_name": experiment_name,
        "model": model_config,
        "training": training_config,
        "tokenizer": config.get("tokenizer", {}),
        "start_step": start_step,
        "task": "train",
        "device": device,
    }

    logger = ExperimentLogger(experiment_name, config)
    checkpoint_dir = logger.run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"实验目录: {logger.run_dir}")

    start_time = time.perf_counter()

    for step in range(start_step, num_steps + 1):
        x, y = get_batch(dataset_train, batch_size, context_length=context_length, device=device)

        logits = model(x)

        loss = cross_entropy(logits, y)

        optimizer.zero_grad()

        loss.backward()

        gradient_clipping(model.parameters(), max_l2_norm=max_l2_norm)

        lr = lr_cosine_schedule(step - 1, max_learning_rate, min_learning_rate, warmup_iters, num_steps)

        optimizer.param_groups[0]["lr"] = lr

        optimizer.step()

        if step % log_interval == 0:
            tokens_seen = batch_size * step * context_length
            val_loss = None

            if step % eval_interval == 0 and dataset_valid is not None:
                val_loss = estimate_loss(
                    model=model,
                    dataset_valid=dataset_valid,
                    batch_size=eval_batch_size,
                    eval_iters=eval_iters,
                    context_length=context_length,
                    device=device,
                )

            elapsed = time.perf_counter() - start_time
            log_message = (
                f"step:{step}, loss:{loss.item():.6f}, lr:{lr:.6e}, "
                f"tokens:{tokens_seen}, elapsed:{format_duration(elapsed)}"
            )
            if val_loss is not None:
                log_message += f", val_loss:{val_loss:.6f}"

            print(log_message)

            logger.log(
                step=step,
                train_loss=loss.item(),
                val_loss=val_loss,
                lr=lr,
                tokens_seen=tokens_seen,
            )

        if step % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step, checkpoint_dir / f"step_{step:06d}.pt")

    end_time = time.perf_counter()

    total_time = end_time - start_time

    print(f"训练用时:{format_duration(total_time)}")

    logger.close()


def generate_text_on_TinyStories(config: dict[str, Any]) -> None:
    generation_config = config["generation"]
    model_config = config["model"]
    device = resolve_device(generation_config["device"])
    tokenizer = load_tokenizer(config, generation_config["tokenizer_path"])

    model = build_model(config, device)
    optimizer = AdamW(model.parameters())
    load_checkpoint(generation_config["checkpoint"], model, optimizer, map_location=device)

    input_ids = torch.tensor(
        tokenizer.encode(generation_config["prompt"]),
        dtype=torch.int64,
        device=torch.device(device),
    )

    end_token_id = tokenizer.id[generation_config["end_token"].encode("utf-8")]

    generated_ids = decoding(
        model=model,
        input_ids=input_ids,
        end_tokens=[end_token_id],
        max_generation_length=generation_config["max_generation_length"],
        temperature=generation_config["temperature"],
        top_p=generation_config["top_p"],
        context_length=model_config["context_length"],
    )

    text = tokenizer.decode(generated_ids.tolist())
    print(text)


def main() -> None:
    config = load_config()
    task = config["task"]
    task_handlers = {
        "train_bpe": train_bpe_on_TinyStories,
        "encode_train": lambda cfg: encode_tokens(cfg, "train"),
        "encode_valid": lambda cfg: encode_tokens(cfg, "valid"),
        "train": training_together_on_TinyStories,
        "generate_text": generate_text_on_TinyStories,
    }
    if task not in task_handlers:
        raise ValueError(f"未知任务: {task}")
    task_handlers[task](config)


if __name__ == "__main__":
    main()
