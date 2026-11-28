"""Fine-tune a causal language model on packed instruction-tuning data."""

import argparse
import math
import random
import time
from itertools import batched
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.data_utils import PackedSFTDataset

WANDB_PROJECT = "cs336-sft"


def parse_args() -> argparse.Namespace:
    data_dir = Path("data/safety_augmented_ultrachat_200k_single_turn")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Llama-3.1-8B")
    parser.add_argument("--train-data", type=Path, default=data_dir / "train.jsonl.gz")
    parser.add_argument("--val-data", type=Path, default=data_dir / "test.jsonl.gz")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sft"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Sequences per forward pass."
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--log-every", type=int, default=1, help="Optimizer updates between logs."
    )
    parser.add_argument(
        "--eval-every", type=int, default=100, help="Updates between evaluations."
    )
    parser.add_argument("--seed", type=int, default=336)
    args = parser.parse_args()

    for name in (
        "epochs",
        "seq_length",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_grad_norm",
        "log_every",
        "eval_every",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be nonnegative and finite")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup-ratio must be in [0, 1)")
    return args


def batch_loss(
    model: torch.nn.Module, batch: dict[str, torch.Tensor], device: torch.device
) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids=input_ids, use_cache=False).logits
    # PackedSFTDataset aligns each label with its predicting position.
    return F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten())


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_sum = torch.zeros((), device=device)
    token_count = 0
    for batch in loader:
        tokens = batch["labels"].numel()
        loss_sum += batch_loss(model, batch, device) * tokens
        token_count += tokens
    return float(loss_sum / token_count)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise SystemExit(f"Choose a new or empty output directory: {output_dir}")

    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise SystemExit("Choose a CPU or CUDA device.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Use --device cpu for a smoke test.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[INIT ] Loading {args.model} on {device}", flush=True)
    model, tokenizer = get_model_and_tokenizer(args.model, str(device))
    print(f"[INIT ] Packing {args.train_data}", flush=True)
    train_data = PackedSFTDataset(
        tokenizer, args.train_data, args.seq_length, shuffle=True
    )
    print(f"[INIT ] Packing {args.val_data}", flush=True)
    val_data = PackedSFTDataset(
        tokenizer, args.val_data, args.seq_length, shuffle=False
    )
    if not len(train_data) or not len(val_data):
        raise SystemExit("Both datasets must contain a complete packed sequence.")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = args.epochs * updates_per_epoch
    warmup_updates = int(args.warmup_ratio * total_updates)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_updates, total_updates
    )

    run_name = f"sft-{Path(args.model).name}-seed-{args.seed}"
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    banner = [
        "╭─ CS336 · SFT Training ".ljust(71, "─"),
        f"│ Run          {run_name}",
        f"│ Model        {args.model}",
        f"│ Device       {device} · {dtype} · seed {args.seed}",
        "│",
        (
            f"│ Training     {args.epochs} epoch(s) · {total_updates:,} updates · "
            f"{len(train_data):,} sequences"
        ),
        (
            f"│ Batch        {args.batch_size} sequences × "
            f"{args.gradient_accumulation_steps} accumulation steps "
            f"= {effective_batch_size} sequences per full update"
        ),
        f"│ Context      {args.seq_length:,} tokens per sequence",
        (
            f"│ Optimizer    AdamW · lr {args.learning_rate:g} · "
            f"weight decay {args.weight_decay:g} · grad clip {args.max_grad_norm:g}"
        ),
        f"│ Schedule     cosine decay · {warmup_updates:,} warmup updates",
        "│",
        (
            f"│ Validation   every {args.eval_every} updates and at the end · "
            f"{len(val_data):,} sequences"
        ),
        f"│ Logging      W&B {WANDB_PROJECT} · train every {args.log_every} updates",
        f"│ Checkpoint   {output_dir}",
        "╰" + "─" * 70,
    ]
    print("\n".join(banner), flush=True)

    training_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "model"
    }
    training_config.update(
        effective_batch_size=effective_batch_size,
        total_updates=total_updates,
        warmup_updates=warmup_updates,
        train_sequences=len(train_data),
        val_sequences=len(val_data),
        dtype=dtype,
    )
    with wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config={"model_id": args.model, "training": training_config},
    ) as run:
        print("[READY] Starting training\n", flush=True)
        step = 0
        step_width = len(str(total_updates))
        model.train()
        for epoch in range(1, args.epochs + 1):
            for microbatches in batched(train_loader, args.gradient_accumulation_steps):
                group_tokens = sum(batch["labels"].numel() for batch in microbatches)
                update_loss = torch.zeros((), device=device)
                for batch in microbatches:
                    weight = batch["labels"].numel() / group_tokens
                    weighted_loss = batch_loss(model, batch, device) * weight
                    weighted_loss.backward()
                    update_loss += weighted_loss.detach()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm, error_if_nonfinite=True
                )
                learning_rate = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                log_step = step in (1, total_updates) or step % args.log_every == 0
                eval_step = step == total_updates or step % args.eval_every == 0
                if not log_step and not eval_step:
                    continue

                progress = f"{step:0{step_width}d}/{total_updates}"
                log_data: dict[str, int | float] = {}
                if log_step:
                    train_loss = float(update_loss)
                    gradient_norm = float(grad_norm)
                    log_data = {
                        "train/loss": train_loss,
                        "train/gradient_norm": gradient_norm,
                        "train/learning_rate": learning_rate,
                        "train/tokens": group_tokens,
                        "train/epoch": epoch,
                    }
                    print(
                        f"[TRAIN {progress}] loss={train_loss:.4f}  "
                        f"grad={gradient_norm:.3f}  lr={learning_rate:.2e}  "
                        f"tokens={group_tokens:,}  epoch={epoch}/{args.epochs}",
                        flush=True,
                    )
                if eval_step:
                    print(f"[EVAL  {progress}] Running validation…", flush=True)
                    eval_start = time.monotonic()
                    val_loss = evaluate(model, val_loader, device)
                    model.train()
                    elapsed = time.monotonic() - eval_start
                    log_data["eval/loss"] = val_loss
                    print(
                        f"[EVAL  {progress}] loss={val_loss:.4f}  {elapsed:.1f}s",
                        flush=True,
                    )

                run.log(log_data, step=step)

        print(f"[SAVE ] Writing model and tokenizer to {output_dir}", flush=True)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".sft-", dir=output_dir.parent) as staging:
            checkpoint = Path(staging)
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            checkpoint.rename(output_dir)
        print(f"[DONE ] Saved model and tokenizer to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
