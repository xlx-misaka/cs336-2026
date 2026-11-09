"""Personal 36M model experiment for the current 711.5M-token corpus.

This replaces the course model/budget for a personal single-GPU experiment.
Batch 128 is a fallback after batch 192 ran out of memory on RTX 6000D (84 GB).
The smaller batch still requires a full training and validation memory check.
See README.md for the original course settings and lower-memory fallbacks.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathsConfig:
    train_bin: Path
    valid_bin: Path
    model_output: Path


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 512
    d_model: int = 256
    d_ff: int = 768
    num_layers: int = 12
    num_heads: int = 4
    rope_theta: float = 10000.0


@dataclass
class TrainingConfig:
    seed: int = 0
    dtype: str = "bfloat16"
    # Large-vocabulary loss dominates memory; test training and validation on the target GPU.
    train_batch_size: int = 128
    eval_batch_size: int = 128
    train_steps: int = 2_048
    gradient_accumulation_steps: int = 6  # Effective batch: 128 * 6 = 768 sequences.
    compile: bool = True
    eval_iterations: int = 192  # 12,582,912 sampled tokens per evaluation.
    eval_interval: int = 256
    max_grad_norm: float | None = 1.0
    device: str = "cuda"
    lr: float = 6e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9
    wandb_project: str | None = None
    wandb_entity: str | None = None
    log_interval: int = 20
    save_checkpoints: bool = False


@dataclass
class Config:
    paths: PathsConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
