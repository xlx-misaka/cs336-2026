# TODO · 在 AutoDL 上跑 Llama 3.1 8B SFT

完成 `Problem (sft)`。用 Llama 3.1 8B Base 全参数微调 **1 个 epoch**，序列长度 **512**，有效 batch size **32**。交付训练配置、最终验证 loss、学习曲线，以及用于后续评测和 DPO 的模型与 tokenizer。 [课程题面 §4.2.2](cs336_spring2026_assignment5_supplement_safety_rlhf.pdf) · [训练脚本](scripts/train_sft.py)

## 1. 租卡和估时

| 单卡选择 | 预计耗时 | 怎么选 |
| --- | --- | --- |
| **B200 180 GB** | **3–5 小时** | 首选。课程标注 3 B200 小时，额外留出分词、验证和保存的时间 |
| **H200 141 GB** | **5–9 小时** | B200 缺货或整次租金更低时选，显存余量充足 |
| RTX PRO 6000 Blackwell 96 GB | **8–15 小时** | 小时价明显更低时再考虑。先确认 FlashAttention-2 能运行并观察前几步的显存 |

这些是根据讲义预算和硬件规格给出的排期估算，尚未实测全量训练，下载时间另算。比较租金时用 **实例小时价 × 预计耗时**。启动后用终端的总更新步数和稳定后的每步用时修正估算，另计验证和保存时间。 [AutoDL 选卡](https://www.autodl.com/docs/gpu/) · [NVIDIA B200/H200 规格](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html) · [RTX PRO 6000 规格](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)

当前脚本把权重、梯度和 AdamW 状态放在同一张卡上，三者合计约 **60 GiB**，运行时还会用到激活、logits 和优化器临时显存。80 GB 卡比较紧。多租一张 32 GB 卡也无法为当前单卡脚本分摊显存。建议主机内存至少 **32 GB**，输入已备齐时给输出盘留 **25 GB** 以上。 [PyTorch AdamW 显存说明](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

## 2. 准备环境和数据

- [ ] 建议在 `tmux new -s sft` 会话里跑。进入作业目录，检查 GPU、磁盘和输入文件。路径按租机上的位置改。

~~~bash
cd /root/autodl-tmp/cs336-2026/assignments/assignment5-alignment
SFT_MODEL=models/Llama-3.1-8B
SFT_DATA=data/safety_augmented_ultrachat_200k_single_turn

nvidia-smi
df -h .
ls -lh "$SFT_MODEL"/config.json "$SFT_MODEL"/*.safetensors \
  "$SFT_DATA"/train.jsonl.gz "$SFT_DATA"/test.jsonl.gz
~~~

- [ ] 新环境执行 `uv sync --extra gpu`，安装 CUDA 版 PyTorch、FlashAttention 和 W&B。数据可直接复制 `.jsonl.gz`，脚本会边读边解压。模型缺失时，拿到 Hugging Face 访问权限后执行下面两条命令。 [Llama 3.1 8B Base](https://huggingface.co/meta-llama/Llama-3.1-8B)

~~~bash
uv run --no-sync hf auth login
uv run --no-sync hf download meta-llama/Llama-3.1-8B --local-dir "$SFT_MODEL"
~~~

## 3. 正式训练

- [ ] 运行以下命令。默认用 BF16、FlashAttention-2、AdamW、学习率 `2e-5`、余弦衰减、`3%` warmup、权重衰减 `0.1` 和梯度裁剪 `1.0`。

~~~bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
  uv run --no-sync python scripts/train_sft.py \
  --model "$SFT_MODEL" \
  --train-data "$SFT_DATA/train.jsonl.gz" \
  --val-data "$SFT_DATA/test.jsonl.gz" \
  --output-dir outputs/sft \
  --epochs 1 --seq-length 512 --batch-size 2 \
  --gradient-accumulation-steps 16
~~~

`CUDA_VISIBLE_DEVICES=0` 选择 GPU 0。每次前向处理 2 条序列，累积 16 批再更新，得到 `2 × 16 = 32`。`--no-sync` 使用已经装好的环境。输出目录须为新目录或空目录。

- [ ] 确认终端出现 `[TRAIN ...]`。可在另一个终端用 `nvidia-smi -l 1` 看前几个更新的峰值显存。默认每 100 次参数更新和训练结束时各验证一次。
- [ ] 需要在线记录时，先执行 `uv run --no-sync wandb login`，再把启动命令里的 `offline` 改为 `online`。离线训练结束后联网，可上传终端显示的实际 run 目录。

~~~bash
ls -d wandb/offline-run-*
uv run --no-sync wandb sync wandb/offline-run-实际目录名
~~~

## 4. 保存作业交付

- [ ] 在 W&B 的 `cs336-sft` 项目保存 `train/loss` 和 `eval/loss` 曲线，横轴用参数更新步数。记录最后一次 `[EVAL ...]` 的 loss 和实际训练配置。
- [ ] 确认终端出现 `[DONE ]`，保存整个 `outputs/sft/`，包括模型分片、配置和 tokenizer。释放实例前，把 checkpoint 和 W&B run 链接或离线目录备份到会保留的存储。
