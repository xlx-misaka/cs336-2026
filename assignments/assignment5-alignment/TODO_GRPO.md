# TODO · 在 AutoDL 双卡上跑 GRPO 实验

OLMo-2-0425-1B 在 GSM8K 上训练。脚本固定用 **GPU 0 训练、GPU 1 生成**，并自行启动 vLLM。现有报告已记录 50 步正确性检查和标准 on-policy 的四个 seed，最终验证准确率平均 **45.66%**。优先复用对应的 W&B 曲线和样本。[标准 GRPO 报告](outputs/grpo_experiments_standard_on_policy.pdf)

## 显存要求与租卡建议

| 选择 | 同机显存 | 怎么选 |
| --- | --- | --- |
| 预算优先 | **2× RTX 4090 24 GB** | 每张 24 GB 可作为试跑起点。先看两张卡在加载、训练和验证时的峰值 |
| 余量优先 | **2× RTX 5090 32 GB** | 正式跑多组变体时优先考虑，尤其是使用 FP32 log-prob 的 GSPO |

GPU 0 要容纳权重、梯度、优化器状态和训练激活，显存峰值随回答长度变化。GPU 1 主要用于 vLLM 权重和 KV cache。以上是估算，可在另一个终端用 `nvidia-smi -l 1` 观察两张卡的用量，跑到 GSPO 时再核对一次。比较租金时，用整台实例的小时价乘以预计运行时间。[AutoDL GPU 选型](https://www.autodl.com/docs/gpu/) · [NVIDIA 5090 与 4090 规格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)

## 1. 准备环境

- [ ] 在 AutoDL 的 Bash 终端进入作业根目录，检查两张 GPU、模型和数据。

~~~bash
cd /root/autodl-tmp/cs336-2026/assignments/assignment5-alignment
nvidia-smi
ls -lh models/OLMo-2-0425-1B/config.json \
  data/gsm8k/train.jsonl data/gsm8k/test.jsonl
find models/OLMo-2-0425-1B -maxdepth 1 -name '*.safetensors'
~~~

`nvidia-smi` 核对 GPU 0、1 和空闲显存。`ls` 检查配置和数据文件，`find` 检查本地权重分片。

- [ ] 新环境用 `uv sync` 装依赖。模型缺失时，再运行 `hf download`。

~~~bash
# 新环境执行
uv sync --extra gpu

# 模型缺失时执行
uv run --no-sync hf download allenai/OLMo-2-0425-1B \
  --local-dir models/OLMo-2-0425-1B
~~~

`--extra gpu` 安装 vLLM、W&B 和训练依赖，`--no-sync` 使用已有环境。`--local-dir` 必须与脚本里的模型目录一致。[Ai2 模型页](https://huggingface.co/allenai/OLMo-2-0425-1B)

## 2. 选择实验入口

脚本当前直接运行只会做 **学习率消融**。到 `__main__` 里每次只启用表中的一项。Off-policy 那行目前只有函数名，取消注释后还要补上 `()`。各组共用 GPU 和端口 `8000`，按顺序运行。[实验入口](cs336_alignment/grpo_experiments.py#L320)

| 实验 | 函数 | 要跑的配置 |
| --- | --- | --- |
| 双卡试跑 | `_run_smoke_on_policy` | 2 步，每步验证 32 题 |
| 正确性检查 | `_run_correctness_check_on_policy` | 50 步，已有报告 |
| 标准四种子 | `_run_4_seed_experiments_on_policy` | zero-shot `r1_zero`，seed 0–3，已有报告 |
| 学习率消融 | `_run_learning_rate_ablation_on_policy` | `3e-6`、`1e-5`、`3e-5`、`1e-4`，当前每档 seed 0 |
| 提示词消融 | `_run_prompt_ablation_on_policy` | `question_only` 和 `r1_zero_three_shot_gsm8k`，各 4 个 seed |
| On-policy 变体 | `_run_variant_ablation_on_policy` | GRPO_constant、Dr. GRPO、RFT、MaxRL，各 4 个 seed |
| Off-policy 变体 | `_run_variant_ablation_off_policy` | naive、noclip、GRPO clip、GSPO，各 4 个 seed |

- [ ] 租用新实例时先试跑 2 步，再按已有结果补跑。若后续变体采用调优后的学习率，手动设置并核对对照组。脚本默认用 `1e-5`。
- [ ] 正式 run 为 **200 个 rollout step**，每 10 步验证、每 40 步记录训练回答。每步生成 256 条回答，on-policy 用一个训练批，off-policy 切成 32 批，每批 8 条。每轮完整验证生成 1,024 题 × 8 条回答。[课程题面](cs336_spring2026_assignment5_alignment.pdf)

## 3. 运行并保存日志

需要断开 SSH 时，先用 `tmux new -s grpo` 打开会话，再进入作业目录。在线记录前用下面的命令登录 W&B。网络不稳定时，把启动命令中的 `WANDB_MODE=online` 改为 `WANDB_MODE=offline`，数据会先保存在本地。

~~~bash
uv run --no-sync wandb login
~~~

~~~bash
mkdir -p outputs/grpo_logs
GRPO_JOB=learning-rate  # 按本次选中的实验修改
GRPO_LOG="outputs/grpo_logs/$GRPO_JOB-$(date +%Y%m%d-%H%M%S).log"
set -o pipefail

WANDB_MODE=online uv run --no-sync python -m cs336_alignment.grpo_experiments \
  2>&1 | tee "$GRPO_LOG"
~~~

- `python -m cs336_alignment.grpo_experiments` 运行脚本 `__main__` 中启用的实验。先按上表选入口，例如试跑时只启用 `_run_smoke_on_policy()`。
- `GRPO_JOB` 只用于终端日志的命名。日期后缀方便区分多次运行，`tee` 会同时在屏幕上显示并写入日志。Bash 的 `set -o pipefail` 让训练失败时整条管道也报告失败。

- [ ] 启动后核对 run 名、seed、prompt、学习率和两张卡的分工。确认出现 `[TRAIN ...]` 和 `[EVAL ...]`。
- [ ] 如果用离线模式，训练结束后登录 W&B，用下面的命令上传。第一条列出实际目录，第二条中的目录名需要替换。[W&B 同步说明](https://github.com/wandb/docs/blob/main/models/ref/cli/wandb-offline.mdx)

~~~bash
ls -d wandb/offline-run-*
uv run --no-sync wandb sync wandb/offline-run-YYYYMMDD_HHMMSS-RUN_ID
~~~

当前入口每次从 OLMo 基模型初始化，未提供 checkpoint 恢复参数。中途退出时，保留已有日志，按实验和 seed 核对缺少哪些 run。

## 4. 核对作业结果

- [ ] 在 W&B 的 `cs336-grpo` 项目按 run 名、seed 和配置筛选。核对各正式 run 的第 200 步验证结果。若 `eval/failed=1`，该步验证记为缺失。
- [ ] 标准四种子图保留各 seed 或波动区间。曲线包括 `train/loss`、`train/gradient_norm`、`train/token_entropy`、训练与验证的总奖励和格式奖励，以及 `eval/mean_response_length`（字符）。保留早期和后期的 `rollouts`。
- [ ] 学习率消融画各档最终验证奖励。提示词消融对比三个 prompt 的平均表现和种子间波动。两组算法变体按 rollout step 和相同学习率对比，额外比较 GRPO clip 与 GSPO 的 `train/clip_fraction`。
- [ ] 保存终端日志、W&B run 链接、图表和回答样本。离线运行先同步或备份离线目录，再释放实例。
