# TODO · 在 AutoDL 双卡上运行 GRPO 实验

使用 OLMo-2-0425-1B 在 GSM8K 上运行标准 on-policy、学习率消融、提示词消融和两组算法变体。训练用 GPU 0，vLLM 用 GPU 1 生成回答。脚本会自行启动本机的 vLLM 服务并同步权重。

先核对已完成的实验。现有报告记录了 50 步正确性检查和 seed 0–3 的四条正式曲线，最终验证准确率平均为 45.66%。如果对应的 W&B 曲线和样本仍可访问，可以直接复用。[标准 on-policy 报告](outputs/grpo_experiments_standard_on_policy.pdf)

## 1. 检查远程环境和数据

- [ ] 在同一台 AutoDL 实例上确认两张 GPU 都可见，并有足够的空闲显存。
- [ ] 进入远程作业目录，检查模型和 GSM8K 文件。

~~~bash
cd /root/autodl-tmp/cs336-2026/assignments/assignment5-alignment
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
ls -lh models/OLMo-2-0425-1B/config.json models/OLMo-2-0425-1B/tokenizer.json
find models/OLMo-2-0425-1B -maxdepth 1 -name '*.safetensors'
wc -l data/gsm8k/train.jsonl data/gsm8k/test.jsonl
~~~

当前项目的数据文件分别有 7,473 条训练样本和 1,319 条测试样本。脚本从训练文件抽取 6,400 题，从测试文件抽取 1,024 题做验证。

- [ ] 新实例没有 GPU 环境时，运行 `uv sync --extra gpu`。如果现有环境能正常运行 vLLM，直接复用。
- [ ] 模型目录缺失时，从 Hugging Face 下载到脚本使用的路径。

~~~bash
uv run --no-sync hf download allenai/OLMo-2-0425-1B \
  --local-dir models/OLMo-2-0425-1B
~~~

[项目依赖](pyproject.toml) · [Ai2 模型页](https://huggingface.co/allenai/OLMo-2-0425-1B) · [Hugging Face 下载命令](https://huggingface.co/docs/huggingface_hub/en/guides/cli#hf-download)

## 2. 选择要运行的实验

脚本的 `__main__` 目前只启用了**学习率消融**。跑其他实验时，在文件末尾切换到相应的现有函数调用。off-policy 那行目前只有函数名，启用时需要去掉注释并补上调用括号 `()`。[实验入口](cs336_alignment/grpo_experiments.py#L320)

| 任务 | 现有入口 | 当前配置 |
| --- | --- | --- |
| 双卡试跑 | `_run_smoke_on_policy` | seed 0，2 个 rollout step，每步验证 32 题 |
| 标准 GRPO 正确性检查 | `_run_correctness_check_on_policy` | seed 0，50 步，每 10 步验证并记录回答 |
| 标准 GRPO 四种子 | `_run_4_seed_experiments_on_policy` | zero-shot `r1_zero`，seed 0–3，各 200 步 |
| 学习率消融 | `_run_learning_rate_ablation_on_policy` | `3e-6`、`1e-5`、`3e-5`、`1e-4`，当前每档 seed 0 |
| 提示词消融 | `_run_prompt_ablation_on_policy` | `question_only`、`r1_zero_three_shot_gsm8k`，每种 4 个 seed，与原先的 zero-shot `r1_zero` 比较 |
| On-policy 变体 | `_run_variant_ablation_on_policy` | GRPO_constant、Dr. GRPO、RFT、MaxRL，每种 4 个 seed |
| Off-policy 变体 | `_run_variant_ablation_off_policy` | naive、noclip、GRPO clip、GSPO，每种 4 个 seed |

- [ ] 先用 2 步试跑检查远程环境，再根据已有结果选择要补跑的正式实验。
- [ ] 学习率消融完成后，记录后续要用的学习率。当前提示词消融和两组算法变体都默认用 `1e-5`，不会自动读取消融结果。如果后续算法变体改用调优后的学习率，核对对照组和被比较组使用的是同一档学习率。
- [ ] 按顺序跑各组。当前入口共用两张卡和 vLLM 端口 `8000`，新的训练进程启动时会尝试结束端口匹配的旧 vLLM 服务。

正式配置每次生成 **32 题 × 8 条回答 = 256 条回答**。On-policy 将这 256 条回答放入一个训练批，梯度累积 32 次。Off-policy 将完整的一批回答切成 32 个训练批，每批 8 条，梯度累积为 1。两类实验都按 **rollout step** 记录日志，正式 run 为 200 步，每 10 步验证，每 40 步记录训练回答。当前脚本每轮完整验证生成 **1,024 × 8 = 8,192 条回答**。[课程题面](cs336_spring2026_assignment5_alignment.pdf)

## 3. 运行当前选中的实验

- [ ] 选择 W&B 记录方式。需要在网页实时查看时先登录，网络不稳定时使用离线模式。

~~~bash
# 需要在线记录时运行，按提示输入自己的 API key
uv run --no-sync wandb login
~~~

- [ ] 如果要断开 SSH，且实例已安装 `tmux`，先运行 `tmux new -s grpo`，在会话内进入作业目录。训练开始后可按 Ctrl+B 再按 D 退出会话界面，之后用 `tmux attach -t grpo` 回到会话。
- [ ] 给本次实验取一个日志标签，与文件末尾启用的入口对应。下面用当前默认的学习率消融举例。

~~~bash
mkdir -p outputs/grpo_logs
GRPO_JOB=learning-rate
GRPO_LOG="outputs/grpo_logs/$GRPO_JOB-$(date +%Y%m%d-%H%M%S).log"
set -o pipefail

WANDB_MODE=online uv run --no-sync python -m cs336_alignment.grpo_experiments \
  2>&1 | tee "$GRPO_LOG"
~~~

要离线记录，把命令里的 `WANDB_MODE=online` 改为 `WANDB_MODE=offline`。本次实验结束后再上传，使用实际的离线 run 目录。

~~~bash
ls -d wandb/offline-run-*
# 把示例目录名换成实际要上传的 run
uv run --no-sync wandb sync wandb/offline-run-YYYYMMDD_HHMMSS-RUN_ID
~~~

[W&B 离线记录与同步](https://github.com/wandb/docs/blob/main/models/ref/cli/wandb-offline.mdx) · [W&B 同步命令](https://github.com/wandb/wandb/blob/main/wandb/cli/cli.py)

## 4. 检查进度和输出

- [ ] 启动信息里核对 run 名、seed、prompt、学习率、rollout 步数，以及 `policy cuda:0` 和 `vLLM cuda:1`。
- [ ] 看到 `[READY] Starting training` 后，检查是否依次出现 `[TRAIN]` 和 `[EVAL]`。另开终端可用 `nvidia-smi -l 2` 查看显存占用。
- [ ] 检查 W&B 的 `cs336-grpo` 项目。标准四种子、提示词消融和变体会用相同的 group 标签，筛选时对照 run 名、seed、prompt 和配置。
- [ ] 核对每个正式 run 是否有第 200 步的验证结果。出现 `eval/failed=1` 的步应标为验证缺失。

| 核对项 | 记录位置或字段 |
| --- | --- |
| 训练曲线 | `train/loss`、`train/gradient_norm`、`train/token_entropy`、`train/mean_reward`、`train/mean_format_reward` |
| 验证曲线 | `eval/mean_reward`、`eval/mean_format_reward`、`eval/accuracy`、`eval/mean_response_length`，回答长度按字符计 |
| 裁剪比例 | `train/clip_fraction`，用于比较 GRPO clip 和 GSPO |
| 回答样本 | W&B 的 `rollouts` 表，默认在第 40、80、120、160、200 步记录 |
| 终端日志 | `outputs/grpo_logs/` |
| 离线记录 | `wandb/offline-run-*` |

如果中途退出，已写入的日志可以保留。当前训练入口每次从 OLMo 基模型初始化，未提供 checkpoint 恢复参数。重跑前先按配置和 seed 找出已经完成的 run。

## 5. 整理到作业报告

- [ ] **标准 GRPO** 核对 50 步检查、四 seed 曲线、最终平均准确率是否达到 25%，以及早期和后期的回答样本。
- [ ] **学习率消融** 画最终验证奖励随学习率变化的图，标出发散或未完成的 run，并记录每档用了几个 seed。
- [ ] **提示词消融** 对比三个 prompt 的平均验证表现和种子间波动。
- [ ] **On-policy 变体** 用相同学习率和 zero-shot `r1_zero` 比较四个变体与标准 GRPO，保留跨 seed 的波动。
- [ ] **Off-policy 变体** 用 rollout step 对齐曲线，对照标准 GRPO 的验证表现，并检查 GRPO clip 和 GSPO 的裁剪比例。
- [ ] 保存终端日志、W&B run 链接、指标和回答样本。离线运行时，在释放实例前把离线目录复制出来或确认已经同步。
