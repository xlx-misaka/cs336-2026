# TODO · 在 AutoDL 双 A800 上运行 AlpacaEval

用本地 Llama 3.3 70B Instruct 裁判比较 **Llama 3.1 8B 的 805 条零样本回答**和 **GPT-4 Turbo 的参考回答**。直接复用已有结果，评测命令会通过 vLLM 加载裁判，无需单独启动服务。

## 1. 检查远程实例和输入

- [ ] 确认同一台实例上有两张 **A800 80 GB**，并检查两张卡的空闲显存。
- [ ] 进入远程作业目录。如果项目放在其他位置，替换下面的路径。
- [ ] 确认模型目录中有权重索引、tokenizer 和全部权重分片。

在 **AutoDL 的 Bash 终端**执行：

~~~bash
cd /root/autodl-tmp/cs336-2026/assignments/assignment5-alignment
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

JUDGE_MODEL=/root/autodl-tmp/models/Llama-3.3-70B-Instruct
# 如果模型在共享盘，改用下面的路径
# JUDGE_MODEL=/root/autodl-fs/models/Llama-3.3-70B-Instruct
ls -lh "$JUDGE_MODEL"/config.json \
  "$JUDGE_MODEL"/tokenizer.json \
  "$JUDGE_MODEL"/model.safetensors.index.json
find "$JUDGE_MODEL" -maxdepth 1 -name 'model-*.safetensors' | wc -l
~~~

原版 BF16 权重应有 **30 个 Safetensors 分片**，总大小约 **141 GB**。[Meta 权重文件列表](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/tree/main)

- [ ] 确认模型输出和参考输出都已同步到远程实例。

~~~bash
ls -lh outputs/alpaca_eval_baseline.json \
  data/alpaca_eval/alpaca_eval_gpt4_turbo.json

# 如果安装了 jq，两个文件都应输出 805
jq length outputs/alpaca_eval_baseline.json \
  data/alpaca_eval/alpaca_eval_gpt4_turbo.json
~~~

## 2. 准备项目环境

- [ ] 如果是新环境，安装项目的 GPU 依赖。项目的 `gpu` extra 包含 `alpaca-eval` 和 `vllm`。如果实例已有可用的 GPU 环境，可以跳过同步。

~~~bash
uv sync --extra gpu
~~~

- [ ] 确认当前环境里能找到评测命令。

~~~bash
uv run --no-sync alpaca_eval -- --help
~~~

`--no-sync` 让正式评测复用已准备好的环境，避免在启动评测时再次同步依赖。如果提示找不到命令，先完成上面的 GPU 依赖安装。[uv 环境同步说明](https://docs.astral.sh/uv/concepts/projects/sync/)

## 3. 配置 70B 裁判

- [ ] 打开远程项目的 [裁判配置](scripts/alpaca_eval_vllm_llama3_3_70b_fn/configs.yaml)，把 `completions_kwargs.model_name` 改成刚才检查过的实际模型目录。

本地数据盘的配置示例：

~~~yaml
    model_name: "/root/autodl-tmp/models/Llama-3.3-70B-Instruct"
~~~

- [ ] 核对 `model_kwargs.tensor_parallel_size` 是 `2`，让 vLLM 将模型分到两张卡上。先按课程配置使用 `max_model_len: 7000`。

配置中的 `vllm_local_completions` 会直接从本地模型目录加载权重。GPT-4 Turbo 的回答来自参考 JSON 文件，这次评测无需 OpenAI API key。[vLLM 多卡推理说明](https://docs.vllm.ai/en/v0.19.0/serving/parallelism_scaling/)

## 4. 运行评测并保存日志

- [ ] 如果要断开 SSH，先进入 `tmux` 会话（实例已安装时），然后在其中进入作业目录。
- [ ] 从作业目录运行下面的命令。如果实例上两张可见 GPU 的编号不同，调整 `CUDA_VISIBLE_DEVICES`。

~~~bash
mkdir -p outputs/alpaca_eval_judge
set -o pipefail

CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync alpaca_eval \
  --model_outputs outputs/alpaca_eval_baseline.json \
  --reference_outputs data/alpaca_eval/alpaca_eval_gpt4_turbo.json \
  --annotators_config scripts/alpaca_eval_vllm_llama3_3_70b_fn \
  --base-dir . \
  --output_path outputs/alpaca_eval_judge \
  2>&1 | tee outputs/alpaca_eval_judge/run.log
~~~

启动时会先从磁盘加载 70B 权重，然后才开始生成裁判结果。可以另开一个 SSH 终端查看日志和显存占用。

~~~bash
# 在项目目录查看日志
tail -f outputs/alpaca_eval_judge/run.log
# 在另一个终端查看两张卡的使用情况
nvidia-smi -l 2
~~~

## 5. 核对并保存结果

- [ ] **(c)** 打开 `outputs/alpaca_eval_judge/leaderboard.csv`，记录 `llama-3.1-8b-base` 对应的 `win_rate` 和 `length_controlled_winrate`。
- [ ] **(d)** 打开 `outputs/alpaca_eval_judge/annotations.json`，核对逐题标注覆盖范围。后续从裁判偏好 GPT-4 Turbo 的条目中随机抽 10 条，对照两边的原文。
- [ ] 保存 `leaderboard.csv`、`annotations.json`、`run.log` 和这次使用的裁判配置，再关机。

~~~bash
ls -lh outputs/alpaca_eval_judge
head -n 4 outputs/alpaca_eval_judge/leaderboard.csv
# 如果安装了 jq，用这条命令核对逐题结果条数
jq length outputs/alpaca_eval_judge/annotations.json
~~~

[课程补充作业题面](cs336_spring2026_assignment5_supplement_safety_rlhf.pdf) · [AlpacaEval CLI 文档](https://github.com/tatsu-lab/alpaca_eval#evaluating-a-model)

## 常见问题

| 现象 | 先检查什么 |
| --- | --- |
| `alpaca_eval` 找不到 | 确认当前目录是作业根目录，并且已经运行 `uv sync --extra gpu`。 |
| 报模型路径或 tokenizer 找不到 | 检查裁判配置的 `model_name`。若用共享盘，先确认 `/root/autodl-fs` 已挂载。 |
| 找不到裁判配置或 prompt | 从作业根目录运行，并保留命令里的 `--annotators_config` 和 `--base-dir .`。 |
| 裁判启动时报显存不足 | 先用 `nvidia-smi` 确认两张卡都可见且空闲。如果日志显示在 CUDA graph 或并发分配阶段耗尽显存，可尝试将配置中的 `batch_size` 从 `900` 降到 `64`，并记录这一改动。 |
