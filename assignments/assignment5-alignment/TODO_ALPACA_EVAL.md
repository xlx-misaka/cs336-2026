# TODO · 在 AutoDL 双 A800 上跑 AlpacaEval

输入是已有的 **805 条 8B 回答**和 **805 条 GPT-4 Turbo 参考回答**。vLLM 会在评测进程里自动加载 70B 裁判。

## 显存要求与租卡建议

- 原版 Llama 3.3 70B Instruct BF16 权重约 **131 GiB**。双卡平分后每张光权重约 **66 GiB**，还要留空间给 KV cache 和运行时。
- 推荐同机 **2× A800 80 GB**，下单时核对每张卡的容量。如果同地区双 A100 80 GB 的整机小时价更低，也可选它。SST 的 70B 评分可共用这两张卡和同一份权重，两个评测依次运行。

[AutoDL GPU 选型](https://www.autodl.com/docs/gpu/) · [70B 权重](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/tree/main) · [vLLM 显存说明](https://docs.vllm.ai/en/v0.19.1/configuration/conserving_memory/)

## 1. 检查远程环境和文件

- [ ] 在 AutoDL 的 Bash 终端进入作业根目录，检查两张卡和输入文件。模型在共享盘时，把 `JUDGE_MODEL` 的路径前缀换成 `/root/autodl-fs`。

~~~bash
cd /root/autodl-tmp/cs336-2026/assignments/assignment5-alignment
nvidia-smi
ls -lh outputs/alpaca_eval_baseline.json \
  data/alpaca_eval/alpaca_eval_gpt4_turbo.json

JUDGE_MODEL=/root/autodl-tmp/models/Llama-3.3-70B-Instruct
ls -lh "$JUDGE_MODEL"/config.json "$JUDGE_MODEL"/tokenizer.json \
  "$JUDGE_MODEL"/model.safetensors.index.json
find "$JUDGE_MODEL" -maxdepth 1 -name 'model-*.safetensors' | wc -l
~~~

`nvidia-smi` 查看卡号和显存占用，`ls` 检查输入文件，`find … | wc -l` 数权重分片。原版共有 **30 个分片**，约 **141 GB**。这份权重也可用于 SST 评分。[Meta 权重文件](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/tree/main) · [SST TODO](TODO_SST_BASELINE.md)

- [ ] 新环境先装 AlpacaEval 和 vLLM 等 GPU 依赖。正式命令用 `--no-sync` 复用该环境。

~~~bash
uv sync --extra gpu
~~~

## 2. 设置裁判并运行

- [ ] 在 `configs.yaml` 中，将 `completions_kwargs.model_name` 改成实际的 70B 绝对路径。下面只展示要替换的那一行。确认 `model_kwargs.tensor_parallel_size: 2`，它让 vLLM 平分模型。首次运行沿用 `max_model_len: 7000`。[裁判配置](scripts/alpaca_eval_vllm_llama3_3_70b_fn/configs.yaml)

~~~yaml
    model_name: "/root/autodl-tmp/models/Llama-3.3-70B-Instruct"
~~~

- [ ] 在项目根目录运行。需要断开 SSH 时，可在 `tmux` 会话内启动。

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

- `CUDA_VISIBLE_DEVICES=0,1` 指定可见的两张 GPU。权重路径和多卡分配由上面的 YAML 读取。
- `--model_outputs` 指定待评测的 8B 回答，`--reference_outputs` 指定 GPT-4 Turbo 参考回答。
- `--annotators_config` 选用课程的本地 70B 裁判，`--base-dir .` 以当前作业目录查找配置，`--output_path` 指定结果目录。
- `tee` 一边显示一边保存终端日志。Bash 的 `set -o pipefail` 保留评测进程失败时的退出状态。

找不到 `alpaca_eval` 时检查 GPU 依赖。加载失败时先核对模型目录和两张卡的空闲显存。如果日志明确显示 CUDA graph 或并发分配显存不足，可尝试把裁判配置的 `batch_size` 从 `900` 降到 `64`，并记录改动。

## 3. 核对结果

- [ ] **(c)** 从 `outputs/alpaca_eval_judge/leaderboard.csv` 记录 `llama-3.1-8b-base` 的 `win_rate` 和 `length_controlled_winrate`。
- [ ] **(d)** 从同目录的 `annotations.json` 中，随机抽 10 条裁判偏好 GPT-4 Turbo 的记录，复查两边的回答。
- [ ] 保存 `leaderboard.csv`、`annotations.json`、`run.log` 和实际使用的裁判配置，再关机。

~~~bash
head -n 4 outputs/alpaca_eval_judge/leaderboard.csv
# 可选，核对逐题标注条数
jq length outputs/alpaca_eval_judge/annotations.json
~~~

`head` 查看排行榜表头和结果，`jq length` 数逐题标注条数。不装 `jq` 时，可以直接打开 JSON 查看。

[课程题面](cs336_spring2026_assignment5_supplement_safety_rlhf.pdf) · [AlpacaEval CLI 文档](https://github.com/tatsu-lab/alpaca_eval#evaluating-a-model)
