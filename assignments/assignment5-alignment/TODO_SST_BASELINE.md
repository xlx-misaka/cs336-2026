# TODO · 在 AutoDL 上跑 SimpleSafetyTests

## 显存要求与租卡建议

| 作业步骤 | 显存估算 | AutoDL 选卡 |
| --- | --- | --- |
| (a) 生成回答、(b) 记录速度 | 8B BF16 权重约 15 GiB，另需 KV cache 和运行时显存。建议单卡 **24 GB** | 先看 **1× RTX 3090 24 GB** 的价格，也可选 **1× RTX 4090 24 GB** |
| (c) 裁判评分 | 70B BF16 权重约 131 GiB，双卡平分后每张光权重约 66 GiB | 同机 **2× A800 80 GB**。双 A100 80 GB 也可，按实例价格选 |

只生成回答时，按单张 24 GB 卡准备。如果已经租了双 A800，先用其中一张生成，再用两张卡依次跑 SST 和 AlpacaEval 的 70B 评分，复用同一份裁判权重。[AutoDL GPU 选型](https://www.autodl.com/docs/gpu/) · [8B 权重](https://huggingface.co/meta-llama/Llama-3.1-8B/tree/main) · [70B 权重](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/tree/main) · [AlpacaEval TODO](TODO_ALPACA_EVAL.md)

## 1. 检查远程环境

- [ ] 在 AutoDL 的 Bash 终端进入作业根目录。项目在其他位置时，替换 `cd` 的路径。

~~~bash
cd /root/autodl-tmp/cs336-2026/assignments/assignment5-alignment
nvidia-smi
ls -lh models/Llama-3.1-8B/config.json \
  data/simple_safety_tests/simple_safety_tests.csv
~~~

`nvidia-smi` 查看 GPU 编号和显存占用，`ls` 核对默认模型和数据路径。8B 模型放在其他目录时，在生成命令中加 `--model /你的模型目录`。

- [ ] 如果是新环境，先安装 GPU 依赖。

~~~bash
uv sync --extra gpu
~~~

`--extra gpu` 安装包括 vLLM 在内的 GPU 依赖。后续的 `--no-sync` 直接使用已准备好的环境。

## 2. 用 8B 生成回答

- [ ] 可先试跑 2 条。全量命令会处理全部 100 条指令，并将生成耗时和 `examples/s` 写入日志。

~~~bash
# 可选试跑
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m cs336_alignment.sst_baseline \
  --max-num-seqs 8 --limit 2 --output outputs/sst_baseline_smoke.jsonl

# 正式生成
mkdir -p outputs
set -o pipefail
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m cs336_alignment.sst_baseline \
  --max-num-seqs 8 2>&1 | tee outputs/sst_baseline.log
~~~

- `CUDA_VISIBLE_DEVICES=0` 让 8B 生成只占用第 0 张卡。编号以 `nvidia-smi` 为准。
- `--max-num-seqs 8` 限制并发数，`--limit 2` 才是试跑的样本数。试跑用 `--output` 单独存文件，正式结果默认写到 `outputs/sst_baseline.jsonl`。
- `2>&1 | tee` 把终端输出同时存入日志。Bash 的 `set -o pipefail` 让脚本失败时整条管道也返回失败状态。

[生成脚本](cs336_alignment/sst_baseline.py)

## 3. 用双 A800 评分

- [ ] 等 8B 进程退出后，确认两张卡空闲，且 `wc -l outputs/sst_baseline.jsonl` 显示 **100 行**。把 `JUDGE_MODEL` 改成 70B 的实际目录。如果模型在共享盘，把路径前缀换成 `/root/autodl-fs`。

~~~bash
JUDGE_MODEL=/root/autodl-tmp/models/Llama-3.3-70B-Instruct
ls -lh "$JUDGE_MODEL"/config.json "$JUDGE_MODEL"/model.safetensors.index.json

mkdir -p outputs
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync python scripts/evaluate_safety.py \
  --input-path outputs/sst_baseline.jsonl \
  --model-name-or-path "$JUDGE_MODEL" \
  --num-gpus 2 \
  --output-path outputs/sst_judge.jsonl \
  2>&1 | tee outputs/sst_judge.log
~~~

- `--model-name-or-path "$JUDGE_MODEL"` 让裁判加载本地权重。这里通过命令行指定路径。
- `CUDA_VISIBLE_DEVICES=0,1` 让进程看到两张卡，`--num-gpus 2` 让 vLLM 把 70B 权重分到这两张卡上。
- `--input-path` 读取全量 8B 回答，`--output-path` 保存裁判的逐题标注，两份 JSONL 可分别复查。

## 4. 保存作业所需结果

- [ ] **(b)** 从 `sst_baseline.log` 记录生成耗时和吞吐量。
- [ ] **(c)** 从 `sst_judge.log` 读取 `safe:`。核对 `sst_judge.jsonl` 有 100 条记录。安全率为 `metrics.safe == 1` 的条数除以标注总数。
- [ ] **(d)** 从 `metrics.safe == 0` 的记录随机抽 10 条复查。如果不足 10 条，报告实际数量。
- [ ] 保存两个 JSONL 和两个日志文件，再关机。

~~~bash
wc -l outputs/sst_baseline.jsonl outputs/sst_judge.jsonl
grep 'safe:' outputs/sst_judge.log
~~~

`wc -l` 核对生成和评分的条数，`grep` 从日志中提取安全率。

[课程题面](cs336_spring2026_assignment5_supplement_safety_rlhf.pdf) · [评分脚本](scripts/evaluate_safety.py)
