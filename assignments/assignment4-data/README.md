# CS336 Spring 2026 Assignment 4: Data

> 本地已完成数据处理与 36M 模型训练，结果和用法见[本地实现与运行记录](#本地实现与运行记录)。当前为个人单卡实验配置；课程正式比较需恢复规定的模型与训练设置。以下英文保留课程原始说明。

For a full description of the assignment, see the assignment handout at
[cs336_assignment4_data.pdf](./cs336_assignment4_data.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336_basics`](./cs336_basics): This module contains the staff
  implementation of the language model from assignment 1. You will use this training code
  to train an LM on your filtered data. You should not modify the training logic, since
  your leaderboard submission must use it exactly.
- [`./cs336_data`](./cs336_data): This folder is basically empty! This is the
  module where you will implement code to filter and process the data.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   └── ... an optimized training implementation ...
├── cs336_data  # TODO(you): code that you'll write for assignment 4
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 4 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 4 ...
```

As in previous assignments, we use `uv` to manage dependencies.

## Downloading data

### For students

Data is available at `/shared-data`, such as `uv run modal shell ./scripts/download_data.py::main --cmd "ls -l /shared-data"`

To only download the files needed for running offline, run `uv run scripts/download_data.py --offline-only`.
To download all data for the students, run `uv run modal run scripts/download_data.py`

### For non-students

To only download the files needed for running offline, run `uv run scripts/download_data.py --offline-only`.

Implement the method `is_english` in [`./cs336_data/wet_files.py`](./cs336_data/wet_files.py) before downloading the full non-offline data.

#### Modal

Remove `environment_name` from `shared_data_volume` in [`./cs336_data/modal_utils.py`](./cs336_data/modal_utils.py)

```python
uv run modal run scripts/download_data.py
```

#### Non-modal

Change the path in [`./cs336_data/common.py`](./cs336_data/common.py) and run `uv run scripts/download_data.py`.

Consider changing `n_files` for `EnglishWetFiles` in [`./cs336_data/wet_files.py`](./cs336_data/wet_files.py) if you want to download less than 2.5k WET files.

## Training on Modal

The Modal entrypoint in [`./scripts/train.py`](./scripts/train.py) contains the full training config; pass the path to your GPT-2-tokenized training data with `--train-bin`.

The final training run uses 8 B200 GPUs:

```sh
uv run modal run scripts/train.py --train-bin /root/data/your_data.bin
```

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a zip with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.

## 本地实现与运行记录

更新日期：2026-11-09。当前已完成语料过滤、分词和个人 36M 模型训练；尚未完成固定课程配置下的数据质量对比。

### 实现情况

| 功能 | 入口 |
| --- | --- |
| HTML 正文提取、语言识别 | [extract_text.py](cs336_data/extract_text.py)、[language_identification.py](cs336_data/language_identification.py) |
| 邮箱 / 电话 / IP 遮蔽、有害内容分类 | [mask_pii.py](cs336_data/mask_pii.py)、[harmful_content.py](cs336_data/harmful_content.py) |
| Gopher 规则、质量分类器与正文复核 | [quality.py](cs336_data/quality.py)、[recall_text.py](cs336_data/recall_text.py) |
| 精确行去重、MinHash + LSH 文档去重 | [deduplication.py](cs336_data/deduplication.py) |
| 并行过滤、分词、模型训练 | [filter_data.py](scripts/filter_data.py)、[tokenize_texts.py](scripts/tokenize_texts.py)、[train.py](scripts/train.py) |

核心函数已接入 [tests/adapters.py](tests/adapters.py)。最近一次完整测试记录为 **86 passed、2 warnings**；警告来自并行测试的 `fork` 弃用提示。通过测试不代表过滤质量或模型效果已经得到证明。

质量分类器以 Wikipedia 引用的**外部网页**为正样本、Common Crawl 网页为负样本，经正文提取及英文过滤后按来源标注。共有 7,622 篇，训练 / 验证为 6,145 / 1,477；这是来源弱标签，不等于人工质量标签。训练入口为 [train_quality_classifier.py](scripts/train_quality_classifier.py)。

### 数据与过滤结果

语料来自 Common Crawl `CC-MAIN-2026-17`：128 个原始 WET，按每 4 个一组进行英文预过滤，得到 32 个分片。默认保留语言为 `en`、置信度至少 0.7 的文本。

| 项目 | 结果 |
| --- | ---: |
| 输入文档 | 817,397 |
| 最终文档 / 保留率 | 547,458 / 66.98% |
| 最终空文档 | 0 |
| 最终 JSONL 分片 | 32 |
| GPT-2 tokens（含每篇 EOS） | **711,501,653** |
| `uint16` 训练文件 | **1,423,003,306 字节，约 1.42GB** |

处理顺序：模板清理 → Gopher → 质量分类 → NSFW / Toxic → PII 遮蔽 → 整篇精确去重 → 行去重 → 正文复检 → MinHash 文档去重。

- 默认 `recall` 策略用正文、代码和短文特征复核 Gopher / 质量模型的拒绝，减少误删；不绕过有害内容判断。
- 默认保守行去重保护识别出的代码与正文；课程 adapter 的精确行去重仍删除所有重复行副本。流水线可用 `--line-dedup-mode strict` 切换行策略。
- 分类分数门限均为 0.8；MinHash 使用 100 个 hash、10 个 band、5-word n-gram，并以真实 Jaccard ≥ 0.8 确认重复。
- 每条 JSONL 保存一篇正文，保留内部换行；中间产物自动清理。分词逐篇追加一个 EOS，流式写入 `np.uint16` 二进制文件。

下表按**第一个删除文档的阶段**归因，占比以全部 269,939 篇丢弃文档为分母：

| 阶段 | 删除文档 | 占全部删除 |
| --- | ---: | ---: |
| 模板与页面状态 | 19,225 | 7.12% |
| Gopher（含正文复核） | 143,437 | 53.14% |
| 质量分类（含正文复核） | 77,293 | 28.63% |
| NSFW | 801 | 0.30% |
| Toxic | 578 | 0.21% |
| PII 遮蔽 | 0 | 0.00% |
| 整篇精确去重 | 8,874 | 3.29% |
| 保守行去重 | 0 | 0.00% |
| 去重后正文与长度检查 | 7,317 | 2.71% |
| MinHash 去重 | 12,414 | 4.60% |

行去重实际删除了 54,345,514 行，其产生的 42 篇空文计入后续复检；“删除文档为 0”不代表文本未改变。PII 遮蔽修改了 244,091 篇文档。

完整记录：[过滤统计](local-shared-data/filtered_corpus/20261109_163531/filter_stats.json)、[分词统计](local-shared-data/tokenized_corpus/20261109_163531/tokenized_ids.stats.json)。这些是本地产物，不随代码仓库自动提供。

### 过滤与分词用法

在 `assignment4-data` 根目录运行，先完成过滤再分词：

```sh
uv run scripts/filter_data.py
uv run scripts/tokenize_texts.py
```

过滤从代码中配置的 Furu manifest 读取 WET；换数据用 `--manifest` 指定新文件。结果分别保存到 `local-shared-data/filtered_corpus/<运行名>/` 和 `local-shared-data/tokenized_corpus/<运行名>/`。有多个实验时，建议明确指定分词输入：

```sh
uv run scripts/tokenize_texts.py \
  --input-dir local-shared-data/filtered_corpus/20261109_163531 \
  --output local-shared-data/tokenized_corpus/20261109_163531/tokenized_ids.bin
```

分词默认按目录名选择已完成的最新过滤结果；上述命令会覆盖同名 token 文件。正文不需要重新合并成 WET。

### 当前训练配置

配置入口为 [train_config.py](cs336_basics/train_config.py)；训练脚本不读取 `configs/*.yaml`。**当前为个人小模型实验，不能替代课程规定的约 430M 模型正式比较。**

| 项目 | 当前设置 |
| --- | --- |
| 参数量 | **35,961,600，约 36M** |
| 层数 / 宽度 / FFN / 注意力头 | 12 / 256 / 768 / 4 |
| 词表 / 上下文 | 50,257 / 512 |
| GPU | 单张 RTX 6000D，84GB |
| 精度 / 编译 | BF16 autocast / `torch.compile` |
| 训练 batch / 梯度累积 | 128 / 6，有效 batch 768 |
| 更新次数 / 累计训练 tokens | 2,048 / **805,306,368** |
| 学习率 / warmup / weight decay | `6e-4` / 0.03 / 0.1 |
| 验证 batch / 批次数 / 间隔 | 128 / 192 / 每 256 步 |

训练量约为现有语料量的 **1.13 倍**。训练随机抽取连续 512-token 窗口，允许重叠、重复和跨文档 EOS，不保证逐篇完整遍历。36M 是本次实验基线，尚未证明是该数据量的最优模型。

AutoDL 上本次数据放在 `/root/autodl-tmp/`，训练命令为：

```sh
uv run scripts/train.py \
  --train-bin /root/autodl-tmp/tokenized_ids.bin \
  --valid-bin /root/autodl-tmp/tokenized_paloma_c4_100_domains_validation.bin
```

已完成单卡训练，训练中截图显示约 **1.12 秒/更新、58GiB 显存、接近 100% GPU 利用率**；这不是精确全程耗时或峰值显存测量。batch 192 曾在反向传播时 OOM，当前 batch 128 已跑通训练和多轮验证。

远端最终输出为 `/root/data/output/your_data/{model.pt,model_config.json}`，本地副本位于 `output/`。当前未启用 W&B 和中间 checkpoint；只保存最终权重与模型配置，不支持完整断点续训。

课程原始设置为 24 层、宽度 1,024、FFN 3,072、16 个头，约 430M 参数；8 张 B200，每卡 batch 128、累积 1 次、更新 16,384 次，累计约 85.9 亿 tokens；验证 batch 128、1,000 批、间隔 2,000 步。恢复课程比较时需同时恢复模型和训练预算，不能直接用当前默认值运行八卡 Modal 命令。

### 模型效果与生成

本地最终权重已通过加载及参数有限性检查。对 C4 100 验证文件做固定种子 336、FP32 CPU 抽样复核，结果如下：

| 指标 | 结果 |
| --- | ---: |
| 抽样窗口 / 长度 | 32 / 512 |
| 抽样 tokens | 16,384 |
| 验证 loss | **4.7308** |
| 验证 perplexity | **113.38** |

这是小样本诊断，不是完整验证或课程排名结果。两段英文续写已出现一些英文句式，但存在偏离提示、重复短语和语义不连贯的问题；目前不能据此宣称过滤改善了模型效果。

在 AutoDL / CUDA 环境生成文本，模型路径指向同时包含权重和配置的目录：

```sh
uv run scripts/generate_with_gpt2_tok.py /root/data/output/your_data \
  --prompt "Once upon a time, there was a" \
  --num-samples 3 --max-new-tokens 128 --temperature 0.7 --device cuda:0
```

后续重点：修正短提示页、模板和 SEO 内容的漏筛；保存完整验证记录与训练曲线；在相同模型和预算下比较不同语料。C4 验证文本不得加入训练集，训练与验证语料的近重复排查尚未完成。
