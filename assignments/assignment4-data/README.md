# CS336 Spring 2026 Assignment 4: Data

> 当前实现、处理结果和运行方法见[本地实现与运行记录](#本地实现与运行记录)。本地训练配置已适配单卡；使用原始八卡 Modal 命令前，请先阅读[当前训练配置](#当前训练配置)。以下英文保留课程原始说明。

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

记录日期：2026-11-09（Asia/Shanghai）。本节描述当前代码和已有运行产物；模型训练效果以实际记录的验证 loss / perplexity 为准。

### 作业实现情况

| 内容 | 实现位置 | 当前状态 |
| --- | --- | --- |
| 从 HTML bytes 提取正文 | [extract_text.py](cs336_data/extract_text.py) | 已实现：优先 UTF-8 解码，失败时检测编码，再用 Resiliparse 提取文本；失败可返回 `None` |
| 语言识别与英文预过滤 | [language_identification.py](cs336_data/language_identification.py)、[wet_files.py](cs336_data/wet_files.py) | 已实现：fastText `lid.176.bin`；WET 预过滤保留 `en` 且置信度至少 0.7 的文档 |
| 邮箱、电话、IP 遮蔽 | [mask_pii.py](cs336_data/mask_pii.py) | 已实现：返回替换后的文本及替换次数，附电话格式与误替换回归测试 |
| NSFW / Toxic 分类 | [harmful_content.py](cs336_data/harmful_content.py) | 已接入两个 fastText 预训练分类器，返回标签与分数 |
| Gopher 规则与质量分类器 | [quality.py](cs336_data/quality.py)、[train_quality_classifier.py](scripts/train_quality_classifier.py) | 已实现规则、分类器训练与加载；本地已有训练好的质量模型 |
| 课程精确行去重 | [deduplication.py](cs336_data/deduplication.py) | 已实现并接入 adapter；删除语料中出现多次的行的所有副本 |
| MinHash + LSH 模糊文档去重 | [deduplication.py](cs336_data/deduplication.py) | 已实现：文本规范化、词 n-gram、MinHash、LSH 候选、真实 Jaccard 与并查集聚簇 |
| 并行语料过滤与统计 | [filter_data.py](scripts/filter_data.py)、[recall_text.py](cs336_data/recall_text.py) | 已完成一次全量处理；提供严格策略与优先减少误删的策略 |
| GPT-2 分词与二进制序列化 | [tokenize_texts.py](scripts/tokenize_texts.py) | 已完成全量分词；每篇追加 EOS，以 `np.uint16` 和 `tofile` 保存 |
| 语言模型训练与 C4 100 对比 | [train.py](scripts/train.py)、[train_config.py](cs336_basics/train_config.py) | 训练入口已有，当前为单卡实验配置；最佳验证 loss、学习曲线和正式对比结果待补充 |

上述函数均已通过 [tests/adapters.py](tests/adapters.py) 接入对应课程测试。2026-11-09 重新运行完整测试，结果为 **86 passed，2 warnings**；两条警告来自并行 MinHash 测试中的 Python `fork` 弃用提示。功能测试和 sanity check 通过，不代表分类器已经具备可靠的人工质量判别能力，也不表示所有书面 deliverable 已完成。

```sh
uv run pytest
```

### 质量分类器训练数据

质量分类器的正样本来自英文 Wikipedia 引用的外部网页，负样本来自 Common Crawl 的 `CC/example.warc.gz`；标签按来源生成。这里的 `wiki` 指“Wikipedia 引用的外部网页”，并非直接使用 Wikipedia 文章正文。

处理时保留 HTTP 200 的 `text/html` 响应，经 HTML 正文提取后保留非空、语言为 `en` 且置信度大于 0.7 的文本。当前训练脚本没有对这些分类器训练样本额外应用 Gopher、NSFW、Toxic 或正文去重。

| 数据划分 | CC / 负样本 | Wiki 引用页 / 正样本 | 合计 |
| --- | ---: | ---: | ---: |
| 训练集 | 3,244 | 2,901 | 6,145 |
| 验证集 | 756 | 721 | 1,477 |
| 总计 | 4,000 | 3,622 | 7,622 |

脚本抽取最多 10,000 个 URL，每类文本上限为 4,000，使用种子 336，逐文档以 0.2 的概率划入验证集，没有按域名分组。训练调用 `fasttext.train_supervised`，模型保存在 `local-shared-data/classifiers/quality_classifier.bin`。

训练脚本顶部保留了一次历史评估快照：来源标签验证准确率约 81.79%；该指标不是人工确认的“高质量文本准确率”。快照还记录了重复正文、跨训练/验证重复和标签冲突，更换数据或模型后需要重新评估。

### 训练语料从哪里来

语言模型语料来自 Common Crawl `CC-MAIN-2026-17` 的 WET 文本。当前下载设置为 `n_files=128`、`group_size=4`、`shuffle_seed=336`，每 4 个原始 WET 合成一个英文预过滤分组，当前 manifest 列出 **32 个英文 WET 输出分片**。

本地数据根目录是 `local-shared-data/`，由 [common.py](cs336_data/common.py) 根据本地 / Modal 环境选择。Furu 管理原始处理缓存，长路径用于区分任务、参数和缓存结果；下游通过 manifest 读取文件，不需要逐个手填路径。

当前过滤脚本默认读取的 manifest：

```text
local-shared-data/furu/cs336_data/wet_files/EnglishWetFiles/
  ae4fa708c26a65eb662a/c22742cf4085f62d0f56/result/manifest.json
```

如果重新下载后生成了新 manifest，使用 `--manifest` 指向新文件。上面的两行仅为展示换行，实际路径是连续的一条路径。

WET 的 `conversion` 记录已经是提取后的纯文本，最终语料过滤时直接用 FastWARC 遍历并解码，不再调用 HTML 正文提取函数。HTML 提取函数用于前面的 WARC 实验和质量分类器数据准备。

### 过滤与去重流程

当前默认流程：

1. 从英文 WET 分片逐条读取正文，按 WET 分片分配过滤进程。
2. 清理独立导航、页脚和部分 Cookie 模板，识别明确的付费墙、失效职位、空搜索和归档预览等页面状态。
3. 应用 Gopher 规则：50–100,000 个词、平均词长 3–10、以省略号结尾的行不超过 30%、至少 80% 的词包含字母。规则用 NLTK 分词。
4. 应用质量分类器：`cc` 且分数至少 0.8 时进入拒绝判断。
5. 应用 NSFW 与 Toxic 分类器：对应有害标签且分数至少 0.8 时删除。
6. 遮蔽邮箱、电话和 IP，记录各类替换次数。
7. 跨所有分片做整篇精确去重：对当前处理后的整篇文本计算 SHA-256，保留第一次出现的文档。
8. 跨所有分片进行行级去重。默认保守模式保护识别出的代码行与空行，每篇内的重复正文行留一份，删除跨文档共享的非正文短行。
9. 去重后复检空文、短文和页面状态。
10. 对整篇文档进行 MinHash + LSH 候选检索，以真实词 n-gram Jaccard 相似度确认，聚簇后保留最早的文档。

默认 `--quality-policy recall` 使用 [recall_text.py](cs336_data/recall_text.py) 的正文、代码和短文特征复核 Gopher / 质量模型的拒绝；这两个阶段拒绝的文本可能被保留。该复核不绕过 NSFW / Toxic 判断。`--quality-policy strict` 使用原 Gopher / 质量拒绝条件及硬长度门限。

默认 `--line-dedup-mode conservative` 用于减少正文和代码破坏；`--line-dedup-mode strict` 删除去重输入中所有重复原始行。课程 adapter 调用的独立精确行去重函数仍保持“删除所有重复副本”的语义。流水线在行去重前额外做整篇去重，所以即使两个选项都选 `strict`，也不是对早期流水线的逐项复现。

| 参数 | 当前默认值 |
| --- | ---: |
| WET 过滤进程 / MinHash 签名进程 | 2 / 2 |
| 质量、NSFW、Toxic 分数门限 | 各 0.8 |
| 去重后最小词数 | 50；`recall` 可豁免部分短文 |
| MinHash 数 / LSH bands | 100 / 10 |
| 词 n-gram 长度 | 5 |
| 真实 Jaccard 门限 | 0.8 |

Gopher 使用 NLTK 分词计数；去重后的长度检查使用空白分隔词计数，两者的“50 词”不一定对应相同的文本长度。

MinHash 比较前统一小写，做 NFD Unicode 规范化、去除组合重音、把标点替换为空格并合并空白。该规范化仅用于相似度计算，不直接覆盖训练正文。签名使用不同 seed 的 MurmurHash；主进程管理 LSH、真实 Jaccard 与聚簇，候选文档按需读取 n-gram，不把全部正文和 n-gram 集合常驻内存。签名索引、桶和文档位置仍随文档数增长。

过滤与精确去重的中间文件放在临时目录，结束后清理。每次运行生成独立时间戳目录，仅保留最终 `wet_XXXX.jsonl` 和 `filter_stats.json`。每条 JSONL 记录包含一个 `text` 字段，对应一篇文档；正文内部换行、空行和缩进通过 JSON 转义保存，读取后恢复。

### 当前全量处理结果

运行目录：`20261109_163531`。以下为实际统计，不再使用早期 8–9 亿 token 的外推估计。

| 项目 | 实际结果 |
| --- | ---: |
| 输入英文 WET 分片 | 32 个 |
| 输入文档数 | 817,397 |
| 最终文档数 | 547,458 |
| 最终空文档数 | 0 |
| 文档保留比例 | 66.98% |
| 全部丢弃文档数 | 269,939 |
| 最终 JSONL 分片 | 32 个 |
| GPT-2 token 数（包含每篇 EOS） | **711,501,653** |
| `uint16` 二进制字节数 | **1,423,003,306**，约 1.42 GB |

统计文件：

- [过滤统计](local-shared-data/filtered_corpus/20261109_163531/filter_stats.json)
- [分词统计](local-shared-data/tokenized_corpus/20261109_163531/tokenized_ids.stats.json)

这些是本地运行产物，不随代码仓库自动提供；其他环境需要下载、处理或复制数据。

各阶段贡献按“第一个删除该文档的阶段”归因，比例的分母为全部 **269,939 篇丢弃文档**，不是该阶段输入数：

| 阶段 | 输入文档 | 保留文档 | 删除文档 | 占全部删除的比例 |
| --- | ---: | ---: | ---: | ---: |
| 模板清理与页面状态检查 | 817,397 | 798,172 | 19,225 | 7.12% |
| Gopher（含正文复核） | 798,172 | 654,735 | 143,437 | 53.14% |
| 质量分类器（含正文复核） | 654,735 | 577,442 | 77,293 | 28.63% |
| NSFW | 577,442 | 576,641 | 801 | 0.30% |
| Toxic | 576,641 | 576,063 | 578 | 0.21% |
| PII 遮蔽 | 576,063 | 576,063 | 0 | 0.00% |
| 整篇精确去重 | 576,063 | 567,189 | 8,874 | 3.29% |
| 保守行级去重 | 567,189 | 567,189 | 0 | 0.00% |
| 去重后正文与长度检查 | 567,189 | 559,872 | 7,317 | 2.71% |
| MinHash 文档去重 | 559,872 | 547,458 | 12,414 | 4.60% |

PII 遮蔽修改了 244,091 篇文档，替换邮箱 273,036 次、电话 525,652 次、IP 13,690 次。各类受影响文档可能重叠，不能直接相加。

行级去重删除了 54,345,514 行、1,070,190,136 个字符，并产生 42 篇空文；这些空文在后续复检阶段删除。因此行级去重的“删除文档数为 0”不代表它没有改变文本。复检阶段合计删除 42 篇空文、6,904 篇短文和 371 篇模板文档。该阶段保留了 12,221 篇低于门限的短文，这是进入最后 MinHash 前的数量。

`rescued_at_stage` 表示该阶段复核放行的次数，不是最终新增保留文档数，也不是人工确认的“纠正误删”数量。

### 运行过滤与分词

在 `assignment4-data` 根目录运行。先等过滤完成，再运行分词：

```sh
uv run scripts/filter_data.py
uv run scripts/tokenize_texts.py
```

过滤可显式指定新输出目录、进程数和策略，例如：

```sh
uv run scripts/filter_data.py \
  --output-dir local-shared-data/filtered_corpus/recall_run_01 \
  --workers 2 \
  --hash-workers 2 \
  --quality-policy recall \
  --line-dedup-mode conservative
```

`--output-dir` 必须是尚不存在的目录。自定义目录名建议采用时间戳；分词脚本默认按目录名排序选择带完成报告的最新目录，自定义名称不一定按实际完成时间排序。

已有多个运行结果时，建议明确指定分词输入和输出。对当前已完成运行重新分词的命令是：

```sh
uv run scripts/tokenize_texts.py \
  --input-dir local-shared-data/filtered_corpus/20261109_163531 \
  --output local-shared-data/tokenized_corpus/20261109_163531/tokenized_ids.bin
```

该命令会覆盖对应的已有 `.bin` 和分词统计文件。重新过滤不会自动改写此前的运行目录。

分词逐条解析 JSONL，使用 GPT-2 tokenizer 编码完整正文，每篇末尾追加一个 EOS；不会给每个正文行单独追加 EOS，也不会把 JSON 的字段名或转义格式编码进训练数据。结果逐条转换为 `np.uint16`，通过 `ids_array.tofile(output_file)` 写入同一个二进制流，不汇总全部 token 到内存。所有文档可长于训练上下文，训练时再抽取窗口。

### 当前训练配置

生效配置为 [cs336_basics/train_config.py](cs336_basics/train_config.py)。当前 [scripts/train.py](scripts/train.py) 的入口构造 dataclass `Config`，通过 `--train-bin`、`--valid-bin` 指定数据；它不会自动读取 `configs/config.yaml` 或 `configs/experiment/your_data.yaml`。

模型参数保持课程的约 **430M**：词表 50,257，层数 24，`d_model=1024`，`d_ff=3072`，注意力头数 16，上下文 512，`rope_theta=10000`。按当前模型实现统计总参数为 **430,132,224**。

当前 batch 设置已为单卡实验调整；下面同时列出课程原始八卡设置，便于核对预算：

| 项目 | 当前单卡实验 | 课程原始八卡设置 |
| --- | ---: | ---: |
| GPU 数量 | 1 | 8 张 B200 |
| `train_batch_size`（每卡小批次） | 16 | 128 |
| `gradient_accumulation_steps` | 64 | 1 |
| 全局有效 batch（序列） | 1,024 | 1,024 |
| `eval_batch_size` | 16 | 128 |
| `eval_iterations` | 8,000 | 1,000 |
| `train_steps`（参数更新次数） | 16,384 | 16,384 |
| 每次更新处理的 token | 524,288 | 524,288 |
| 累计训练 token | 8,589,934,592 | 8,589,934,592 |

共同参数：`seed=0`、`dtype=bfloat16`、`compile=True`、`device=cuda`、学习率 `6e-4`、warmup 比例 0.03、weight decay 0.1、Adam betas `(0.9, 0.98)`、epsilon `1e-9`、梯度裁剪范数 1.0、每 2,000 步评估一次、每 20 步记录训练信息。

预算的计算方式为：

```text
GPU 数 × train_batch_size × gradient_accumulation_steps × context_length × train_steps
当前单卡：1 × 16 × 64 × 512 × 16,384 = 8,589,934,592 tokens
```

当前语料为 711,501,653 tokens，完整训练相当于约 **12.07 遍语料量**的重复采样。[get_batch](cs336_basics/data.py) 随机选择起点，再读取连续 512 个 token；标签窗口向后移动一位。采样可以重复或重叠，也可能跨越文档 EOS，不保证按顺序完整遍历数据。

每个小批次处理 8,192 tokens；累积 64 个小批次的梯度后才更新一次参数。累积次数是有效 batch 的组成部分，不是把同一批样本训练 64 遍。

两组验证配置都评估 `128,000 × 512 = 65,536,000` 个采样 token。保持有效 batch、训练 token 和验证采样量相同，不保证数值轨迹完全一致，也不自动意味着单卡适配满足课程正式提交要求；正式比较应遵循手册固定模型和训练流程的规定。

**当前单卡配置不能直接套用原始八卡 Modal 命令。** Modal 入口会启动 8 个进程，但读取当前的 `16 × 64` 设置，总训练量将变为 **68,719,476,736 tokens**，是原始课程预算的 8 倍。按课程八卡方案运行前，应核对并恢复该列的每卡 batch、累积次数及验证配置。

### 单卡训练入口与产物

训练需要可用 CUDA 环境，以及分词后的训练、验证文件。在租卡机器上保持以下相对目录，或把参数替换为实际文件路径：

```sh
uv run scripts/train.py \
  --train-bin local-shared-data/tokenized_corpus/20261109_163531/tokenized_ids.bin \
  --valid-bin local-shared-data/tokenized_paloma_c4_100_domains_validation.bin
```

训练数据通过 `np.memmap` 按批读取，不把整个 token 文件一次性放入显存。80/85GB 显卡上的峰值显存和实际训练时长尚未实测，先做训练与验证的短程压测，再投入完整预算。

训练输出路径目前由 `scripts/train.py` 中的 `DEFAULT_MODAL_MODEL_OUTPUT` 固定为 `/root/data/output/your_data`；运行用户需要对此目录有写权限。当前 CLI 不提供训练输出目录或 batch 的覆盖参数，不能只修改 `your_data.yaml` 来改变这些行为。

当前 `wandb_project`、`wandb_entity` 均为 `None`，未启用 W&B 记录。配置二者后才能初始化对应的 W&B 实验。当前 `save_checkpoints=False`，训练正常结束时保存最终 `model.pt` 和 `model_config.json`；默认不保存各次验证时的中间模型。即使开启中间保存，当前脚本也只写模型权重与配置，没有完整的优化器 / 随机数状态恢复流程，不应当作完整断点续训方案。

C4 100 验证文件保持独立，不能把其文本或 token 拼入训练集。当前流程未主动使用该文件生成训练正文；这不等于已经完成训练语料与验证语料的近重复排查。

### 已知质量问题与待完成验证

功能测试之外，做过一次冻结规则后的抽样检查：从此前未用于调参的两个 WET 输出分片各抽 500 篇，排除与早期审查文本完全相同的记录后，共检查 1,000 篇。默认策略最终保留 735 篇，严格策略保留 467 篇。两者同时改变了质量策略和行去重方式，不是单因素消融实验，这些保留率也不能当成全量语料的质量准确率。

随机内容抽查结果：

| 抽查对象 | 数量 | 观察结果 |
| --- | ---: | --- |
| 默认保留文档 | 20 | 11 篇正文可用，5 篇混杂或边界样本，4 篇明显噪声 |
| 默认丢弃文档 | 20 | 17 篇删除合理，3 篇属于馆藏、房源等结构化信息边界 |
| 该批全部保留短文 | 14 | 13 篇是提示页、重定向、维护页、标题等非目标正文，1 篇是课程搜索入口 |

正文复核能保住科学论文、技术问答和完整条款，也会放行部分下载诱导模板、博彩关键词拼接、空购物页面。短文“句子完整”与足够多的英文正文词，并不保证存在有用的主要内容。上述为内容审查判断，不是独立人工金标，也未核实文章事实或证明语言模型训练收益。

该批样本的 735 篇输出逐篇完成了 GPT-2 编码 / 解码还原、EOS 和 `uint16` 文件大小检查；对进入保守行去重的 737 篇文档，没有发现被该启发式识别出的代码行或独有正文行丢失。这只能证明这些检查覆盖到的行为，不能保证所有有用内容都被识别。

后续需要补充：

- 修正短文豁免和模板漏筛，并重新生成对应的过滤结果与 token 文件。
- 在目标 GPU 上记录峰值显存、吞吐和完整训练耗时。
- 用相同模型、预算与验证设置比较不同语料版本的 C4 100 loss / perplexity。
- 保存最佳验证 loss、训练曲线、模型产物与生成样本；当前文档不宣称全量过滤已改善验证 perplexity。
