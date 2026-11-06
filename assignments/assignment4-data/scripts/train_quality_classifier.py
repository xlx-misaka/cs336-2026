# 质量分类器训练记录（静态快照，评估时间：2026-11-06 10:37，Asia/Shanghai）
#
# 数据来源与处理：
# - 正样本：Wikipedia 引用的外部网页，标签 __label__high-quality。
# - 负样本：local-shared-data/CC/example.warc.gz，标签 __label__low-quality。
# - 保留 HTTP 200 的 text/html 响应，经正文提取后非空，语言为 en 且分数 > 0.7。
# - 标签按来源生成；当前训练数据未应用 Gopher、NSFW、toxic 过滤或正文去重。
# - 抽样 URL 数为 10,000；每类文本上限为 4,000，实际为 CC 4,000、Wiki 3,622。
# - 数据随机种子为 336；逐文档以 0.2 的概率分入验证集，未按域名分组。
#
# 实际数据规模：
#                    CC（负）    Wiki（正）    合计
#   训练集             3,244       2,901       6,145
#   验证集               756         721       1,477
#   总计               4,000       3,622       7,622
#
# 训练配置与产物：
# - 调用 fasttext.train_supervised，未显式覆盖训练超参数。
# - 当前模型：dim=100，epoch=5，wordNgrams=1，minCount=1，loss=softmax。
# - 不使用字符 n-gram（minn=0，maxn=0）；模型训练 seed 未在脚本中显式指定。
# - 模型：data/quality_classifier_data/quality_classifier.bin，约 221 MiB。
# - 模型文件最后修改时间：2026-11-05 23:42:13（Asia/Shanghai）。
#
# 实测结果（根据来源标签评估，不代表人工确认的文本质量）：
# - 训练集准确率：82.69%；验证集准确率：81.79%；多数类基线：51.18%。
# - 验证集高质量类：Precision=80.13%，Recall=83.36%，F1=81.71%。
# - 验证集混淆计数：Wiki->Wiki 601，Wiki->CC 120，CC->Wiki 149，CC->CC 607。
# - uv run pytest -k test_classify_quality：通过（sanity check）。
#
# 本次评估发现的数据局限：
# - 训练集有 232 条额外的完全重复正文记录，以及 2 种正文存在正负标签冲突。
# - 训练与验证共享 13 种完全相同的正文，影响 62 条验证记录（约 4.2%）。
# - 排除与训练正文完全重复的验证记录后，准确率为 81.55%。
# - 正样本中仍有网络验证、页面迁移、域名过期等缺乏有效正文的页面。
#
# 此注释不会自动更新；更换数据、训练配置或模型后，应重新评估并更新记录。

import gzip
import random
import subprocess
from pathlib import Path

import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
from more_itertools import sample

from cs336_data.common import get_shared_assets_path
from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.language_identification import identify_language

_ROOT_PATH = Path("data/quality_classifier_data")
_ROOT_PATH.mkdir(parents=True, exist_ok=True)

_SEED = 336
_MAX_NUM_WIKI_URLS = 10000
_MAX_NUM_WIKI_TEXTS = 4000
_MAX_NUM_CC_TEXTS = 4000

random.seed(_SEED)


def get_wiki_urls():
    """
    Load a specified number of Wikipedia URLs from a compressed text file.
    """
    wiki_path = get_shared_assets_path() / "wiki" / "enwiki-20260501-extracted_urls.txt.gz"
    with gzip.open(wiki_path, mode="rt", encoding="utf-8") as file:
        iterator = iter(file)
        selected_urls = sample(iterator, _MAX_NUM_WIKI_URLS)

    saved_path = _ROOT_PATH / "sampled_wiki_urls.txt"
    with open(saved_path, mode="w", encoding="utf-8") as file:
        file.writelines(url.strip() + "\n" for url in selected_urls)


def wget_wiki_urls():
    """
    Download a list of URLs using wget.
    """
    wiki_urls_path = _ROOT_PATH / "sampled_wiki_urls.txt"
    if not wiki_urls_path.exists():
        get_wiki_urls()

    wiki_warc_path = _ROOT_PATH / "sampled_wiki"
    command = [
        "wget",
        "--timeout=5",
        "-i",
        wiki_urls_path,
        f"--warc-file={wiki_warc_path}",
        "-O",
        "/dev/null",
    ]

    subprocess.run(command, check=False)


def get_wiki_warc():
    """
    get the WARC file from the Common Crawl dataset.
    """
    wiki_warc_path = _ROOT_PATH / "sampled_wiki.warc.gz"
    if not wiki_warc_path.exists():
        wget_wiki_urls()
    return wiki_warc_path


def get_cc_warc():
    """
    get the Common Crawl WARC file.
    """
    cc_warc_path = get_shared_assets_path() / "CC" / "example.warc.gz"
    return cc_warc_path


def get_htlm_en_texts_form_warc(warc_path: Path | str):
    """
    Get the HTML bytes from a WARC file.
    """
    warc_path = Path(warc_path)
    with open(warc_path, "rb") as f:
        for record in ArchiveIterator(f):
            if record.record_type == WarcRecordType.response and record.http_content_type == "text/html":
                if not record.http_headers or record.http_headers.status_code != 200:
                    continue
                html_bytes = record.reader.read()
                text = extract_text_from_html_bytes(html_bytes)
                if text is None or not text.strip():
                    continue
                label, score = identify_language(text)
                if label == "en" and score > 0.7:
                    yield text


def label_text(source: str, text: str):
    """
    Label the text by source.
    """
    label = "__label__low-quality" if source == "cc" else "__label__high-quality"
    return f"{label} {text.replace('\n', ' ')}"


def get_data_set():
    """
    Generate a data set from the CC and Wiki WARC files.
    """
    train_set_path = _ROOT_PATH / "train_set.txt"
    valid_set_path = _ROOT_PATH / "valid_set.txt"

    cc_warc_path = get_cc_warc()
    wiki_warc_path = get_wiki_warc()

    cc_texts = get_htlm_en_texts_form_warc(cc_warc_path)
    wiki_texts = get_htlm_en_texts_form_warc(wiki_warc_path)
    generator = random.Random(_SEED)
    cnt_cc = 0
    cnt_wiki = 0
    with open(train_set_path, "w") as train_set_file, open(valid_set_path, "w") as valid_set_file:
        for text in cc_texts:
            if cnt_cc == _MAX_NUM_CC_TEXTS:
                break
            cnt_cc += 1
            text = label_text("cc", text)
            if generator.random() < 0.2:
                valid_set_file.write(text + "\n")
            else:
                train_set_file.write(text + "\n")
        for text in wiki_texts:
            if cnt_wiki == _MAX_NUM_WIKI_TEXTS:
                break
            cnt_wiki += 1
            text = label_text("wiki", text)
            if generator.random() < 0.2:
                valid_set_file.write(text + "\n")
            else:
                train_set_file.write(text + "\n")

    return train_set_path, valid_set_path


def run():
    """
    Run the script.
    """
    train_set_path, valid_set_path = get_data_set()
    model = fasttext.train_supervised(input=str(train_set_path))
    print("=" * 80)
    model.save_model(str(_ROOT_PATH / "quality_classifier.bin"))
    print("Model saved to quality_classifier.bin")

    print("=" * 80)
    n, precision, recall = model.test(str(valid_set_path))
    print(f"Test results: n={n}, precision={precision}, recall={recall}")
    print("=" * 80)


if __name__ == "__main__":
    run()
