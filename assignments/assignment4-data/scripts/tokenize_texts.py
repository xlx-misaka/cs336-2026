"""将过滤后的正文流式编码为 GPT-2 token，保存为训练脚本可读取的 uint16 二进制。

在 assignment4-data 项目根目录运行：
    uv run scripts/tokenize_texts.py
    uv run scripts/tokenize_texts.py --input-dir path/to/minhash_dedup --output /tmp/train.bin

默认读取最近一次已完成过滤的 minhash_dedup/，也可通过 --input-dir 指定。
新格式：每个 JSONL 记录为 {"text": "正文"}，解析后保留正文内部换行、空行和缩进。
旧格式：按 filter_stats.json 的标记读取一行一文档 TXT；旧的无标记报告按一文件一文档。
没有报告的 TXT 默认一行一文档。每个目录只应存放一种格式，避免重复读取。
流程：逐条读取非空正文 → 多进程完整编码 → 每篇追加一个 EOS → uint16.tofile。
只对正文编码，不把 JSON 字段名、引号或转义符写入模型输入；不对每个正文行追加 EOS。
默认输出 local-shared-data/tokenized_texts/运行目录名/tokenized_ids.bin，已有同名文件会覆盖。
旁边的 tokenized_ids.stats.json 记录非空文档数、包含 EOS 的 token 数和文件字节数。
每个 token 占两个字节；两个 worker 各加载一次 tokenizer，不汇总全部正文和编码结果到列表。
"""

import argparse
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer

from cs336_data.common import get_shared_assets_path

NUM_CPUS = 2
CHUNK_SIZE = 100
tokenizer = None


def _load_tokenizer():
    """Load once in each worker, without creating extra tokenizer threads."""
    global tokenizer
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")


def iter_documents(input_dir):
    """Decode JSONL text records, or read legacy TXT document layouts."""
    jsonl_paths = sorted(input_dir.glob("*.jsonl"))
    if jsonl_paths:
        if any(input_dir.glob("*.txt")):
            raise ValueError(f"Mixed JSONL and TXT shards in {input_dir}; use separate input directories.")
        for path in jsonl_paths:
            with path.open(encoding="utf-8") as source:
                for record in source:
                    text = json.loads(record)["text"]
                    if not isinstance(text, str):
                        raise TypeError(f"JSONL text must be a string: {path}")
                    if text.strip():
                        yield text
        return
    stats_path = input_dir.parent / "filter_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    line_documents = not stats or stats.get("text_format") == "one_document_per_line"
    for path in sorted(input_dir.glob("*.txt")):
        with path.open(encoding="utf-8") as source:
            if line_documents:
                for line in source:
                    if line.strip():
                        yield line.rstrip("\r\n")
            else:
                text = source.read()
                if text.strip():
                    yield text


def tokenize_text_and_add_eos(text):
    assert tokenizer is not None
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids.append(tokenizer.eos_token_id)
    return ids


def _latest_input_dir():
    processed = get_shared_assets_path() / "processed_wet"
    candidates = sorted(
        path for path in processed.glob("*/minhash_dedup") if (path.parent / "filter_stats.json").is_file()
    )
    if not candidates:
        raise FileNotFoundError("No completed filtering run found; specify --input-dir.")
    return candidates[-1]


def run(argv=None):
    parser = argparse.ArgumentParser(description="Serialize GPT-2 tokens as raw uint16 IDs.")
    parser.add_argument(
        "--input-dir", type=Path, help="JSONL or legacy TXT directory; defaults to the latest completed run."
    )
    parser.add_argument("--output", type=Path, help="Output .bin path.")
    args = parser.parse_args(argv)
    input_dir = args.input_dir if args.input_dir is not None else _latest_input_dir()
    if not input_dir.is_dir() or not any(path.suffix in {".jsonl", ".txt"} for path in input_dir.iterdir()):
        raise FileNotFoundError(f"No JSONL or TXT shards found in {input_dir}")
    output_path = (
        args.output or get_shared_assets_path() / "tokenized_texts" / input_dir.parent.name / "tokenized_ids.bin"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokens = documents = 0
    with multiprocessing.Pool(NUM_CPUS, initializer=_load_tokenizer) as pool, output_path.open("wb") as output_file:
        results = pool.imap(tokenize_text_and_add_eos, iter_documents(input_dir), chunksize=CHUNK_SIZE)
        for result in tqdm(results, desc="Tokenizing documents", unit="doc"):
            ids = np.asarray(result, dtype=np.uint16)
            ids.tofile(output_file)
            tokens += ids.size
            documents += 1

    file_bytes = output_path.stat().st_size
    if file_bytes != tokens * np.dtype(np.uint16).itemsize:
        raise RuntimeError("Output size does not match the token count.")
    report = {
        "input_dir": str(input_dir.resolve()),
        "output_path": str(output_path.resolve()),
        "documents": documents,
        "tokens": tokens,
        "file_bytes": file_bytes,
        "dtype": "uint16",
    }
    output_path.with_suffix(".stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Tokenized {documents} documents into {tokens} tokens: {output_path}")


if __name__ == "__main__":
    run()
