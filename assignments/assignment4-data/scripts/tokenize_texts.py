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
    """Read TXT lines; old runs without the format marker use one file per document."""
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
    parser.add_argument("--input-dir", type=Path, help="TXT directory; defaults to the latest completed filtering run.")
    parser.add_argument("--output", type=Path, help="Output .bin path.")
    args = parser.parse_args(argv)
    input_dir = args.input_dir if args.input_dir is not None else _latest_input_dir()
    if not input_dir.is_dir() or not any(input_dir.glob("*.txt")):
        raise FileNotFoundError(f"No TXT shards found in {input_dir}")
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
