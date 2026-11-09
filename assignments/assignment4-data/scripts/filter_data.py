"""并行过滤 WET，减少可读正文误删，并对整篇文档进行 MinHash 去重。

在 assignment4-data 项目根目录运行：
    uv run scripts/filter_data.py
    uv run scripts/filter_data.py --output-dir /tmp/filtered-demo --workers 2 --hash-workers 2 --min-words 50

输入：furu manifest.json 中列出的英文 WET，以及现有质量、NSFW、Toxic 分类器。
流程：WET 正文 → 界面模板清理 → Gopher/质量信号 → NSFW → Toxic → PII 遮蔽，
      然后整篇精确去重、行级去重、正文复检和 MinHash 文档去重。
输出：默认 local-shared-data/filtered_corpus/北京时间戳/，仅包含最终的
      wet_XXXX.jsonl 和 filter_stats.json，每个 WET 对应一个 JSONL 分片。
每行是 {"text": "正文"}。过滤和精确去重的中间文件使用临时目录，退出时自动清理。
正文内部换行、空行、制表符由 JSON 转义保存，读取后恢复；正文首尾仍由读取函数 strip。
默认 --quality-policy recall：可读正文/代码为 Gopher 或质量模型拒绝提供复核证据，
短文有完整句子时可保留；这不是置信度校准。strict 使用原 Gopher/质量拒绝条件和硬长度门限。
默认 --line-dedup-mode conservative：保留代码及有用正文，文档内重复正文留一份；
strict 删除所有重复原始行。课程独立去重和 Gopher adapter 保持原语义。
空文始终删除；--min-words 默认 50，recall 模式保留有正文证据的完整短文。
每个拒绝、模板修改和规则复核都记录统计；MinHash 每个连通分组保留最早的一篇。
过滤和 MinHash 签名计算分别使用进程池；全局行统计、LSH、Jaccard 和聚簇由主进程处理。
"""

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from fastwarc import ArchiveIterator, WarcRecordType
from tqdm import tqdm

from cs336_data.common import get_shared_assets_path
from cs336_data.deduplication import _get_n_gram_set, _jaccard, _minhash, _norm_text, _UnionFind
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech
from cs336_data.mask_pii import mask_emails, mask_ips, mask_phone_numbers
from cs336_data.quality import classify_quality, gopher_quality_filter
from cs336_data.text_quality import (
    clean_boilerplate,
    is_code_line,
    page_rejection_reason,
    prose_word_count,
    text_signals,
)

# Path to the manifest file.
MANIFEST_PATH = Path(
    "local-shared-data/furu/cs336_data/wet_files/EnglishWetFiles/"
    "ae4fa708c26a65eb662a/c22742cf4085f62d0f56/result/manifest.json"
)

# Parallelism parameters.
NUM_CPUS = 2
FILTER_STAGES = ("boilerplate", "gopher", "quality", "nsfw", "toxic")
PII_TYPES = ("emails", "phone_numbers", "ips")

# filter parameters
QUALITY_THRESHOLD = 0.8
NSFW_THRESHOLD = 0.8
TOXIC_THRESHOLD = 0.8
MIN_FINAL_WORDS = 50

# minhash parameters
NUM_HASHES = 100
NUM_BANDS = 10
NGRAMS = 5
JACCARD_THRESHOLD = 0.8
TEXT_FORMAT = "jsonl_text"


def get_wet_pathes(manifest_path=MANIFEST_PATH):
    """Load the data pathes from the manifest file."""
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    pathes = [Path(entry["$furu"]["value"]) for entry in entries]
    return pathes


def load_texts_from_wet(wet_path):
    """Load the texts from a WET file."""
    with wet_path.open("rb") as f:
        record_iter = ArchiveIterator(
            f,
            record_types=WarcRecordType.conversion,
        )
        for record in record_iter:
            text_bytes = record.reader.read()
            text = text_bytes.decode("utf-8", errors="replace")
            yield text.strip()


def filter_text(text: str, counts: Counter[str] | None = None, quality_policy="recall"):
    """Combine explicit page-state checks, quality signals and unchanged harmful-content gates."""
    if counts is None:
        counts = Counter()
    counts["input_documents"] += 1

    original_text = text
    text, removed = clean_boilerplate(text)
    counts["boilerplate_modified_documents"] += int(removed > 0)
    counts["boilerplate_removed_lines"] += removed
    signals = text_signals(text)
    reason = page_rejection_reason(text, signals, removed)
    if reason:
        counts["boilerplate_discarded"] += 1
        counts[f"boilerplate_reason_{reason}"] += 1
        return None
    readable = signals.substantial_body or signals.readable_short
    rescue = quality_policy == "recall" and readable

    # gopher quality filter
    if not gopher_quality_filter(text):
        # Cleaning interface text can move an otherwise valid document across a numeric rule boundary.
        cleanup_pass = (
            not rescue
            and quality_policy == "recall"
            and removed > 0
            and signals.prose_words > 0
            and gopher_quality_filter(original_text)
        )
        if not (rescue or cleanup_pass):
            counts["gopher_discarded"] += 1
            return None
        counts["gopher_rescued"] += 1
        counts["gopher_preclean_pass"] += int(cleanup_pass)

    # quality classifier
    label, score = classify_quality(text)
    if label == "cc" and score >= QUALITY_THRESHOLD:
        if not rescue:
            counts["quality_discarded"] += 1
            return None
        counts["quality_rescued"] += 1

    # harmful content classifier
    label, score = classify_nsfw(text)
    if label == "nsfw" and score >= NSFW_THRESHOLD:
        counts["nsfw_discarded"] += 1
        return None
    label, score = classify_toxic_speech(text)
    if label == "toxic" and score >= TOXIC_THRESHOLD:
        counts["toxic_discarded"] += 1
        return None

    # mask pii
    modified = False
    for name, mask in zip(PII_TYPES, (mask_emails, mask_phone_numbers, mask_ips), strict=True):
        text, num_masked = mask(text)
        counts[f"{name}_masked_instances"] += num_masked
        counts[f"{name}_modified_documents"] += int(num_masked > 0)
        modified |= num_masked > 0
    counts["pii_modified_documents"] += int(modified)
    counts["kept_documents"] += 1

    return text


def process_single_wet(wet_index: int, wet_path, output_root: Path, quality_policy="recall"):
    """Write one JSONL shard per WET, preserving document line breaks."""
    output_dir = output_root / "filtered"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"wet_{wet_index:04d}.jsonl"
    counts = Counter()
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for text in load_texts_from_wet(wet_path):
            text = filter_text(text, counts, quality_policy)
            if text is not None:
                output_file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"WET {wet_index}: kept {counts['kept_documents']}/{counts['input_documents']} documents.")
    return {
        "path": output_path,
        "wet_index": wet_index,
        "wet_path": str(wet_path),
        "counts": dict(counts),
    }


def _stage_stats(name, entered, kept, **details):
    return {
        "name": name,
        "input_documents": entered,
        "kept_documents": kept,
        "discarded_documents": entered - kept,
        **details,
    }


def _filter_stage_stats(counts):
    stages = []
    entered = counts["input_documents"]
    for name in FILTER_STAGES:
        kept = entered - counts[f"{name}_discarded"]
        stage = _stage_stats(name, entered, kept)
        if name == "boilerplate":
            stage.update(
                modified_documents=counts["boilerplate_modified_documents"],
                removed_lines=counts["boilerplate_removed_lines"],
                by_reason={
                    key.removeprefix("boilerplate_reason_"): value
                    for key, value in counts.items()
                    if key.startswith("boilerplate_reason_")
                },
            )
        if name in {"gopher", "quality"}:
            stage["rescued_at_stage"] = counts[f"{name}_rescued"]
        if name == "gopher":
            stage["preclean_passes"] = counts["gopher_preclean_pass"]
        stages.append(stage)
        entered = kept
    assert entered == counts["kept_documents"]
    stages.append(
        _stage_stats(
            "pii_masking",
            entered,
            entered,
            modified_documents=counts["pii_modified_documents"],
            by_type={
                name: {
                    "modified_documents": counts[f"{name}_modified_documents"],
                    "masked_instances": counts[f"{name}_masked_instances"],
                }
                for name in PII_TYPES
            },
        )
    )
    return stages


def _read_documents(path):
    with path.open(encoding="utf-8") as source:
        for record in source:
            yield json.loads(record)["text"]


def _exact_document_deduplication(input_files, output_directory):
    """Retain one exact document before counting shared lines, avoiding loss of every copy."""
    output_directory.mkdir(parents=True, exist_ok=True)
    seen = set()
    entered = kept = 0
    for path in input_files:
        with (output_directory / path.name).open("w", encoding="utf-8", newline="\n") as destination:
            for text in _read_documents(path):
                entered += 1
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                if digest not in seen:
                    seen.add(digest)
                    destination.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    kept += 1
    return _stage_stats("exact_document_deduplication", entered, kept)


def _exact_line_deduplication(input_files, output_directory, mode="strict"):
    """Strict course line semantics or conservative body/code preservation for the pipeline."""
    if mode not in {"strict", "conservative"}:
        raise ValueError("Unknown line deduplication mode")
    output_directory.mkdir(parents=True, exist_ok=True)
    line_counts = Counter()
    for path in input_files:
        for text in _read_documents(path):
            # Match text-mode file iteration, including universal newline handling.
            keys = [hash(line) for line in StringIO(text, newline=None)]
            line_counts.update(set(keys) if mode == "conservative" else keys)

    documents = modified = removed_lines = removed_characters = empty = 0
    for path in input_files:
        with (output_directory / path.name).open("w", encoding="utf-8", newline="\n") as destination:
            for text in _read_documents(path):
                lines = list(StringIO(text, newline=None))
                unique_lines = []
                seen = set()
                for line in lines:
                    key = hash(line)
                    if mode == "strict":
                        keep = line_counts[key] == 1
                    else:
                        protected = not line.strip() or is_code_line(line)
                        keep = protected or (key not in seen and (line_counts[key] == 1 or prose_word_count(line) > 0))
                    if keep:
                        unique_lines.append(line)
                    seen.add(key)
                result = "".join(unique_lines)
                documents += 1
                modified += int(len(lines) != len(unique_lines))
                removed_lines += len(lines) - len(unique_lines)
                removed_characters += sum(map(len, lines)) - len(result)
                empty += int(not result.strip())
                destination.write(json.dumps({"text": result}, ensure_ascii=False) + "\n")
    return _stage_stats(
        "exact_line_deduplication",
        documents,
        documents,
        modified_documents=modified,
        removed_lines=removed_lines,
        removed_characters=removed_characters,
        empty_output_documents=empty,
        mode=mode,
    )


def _filter_short_documents(input_files, output_directory, min_words, preserve_readable=False):
    """Recheck page state and length after line removal; optionally retain complete short prose."""
    output_directory.mkdir(parents=True, exist_ok=True)
    entered = kept = empty = short = templates = rescued = 0
    for path in input_files:
        with (output_directory / path.name).open("w", encoding="utf-8", newline="\n") as destination:
            for text in _read_documents(path):
                entered += 1
                signals = text_signals(text) if preserve_readable else None
                reason = page_rejection_reason(text, signals) if signals is not None else None
                if not text.strip():
                    empty += 1
                elif reason:
                    templates += 1
                elif len(text.split()) < min_words and not (signals and signals.readable_short):
                    short += 1
                else:
                    destination.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    kept += 1
                    rescued += int(len(text.split()) < min_words)
    return _stage_stats(
        "post_dedup_quality",
        entered,
        kept,
        empty_discarded=empty,
        short_discarded=short,
        template_discarded=templates,
        readable_short_kept=rescued,
        min_words=min_words,
    )


def _document_signature(text):
    ngrams = _get_n_gram_set(_norm_text(text), NGRAMS)
    return _minhash(ngrams, NUM_HASHES) if ngrams is not None else None


def _minhash_document_deduplication(input_files, output_directory, workers=1):
    """Compare decoded texts, keeping byte offsets into JSONL shards for bounded text storage."""
    if NUM_HASHES <= 0 or NUM_BANDS <= 0 or NUM_HASHES % NUM_BANDS:
        raise ValueError("NUM_HASHES must be positive and divisible by positive NUM_BANDS")
    if workers < 1:
        raise ValueError("Hash workers must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    rows_per_band = NUM_HASHES // NUM_BANDS
    # Store only locations, not all documents or all n-gram sets.
    documents = []
    for shard_index, path in enumerate(input_files):
        offset = 0
        with path.open("rb") as source:
            for line in source:
                documents.append((shard_index, offset, len(line)))
                offset += len(line)

    groups = _UnionFind(len(documents))
    buckets = [{} for _ in range(NUM_BANDS)]
    with ExitStack() as stack:
        sources = [stack.enter_context(path.open("rb")) for path in input_files]
        destinations = [stack.enter_context((output_directory / path.name).open("wb")) for path in input_files]

        def read_document(index):
            shard_index, offset, length = documents[index]
            source = sources[shard_index]
            source.seek(offset)
            return source.read(length)

        def read_ngrams(index):
            text = json.loads(read_document(index))["text"]
            return _get_n_gram_set(_norm_text(text), NGRAMS)

        # A separate file iterator feeds workers; it never shares seek handles with Jaccard reads.
        texts = (text for path in input_files for text in _read_documents(path))
        if workers == 1:
            signatures = map(_document_signature, texts)
        else:
            pool = stack.enter_context(multiprocessing.Pool(workers))
            signatures = pool.imap(_document_signature, texts, chunksize=16)
        for index, signature in enumerate(tqdm(signatures, total=len(documents), desc="MinHash documents", unit="doc")):
            if signature is None:
                continue
            candidates = set()
            for band_index, band in enumerate(buckets):
                start = band_index * rows_per_band
                bucket = band.setdefault(tuple(signature[start : start + rows_per_band]), [])
                candidates.update(bucket)
                bucket.append(index)
            # A pair is checked at most once, even if it matched multiple bands.
            ngrams = None
            for candidate in sorted(candidates):
                left, right = groups.find(candidate), groups.find(index)
                if left == right:
                    continue
                if ngrams is None:
                    ngrams = read_ngrams(index)
                    assert ngrams is not None, "A signed document must have n-grams"
                other_ngrams = read_ngrams(candidate)
                if other_ngrams is not None and _jaccard(ngrams, other_ngrams) >= JACCARD_THRESHOLD:
                    groups.merge(min(left, right), max(left, right))

        kept = 0
        for index, (shard_index, _, _) in enumerate(documents):
            if groups.find(index) == index:
                destinations[shard_index].write(read_document(index))
                kept += 1
    return _stage_stats("minhash_document_deduplication", len(documents), kept)


def run(argv=None):
    parser = argparse.ArgumentParser(description="Filter and deduplicate WET documents into JSONL shards.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="Input furu WET manifest.")
    parser.add_argument("--output-dir", type=Path, help="New output directory; defaults to a timestamped directory.")
    parser.add_argument("--workers", type=int, default=NUM_CPUS, help="Number of WET filtering processes.")
    parser.add_argument("--hash-workers", type=int, default=NUM_CPUS, help="Number of MinHash signature processes.")
    parser.add_argument(
        "--quality-policy",
        choices=("recall", "strict"),
        default="recall",
        help="Recall policy reviews rule/model rejections using body evidence.",
    )
    parser.add_argument(
        "--line-dedup-mode",
        choices=("conservative", "strict"),
        default="conservative",
        help="Conservative mode preserves prose/code; strict removes all repeated lines.",
    )
    parser.add_argument(
        "--min-words", type=int, default=MIN_FINAL_WORDS, help="Minimum whitespace words after deduplication."
    )
    args = parser.parse_args(argv)
    if args.workers < 1 or args.hash_workers < 1:
        parser.error("--workers and --hash-workers must be positive")
    if args.min_words < 0:
        parser.error("--min-words cannot be negative")
    wet_pathes = get_wet_pathes(args.manifest)
    # Generate one experiment directory in the parent and pass it to every worker.
    output_root = args.output_dir or (
        get_shared_assets_path() / "filtered_corpus" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    )
    output_root.mkdir(parents=True, exist_ok=False)
    filtered_shards = []
    counts = Counter()
    per_wet_file = []
    with TemporaryDirectory(prefix=".intermediate_", dir=output_root) as temporary:
        work_root = Path(temporary)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_single_wet, wet_index, wet_path, work_root, args.quality_policy)
                for wet_index, wet_path in enumerate(wet_pathes)
            ]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                result = future.result()
                filtered_shards.append(result["path"])
                counts.update(result["counts"])
                per_wet_file.append({key: result[key] for key in ("wet_index", "wet_path", "counts")})

        # Shut down workers before building the global line counter.
        filtered_shards.sort()
        document_directory = work_root / "document_dedup"
        document_stats = _exact_document_deduplication(filtered_shards, document_directory)
        document_shards = [document_directory / path.name for path in filtered_shards]
        exact_directory = work_root / "exact_dedup"
        exact_stats = _exact_line_deduplication(document_shards, exact_directory, args.line_dedup_mode)
        exact_shards = [exact_directory / path.name for path in filtered_shards]
        clean_directory = work_root / "length_checked"
        length_stats = _filter_short_documents(
            exact_shards, clean_directory, args.min_words, preserve_readable=args.quality_policy == "recall"
        )
        clean_shards = [clean_directory / path.name for path in filtered_shards]
        minhash_stats = _minhash_document_deduplication(clean_shards, output_root, args.hash_workers)

    stages = _filter_stage_stats(counts)
    assert document_stats["input_documents"] == counts["kept_documents"]
    stages.append(document_stats)
    assert exact_stats["input_documents"] == document_stats["kept_documents"]
    stages.append(exact_stats)

    assert length_stats["input_documents"] == exact_stats["kept_documents"]
    stages.append(length_stats)
    assert minhash_stats["input_documents"] == length_stats["kept_documents"]
    stages.append(minhash_stats)
    final_documents = minhash_stats["kept_documents"]
    final_empty = 0
    total_discarded = counts["input_documents"] - final_documents
    assert sum(stage["discarded_documents"] for stage in stages) == total_discarded
    for stage in stages:
        stage["discard_contribution_percent"] = (
            100 * stage["discarded_documents"] / total_discarded if total_discarded else None
        )

    report = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "manifest_path": str(args.manifest.resolve()),
        "output_root": str(output_root.resolve()),
        "input_wet_files": len(wet_pathes),
        "input_documents": counts["input_documents"],
        "text_format": TEXT_FORMAT,
        "final_output_files": len(filtered_shards),
        "final_output_documents": final_documents,
        "final_empty_documents": final_empty,
        "final_nonempty_documents": final_documents - final_empty,
        "total_discarded_documents": total_discarded,
        "counting_notes": {
            "discard_attribution": "First rejecting stage in execution order; each document is counted once.",
            "pii": "Masking changes text without discarding documents; per-type document counts can overlap.",
            "storage": "Only final JSONL shards and this report are kept; intermediate files are temporary.",
            "line_breaks": "JSON escapes preserve document line breaks; deduplication operates on decoded text.",
            "exact_deduplication": "Exact documents are kept once; line behavior follows line_dedup_mode.",
            "post_dedup_quality": "Page-state checks run again; recall policy can retain complete short prose.",
            "rescues": "Counts indicate passes at each stage, not gold-label accuracy or final retained counts.",
            "zero_discards": "Contribution percentages are null when no documents were discarded.",
        },
        "parameters": {
            "workers": args.workers,
            "hash_workers": args.hash_workers,
            "min_final_words": args.min_words,
            "quality_policy": args.quality_policy,
            "line_dedup_mode": args.line_dedup_mode,
            "quality_threshold": QUALITY_THRESHOLD,
            "nsfw_threshold": NSFW_THRESHOLD,
            "toxic_threshold": TOXIC_THRESHOLD,
            "num_hashes": NUM_HASHES,
            "num_bands": NUM_BANDS,
            "ngrams": NGRAMS,
            "jaccard_threshold": JACCARD_THRESHOLD,
        },
        "stages": stages,
        "per_wet_file": sorted(per_wet_file, key=lambda item: item["wet_index"]),
    }
    stats_path = output_root / "filter_stats.json"
    stats_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nStage                              Input       Kept  Discarded  Share of all discards")
    for stage in stages:
        percent = stage["discard_contribution_percent"]
        share = "N/A" if percent is None else f"{percent:.2f}%"
        print(
            f"{stage['name']:<34} {stage['input_documents']:>7} {stage['kept_documents']:>10} "
            f"{stage['discarded_documents']:>10} {share:>12}"
        )
    print(f"PII: modified {counts['pii_modified_documents']} documents.")
    print(f"Exact-line deduplication: removed {exact_stats['removed_lines']} lines.")
    print(
        f"Post-deduplication: discarded {length_stats['empty_discarded']} empty "
        f"and {length_stats['short_discarded']} short documents; "
        f"kept {length_stats['readable_short_kept']} readable short documents."
    )
    print(
        f"Final output: {final_documents} document records ({final_empty} empty) "
        f"in {len(filtered_shards)} JSONL shards."
    )
    print(f"Statistics written to {stats_path}")


if __name__ == "__main__":
    run()
