import concurrent.futures
import json
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

# Path to the manifest file.
MANIFEST_PATH = Path(
    "local-shared-data/furu/cs336_data/wet_files/EnglishWetFiles/"
    "ae4fa708c26a65eb662a/c22742cf4085f62d0f56/result/manifest.json"
)

# Parallelism parameters.
NUM_CPUS = 2
FILTER_STAGES = ("gopher", "quality", "nsfw", "toxic")
PII_TYPES = ("emails", "phone_numbers", "ips")

# filter parameters
QUALITY_THRESHOLD = 0.8
NSFW_THRESHOLD = 0.8
TOXIC_THRESHOLD = 0.8

# minhash parameters
NUM_HASHES = 100
NUM_BANDS = 10
NGRAMS = 5
JACCARD_THRESHOLD = 0.8
TEXT_FORMAT = "one_document_per_line"


def get_wet_pathes():
    """Load the data pathes from the manifest file."""
    entries = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
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


def filter_text(text: str, counts: Counter[str] | None = None):
    """Filter out unwanted characters and words."""
    if counts is None:
        counts = Counter()
    counts["input_documents"] += 1

    # gopher quality filter
    if not gopher_quality_filter(text):
        counts["gopher_discarded"] += 1
        return None

    # quality classifier
    label, socre = classify_quality(text)
    if label == "cc" and socre >= QUALITY_THRESHOLD:
        counts["quality_discarded"] += 1
        return None

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


def process_single_wet(wet_index: int, wet_path, output_root: Path, raw_directory: Path):
    """Write TXT shards and temporarily retain original document line breaks."""
    output_dir = output_root / "filtered"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"wet_{wet_index:04d}.txt"
    raw_path = raw_directory / f"wet_{wet_index:04d}.jsonl"
    counts = Counter()
    with (
        output_path.open("w", encoding="utf-8", newline="\n") as output_file,
        raw_path.open("w", encoding="utf-8") as raw_file,
    ):
        for text in load_texts_from_wet(wet_path):
            text = filter_text(text, counts)
            if text is not None:
                raw_file.write(json.dumps(text, ensure_ascii=False) + "\n")
                output_file.write(" ".join(text.splitlines()).strip() + "\n")
    print(f"WET {wet_index}: kept {counts['kept_documents']}/{counts['input_documents']} documents.")
    return {
        "path": output_path,
        "raw_path": raw_path,
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
        stages.append(_stage_stats(name, entered, kept))
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


def _read_raw_documents(path):
    with path.open(encoding="utf-8") as source:
        for record in source:
            yield json.loads(record)


def _exact_line_deduplication(input_files, output_directory):
    """Keep only corpus-unique original lines, then serialize one document per TXT line."""
    output_directory.mkdir(parents=True, exist_ok=True)
    line_counts = Counter()
    for path in input_files:
        for text in _read_raw_documents(path):
            # Match text-mode file iteration, including universal newline handling.
            line_counts.update(hash(line) for line in StringIO(text, newline=None))

    documents = modified = removed_lines = removed_characters = empty = 0
    for path in input_files:
        with (output_directory / path.with_suffix(".txt").name).open(
            "w", encoding="utf-8", newline="\n"
        ) as destination:
            for text in _read_raw_documents(path):
                lines = list(StringIO(text, newline=None))
                unique_lines = [line for line in lines if line_counts[hash(line)] == 1]
                result = "".join(unique_lines)
                documents += 1
                modified += int(len(lines) != len(unique_lines))
                removed_lines += len(lines) - len(unique_lines)
                removed_characters += sum(map(len, lines)) - len(result)
                empty += int(not result.strip())
                destination.write(" ".join(result.splitlines()).strip() + "\n")
    return _stage_stats(
        "exact_line_deduplication",
        documents,
        documents,
        modified_documents=modified,
        removed_lines=removed_lines,
        removed_characters=removed_characters,
        empty_output_documents=empty,
    )


def _minhash_document_deduplication(input_files, output_directory):
    """Reuse document MinHash/Jaccard operations with byte offsets into TXT shards."""
    if NUM_HASHES <= 0 or NUM_BANDS <= 0 or NUM_HASHES % NUM_BANDS:
        raise ValueError("NUM_HASHES must be positive and divisible by positive NUM_BANDS")
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
            return _get_n_gram_set(_norm_text(read_document(index).decode("utf-8")), NGRAMS)

        for index in tqdm(range(len(documents)), desc="MinHash documents", unit="doc"):
            ngrams = read_ngrams(index)
            if ngrams is None:
                continue
            signature = _minhash(ngrams, NUM_HASHES)
            candidates = set()
            for band_index, band in enumerate(buckets):
                start = band_index * rows_per_band
                bucket = band.setdefault(tuple(signature[start : start + rows_per_band]), [])
                candidates.update(bucket)
                bucket.append(index)
            # A pair is checked at most once, even if it matched multiple bands.
            for candidate in sorted(candidates):
                left, right = groups.find(candidate), groups.find(index)
                if left == right:
                    continue
                other_ngrams = read_ngrams(candidate)
                if other_ngrams is not None and _jaccard(ngrams, other_ngrams) >= JACCARD_THRESHOLD:
                    groups.merge(min(left, right), max(left, right))

        kept = 0
        for index, (shard_index, _, _) in enumerate(documents):
            if groups.find(index) == index:
                destinations[shard_index].write(read_document(index))
                kept += 1
    return _stage_stats("minhash_document_deduplication", len(documents), kept)


def run():
    # Generate one experiment directory in the parent and pass it to every worker.
    output_root = (
        get_shared_assets_path() / "processed_wet" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    )
    output_root.mkdir(parents=True, exist_ok=True)
    wet_pathes = get_wet_pathes()
    filtered_shards = []
    raw_shards = []
    counts = Counter()
    per_wet_file = []
    output_dir1 = output_root / "exact_dedup"
    # One temporary file per WET preserves paragraph boundaries for both line-counting passes.
    with TemporaryDirectory(prefix=".raw_documents_", dir=output_root) as temporary:
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
            futures = [
                executor.submit(process_single_wet, wet_index, wet_path, output_root, Path(temporary))
                for wet_index, wet_path in enumerate(wet_pathes)
            ]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                result = future.result()
                filtered_shards.append(result["path"])
                raw_shards.append(result["raw_path"])
                counts.update(result["counts"])
                per_wet_file.append({key: result[key] for key in ("wet_index", "wet_path", "counts")})

        # Shut down workers before building the global line counter.
        exact_stats = _exact_line_deduplication(sorted(raw_shards), output_dir1)
    filtered_shards.sort()
    stages = _filter_stage_stats(counts)
    assert exact_stats["input_documents"] == counts["kept_documents"]
    stages.append(exact_stats)

    exact_shards = [output_dir1 / path.name for path in filtered_shards]
    output_dir2 = output_root / "minhash_dedup"
    minhash_stats = _minhash_document_deduplication(exact_shards, output_dir2)
    assert minhash_stats["input_documents"] == exact_stats["kept_documents"]
    stages.append(minhash_stats)
    final_documents = minhash_stats["kept_documents"]
    # Empty records have no n-grams and are retained by MinHash; tokenization skips them.
    final_empty = exact_stats["empty_output_documents"]
    total_discarded = counts["input_documents"] - final_documents
    assert sum(stage["discarded_documents"] for stage in stages) == total_discarded
    for stage in stages:
        stage["discard_contribution_percent"] = (
            100 * stage["discarded_documents"] / total_discarded if total_discarded else None
        )

    report = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "manifest_path": str(MANIFEST_PATH.resolve()),
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
            "storage": "Each WET produces one TXT shard per stage; each line is one document.",
            "line_breaks": "Original line breaks are kept temporarily for line deduplication; TXT outputs use spaces.",
            "exact_deduplication": "Keep only lines occurring once in the filtered corpus; remove all repeated copies.",
            "empty_documents": "Emptied documents remain as blank records; tokenization skips them.",
            "zero_discards": "Contribution percentages are null when no documents were discarded.",
        },
        "parameters": {
            "workers": NUM_CPUS,
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
        f"Final output: {final_documents} document records ({final_empty} empty) in {len(filtered_shards)} TXT shards."
    )
    print(f"Statistics written to {stats_path}")


if __name__ == "__main__":
    run()
