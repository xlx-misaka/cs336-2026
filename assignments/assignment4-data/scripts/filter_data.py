import concurrent.futures
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastwarc import ArchiveIterator, WarcRecordType
from tqdm import tqdm

from cs336_data.common import get_shared_assets_path
from cs336_data.deduplication import exact_deduplication, minhash_deduplication
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


def process_single_wet(wet_index: int, wet_path, output_root: Path):
    """Process a single WET file."""
    output_dir = output_root / "filtered"
    output_dir.mkdir(parents=True, exist_ok=True)
    texts_iter = load_texts_from_wet(wet_path)
    output_pathes = []
    counts = Counter()
    for text_index, text in enumerate(texts_iter):
        text = filter_text(text, counts)
        if text:
            text_name = f"wet_{wet_index:04d}_text_{text_index:04d}.txt"
            with open(output_dir / text_name, "w") as f:
                f.write(text)
                output_pathes.append(output_dir / text_name)
    print(f"WET {wet_index}: kept {counts['kept_documents']}/{counts['input_documents']} documents.")
    return {
        "paths": output_pathes,
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


def _exact_stage_stats(input_files, output_directory):
    """Measure text changes separately from document removals."""
    modified = removed_lines = removed_characters = 0
    empty_names = set()
    for path in input_files:
        before = path.read_text(encoding="utf-8")
        after = (output_directory / path.name).read_text(encoding="utf-8")
        modified += int(before != after)
        before_lines = before.count("\n") + int(bool(before) and not before.endswith("\n"))
        after_lines = after.count("\n") + int(bool(after) and not after.endswith("\n"))
        removed_lines += before_lines - after_lines
        removed_characters += len(before) - len(after)
        if not after.strip():
            empty_names.add(path.name)
    return (
        _stage_stats(
            "exact_line_deduplication",
            len(input_files),
            len(input_files),
            modified_documents=modified,
            removed_lines=removed_lines,
            removed_characters=removed_characters,
            empty_output_documents=len(empty_names),
        ),
        empty_names,
    )


def run():
    # Generate one experiment directory in the parent and pass it to every worker.
    output_root = (
        get_shared_assets_path() / "processed_wet" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    )
    wet_pathes = get_wet_pathes()
    filtered_text_pathes = []
    counts = Counter()
    per_wet_file = []
    # Shut down filtering workers before the memory-intensive deduplication stage.
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
        futures = [
            executor.submit(process_single_wet, wet_index, wet_path, output_root)
            for wet_index, wet_path in enumerate(wet_pathes)
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            filtered_text_pathes.extend(result["paths"])
            counts.update(result["counts"])
            per_wet_file.append({key: result[key] for key in ("wet_index", "wet_path", "counts")})

    assert len(filtered_text_pathes) == counts["kept_documents"]
    stages = _filter_stage_stats(counts)
    output_dir1 = output_root / "exact_dedup"
    output_dir1.mkdir(parents=True, exist_ok=True)
    exact_deduplication(filtered_text_pathes, output_dir1)
    exact_stats, empty_names = _exact_stage_stats(filtered_text_pathes, output_dir1)
    stages.append(exact_stats)

    exact_deduped_text_pathes = [output_dir1 / Path(text_path).name for text_path in filtered_text_pathes]
    output_dir2 = output_root / "minhash_dedup"
    output_dir2.mkdir(parents=True, exist_ok=True)
    minhash_deduplication(exact_deduped_text_pathes, NUM_HASHES, NUM_BANDS, NGRAMS, JACCARD_THRESHOLD, output_dir2)  # type: ignore

    final_paths = [output_dir2 / path.name for path in exact_deduped_text_pathes if (output_dir2 / path.name).is_file()]
    final_empty = sum(path.name in empty_names for path in final_paths)
    stages.append(
        _stage_stats(
            "minhash_document_deduplication",
            len(exact_deduped_text_pathes),
            len(final_paths),
            empty_output_documents=final_empty,
        )
    )
    total_discarded = counts["input_documents"] - len(final_paths)
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
        "final_output_files": len(final_paths),
        "final_empty_documents": final_empty,
        "final_nonempty_documents": len(final_paths) - final_empty,
        "total_discarded_documents": total_discarded,
        "counting_notes": {
            "discard_attribution": "First rejecting stage in execution order; each document is counted once.",
            "pii": "Masking changes text without discarding documents; per-type document counts can overlap.",
            "empty_documents": "Whitespace-only and empty outputs are still retained by the existing pipeline.",
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
    print(f"Final output: {len(final_paths)} files, including {final_empty} empty documents.")
    print(f"Statistics written to {stats_path}")


if __name__ == "__main__":
    run()
