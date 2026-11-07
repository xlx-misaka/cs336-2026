import concurrent.futures
import json
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

# Path to the output directory.
OUTPUT_ROOT = (
    Path(get_shared_assets_path()) / "processed_wet" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
)

# pallelism parameters
NUM_CPUS = 2
executor = concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CPUS)

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


def filter_text(text: str):
    """Filter out unwanted characters and words."""
    # gopher quality filter
    if not gopher_quality_filter(text):
        return None

    # quality classifier
    label, socre = classify_quality(text)
    if label == "cc" and socre >= QUALITY_THRESHOLD:
        return None

    # harmful content classifier
    label, score = classify_nsfw(text)
    if label == "nsfw" and score >= NSFW_THRESHOLD:
        return None
    label, score = classify_toxic_speech(text)
    if label == "toxic" and score >= TOXIC_THRESHOLD:
        return None

    # mask pii
    text = mask_emails(text)[0]
    text = mask_phone_numbers(text)[0]
    text = mask_ips(text)[0]

    return text


def process_single_wet(wet_index: int, wet_path):
    """Process a single WET file."""
    output_dir = OUTPUT_ROOT / "filtered"
    output_dir.mkdir(parents=True, exist_ok=True)
    texts_iter = load_texts_from_wet(wet_path)
    output_pathes = []
    for text_index, text in enumerate(texts_iter):
        text = filter_text(text)
        if text:
            text_name = f"wet_{wet_index:04d}_text_{text_index:04d}.txt"
            with open(output_dir / text_name, "w") as f:
                f.write(text)
                output_pathes.append(output_dir / text_name)
    print(f"Processed {wet_index} WET files.")
    return output_pathes


def run():

    futures = []
    wet_pathes = get_wet_pathes()
    for wet_index, wet_path in enumerate(wet_pathes):
        future = executor.submit(
            process_single_wet,
            wet_index,
            wet_path,
        )
        futures.append(future)

    filtered_text_pathes = []

    for future in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(futures),
    ):
        filtered_text_pathes += future.result()

    output_dir1 = OUTPUT_ROOT / "exact_dedup"
    output_dir1.mkdir(parents=True, exist_ok=True)
    exact_deduplication(filtered_text_pathes, output_dir1)

    exact_deduped_text_pathes = [output_dir1 / Path(text_path).name for text_path in filtered_text_pathes]
    output_dir2 = OUTPUT_ROOT / "minhash_dedup"
    output_dir2.mkdir(parents=True, exist_ok=True)
    minhash_deduplication(exact_deduped_text_pathes, NUM_HASHES, NUM_BANDS, NGRAMS, JACCARD_THRESHOLD, output_dir2)  # pyright: ignore[reportArgumentType]


if __name__ == "__main__":
    run()
