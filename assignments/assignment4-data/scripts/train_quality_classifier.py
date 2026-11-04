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
_NUM_WIKI_URLS = 10000

random.seed(_SEED)


def get_wiki_urls():
    """
    Load a specified number of Wikipedia URLs from a compressed text file.
    """
    wiki_path = get_shared_assets_path() / "wiki" / "enwiki-20260501-extracted_urls.txt.gz"
    with gzip.open(wiki_path, mode="rt", encoding="utf-8") as file:
        iterator = iter(file)
        selected_urls = sample(iterator, _NUM_WIKI_URLS)

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
    with open(train_set_path, "w") as train_set_file, open(valid_set_path, "w") as valid_set_file:
        for text in cc_texts:
            text = label_text("cc", text)
            if generator.random() < 0.2:
                valid_set_file.write(text + "\n")
            else:
                train_set_file.write(text + "\n")
        for text in wiki_texts:
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
