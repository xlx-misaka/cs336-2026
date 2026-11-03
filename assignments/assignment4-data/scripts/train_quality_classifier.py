import gzip
import random
import subprocess
from pathlib import Path

import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
from more_itertools import sample

from cs336_data.common import get_shared_assets_path
from cs336_data.extract_text import extract_text_from_html_bytes

_ROOT_PATH = Path("data/quality_classifier_data")
_ROOT_PATH.mkdir(parents=True, exist_ok=True)

_SEED = 336
_NUM_WIKI_URLS = 5

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
    wiki_warc_path = _ROOT_PATH / "sampled_wiki.warc.gz"
    if not wiki_warc_path.exists():
        wget_wiki_urls()
    return wiki_warc_path


def get_cc_warc():
    cc_warc_path = get_shared_assets_path() / "CC" / "example.warc.gz"
    return cc_warc_path


def get_htlm_texts_form_warc(warc_path: Path | str):
    """
    Get the HTML bytes from a WARC file.
    """
    warc_path = Path(warc_path)
    with open(warc_path, "rb") as f:
        for record in ArchiveIterator(f):
            if record.record_type == WarcRecordType.response and record.http_content_type == "text/html":
                html_bytes = record.reader.read()
                text = extract_text_from_html_bytes(html_bytes)
                yield text


def label_text(source: str, text: str):
    """
    Label the text based on the source.
    """
    label = "__label__negative" if source == "cc" else "__label__positive"
    return f"{label} {text.replace('\n', ' ')}"


def get_train_set():

    train_set_path = _ROOT_PATH / "train_set.txt"
    if train_set_path.exists():
        return train_set_path

    cc_warc_path = get_cc_warc()
    wiki_warc_path = get_wiki_warc()

    cc_texts = get_htlm_texts_form_warc(cc_warc_path)
    wiki_texts = get_htlm_texts_form_warc(wiki_warc_path)

    with open(train_set_path, "w") as f:
        for text in cc_texts:
            if text is None:
                continue
            f.write(label_text("cc", text) + "\n")
        for text in wiki_texts:
            if text is None:
                continue
            f.write(label_text("wiki", text) + "\n")


def run():
    """
    Run the script.
    """
    train_set_path = get_train_set()
    model = fasttext.train_supervised(input=train_set_path)
    model.save_model(_ROOT_PATH / "quality_classifier.bin")
    print("Model saved to quality_classifier.bin")


if __name__ == "__main__":
    run()
