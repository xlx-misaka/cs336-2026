import gzip
import random
import subprocess
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType
from more_itertools import sample

from cs336_data.common import get_shared_assets_path
from cs336_data.extract_text import extract_text_from_html_bytes

_SEED = 336
_ROOT_PATH = Path("data/quality_classifier_data")
_NUM_WIKI_URLS = 5

random.seed(_SEED)


def _get_wiki_urls():
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


def _wget_wiki_urls():
    """
    Download a list of URLs using wget.
    """
    wiki_urls_path = _ROOT_PATH / "sampled_wiki_urls.txt"
    if not wiki_urls_path.exists():
        _get_wiki_urls()

    wiki_warc_path = _ROOT_PATH / "subsampled_wiki"
    command = [
        "wget",
        "--timeout=5",
        "-i",
        wiki_urls_path,
        f"--warc-file={wiki_warc_path}",
        "-O",
        "/dev/null",
    ]

    result = subprocess.run(command, check=False)
    return result.returncode


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


def run():
    """
    Run the script.
    """
    wiki_warc_path = _ROOT_PATH / "subsampled_wiki.warc.gz"
    if not wiki_warc_path.exists():
        _wget_wiki_urls()


if __name__ == "__main__":
    run()
