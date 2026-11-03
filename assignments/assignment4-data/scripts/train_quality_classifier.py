import gzip
import random
import subprocess
from pathlib import Path

from more_itertools import sample

from cs336_data.common import get_shared_assets_path

_SEED = 336
_ROOT_PATH = Path("data/quality_classifier_data")

random.seed(_SEED)


def sample_wiki_urls(num_urls: int = 50):
    """
    Load a specified number of Wikipedia URLs from a compressed text file.
    """
    wiki_path = get_shared_assets_path() / "wiki" / "enwiki-20260501-extracted_urls.txt.gz"

    with gzip.open(wiki_path, mode="rt", encoding="utf-8") as file:
        iterator = iter(file)
        selected_urls = sample(iterator, num_urls)

    saved_path = _ROOT_PATH / "sampled_wiki_urls.txt"
    return saved_path.write_text("\n".join([url.strip() for url in selected_urls if url.strip() != ""]))


def wget_urls(urls: list[str]): ...


sample_wiki_urls()
