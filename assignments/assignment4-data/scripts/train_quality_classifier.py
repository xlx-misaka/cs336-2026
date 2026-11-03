import gzip
import random

from more_itertools import sample

from cs336_data.common import get_shared_assets_path

_SEED = 336


random.seed(_SEED)


def load_wiki_urls(num_urls: int = 50) -> list[str]:
    """
    Load a specified number of Wikipedia URLs from a compressed text file.
    """
    wiki_path = get_shared_assets_path() / "wiki" / "enwiki-20260501-extracted_urls.txt.gz"

    with gzip.open(wiki_path, mode="rt", encoding="utf-8") as file:
        iterator = iter(file)
        selected_urls = sample(iterator, num_urls)

    return selected_urls
