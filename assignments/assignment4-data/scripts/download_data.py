import bz2
import gzip
import http.client
import re
import shutil
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

from cs336_data.common import get_shared_assets_path
from cs336_data.modal_utils import VOLUME_MOUNTS, app, build_image
from cs336_data.wet_files import EnglishWetFiles


# Identify download requests for Wikimedia's User-Agent policy.
_HTTP_OPENER = urllib.request.build_opener()
_HTTP_OPENER.addheaders = [("User-Agent", "CS336DataStudy/0.1 (https://github.com/xlx-misaka/cs336-2026)")]
urllib.request.install_opener(_HTTP_OPENER)


def _download_with_retries(url: str, out: Path) -> None:
    """Publish a download only after it finishes; retry transient network failures."""
    partial = out.with_name(out.name + ".part")
    for attempt in range(1, 9):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as f:
                expected = response.headers.get("Content-Length")
                shutil.copyfileobj(response, f)
            if expected is not None and partial.stat().st_size != int(expected):
                raise urllib.error.ContentTooShortError(
                    f"retrieval incomplete: got {partial.stat().st_size} out of {expected} bytes", None
                )
            partial.replace(out)
            return
        except urllib.error.HTTPError as error:
            error.close()
            if error.code not in (408, 429, 500, 502, 503, 504):
                raise
            failure = error
        except (urllib.error.URLError, TimeoutError, ConnectionError, ssl.SSLError, http.client.IncompleteRead) as error:
            failure = error
        if attempt == 8:
            raise failure
        delay = min(2**attempt, 30)
        print(f"[wiki] download attempt {attempt}/8 failed: {failure}; retrying in {delay}s", flush=True)
        time.sleep(delay)


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, timeout=60 * 60 * 12, max_containers=128)
def extract_wiki_urls(shard: str) -> list[str]:
    cache = None
    if modal.is_local():
        cache = get_shared_assets_path() / "wiki" / ".url-shards" / f"{shard}.urls.gz"
        if cache.exists():
            print(f"[wiki] using cached URLs for {shard}", flush=True)
            with gzip.open(cache, "rt", encoding="utf-8") as f:
                return [line.rstrip("\n") for line in f]

    dump_date = "20260501"
    tmp_dir = Path("/tmp/wiki")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dump = tmp_dir / shard
    url_re = re.compile(
        r"\b(?:https?|telnet|gopher|file|wais|ftp):[\w/#~:.?+=&%@!\-.:?\\-]+?(?=[.:?\-]*(?:[^\w/#~:.?+=&%@!\-.:?\-]|$))"
    )

    print(f"[wiki] downloading {shard}", flush=True)
    _download_with_retries(f"https://dumps.wikimedia.org/enwiki/{dump_date}/{shard}", dump)
    urls = []
    with bz2.open(dump, "rt", errors="ignore") as f:
        for line in f:
            if refs := re.search("&lt;ref&gt(.*)&lt;/ref&gt;", line):
                urls.extend(url_re.findall(refs.group(0)))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache_partial = cache.with_name(cache.name + ".part")
        with gzip.open(cache_partial, "wt", encoding="utf-8") as f:
            for url in urls:
                f.write(url + "\n")
        cache_partial.replace(cache)
    dump.unlink(missing_ok=True)
    return urls


def download_offline_files(*, root_path: Path) -> None:
    paloma_out = root_path / "tokenized_paloma_c4_100_domains_validation.bin"
    if not paloma_out.exists():
        print(f"[huggingface] downloading {paloma_out.name}", flush=True)
        urllib.request.urlretrieve(
            "https://huggingface.co/datasets/brunborg/cs336-a4/resolve/main/tokenized_paloma_c4_100_domains_validation.bin",
            paloma_out,
        )

    cc = root_path / "CC"
    cc.mkdir(parents=True, exist_ok=True)
    for kind, out_name in [("warc", "example.warc.gz"), ("wet", "example.warc.wet.gz")]:
        out = cc / out_name
        if not out.exists():
            print(f"[cc] downloading {out_name}", flush=True)
            with urllib.request.urlopen(
                f"https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-12/{kind}.paths.gz"
            ) as r:
                first_path = gzip.decompress(r.read()).decode().splitlines()[0]
            urllib.request.urlretrieve(f"https://data.commoncrawl.org/{first_path}", out)

    for rel_path, url in [
        (
            "classifiers/lid.176.bin",
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
        ),
        (
            "classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin",
            "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech/resolve/main/model.bin",
        ),
        (
            "classifiers/dolma_fasttext_nsfw_jigsaw_model.bin",
            "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-nsfw/resolve/main/model.bin",
        ),
    ]:
        out = root_path / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            print(f"[file] downloading {rel_path}", flush=True)
            urllib.request.urlretrieve(url, out)


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, timeout=60 * 60 * 12)
def main(offline_only: bool = False):
    root_path = get_shared_assets_path()
    download_offline_files(root_path=root_path)
    if offline_only:
        return

    dump_date = "20260501"
    base_url = f"https://dumps.wikimedia.org/enwiki/{dump_date}/"
    wiki_out = root_path / "wiki/enwiki-20260501-extracted_urls.txt.gz"
    if not wiki_out.exists():
        wiki_out.parent.mkdir(parents=True, exist_ok=True)
        index = Path("/tmp") / f"enwiki-{dump_date}-index.html"
        _download_with_retries(base_url, index)
        html = index.read_text(encoding="utf-8")
        shards = sorted(
            set(re.findall(rf"enwiki-{dump_date}-pages-articles-multistream[0-9]+\.xml-p[0-9]+p[0-9]+\.bz2", html))
        )
        if not shards:
            raise ValueError(f"No Wikipedia shards found at {base_url}")
        tmp_out = Path("/tmp") / wiki_out.name
        tmp_out.unlink(missing_ok=True)
        print(f"[wiki] extracting {len(shards)} shards", flush=True)
        with gzip.open(tmp_out, "wt", encoding="utf-8") as f:
            for urls in (
                (extract_wiki_urls.local(shard) for shard in shards)
                if modal.is_local()
                else extract_wiki_urls.map(shards)
            ):
                for url in urls:
                    f.write(url + "\n")
        final_partial = wiki_out.with_name(wiki_out.name + ".part")
        shutil.copy2(tmp_out, final_partial)
        final_partial.replace(wiki_out)
        tmp_out.unlink(missing_ok=True)
        print(f"[wiki] wrote {wiki_out}", flush=True)

    english_wet_files = EnglishWetFiles()
    wet_file_paths = english_wet_files.load_or_create()
    print(f"downloaded {len(wet_file_paths)} including {wet_file_paths[0]=}")


@app.local_entrypoint()
def modal_main(offline_only: bool = False):
    main.remote(offline_only=offline_only)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Only download files needed for running the assignment offline; skip full WET/wiki data creation.",
    )
    args = parser.parse_args()
    main.local(offline_only=args.offline_only)
