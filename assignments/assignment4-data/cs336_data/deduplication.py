import os
import unicodedata
from collections import Counter

import mmh3


def exact_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike):
    "Exact line deduplication algorithm."
    cnt = Counter()
    for file in input_files:
        with open(file) as f:
            for line in f:
                cnt[hash(line)] += 1

    for file in input_files:
        with open(file) as f, open(os.path.join(output_directory, os.path.basename(file)), "w") as out_f:
            for line in f:
                if cnt[hash(line)] > 1:
                    line = ""
                out_f.write(line)


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    """
    Minhash deduplication algorithm.
    """
    if num_hashes % num_bands != 0:
        raise ValueError("num_hashes must be divisible by num_bands")
    r = num_hashes // num_bands
    buckets = [{} for _ in range(num_bands)]

    for file_index, input_path in enumerate(input_files):
        with open(input_path) as f:
            text = _norm_text(f.read())
            ngram_set = _get_n_gram_set(text, ngrams)
            if ngram_set is None:
                continue
            minhashes = _minhash(ngram_set, num_hashes)
            for band_index in range(num_bands):
                key = tuple(minhashes[band_index * r : (band_index + 1) * r])
                if key not in buckets[band_index]:
                    buckets[band_index][key] = []
                buckets[band_index][key].append(file_index)

    uf = _UnionFind(len(input_files))
    for bucket in buckets:
        for file_indices in bucket.values():
            if len(file_indices) < 2:
                continue
            n_gram_sets = []
            for file_index in file_indices:
                with open(input_files[file_index]) as f:
                    n_gram_sets.append(_get_n_gram_set(_norm_text(f.read()), ngrams))
            for i in range(len(n_gram_sets) - 1):
                for j in range(i + 1, len(n_gram_sets)):
                    if _jaccard(n_gram_sets[i], n_gram_sets[j]) >= jaccard_threshold:
                        uf.merge(file_indices[i], file_indices[j])

    for file_index, parent_index in enumerate(uf.parent):
        if file_index == parent_index:
            with (
                open(os.path.join(output_directory, os.path.basename(input_files[file_index])), "w") as out_f,
                open(input_files[file_index]) as in_f,
            ):
                out_f.write(in_f.read())


def _norm_text(text: str) -> str:
    """
    Normalizes text  by lowercasing, removing punctuation, normalizing whitespaces,
    and removing accents, and applying NFD unicode normalization.
    """
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = "".join(char if not unicodedata.category(char).startswith("P") else " " for char in text)
    return " ".join(text.split())


def _get_n_gram_set(text: str, n: int) -> set[str] | None:
    """
    Generates n-grams from the given text.
    """
    words = text.split(" ")
    if len(words) < n or not text:
        return None
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _minhash(ngram_set: set[str], num_hashes: int) -> list[int]:
    """
    Generates minhash values for the given n-gram set.
    """
    if not ngram_set:
        raise ValueError("N-gram set must not be empty")
    ngram_list = list(ngram_set)
    minhashes = [0] * num_hashes
    seeds = list(range(num_hashes))
    for i, seed in enumerate(seeds):
        minhashes[i] = mmh3.hash128(ngram_list[0], seed=seed)
        for j in range(1, len(ngram_set)):
            minhashes[i] = min(minhashes[i], mmh3.hash128(ngram_list[j], seed=seed))
    return minhashes


def _jaccard(ngram_set1: set[str], ngram_set2: set[str]) -> float:
    """
    Calculates the Jaccard similarity between two n-gram sets.
    """
    if not ngram_set1 or not ngram_set2:
        raise ValueError("N-gram sets must not be empty")
    intersection = len(ngram_set1 & ngram_set2)
    union = len(ngram_set1 | ngram_set2)
    return intersection / union


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        acopy = a
        while a != self.parent[a]:
            a = self.parent[a]
        while acopy != a:
            self.parent[acopy], acopy = a, self.parent[acopy]
        return a

    def merge(self, a, b):
        self.parent[self.find(b)] = self.find(a)
