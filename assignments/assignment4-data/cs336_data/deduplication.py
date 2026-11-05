import os
from collections import Counter


def exact_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike):
    "Exact line deduplication algorithm."
    cnt = Counter()
    for file in input_files:
        with open(file) as f:
            for line in f:
                cnt[hash(line)] += 1

    for file in input_files:
        with open(file) as f:
            for line in f:
                if cnt[hash(line)] > 1:
                    line = ""
                with open(os.path.join(output_directory, os.path.basename(file)), "a") as out_f:
                    out_f.write(line)
