import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: bytes | list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    Boundaries are placed at the start of any of the split special tokens.
    May return fewer chunks if the boundaries end up overlapping.
    """
    if isinstance(split_special_tokens, bytes):
        split_special_tokens = [split_special_tokens]
    tokens = list(split_special_tokens)
    assert tokens, "Must provide at least one split special token"
    assert all(isinstance(token, bytes) for token in tokens), "Must represent special tokens as bytestrings"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the earliest split special token in the mini chunk
            found_at = -1
            for token in tokens:
                match = mini_chunk.find(token)
                if match != -1 and (found_at == -1 or match < found_at):
                    found_at = match
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage
with open(..., "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(
        f,
        num_processes,
        [b"<|endoftext|>", b"<|startoftext|>"],
    )

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
