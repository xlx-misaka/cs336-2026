import gzip
from itertools import chain
import json
from os import PathLike
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenizes the prompts and outputs and returns a dictionary of tensors."""
    if not prompt_strs:
        raise ValueError("prompt_strs cannot be empty.")

    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length.")

    prompts_ids = [
        torch.Tensor(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in prompt_strs
    ]
    outputs_ids = [
        torch.Tensor(tokenizer.encode(output, add_special_tokens=False))
        for output in output_strs
    ]

    batch_size = len(prompt_strs)
    max_prompt_and_output_len = max(
        len(prompt_ids) + len(output_ids)
        for prompt_ids, output_ids in zip(prompts_ids, outputs_ids)
    )

    if max_prompt_and_output_len == 0:
        raise ValueError("prompt_strs and output_strs must contain non-empty strings.")

    ids = torch.empty(batch_size, max_prompt_and_output_len - 1)
    labels = torch.empty(batch_size, max_prompt_and_output_len - 1)
    response_mask = torch.empty(batch_size, max_prompt_and_output_len - 1)

    pad_token_id = tokenizer.pad_token_id
    assert isinstance(pad_token_id, int)

    for i, (prompt_ids, output_ids) in enumerate(zip(prompts_ids, outputs_ids)):
        prompt_output_len = len(prompt_ids) + len(output_ids)
        ids[i, :] = torch.cat(
            [
                prompt_ids,
                output_ids,
                torch.full(
                    (max_prompt_and_output_len - prompt_output_len,),
                    pad_token_id,
                ),
            ]
        )[:-1]
        labels[i, :] = torch.cat(
            [
                prompt_ids,
                output_ids,
                torch.full(
                    (max_prompt_and_output_len - prompt_output_len,),
                    pad_token_id,
                ),
            ]
        )[1:]
        response_mask[i, :] = torch.cat(
            [
                torch.full((len(prompt_ids),), False),
                torch.full((len(output_ids),), True),
                torch.full((max_prompt_and_output_len - prompt_output_len,), False),
            ]
        )[1:]

    return {
        "ids": ids.to(torch.int64),
        "labels": labels.to(torch.int64),
        "response_mask": response_mask.to(torch.bool),
    }


SFT_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts_safety" / "alpaca_sft.prompt"
)


class PackedSFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | PathLike[str],
        seq_length: int,
        shuffle: bool,
    ) -> None:
        open_fn = gzip.open if Path(dataset_path).suffix == ".gz" else open

        sft_prompt = SFT_PROMPT_PATH.read_text()

        ids = []

        with open_fn(dataset_path, "rt") as f:
            for line in f:
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                text = sft_prompt.format(
                    instruction=row["prompt"],
                    response=row["response"],
                )
                ids.append(tokenizer.encode(text) + [tokenizer.eos_token_id])

        if shuffle:
            random.shuffle(ids)

        ids = list(chain.from_iterable(ids))

        self.metadata: list[dict[str, Any]] = []
        for i in range(0, len(ids) - seq_length, seq_length):
            self.metadata.append(
                {
                    "input_ids": ids[i : i + seq_length],
                    "labels": ids[i + 1 : i + 1 + seq_length],
                }
            )

        super().__init__()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, i) -> dict[str, Any]:
        return self.metadata[i]
