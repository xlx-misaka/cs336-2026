import torch
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    prompts_ids = [
        torch.Tensor(
            tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompt_strs
        )
    ]
    outputs_ids = [
        torch.Tensor(
            tokenizer.encode(output, add_special_tokens=False) for output in output_strs
        )
    ]

    batch_size = len(prompt_strs)
    max_prompt_and_output_len = max(
        len(prompt_ids) + len(output_ids)
        for prompt_ids, output_ids in zip(prompts_ids, outputs_ids)
    )

    input_ids = torch.empty(batch_size, max_prompt_and_output_len - 1)
    labels = torch.empty(batch_size, max_prompt_and_output_len - 1)
    response_mask = torch.empty(batch_size, max_prompt_and_output_len - 1)

    for i, (prompt_ids, output_ids) in enumerate(zip(prompts_ids, outputs_ids)):
        prompt_output_len = len(prompt_ids) + len(output_ids)
        input_ids[i, :] = (
            prompt_ids
            + output_ids[0:-1]
            + torch.full(
                (max_prompt_and_output_len - prompt_output_len,),
                tokenizer.pad_token_type_id,
            )
        )
        labels[i, :] = (
            prompt_ids[1:]
            + output_ids
            + torch.full(
                (max_prompt_and_output_len - prompt_output_len,),
                tokenizer.pad_token_type_id,
            )
        )
        response_mask[i, :] = (
            torch.full((len(prompts_ids) - 1,), False)
            + torch.full((len(outputs_ids),), True)
            + torch.full((max_prompt_and_output_len - prompt_output_len,), False)
        )

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}
