import torch


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:

    logits = torch.nn.functional.log_softmax(model(input_ids).logits, dim=-1)
    log_probs = logits.gather(dim=-1, index=labels[:, :, None])[:, :, 0]

    token_entropy = torch.empty_like(log_probs)
    if return_token_entropy:
        token_entropy = -torch.sum(logits * logits.exp(), dim=-1)

    return {
        "log_probs": log_probs,
        "token_entropy": token_entropy,
    }
