import argparse
import json
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import TYPE_CHECKING, Any

import regex as re

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput


def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    response = re.split(
        r"^[ \t]*#\s*Query\s*:",
        model_output,
        maxsplit=1,
        flags=re.I | re.M,
    )[0]
    response = response.replace("**", "").replace("`", "").strip()

    answer_marker = r"answer(?:\s+is\s+|\s*:\s*)"
    qualified_prefix = r"\b(?:(correct|final|wrong|incorrect)\s+)?" + answer_marker
    option_pattern = (
        r"(?:option\s+)?\(?([A-D])"
        r"(?:\)|(?=[ \t]*(?:[.!?,:;\n]|$)|[ \t]+because\b))"
    )
    answer_matches = re.findall(
        qualified_prefix + option_pattern,
        response,
        flags=re.I,
    )

    final_letters = {
        letter.upper()
        for qualifier, letter in answer_matches
        if qualifier.casefold() == "final"
    }
    if final_letters:
        return next(iter(final_letters)) if len(final_letters) == 1 else None

    letters = {
        letter.upper()
        for qualifier, letter in answer_matches
        if qualifier.casefold() not in {"wrong", "incorrect"}
    }
    if letters:
        return next(iter(letters)) if len(letters) == 1 else None

    leading_option = re.match(r"^\(?([A-D])(?:[.)](?=\s|$)|$)", response, re.I)
    if leading_option:
        return leading_option.group(1).upper()

    answer_prefix = r"\b(?:(?:correct|final)\s+)?" + answer_marker
    answer_text = re.sub(r"^(?:the\s+)?" + answer_prefix, "", response, flags=re.I)
    normalized = " ".join(answer_text.casefold().rstrip(" .!?").split())

    matches = [
        letter
        for letter, option in zip("ABCD", mmlu_example["options"])
        if " ".join(option.casefold().rstrip(" .!?").split()) == normalized
    ]

    return matches[0] if len(matches) == 1 else None


def parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    number_pattern = (
        r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
        r"(?:[eE][+-]?\d+)?"
    )
    numbers = re.findall(number_pattern, model_output.replace("\u2212", "-"))
    if not numbers:
        return None

    return numbers[-1].replace(",", "").removeprefix("+")


def parse_zero_shot_args(
    task_name: str,
    *,
    model: Path,
    dataset: Path,
    output: Path,
    max_tokens: int = 512,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Evaluate zero-shot {task_name}.")
    parser.add_argument("--model", default=str(model))
    parser.add_argument("--dataset", type=Path, default=dataset)
    parser.add_argument("--output", type=Path, default=output)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N examples.")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=max_tokens)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    if not 0 < args.max_tokens < args.max_model_len:
        parser.error("--max-tokens must be positive and smaller than --max-model-len.")
    if args.max_num_seqs <= 0:
        parser.error("--max-num-seqs must be positive.")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1].")
    return args


def generate_zero_shot(
    args: argparse.Namespace, task_name: str, prompts: list[str]
) -> tuple[list["RequestOutput"], dict[str, Any], float]:
    """Generate one response per prompt and time the generation call."""
    from vllm import LLM, SamplingParams

    sampling = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "stop": ["# Query:"],
    }
    sampling_params = SamplingParams(**sampling)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    print(
        f"Evaluating {args.model} on {len(prompts):,} {task_name} examples...",
        flush=True,
    )
    start = perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    return outputs, sampling, perf_counter() - start


def evaluate_zero_shot(
    args: argparse.Namespace,
    task_name: str,
    examples: list[dict[str, Any]],
    prompts: list[str],
    score_response: Callable[[dict[str, Any], str], tuple[str | None, bool]],
) -> dict[str, Any]:
    """Collect per-example scores for a benchmark with reference answers."""
    outputs, sampling, generation_seconds = generate_zero_shot(args, task_name, prompts)
    samples = []
    for example, output in zip(examples, outputs, strict=True):
        completion = output.outputs[0]
        prediction, correct = score_response(example, completion.text)
        samples.append(
            {
                **example,
                "response": completion.text,
                "prediction": prediction,
                "correct": correct,
                "finish_reason": completion.finish_reason,
            }
        )

    num_correct = sum(sample["correct"] for sample in samples)
    return {
        "model": args.model,
        "dataset": str(args.dataset),
        "sampling": sampling,
        "num_examples": len(samples),
        "num_correct": num_correct,
        "num_parse_failures": sum(sample["prediction"] is None for sample in samples),
        "accuracy": num_correct / len(samples),
        "generation_seconds": generation_seconds,
        "examples_per_second": len(samples) / generation_seconds,
        "samples": samples,
    }


def save_results(path: Path, results: dict[str, Any] | list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary_path = Path(directory) / path.name
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(path)


def report_zero_shot_results(path: Path, results: dict[str, Any]) -> None:
    save_results(path, results)
    print(
        f"Accuracy: {results['accuracy']:.2%} "
        f"({results['num_correct']:,}/{results['num_examples']:,}) | "
        f"Parse failures: {results['num_parse_failures']:,} | "
        f"Generation: {results['generation_seconds']:.2f}s "
        f"({results['examples_per_second']:.2f} examples/s)"
    )
    print(f"Results saved to {path}")
