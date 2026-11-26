import argparse
import json
import random
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from cs336_alignment.eval_utils import (
    evaluate_zero_shot,
    parse_gsm8k_response,
    parse_zero_shot_args,
    report_zero_shot_results,
)

PROJECT_PATH = Path(__file__).resolve().parent.parent
PROMPT_PATH = Path(__file__).resolve().parent / "prompts_safety"


def parse_args() -> argparse.Namespace:
    return parse_zero_shot_args(
        "GSM8K",
        model=PROJECT_PATH / "models" / "Llama-3.1-8B",
        dataset=PROJECT_PATH / "data" / "gsm8k" / "test.jsonl",
        output=PROJECT_PATH / "outputs" / "gsm8k_baseline.json",
    )


def load_gsm8k_examples(dataset_path: Path, limit: int | None) -> list[dict[str, str]]:
    """Load questions and reference answers from the course GSM8K JSONL."""
    examples = []
    with dataset_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            solution, marker, answer = row["answer"].rpartition("####")
            answer = answer.strip().replace(",", "")
            try:
                valid_answer = bool(marker) and Decimal(answer).is_finite()
            except InvalidOperation:
                valid_answer = False
            if not valid_answer:
                raise ValueError(
                    f"Invalid GSM8K answer at {dataset_path}:{line_number}"
                )
            examples.append(
                {
                    "question": row["question"],
                    "answer": answer,
                    "reference_solution": solution.strip(),
                }
            )
            if len(examples) == limit:
                break

    if not examples:
        raise ValueError(f"No GSM8K examples found in {dataset_path}")
    return examples


def format_prompts(examples: list[dict[str, str]]) -> list[str]:
    task = (PROMPT_PATH / "gsm8k_zero_shot.prompt").read_text(encoding="utf-8")
    system = (PROMPT_PATH / "zero_shot_system_prompt.prompt").read_text(
        encoding="utf-8"
    )
    return [
        system.format(instruction=task.format(question=example["question"]))
        for example in examples
    ]


def score_response(example: dict[str, Any], response: str) -> tuple[str | None, bool]:
    prediction = parse_gsm8k_response(response)
    if prediction is None:
        return None, False
    try:
        return prediction, Decimal(prediction) == Decimal(example["answer"])
    except InvalidOperation:
        return None, False


def benchmark(args: argparse.Namespace) -> None:
    examples = load_gsm8k_examples(args.dataset, args.limit)
    prompts = format_prompts(examples)
    results = evaluate_zero_shot(args, "GSM8K", examples, prompts, score_response)
    failures = [sample for sample in results["samples"] if sample["prediction"] is None]
    incorrect = [
        sample
        for sample in results["samples"]
        if sample["prediction"] is not None and not sample["correct"]
    ]
    results["parse_failure_examples"] = failures[:3]
    results["incorrect_examples"] = random.Random(0).sample(
        incorrect, min(10, len(incorrect))
    )
    report_zero_shot_results(args.output, results)


if __name__ == "__main__":
    benchmark(parse_args())
