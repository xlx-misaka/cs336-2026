import argparse
import csv
from pathlib import Path
from typing import Any

from cs336_alignment.eval_utils import (
    evaluate_zero_shot,
    parse_mmlu_response,
    parse_zero_shot_args,
    report_zero_shot_results,
)

PROMPT_PATH = Path(__file__).parent / "prompts_safety"
MODEL_PATH = Path(__file__).parent.parent / "models"
DATA_PATH = Path(__file__).parent.parent / "data"
OUTPUT_PATH = Path(__file__).parent.parent / "outputs"


def parse_args() -> argparse.Namespace:
    return parse_zero_shot_args(
        "MMLU",
        model=MODEL_PATH / "Llama-3.1-8B",
        dataset=DATA_PATH / "mmlu" / "test",
        output=OUTPUT_PATH / "mmlu_baseline.json",
    )


def load_mmlu_examples(dataset_path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Read the course's headerless MMLU test CSVs in a stable order."""
    examples = []
    files = sorted(dataset_path.glob("*_test.csv"))
    if not files:
        raise FileNotFoundError(f"No MMLU test CSVs found in {dataset_path}")

    for path in files:
        subject = path.name.removesuffix("_test.csv")
        with path.open(encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                if not row:
                    continue
                if len(row) != 6 or row[-1].strip() not in {"A", "B", "C", "D"}:
                    raise ValueError(f"Invalid MMLU row at {path}:{reader.line_num}")
                question, *options, answer = row
                examples.append(
                    {
                        "subject": subject,
                        "question": question,
                        "options": options,
                        "answer": answer.strip(),
                    }
                )
                if len(examples) == limit:
                    return examples

    if not examples:
        raise ValueError(f"No MMLU examples found in {dataset_path}")
    return examples


def format_prompts(examples: list[dict[str, Any]]) -> list[str]:
    task = (PROMPT_PATH / "mmlu_zero_shot.prompt").read_text(encoding="utf-8")
    system = (PROMPT_PATH / "zero_shot_system_prompt.prompt").read_text(
        encoding="utf-8"
    )
    prompts = []
    for example in examples:
        instruction = task.format(
            subject=example["subject"],
            question=example["question"],
            options=example["options"],
        )
        prompts.append(system.format(instruction=instruction))
    return prompts


def score_response(example: dict[str, Any], response: str) -> tuple[str | None, bool]:
    prediction = parse_mmlu_response(example, response)
    return prediction, prediction == example["answer"]


def benchmark(args: argparse.Namespace) -> None:
    examples = load_mmlu_examples(args.dataset, args.limit)
    prompts = format_prompts(examples)
    results = evaluate_zero_shot(args, "MMLU", examples, prompts, score_response)
    report_zero_shot_results(args.output, results)


if __name__ == "__main__":
    benchmark(parse_args())
