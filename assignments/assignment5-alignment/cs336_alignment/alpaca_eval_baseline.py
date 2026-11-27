# Generation: 150.46s (5.35 examples/s) | Reached length limit: 92
# Predictions saved to /root/autodl-tmp/cs336-2026/assignments/assignment5-alignment/outputs/alpaca_eval_baseline.json

import argparse
import json
from pathlib import Path

from cs336_alignment.eval_utils import (
    generate_zero_shot,
    parse_zero_shot_args,
    save_results,
)

PROJECT_PATH = Path(__file__).resolve().parent.parent
PROMPT_PATH = Path(__file__).resolve().parent / "prompts_safety"


def parse_args() -> argparse.Namespace:
    return parse_zero_shot_args(
        "AlpacaEval",
        model=PROJECT_PATH / "models" / "Llama-3.1-8B",
        dataset=PROJECT_PATH / "data" / "alpaca_eval" / "alpaca_eval_gpt4_turbo.json",
        output=PROJECT_PATH / "outputs" / "alpaca_eval_baseline.json",
        max_tokens=1024,
    )


def load_alpaca_examples(dataset_path: Path, limit: int | None) -> list[dict[str, str]]:
    """Load instructions and dataset identifiers from the reference file."""
    with dataset_path.open(encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty JSON array in {dataset_path}")

    examples = []
    for index, row in enumerate(rows[:limit], start=1):
        if not isinstance(row, dict) or not all(
            isinstance(row.get(key), str) and row[key].strip()
            for key in ("instruction", "dataset")
        ):
            raise ValueError(f"Invalid AlpacaEval example {index} in {dataset_path}")
        examples.append({"instruction": row["instruction"], "dataset": row["dataset"]})
    return examples


def format_prompts(examples: list[dict[str, str]]) -> list[str]:
    task = (PROMPT_PATH / "alpaca_eval_zero_shot.prompt").read_text(encoding="utf-8")
    system = (PROMPT_PATH / "zero_shot_system_prompt.prompt").read_text(
        encoding="utf-8"
    )
    return [
        system.format(instruction=task.format(instruction=example["instruction"]))
        for example in examples
    ]


def benchmark(args: argparse.Namespace) -> None:
    examples = load_alpaca_examples(args.dataset, args.limit)
    prompts = format_prompts(examples)
    outputs, _, generation_seconds = generate_zero_shot(args, "AlpacaEval", prompts)

    model_name = Path(args.model).name
    generator = model_name
    if model_name.lower().removeprefix("meta-") == "llama-3.1-8b":
        generator = "llama-3.1-8b-base"

    predictions = []
    num_truncated = 0
    for example, output in zip(examples, outputs, strict=True):
        completion = output.outputs[0]
        predictions.append(
            {**example, "output": completion.text, "generator": generator}
        )
        num_truncated += completion.finish_reason == "length"

    save_results(args.output, predictions)
    examples_per_second = len(predictions) / generation_seconds
    print(
        f"Generation: {generation_seconds:.2f}s ({examples_per_second:.2f} examples/s) | "
        f"Reached length limit: {num_truncated:,}"
    )
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    benchmark(parse_args())
