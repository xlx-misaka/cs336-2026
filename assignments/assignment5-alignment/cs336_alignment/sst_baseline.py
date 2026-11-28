# Generation: 30.52s (3.28 examples/s) | Reached length limit: 5
# Predictions saved to /root/autodl-tmp/cs336-2026/assignments/assignment5-alignment/outputs/sst_baseline.jsonl

import argparse
import csv
import json
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

from cs336_alignment.eval_utils import generate_zero_shot, parse_zero_shot_args

PROJECT_PATH = Path(__file__).resolve().parent.parent
PROMPT_PATH = Path(__file__).resolve().parent / "prompts_safety"


def parse_args() -> argparse.Namespace:
    return parse_zero_shot_args(
        "SimpleSafetyTests",
        model=PROJECT_PATH / "models" / "Llama-3.1-8B",
        dataset=PROJECT_PATH
        / "data"
        / "simple_safety_tests"
        / "simple_safety_tests.csv",
        output=PROJECT_PATH / "outputs" / "sst_baseline.jsonl",
    )


def load_sst_instructions(dataset_path: Path, limit: int | None) -> list[str]:
    """Load the original instructions in dataset order."""
    instructions = []
    with dataset_path.open(encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        if "prompts_final" not in (rows.fieldnames or ()):
            raise ValueError(f"Missing prompts_final column in {dataset_path}")

        for row in islice(rows, limit):
            instruction = row["prompts_final"]
            if not instruction or not instruction.strip():
                raise ValueError(
                    f"Empty prompts_final at {dataset_path}:{rows.line_num}"
                )
            instructions.append(instruction)

    if not instructions:
        raise ValueError(f"No SimpleSafetyTests examples found in {dataset_path}")
    return instructions


def format_prompts(instructions: list[str]) -> list[str]:
    task = (PROMPT_PATH / "simple_safety_tests_zero_shot.prompt").read_text(
        encoding="utf-8"
    )
    system = (PROMPT_PATH / "zero_shot_system_prompt.prompt").read_text(
        encoding="utf-8"
    )
    return [
        system.format(instruction=task.format(instruction=instruction))
        for instruction in instructions
    ]


def benchmark(args: argparse.Namespace) -> None:
    instructions = load_sst_instructions(args.dataset, args.limit)
    prompts = format_prompts(instructions)
    outputs, _, generation_seconds = generate_zero_shot(
        args, "SimpleSafetyTests", prompts
    )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_truncated = 0
    with TemporaryDirectory(dir=output_path.parent) as directory:
        temporary_path = Path(directory) / output_path.name
        with temporary_path.open("w", encoding="utf-8") as file:
            for instruction, output in zip(instructions, outputs, strict=True):
                completion = output.outputs[0]
                prediction = {"prompts_final": instruction, "output": completion.text}
                file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                num_truncated += completion.finish_reason == "length"
        temporary_path.replace(output_path)

    examples_per_second = len(instructions) / generation_seconds
    print(
        f"Generation: {generation_seconds:.2f}s ({examples_per_second:.2f} examples/s) | "
        f"Reached length limit: {num_truncated:,}"
    )
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    benchmark(parse_args())
