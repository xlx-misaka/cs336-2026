import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from cs336_alignment.eval_utils import parse_mmlu_response

PROMPT_PATH = Path(__file__).parent / "prompts_safety"
MODEL_PATH = Path(__file__).parent.parent / "models"
DATA_PATH = Path(__file__).parent.parent / "data"
OUTPUT_PATH = Path(__file__).parent.parent / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot MMLU.")
    parser.add_argument("--model", default=str(MODEL_PATH / "Llama-3.1-8B"))
    parser.add_argument("--dataset", type=Path, default=DATA_PATH / "mmlu" / "test")
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH / "mmlu_baseline.json"
    )
    parser.add_argument("--limit", type=int, help="Evaluate only the first N examples.")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
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


def save_results(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary_path = Path(directory) / path.name
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(path)


def benchmark(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    examples = load_mmlu_examples(args.dataset, args.limit)
    prompts = format_prompts(examples)
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

    print(f"Evaluating {args.model} on {len(examples):,} MMLU examples...", flush=True)
    start = perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    generation_seconds = perf_counter() - start

    samples = []
    for example, output in zip(examples, outputs, strict=True):
        completion = output.outputs[0]
        prediction = parse_mmlu_response(example, completion.text)
        samples.append(
            {
                **example,
                "response": completion.text,
                "prediction": prediction,
                "correct": prediction == example["answer"],
                "finish_reason": completion.finish_reason,
            }
        )

    num_correct = sum(sample["correct"] for sample in samples)
    num_parse_failures = sum(sample["prediction"] is None for sample in samples)
    accuracy = num_correct / len(samples)
    examples_per_second = len(samples) / generation_seconds
    results = {
        "model": args.model,
        "dataset": str(args.dataset),
        "sampling": sampling,
        "num_examples": len(samples),
        "num_correct": num_correct,
        "num_parse_failures": num_parse_failures,
        "accuracy": accuracy,
        "generation_seconds": generation_seconds,
        "examples_per_second": examples_per_second,
        "samples": samples,
    }
    save_results(args.output, results)

    print(
        f"Accuracy: {accuracy:.2%} ({num_correct:,}/{len(samples):,}) | "
        f"Parse failures: {num_parse_failures:,} | "
        f"Generation: {generation_seconds:.2f}s ({examples_per_second:.2f} examples/s)"
    )
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    benchmark(parse_args())
