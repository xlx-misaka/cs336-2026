import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.vllm_utils import VLLMServer

PROMPT_PATH = Path(__file__).parent / "prompts"
MODEL_PATH = Path(__file__).parent.parent / "models"
DATA_PATH = Path(__file__).parent.parent / "data"
OUTPUT_PATH = Path(__file__).parent.parent / "output"

QUESTION_ONLY_PROMPT = Path(PROMPT_PATH / "question_only.prompt").read_text(
    encoding="utf-8"
)

R1_ZERO_PROMPT = Path(PROMPT_PATH / "r1_zero.prompt").read_text(encoding="utf-8")

R1_ZERO_THREE_SHOT_GSM8K_PROMPT = Path(
    PROMPT_PATH / "r1_zero_three_shot_gsm8k.prompt"
).read_text(encoding="utf-8")

SAMPLING_PARAMS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 512,
    "n": 1,
    "seed": 0,
}


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Run a benchmark.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_PATH / "OLMo-2-0425-1B"),
        help="The model to use for the benchmark.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DATA_PATH / "gsm8k" / "test.jsonl"),
        help="The dataset to use for the benchmark.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH / "prompting_baselines.json",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1].")
    return args


def run_benchmark(
    serve: VLLMServer,
    prompt: str,
    sampling_params: dict[str, Any],
    dataset: list[dict[str, Any]],
    reward_fn: Callable[..., dict[str, float]],
):
    """Runs a benchmark on a given dataset."""
    results: list[dict[str, Any]] = []
    prompts = [prompt.format(question=item["question"]) for item in dataset]
    outputs = serve.generate_completions(prompts, sampling_params, batch_size=32)
    for item, output in zip(dataset, outputs, strict=True):
        question = item["question"]
        answer = item["answer"].rsplit("####", 1)[-1].strip()
        reward = reward_fn(output.text, answer)
        results.append(
            {
                "question": question,
                "answer": answer,
                "response": output.text,
                **reward,
            }
        )

    return results


def benchmark(args):
    serve = VLLMServer(
        args.model,
        gpu=args.gpu,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    with open(args.dataset, encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    if args.limit is not None:
        dataset = dataset[: args.limit]
    if not dataset:
        raise ValueError("The dataset is empty.")

    prompts = [
        ("question_only", QUESTION_ONLY_PROMPT, question_only_reward_fn),
        ("r1_zero", R1_ZERO_PROMPT, r1_zero_reward_fn),
        ("r1_zero_three_shot", R1_ZERO_THREE_SHOT_GSM8K_PROMPT, r1_zero_reward_fn),
    ]

    results = {}
    try:
        serve.start()
        for name, prompt, reward_fn in prompts:
            params = SAMPLING_PARAMS.copy()
            if name != "question_only":
                params["stop"] = ["</answer>"]
                params["include_stop_str_in_output"] = True
            print(f"Running {name} on {len(dataset)} examples...", flush=True)
            samples = run_benchmark(serve, prompt, params, dataset, reward_fn)
            accuracy = sum(item["answer_reward"] for item in samples) / len(samples)
            format_rate = sum(item["format_reward"] for item in samples) / len(samples)
            results[name] = {
                "accuracy": accuracy,
                "format_rate": format_rate,
                "samples": samples,
            }
            print(f"{name}: accuracy={accuracy:.2%}, format_rate={format_rate:.2%}")
    finally:
        serve.stop()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Results saved to {args.output}")
    return results


if __name__ == "__main__":
    benchmark(parse_args())
