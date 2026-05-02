"""Compute Path-B ablation statistics from a base-vs-adapter comparison artifact.

This script is intentionally offline: it does not regenerate model outputs. It
reads ``outputs/evaluation_comparison.json`` from ``training/evaluate_trained_model.py``
and writes a structured cost/quality report with paired statistics.

Run:
    python ablations/run_ablation_analysis.py
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_COMPARISON = ROOT / "outputs" / "evaluation_comparison.json"
DEFAULT_OUTPUT = ROOT / "ablations" / "ablation_report.json"
DEFAULT_BOOTSTRAP_SAMPLES = 10_000


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-price-per-1m", type=float, default=0.0)
    parser.add_argument("--output-price-per-1m", type=float, default=0.0)
    return parser.parse_args()


def load_comparison(path: Path) -> Dict[str, Any]:
    """Load the comparison JSON emitted by the evaluator."""
    if not path.exists():
        raise FileNotFoundError(f"Comparison artifact not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "examples" not in data or not isinstance(data["examples"], list):
        raise ValueError(f"{path} does not look like an evaluation comparison artifact")
    return data


def score_deltas(examples: Sequence[Mapping[str, Any]]) -> List[float]:
    """Return paired fine-tuned minus base overall-score deltas."""
    return [
        float(example["fine_tuned_score"]["overall"]) - float(example["base_score"]["overall"])
        for example in examples
    ]


def bootstrap_ci(values: Sequence[float], samples: int, seed: int) -> Dict[str, float]:
    """Compute a paired bootstrap mean and 95% confidence interval."""
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.fmean(draw))
    means.sort()
    low_index = int(0.025 * samples)
    high_index = int(0.975 * samples)
    return {
        "mean": round(statistics.fmean(values), 6),
        "ci_low": round(means[low_index], 6),
        "ci_high": round(means[min(high_index, samples - 1)], 6),
    }


def paired_sign_flip_p_value(values: Sequence[float], samples: int, seed: int) -> float:
    """Approximate a two-sided paired permutation p value by sign flipping deltas."""
    if not values:
        return 1.0
    observed = abs(statistics.fmean(values))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        permuted = [value if rng.random() < 0.5 else -value for value in values]
        if abs(statistics.fmean(permuted)) >= observed:
            extreme += 1
    return round(extreme / samples, 6)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate for cost logging when tokenizer counts were not recorded."""
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def task_lookup(task_file: Path) -> Dict[str, Mapping[str, Any]]:
    """Load tasks by ID when the comparison artifact points at an available task file."""
    path = task_file if task_file.is_absolute() else ROOT / task_file
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if text[0] in "[{":
        data = json.loads(text)
        if isinstance(data, Mapping):
            data = data.get("tasks", [])
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    return {str(task.get("task_id", "")): task for task in data if isinstance(task, Mapping)}


def prompt_proxy(task: Mapping[str, Any]) -> str:
    """Use the task input as a stable prompt proxy for cost estimation."""
    return json.dumps(task.get("input", {}), sort_keys=True)


def per_task_records(
    data: Mapping[str, Any],
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> List[Dict[str, Any]]:
    """Build per-task quality, latency, token, and cost records."""
    tasks = task_lookup(Path(str(data.get("task_file", ""))))
    rows = []
    for example in data["examples"]:
        task_id = str(example.get("task_id", ""))
        task_found = task_id in tasks
        prompt = prompt_proxy(tasks[task_id]) if task_found else ""
        prompt_tokens = estimate_tokens(prompt)
        base_output_tokens = estimate_tokens(str(example.get("base_output", "")))
        fine_tuned_output_tokens = estimate_tokens(str(example.get("fine_tuned_output", "")))
        base_cost = (prompt_tokens * input_price_per_1m + base_output_tokens * output_price_per_1m) / 1_000_000
        fine_tuned_cost = (
            prompt_tokens * input_price_per_1m + fine_tuned_output_tokens * output_price_per_1m
        ) / 1_000_000
        rows.append(
            {
                "task_id": task_id,
                "base_overall": float(example["base_score"]["overall"]),
                "fine_tuned_overall": float(example["fine_tuned_score"]["overall"]),
                "delta_a": float(example["fine_tuned_score"]["overall"])
                - float(example["base_score"]["overall"]),
                "prompt_only_baseline_score": float(example["base_score"]["overall"]),
                "delta_b_baseline": 0.0,
                "latency_seconds": {
                    "base": example.get("base_latency_seconds"),
                    "fine_tuned": example.get("fine_tuned_latency_seconds"),
                    "note": "not recorded in source artifact" if "base_latency_seconds" not in example else "recorded",
                },
                "token_estimates": {
                    "task_prompt_found": task_found,
                    "prompt": prompt_tokens,
                    "base_output": base_output_tokens,
                    "fine_tuned_output": fine_tuned_output_tokens,
                },
                "cost_estimates_usd": {
                    "base": round(base_cost, 8),
                    "fine_tuned": round(fine_tuned_cost, 8),
                    "delta": round(fine_tuned_cost - base_cost, 8),
                    "price_anchor": {
                        "input_price_per_1m_tokens": input_price_per_1m,
                        "output_price_per_1m_tokens": output_price_per_1m,
                    },
                },
            }
        )
    return rows


def main() -> None:
    """Write the ablation report."""
    args = parse_args()
    data = load_comparison(args.comparison)
    examples = data["examples"]
    deltas = score_deltas(examples)
    stats = bootstrap_ci(deltas, args.bootstrap_samples, args.seed)
    p_value = paired_sign_flip_p_value(deltas, args.bootstrap_samples, args.seed + 1)
    records = per_task_records(data, args.input_price_per_1m, args.output_price_per_1m)

    payload = {
        "comparison_artifact": str(args.comparison),
        "task_file": data.get("task_file"),
        "dataset_size": len(examples),
        "backbone": data.get("base_model"),
        "delta_a": {
            "definition": "fine_tuned_adapter - prompt_only_same_backbone_baseline",
            "point_estimate": round(statistics.fmean(deltas), 6) if deltas else 0.0,
            "paired_bootstrap_95ci": [stats["ci_low"], stats["ci_high"]],
            "paired_sign_flip_p_value": p_value,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "delta_b_prompt_only_baseline": {
            "definition": "prompt-only baseline on the same backbone and same prompt/intervention shape",
            "point_estimate": data.get("base_summary", {}).get("mean_overall"),
            "note": "This is the baseline used for Delta A, not a separate backbone.",
        },
        "cost_pareto": {
            "latency_note": "latency fields are emitted per task; current source artifact did not record timings",
            "token_cost_note": "token counts are character/4 estimates unless the evaluator records tokenizer counts",
            "input_price_per_1m_tokens": args.input_price_per_1m,
            "output_price_per_1m_tokens": args.output_price_per_1m,
        },
        "per_task": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(args.output), "delta_a": payload["delta_a"]}, indent=2))


if __name__ == "__main__":
    main()
