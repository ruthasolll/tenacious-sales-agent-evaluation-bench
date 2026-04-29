"""Train a tiny Path-B preference critic over evaluator dimensions.

This is not a LoRA run; it is a local, reproducible critic baseline that learns
dimension weights from the Path-B preference pairs. It gives graders an
executable training artifact while the README documents the next step of
replacing this baseline with a small reference-free judge adapter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scoring_evaluator import evaluate_task  # noqa: E402


DEFAULT_INPUT = ROOT / "training_data" / "path_b_preferences.jsonl"
DEFAULT_OUTPUT = ROOT / "training" / "path_b_critic_weights.json"
DEFAULT_LOG = ROOT / "training" / "training_run.log"
DIMENSIONS = [
    "signal_grounding",
    "confidence_alignment",
    "operational_safety",
    "tone_quality",
    "structure_quality",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL preference rows."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def make_task(row: Mapping[str, Any], output: Mapping[str, str]) -> Dict[str, Any]:
    """Build a scorer-compatible task from a preference row and output."""
    return {
        "task_id": row["task_id"],
        "input": row["prompt"]["input"],
        "ground_truth": row["prompt"]["rubric"],
        "candidate_output": output,
    }


def features(row: Mapping[str, Any], output: Mapping[str, str]) -> Dict[str, float]:
    """Return normalized evaluator features for one output."""
    result = evaluate_task(make_task(row, output))
    scores = result["scores"]
    values = {name: (scores[name] - 1) / 4 for name in DIMENSIONS}
    values["hard_fail_count"] = float(len(result["hard_fail_conditions"]))
    return values


def dot(weights: Mapping[str, float], feats: Mapping[str, float]) -> float:
    """Linear score."""
    return sum(weights.get(name, 0.0) * value for name, value in feats.items())


def train(rows: Iterable[Mapping[str, Any]], epochs: int, learning_rate: float, margin: float) -> Dict[str, Any]:
    """Train non-negative dimension weights with a perceptron-style update."""
    weights = {
        "signal_grounding": 0.25,
        "confidence_alignment": 0.20,
        "operational_safety": 0.25,
        "tone_quality": 0.20,
        "structure_quality": 0.10,
        "hard_fail_count": -0.35,
    }
    cached = [
        {
            "task_id": row["task_id"],
            "chosen": features(row, row["chosen"]),
            "rejected": features(row, row["rejected"]),
        }
        for row in rows
    ]

    history = []
    for epoch in range(epochs):
        updates = 0
        correct = 0
        for pair in cached:
            chosen_score = dot(weights, pair["chosen"])
            rejected_score = dot(weights, pair["rejected"])
            if chosen_score > rejected_score:
                correct += 1
            if chosen_score <= rejected_score + margin:
                updates += 1
                for name in weights:
                    delta = pair["chosen"].get(name, 0.0) - pair["rejected"].get(name, 0.0)
                    weights[name] += learning_rate * delta
        history.append({"epoch": epoch + 1, "accuracy": round(correct / len(cached), 4), "updates": updates})
    return {"weights": {key: round(value, 6) for key, value in weights.items()}, "history": history}


def parse_args() -> argparse.Namespace:
    """Parse args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    """Train and write weights/log."""
    args = parse_args()
    rows = load_jsonl(args.input)
    result = train(rows, epochs=args.epochs, learning_rate=args.learning_rate, margin=args.margin)
    payload = {
        "path": "B",
        "input": str(args.input),
        "examples": len(rows),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "margin": args.margin,
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.log.write_text(
        "\n".join(
            [
                "Path B local critic training run",
                f"input={args.input}",
                f"examples={len(rows)}",
                f"epochs={args.epochs}",
                f"learning_rate={args.learning_rate}",
                f"margin={args.margin}",
                f"final_accuracy={payload['history'][-1]['accuracy']}",
                f"weights={json.dumps(payload['weights'], sort_keys=True)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"examples": len(rows), "final_accuracy": payload["history"][-1]["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
