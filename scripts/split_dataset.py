"""Deterministically split generated Tenacious-Bench tasks into partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_INPUT = Path("data/generated_tasks.json")
DEFAULT_OUTPUT_DIR = Path("data/splits")


Task = Dict[str, Any]


def load_tasks(path: Path) -> List[Task]:
    """Load a JSON list of tasks."""
    if not path.exists():
        raise FileNotFoundError(f"Missing task file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Task file must contain a JSON list.")
    return data


def stable_task_key(task: Task) -> str:
    """Return a deterministic hash key for a task."""
    payload = json.dumps(task, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_tasks(
    tasks: List[Task],
    train_ratio: float = 0.5,
    dev_ratio: float = 0.3,
) -> Tuple[List[Task], List[Task], List[Task]]:
    """Split tasks deterministically into train, dev, and held-out lists."""
    if not tasks:
        return [], [], []

    ordered = sorted(tasks, key=stable_task_key)
    total = len(ordered)
    train_count = int(total * train_ratio)
    dev_count = int(total * dev_ratio)

    train = ordered[:train_count]
    dev = ordered[train_count : train_count + dev_count]
    held_out = ordered[train_count + dev_count :]
    return train, dev, held_out


def save_splits(train: List[Task], dev: List[Task], held_out: List[Task], output_dir: Path) -> None:
    """Write split files to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.json").write_text(json.dumps(train, indent=2), encoding="utf-8")
    (output_dir / "dev.json").write_text(json.dumps(dev, indent=2), encoding="utf-8")
    (output_dir / "held_out.json").write_text(json.dumps(held_out, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Load generated tasks, split them, and write partition files."""
    args = parse_args()
    tasks = load_tasks(args.input)
    train, dev, held_out = split_tasks(tasks)
    save_splits(train, dev, held_out, args.output_dir)
    print(
        json.dumps(
            {
                "train": len(train),
                "dev": len(dev),
                "held_out": len(held_out),
                "total": len(tasks),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
