"""Check Tenacious-Bench dataset splits for contamination."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


DEFAULT_SPLITS_DIR = Path("data/splits")


Task = Dict[str, Any]
Pair = Tuple[str, str]


def load_split(path: Path) -> List[Task]:
    """Load one split file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Split file must contain a JSON list: {path}")
    return data


def load_splits(splits_dir: Path) -> Dict[str, List[Task]]:
    """Load train, dev, and held-out splits."""
    return {
        "train": load_split(splits_dir / "train.json"),
        "dev": load_split(splits_dir / "dev.json"),
        "held_out": load_split(splits_dir / "held_out.json"),
    }


def task_input_key(task: Task) -> str:
    """Return a canonical string key for a task input."""
    return json.dumps(task.get("input", {}), sort_keys=True, ensure_ascii=False)


def task_output_key(task: Task) -> str:
    """Return a canonical string key for a task output."""
    output = task.get("output") or task.get("candidate_output") or task.get("reference_output") or {}
    if isinstance(output, dict):
        output = f"{output.get('subject', '')}\n{output.get('body', '')}"
    return str(output).strip().lower()


def company_signal_pair(task: Task) -> Pair:
    """Return a normalized company plus signal pair."""
    task_input = task.get("input", {})
    if not isinstance(task_input, dict):
        task_input = {}
    prospect = task_input.get("prospect", {})
    signal_brief = task_input.get("signal_brief", {})
    if not isinstance(prospect, dict):
        prospect = {}
    if not isinstance(signal_brief, dict):
        signal_brief = {}
    company = str(task_input.get("company") or prospect.get("company", "")).strip().lower()
    signal = str(task_input.get("signal") or signal_brief.get("summary", "")).strip().lower()
    return company, signal


def count_duplicate_keys(keys: Iterable[str]) -> int:
    """Count duplicate items beyond the first occurrence."""
    counts = Counter(keys)
    return sum(count - 1 for count in counts.values() if count > 1)


def pair_set(tasks: Iterable[Task]) -> Set[Pair]:
    """Build a set of company-signal pairs for a task list."""
    return {pair for pair in (company_signal_pair(task) for task in tasks) if all(pair)}


def contamination_report(splits: Dict[str, List[Task]]) -> Dict[str, Any]:
    """Return contamination metrics for all splits."""
    all_tasks = [task for tasks in splits.values() for task in tasks]
    input_duplicates = count_duplicate_keys(task_input_key(task) for task in all_tasks)
    output_duplicates = count_duplicate_keys(task_output_key(task) for task in all_tasks)

    train_pairs = pair_set(splits["train"])
    dev_pairs = pair_set(splits["dev"])
    held_out_pairs = pair_set(splits["held_out"])

    train_dev_overlap = train_pairs & dev_pairs
    train_held_out_overlap = train_pairs & held_out_pairs
    dev_held_out_overlap = dev_pairs & held_out_pairs
    all_overlaps = train_dev_overlap | train_held_out_overlap | dev_held_out_overlap

    total_pairs = len(train_pairs | dev_pairs | held_out_pairs)
    overlap_percentage = (len(all_overlaps) / total_pairs * 100) if total_pairs else 0.0

    return {
        "duplicate_inputs": input_duplicates,
        "duplicate_outputs": output_duplicates,
        "overlapping_company_signal_pairs": len(all_overlaps),
        "overlap_percentage": round(overlap_percentage, 4),
        "train_held_out_overlap": len(train_held_out_overlap),
        "split_sizes": {name: len(tasks) for name, tasks in splits.items()},
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    return parser.parse_args()


def main() -> None:
    """Print contamination report and fail on train/held-out overlap."""
    args = parse_args()
    splits = load_splits(args.splits_dir)
    report = contamination_report(splits)
    print(json.dumps(report, indent=2))
    if report["train_held_out_overlap"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
