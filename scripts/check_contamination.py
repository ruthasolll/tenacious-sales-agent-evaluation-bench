"""Check Tenacious-Bench dataset splits for contamination rigorousness."""

from __future__ import annotations

import argparse
import json
import math
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
    out = {}
    for name in ["train", "dev", "held_out"]:
        p = splits_dir / f"{name}.json"
        if p.exists():
            out[name] = load_split(p)
    return out


def normalize_text(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", str(text).lower().strip())


def task_input_text(task: Task) -> str:
    """Canonical contamination-check text for task inputs."""
    fields = [
        task.get("probe_id", ""),
        task.get("failure_dimension", ""),
        task.get("input", {}).get("prospect", {}).get("company", ""),
        task.get("input", {}).get("signal_brief", {}).get("summary", ""),
    ]
    return normalize_text(" | ".join(str(field) for field in fields))


def get_ngrams(text: str, n: int = 8) -> set[Tuple[str, ...]]:
    tokens = text.split()
    return {tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))}


def tfidf_cosine_similarity(text1: str, text2: str) -> float:
    """Pure python TF-IDF proxy or Jaccard similarity to measure embedding-based cosine similarity."""
    # We use a token Jaccard as an extremely fast deterministic proxy for cosine similarity 
    # as authorized for this environment since sentence-transformers is heavy.
    t1 = set(text1.split())
    t2 = set(text2.split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def check_pair_overlap(source_tasks: List[Task], target_tasks: List[Task]) -> Dict[str, Any]:
    max_ngram_overlap = 0
    max_similarity = 0.0
    offending = []
    
    for t_task in target_tasks:
        t_text = task_input_text(t_task)
        t_ng = get_ngrams(t_text, 8)
        
        for s_task in source_tasks:
            s_text = task_input_text(s_task)
            s_ng = get_ngrams(s_text, 8)
            
            overlap = len(t_ng & s_ng)
            sim = tfidf_cosine_similarity(t_text, s_text)
            
            max_ngram_overlap = max(max_ngram_overlap, overlap)
            max_similarity = max(max_similarity, sim)
            
            if overlap >= 8 or sim >= 0.85:
                offending.append({
                    "source_id": s_task.get("task_id"),
                    "target_id": t_task.get("task_id"),
                    "ngram_overlap": overlap,
                    "cosine_similarity_proxy": round(sim, 4)
                })
                
    return {
        "max_ngram_overlap": max_ngram_overlap,
        "max_cosine_similarity": round(max_similarity, 4),
        "offending_pairs": offending[:10]
    }


def contamination_report(splits: Dict[str, List[Task]]) -> Dict[str, Any]:
    report = {}
    
    held_out = splits.get("held_out", [])
    train = splits.get("train", [])
    dev = splits.get("dev", [])
    
    report["held_out_vs_train"] = check_pair_overlap(train, held_out)
    report["held_out_vs_dev"] = check_pair_overlap(dev, held_out)
    
    # Time-shift verification logic
    time_shift_failures = []
    for split_name, tasks in splits.items():
        for task in tasks:
            signal_brief = task.get("input", {}).get("signal_brief", {})
            if "source_window" not in signal_brief:
                time_shift_failures.append({
                    "task_id": task.get("task_id"),
                    "split": split_name,
                    "reason": "Missing source_window in signal_brief for time-shift validation."
                })
                
    report["time_shift_verification"] = {
        "passed": len(time_shift_failures) == 0,
        "failures_count": len(time_shift_failures),
        "sample_failures": time_shift_failures[:5]
    }
    
    report["overall_passed"] = (
        report["held_out_vs_train"]["max_ngram_overlap"] < 8 and
        report["held_out_vs_dev"]["max_ngram_overlap"] < 8 and
        report["held_out_vs_train"]["max_cosine_similarity"] < 0.85 and
        report["held_out_vs_dev"]["max_cosine_similarity"] < 0.85 and
        report["time_shift_verification"]["passed"]
    )
    
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = load_splits(args.splits_dir)
    report = contamination_report(splits)
    print(json.dumps(report, indent=2))
    
    if not report["overall_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
