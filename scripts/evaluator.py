"""Simple evaluator for the Tenacious-Bench seed dev split.

Run:
    python scripts/evaluator.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping


DATASET_PATH = Path("data/seed/splits/dev.json")

BANNED_PHRASES = [
    "world-class",
    "top talent",
    "a-players",
    "rockstar",
    "ninja",
    "wizard",
    "skyrocket",
    "supercharge",
    "10x",
    "i hope this email finds you well",
    "just following up",
    "circling back",
    "quick question",
    "quick chat",
    "synergy",
    "leverage",
    "game-changer",
    "don't miss out",
    "per my last email",
]

CTA_PATTERNS = [
    r"\bwould\b.*\b(useful|help|work)\b",
    r"\bare you open\b",
    r"\bcan we\b",
    r"\bcould we\b",
    r"\b15[- ]minute",
    r"\breply\b",
    r"\bcalendar\b",
]


def check_signal_mention(output: Any, signal: str) -> int:
    """Return 1 when the output mentions the task's signal, otherwise 0."""
    text = normalize_text(output_to_text(output))
    signal_text = normalize_text(signal)
    if not signal_text:
        return 0
    if signal_text in text:
        return 1

    signal_tokens = important_tokens(signal_text)
    if not signal_tokens:
        return 0
    matched = sum(1 for token in signal_tokens if token in text)
    return 1 if matched / len(signal_tokens) >= 0.7 else 0


def check_tone_score(output: Any) -> int:
    """Score Tenacious tone from 1 to 5.

    Tone calibration:
    - Low tone (1-2): uses banned phrases, hype, condescension, or external
      "bench" language. These are style-guide blockers.
    - Medium tone (3): basically professional, but generic, too long, or weakly
      direct. A human would edit before sending.
    - High tone (4-5): concise, professional, honest, non-condescending, and
      free of banned phrases.

    Threshold rationale:
    The Tenacious style guide treats banned phrases and external "bench"
    language as regenerate conditions, so any such hit caps the score at 2.
    Cold outreach is expected to stay near or below 120 words, so messages over
    160 words lose directness. A clean, concise message gets 5.
    """
    text = output_to_text(output)
    normalized = normalize_text(text)
    word_count = count_words(text)

    banned_hits = [phrase for phrase in BANNED_PHRASES if phrase in normalized]
    if banned_hits or re.search(r"\bbench\b", normalized):
        return 2

    condescending = [
        "behind the curve",
        "falling behind",
        "you need to",
        "you should",
        "catch up",
    ]
    if any(phrase in normalized for phrase in condescending):
        return 2

    hype_or_overclaim = ["guarantee", "definitely", "always", "unbeatable"]
    if any(term in normalized for term in hype_or_overclaim):
        return 3

    if word_count == 0:
        return 1
    if word_count > 160:
        return 3
    if word_count > 120:
        return 4
    return 5


def check_cta(output: Any) -> int:
    """Return 1 when the output contains at least one clear call to action."""
    text = normalize_text(output_to_text(output))
    if "?" in text:
        return 1
    return 1 if any(re.search(pattern, text) for pattern in CTA_PATTERNS) else 0


def evaluate_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate one task and return the required score object."""
    task_input = task.get("input", {})
    if not isinstance(task_input, Mapping):
        task_input = {}

    output = task.get("output", "")
    signal = str(task_input.get("signal", ""))

    scores = {
        "signal": check_signal_mention(output, signal),
        "tone": check_tone_score(output),
        "cta": check_cta(output),
    }
    total = scores["signal"] + scores["tone"] + scores["cta"]
    return {
        "task_id": task.get("task_id", ""),
        "scores": scores,
        "total": total,
    }


def load_dataset(path: Path = DATASET_PATH) -> List[Mapping[str, Any]]:
    """Load the seed dev split."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of tasks in {path}")
    return data


def output_to_text(output: Any) -> str:
    """Support string outputs and {'subject', 'body'} outputs."""
    if isinstance(output, Mapping):
        return f"{output.get('subject', '')}\n{output.get('body', '')}"
    return str(output or "")


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def important_tokens(text: str) -> List[str]:
    """Extract useful tokens from a signal string."""
    stopwords = {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "the", "to", "with"}
    tokens = re.findall(r"\b[\w$.-]+\b", text)
    return [token for token in tokens if token not in stopwords and len(token) > 1]


def count_words(text: str) -> int:
    """Count word-like tokens."""
    return len(re.findall(r"\b[\w'-]+\b", text))


def main() -> None:
    """Load the dev split, evaluate each task, and print the average score."""
    tasks = load_dataset()
    results = [evaluate_task(task) for task in tasks]
    average = sum(result["total"] for result in results) / len(results) if results else 0.0

    print(json.dumps({"results": results, "average_score": round(average, 3)}, indent=2))


if __name__ == "__main__":
    main()
