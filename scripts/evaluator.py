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

GENERIC_OUTPUT_PHRASES = [
    "help you",
    "we are excited",
]

CTA_PATTERNS = [
    r"\b(?:15|min|minute|minutes|call|chat|meeting)\b",
    r"\blet me know\b",
    r"\bwould you\b",
    r"\bwould\b.*\b(useful|help|work)\b",
    r"\bare you open\b",
    r"\bcan we\b",
    r"\bcould we\b",
    r"\b15[- ]minute",
    r"\breply\b",
    r"\bcalendar\b",
]


def check_signal_mention(output: Any, signal: str) -> int:
    """Return 1 when the output reflects the task signal, otherwise 0."""
    text = normalize_text(output_to_text(output))
    signal_text = normalize_text(signal)
    if any(phrase in text for phrase in GENERIC_OUTPUT_PHRASES):
        return 0
    if not signal_text:
        return 0
    if signal_text in text:
        return 1

    signal_tokens = important_tokens(signal_text)
    return 1 if signal_tokens and any(token in text for token in signal_tokens) else 0


def check_tone_score(output: Any, signal: str = "") -> int:
    """Score Tenacious tone from 1 to 5.

    Tone calibration:
    - Low tone (1-2): uses banned phrases, hype, condescension, or external
      "bench" language. These are style-guide blockers.
    - Medium tone (3): basically professional, but generic, too long, or weakly
      direct. A human would edit before sending.
    - High tone (4-5): concise, professional, honest, non-condescending, and
      free of banned phrases.

    Threshold rationale:
    The Tenacious style guide treats banned phrases as regenerate conditions,
    so those score 1. External "bench" language and condescension score 2.
    Weakly personalized messages under 60 words score 3, while grounded,
    structured messages can score 5.
    """
    text = output_to_text(output)
    normalized = normalize_text(text)
    word_count = count_words(text)

    banned_hits = [phrase for phrase in BANNED_PHRASES if phrase in normalized]
    generic_spam = [phrase for phrase in GENERIC_OUTPUT_PHRASES if phrase in normalized]
    if banned_hits or generic_spam:
        return 1
    if re.search(r"\bbench\b", normalized):
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
    if word_count < 60:
        return 3
    if word_count > 160:
        return 3
    if word_count > 120:
        return 4

    signal_text = normalize_text(signal)
    signal_tokens = important_tokens(signal_text)
    has_signal_detail = bool(
        signal_text and (signal_text in normalized or any(token in normalized for token in signal_tokens))
    )
    structured_specific = has_signal_detail and "tenacious" in normalized
    if structured_specific:
        return 5
    return 4


def check_cta(output: Any) -> int:
    """Return 1 when the output contains at least one clear call to action."""
    text = normalize_text(output_to_text(output))
    return 1 if any(re.search(pattern, text) for pattern in CTA_PATTERNS) else 0


def evaluate_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate one task and return the required score object."""
    input_data = task.get("input", {})
    if not isinstance(input_data, Mapping):
        input_data = {}

    output = task.get("output", "")
    signal = str(input_data.get("signal", ""))

    signal_score = check_signal_mention(output, signal)
    tone_score = check_tone_score(output, signal)
    cta_score = check_cta(output)

    return {
        "task_id": task.get("task_id", ""),
        "scores": {
            "signal": signal_score,
            "tone": tone_score,
            "cta": cta_score,
        },
        "total": signal_score + tone_score + cta_score,
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
    stopwords = {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "new",
        "of",
        "on",
        "team",
        "the",
        "to",
        "with",
    }
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
