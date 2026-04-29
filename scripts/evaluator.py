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

# -------------------------------
# SCORING CALIBRATION (IMPORTANT)
# -------------------------------
# SIGNAL SCORE:
# 0 = no meaningful reference to input signal
# 1 = explicit or strong semantic reference to signal
#
# TONE SCORE:
# 1 = spammy / generic / salesy / buzzword-heavy
# 3 = acceptable but slightly vague or templated
# 5 = grounded, specific, non-salesy, context-aware
#
# CTA SCORE:
# 0 = no clear next step or call-to-action
# 1 = explicit invitation to respond, meet, or discuss

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
    "number one",
    "#1",
    "per my last email",
]

GENERIC_OUTPUT_PHRASES = [
    "help you",
    "we are excited",
]

CTA_PATTERNS = [
    r"\bwould (?:this|it|that).*(?:help|work|useful)\b",
    r"\bare you open to\b",
    r"\bcan we\b",
    r"\bcould we\b",
    r"\bworth (?:a |quick )?(?:chat|call|discussion)\b",
    r"\blet me know\b",
    r"\bhappy to\b",
    r"\bopen to\b",
    r"\b\d+\s*[- ]?\s*(?:min|minute|minutes)\b",
    r"\bschedule\b",
    r"\bcall\b",
    r"\bchat\b",
    r"\bmeeting\b",
    r"\bcalendar\b",
]

MONTH_TOKENS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}


def check_signal_mention(output: Any, signal: str) -> int:
    """Return 1 when the output reflects the task signal, otherwise 0."""
    text = normalize_text(output_to_text(output))
    signal_text = normalize_text(signal)
    if any(phrase in text for phrase in GENERIC_OUTPUT_PHRASES):
        return 0
    signal_tokens = important_tokens(signal_text)
    if not signal_tokens:
        return 0
    if signal_text in text:
        return 1

    matched_tokens = [token for token in signal_tokens if len(token) > 3 and token in text]
    if not matched_tokens:
        return 0

    anchor_tokens = [token for token in signal_tokens if is_signal_anchor(token)]
    matched_anchors = [token for token in anchor_tokens if token in text]
    if anchor_tokens:
        return 1 if matched_anchors and len(matched_tokens) >= 2 else 0

    return 1 if len(matched_tokens) >= min(2, len(signal_tokens)) else 0


# ----------------------------
# TONE CALIBRATION RUBRIC
# ----------------------------
# 1 -> spammy, generic, or overpromising
#      ("We are excited to help you scale your business")
# 3 -> acceptable but weak personalization
#      ("I saw you're hiring engineers, we can help")
# 5 -> highly specific, grounded in signal, and professionally framed
#      ("I noticed your AI hiring push after the Series C...")
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

    urgency_or_overclaim = ["guarantee", "definitely", "always", "unbeatable", "instantly", "urgently"]
    if any(term in normalized for term in urgency_or_overclaim):
        return 2

    if word_count == 0:
        return 1
    if word_count < 60:
        return 3
    if word_count > 160:
        return 3
    if word_count > 120:
        return 4

    signal_grounded = check_signal_mention(output, signal) == 1
    has_cta = contains_cta(normalized)
    structured_specific = signal_grounded and has_cta and "tenacious" in normalized
    if structured_specific:
        return 5
    return 4


def check_cta(output: Any) -> int:
    """Return 1 when the output contains a clear response, meeting, or discussion CTA."""
    text = normalize_text(output_to_text(output))
    return 1 if contains_cta(text) else 0


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


def is_signal_anchor(token: str) -> bool:
    """Return whether a token carries concrete signal evidence."""
    return any(char.isdigit() for char in token) or token.startswith("$") or token in MONTH_TOKENS


def contains_cta(text: str) -> bool:
    """Return whether normalized text contains a clear CTA."""
    return any(re.search(pattern, text) for pattern in CTA_PATTERNS)


def count_words(text: str) -> int:
    """Count word-like tokens."""
    return len(re.findall(r"\b[\w'-]+\b", text))


def main() -> None:
    """Load the dev split, evaluate each task, and print the average score."""
    tasks = load_dataset()
    results = [evaluate_task(task) for task in tasks]
    average = sum(result["total"] for result in results) / len(results) if results else 0.0

    print(json.dumps({"results": results, "average_score": round(average, 3)}, indent=2))
    print("TASK VARIANCE CHECK:")
    print("signal:", [result["scores"]["signal"] for result in results])
    print("tone:", [result["scores"]["tone"] for result in results])
    print("cta:", [result["scores"]["cta"] for result in results])


if __name__ == "__main__":
    main()
