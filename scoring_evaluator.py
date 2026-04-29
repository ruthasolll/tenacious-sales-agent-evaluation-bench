"""Deterministic scoring evaluator for Tenacious-Bench outreach emails.

The evaluator is intentionally lightweight: it uses only Python's standard
library, simple text heuristics, and explicit score components. It is designed
for dataset filtering, training-data quality scoring, and reward shaping where
stable behavior is more important than model-like nuance.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ScoreDict = Dict[str, float]

DEFAULT_TONE_MARKERS = [
    "direct",
    "grounded",
    "honest",
    "professional",
    "non-condescending",
]

DEFAULT_BANNED_PHRASES = [
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
    "synergize",
    "synergy",
    "leverage",
    "ecosystem",
    "game-changer",
    "disruptor",
    "paradigm shift",
    "you'll regret missing this",
    "don't miss out",
    "per my last email",
]

ASK_PATTERNS = [
    r"\bschedule\b",
    r"\bbook\b",
    r"\bcall\b",
    r"\bmeet\b",
    r"\bmeeting\b",
    r"\bcalendar\b",
    r"\breply\b",
    r"\bsend\b",
    r"\bdiscuss\b",
    r"\bconnect\b",
    r"\b15[- ]minute",
    r"\b30[- ]minute",
    r"\bwould you\b",
    r"\bcould we\b",
    r"\bcan we\b",
    r"\bare you open\b",
    r"\bdo you have\b",
    r"\bnext week\b",
]

WEIGHTS = {
    "tone_score": 0.25,
    "banned_phrase_score": 0.25,
    "signal_presence_score": 0.25,
    "word_count_score": 0.15,
    "ask_intensity_score": 0.10,
}


def evaluate_email(payload: Mapping[str, Any]) -> ScoreDict:
    """Evaluate a generated sales email and return normalized component scores.

    Args:
        payload: Dictionary containing ``email_text`` and optional ``metadata``.
            Metadata may include ``tone_markers_expected``, ``banned_phrases``,
            and ``signals_expected``.

    Returns:
        A dictionary with component scores in the range 0.0 to 1.0 plus an
        ``overall_score`` computed from deterministic weights.
    """
    email_text = str(payload.get("email_text") or "")
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}

    tone_markers = _get_string_list(
        metadata.get("tone_markers_expected"),
        default=DEFAULT_TONE_MARKERS,
    )
    banned_phrases = _get_string_list(
        metadata.get("banned_phrases"),
        default=DEFAULT_BANNED_PHRASES,
    )
    signals_expected = _get_string_list(
        metadata.get("signals_expected"),
        default=[],
    )

    banned_violations = find_banned_phrases(email_text, banned_phrases)

    tone_score = score_tone(email_text, tone_markers)
    if banned_violations:
        tone_score = min(tone_score * 0.45, 0.45)

    scores = {
        "tone_score": tone_score,
        "banned_phrase_score": score_banned_phrases(banned_violations),
        "signal_presence_score": score_signal_presence(email_text, signals_expected),
        "word_count_score": score_word_count(count_words(email_text)),
        "ask_intensity_score": score_ask_intensity(count_asks(email_text)),
    }
    scores["overall_score"] = compute_overall_score(scores)
    return {key: round(value, 4) for key, value in scores.items()}


def score_tone(email_text: str, expected_markers: Sequence[str]) -> float:
    """Score how well the email satisfies expected tone markers.

    Tenacious's five standard markers are scored with deterministic heuristics.
    Unknown custom markers are treated as literal phrases and scored by whether
    they appear in the email text.
    """
    markers = [marker for marker in expected_markers if marker]
    if not markers:
        return 1.0

    normalized = normalize_text(email_text)
    marker_scores = []
    for marker in markers:
        key = normalize_text(marker).replace(" ", "-")
        if key == "direct":
            marker_scores.append(_score_direct(email_text))
        elif key == "grounded":
            marker_scores.append(_score_grounded(email_text))
        elif key == "honest":
            marker_scores.append(_score_honest(email_text))
        elif key == "professional":
            marker_scores.append(_score_professional(email_text))
        elif key in {"non-condescending", "noncondescending"}:
            marker_scores.append(_score_non_condescending(email_text))
        else:
            marker_scores.append(1.0 if normalize_text(marker) in normalized else 0.0)

    return _mean(marker_scores)


def score_banned_phrases(violations: Sequence[str]) -> float:
    """Score banned phrase compliance with a strong violation penalty."""
    count = len(set(violations))
    if count == 0:
        return 1.0
    return clamp01(1.0 - (0.35 * count))


def score_signal_presence(email_text: str, expected_signals: Sequence[str]) -> float:
    """Score whether required prospect signals appear in the email.

    Missing signals are penalized heavily by squaring the coverage ratio. This
    makes partial grounding visible but prevents half-grounded emails from
    receiving a high signal score.
    """
    signals = [signal for signal in expected_signals if signal]
    if not signals:
        return 1.0

    normalized = normalize_text(email_text)
    matched = sum(1 for signal in signals if _contains_signal(normalized, signal))
    coverage = matched / len(signals)
    return clamp01(coverage * coverage)


def score_word_count(word_count: int, target_min: int = 120, target_max: int = 250) -> float:
    """Score word count with a smooth continuous penalty outside the target range."""
    if word_count <= 0:
        return 0.0
    if target_min <= word_count <= target_max:
        return 1.0
    if word_count < target_min:
        return clamp01(word_count / target_min)
    return clamp01(target_max / word_count)


def score_ask_intensity(ask_count: int, ideal_min: int = 1, ideal_max: int = 2) -> float:
    """Score CTA intensity, rewarding one or two asks and penalizing extremes."""
    if ideal_min <= ask_count <= ideal_max:
        return 1.0
    if ask_count <= 0:
        return 0.25
    return clamp01(ideal_max / ask_count)


def compute_overall_score(scores: Mapping[str, float]) -> float:
    """Compute the weighted overall score from normalized component scores."""
    total = 0.0
    for key, weight in WEIGHTS.items():
        total += float(scores.get(key, 0.0)) * weight
    return clamp01(total)


def count_words(text: str) -> int:
    """Count word-like tokens in text."""
    return len(re.findall(r"\b[\w'-]+\b", text))


def count_asks(text: str) -> int:
    """Count likely asks using simple CTA and question-mark heuristics."""
    normalized = normalize_text(text)
    keyword_hits = 0
    for pattern in ASK_PATTERNS:
        if re.search(pattern, normalized):
            keyword_hits += 1

    question_hits = normalized.count("?")
    calendar_hits = len(re.findall(r"https?://\S*calendar\S*|cal\.com/\S+", normalized))
    return keyword_hits + question_hits + calendar_hits


def find_banned_phrases(text: str, banned_phrases: Sequence[str]) -> List[str]:
    """Return banned phrases found in the email, preserving input phrase casing."""
    normalized = normalize_text(text)
    found = []
    for phrase in banned_phrases:
        if phrase and normalize_text(phrase) in normalized:
            found.append(phrase)
    return found


def normalize_text(text: str) -> str:
    """Normalize text for deterministic matching."""
    lowered = str(text).lower()
    return re.sub(r"\s+", " ", lowered).strip()


def clamp01(value: float) -> float:
    """Clamp a numeric value into the inclusive range 0.0 to 1.0."""
    return max(0.0, min(1.0, float(value)))


def _get_string_list(value: Any, default: Sequence[str]) -> List[str]:
    """Return a clean list of strings from metadata, falling back gracefully."""
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Iterable):
        return list(default)
    return [str(item) for item in value if str(item).strip()]


def _contains_signal(normalized_email: str, signal: str) -> bool:
    """Return whether a signal is represented in normalized email text."""
    normalized_signal = normalize_text(signal)
    if not normalized_signal:
        return False
    if normalized_signal in normalized_email:
        return True

    signal_tokens = _important_tokens(normalized_signal)
    if not signal_tokens:
        return False

    matched_tokens = [token for token in signal_tokens if token in normalized_email]
    return len(matched_tokens) / len(signal_tokens) >= 0.75


def _important_tokens(text: str) -> List[str]:
    """Extract non-trivial tokens for approximate signal matching."""
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
        "your",
    }
    tokens = re.findall(r"\b[\w$.-]+\b", text)
    return [token for token in tokens if token not in stopwords and len(token) > 1]


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of normalized values."""
    if not values:
        return 1.0
    return clamp01(sum(values) / len(values))


def _score_direct(email_text: str) -> float:
    """Score directness using length, filler, and ask-count heuristics."""
    word_count = count_words(email_text)
    ask_count = count_asks(email_text)
    filler_penalty = _contains_any(
        email_text,
        ["i hope this email finds you well", "just wanted", "quick question", "quick chat"],
    )
    score = score_word_count(word_count, target_min=50, target_max=160)
    if ask_count == 0:
        score *= 0.75
    elif ask_count > 2:
        score *= max(0.45, 2 / ask_count)
    if filler_penalty:
        score *= 0.7
    return clamp01(score)


def _score_grounded(email_text: str) -> float:
    """Score groundedness with numeric, date, and evidence-language heuristics."""
    normalized = normalize_text(email_text)
    has_number = bool(re.search(r"\b\d+(\.\d+)?\b|\$[\d,]+", normalized))
    has_time_reference = bool(
        re.search(
            r"\b(january|february|march|april|may|june|july|august|september|"
            r"october|november|december|q[1-4]|last \d+ days|\d+ days)\b",
            normalized,
        )
    )
    has_evidence_language = _contains_any(
        normalized,
        ["saw", "noticed", "posted", "roles", "announcement", "funding", "layoff"],
    )
    return _mean([float(has_number), float(has_time_reference), float(has_evidence_language)])


def _score_honest(email_text: str) -> float:
    """Score honesty by rewarding uncertainty and penalizing overclaim language."""
    normalized = normalize_text(email_text)
    overclaim = _contains_any(
        normalized,
        ["guarantee", "absolutely", "definitely", "always", "must be", "will skyrocket"],
    )
    uncertainty = _contains_any(
        normalized,
        ["i cannot tell", "if", "whether", "depends", "route", "confirm capacity"],
    )
    if overclaim and not uncertainty:
        return 0.35
    if overclaim:
        return 0.6
    if uncertainty:
        return 1.0
    return 0.75


def _score_professional(email_text: str) -> float:
    """Score professionalism by penalizing informal or vendor-cliche language."""
    normalized = normalize_text(email_text)
    unprofessional_terms = [
        "hey ",
        "rockstar",
        "ninja",
        "wizard",
        "top talent",
        "world-class",
        "synergy",
        "synergize",
        "game-changer",
    ]
    violations = sum(1 for term in unprofessional_terms if term in normalized)
    return clamp01(1.0 - (0.2 * violations))


def _score_non_condescending(email_text: str) -> float:
    """Score whether the email avoids shaming or superiority framing."""
    normalized = normalize_text(email_text)
    condescending_terms = [
        "behind the curve",
        "falling behind",
        "you need to",
        "you should",
        "catch up",
        "missing",
        "obvious next move",
    ]
    violations = sum(1 for term in condescending_terms if term in normalized)
    return clamp01(1.0 - (0.25 * violations))


def _contains_any(text: str, phrases: Sequence[str]) -> bool:
    """Return True if any phrase appears in text after normalization."""
    normalized = normalize_text(text)
    return any(normalize_text(phrase) in normalized for phrase in phrases)


if __name__ == "__main__":
    example_payload = {
        "email_text": (
            "Subject: Request: 15 minutes on Python hiring\n\n"
            "Hi Maya,\n\n"
            "You closed your $14M Series A in February, and your open Python "
            "roles moved from 2 to 7 in the last 60 days. If recruiting "
            "capacity is the bottleneck, we can discuss managed Python and "
            "data engineers with confirmed availability.\n\n"
            "Would 15 minutes next week be useful?\n\n"
            "Best,\nYabi"
        ),
        "metadata": {
            "signals_expected": ["$14M Series A", "Python roles moved from 2 to 7"],
            "tone_markers_expected": DEFAULT_TONE_MARKERS,
            "banned_phrases": DEFAULT_BANNED_PHRASES,
        },
    }
    print(evaluate_email(example_payload))
