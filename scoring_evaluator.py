"""Deterministic scoring evaluator for Tenacious-Bench v0.1.

The evaluator is intentionally executable without network calls or model
dependencies. It can score a benchmark task plus a candidate outreach draft and
return stable numeric scores for automated grading, dataset filtering, and
Path-B preference-data preparation.

Calibration notes
-----------------
Each dimension is scored on a 1-5 scale:

1 = severe failure. The draft would be blocked or routed to human review.
2 = weak. A relevant idea is present, but the mistake is materially risky.
3 = partial. The draft is usable only after editing or manual verification.
4 = pass. The draft meets the pre-flight checklist with minor imperfections.
5 = strong. The draft is specific, concise, and safe without manual repair.

Thresholds follow the Tenacious style guide: any tone marker below 4/5 triggers
regeneration, so the benchmark uses >=4 on signal grounding, confidence
alignment, operational safety, and tone as the inclusion threshold. Structure
allows >=3 because subject length and word count are deterministic trim issues;
hard-fail conditions still fail the record regardless of the weighted score.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


DEFAULT_EXAMPLES = Path("examples/example_tasks.json")

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
    "our proprietary",
    "our ai-powered",
    "you'll regret missing this",
    "don't miss out",
    "per my last email",
    "i noticed you're a",
]

CONDESCENDING_TERMS = [
    "behind the curve",
    "falling behind",
    "catch up",
    "obvious next move",
    "you need to",
    "you should",
    "your team is missing",
]

ASSERTIVE_WEAK_SIGNAL_TERMS = [
    "you are scaling",
    "you're scaling",
    "you clearly need",
    "you must be hiring",
    "you need engineers",
    "you have a delivery gap",
    "your bottleneck is",
]

CONDITIONAL_TERMS = [
    "if",
    "whether",
    "i cannot tell",
    "i can't tell",
    "might",
    "may",
    "could",
    "would",
    "depends",
    "assuming",
]

HUMAN_ROUTE_TERMS = [
    "confirm capacity",
    "scope",
    "scoping",
    "discovery",
    "delivery lead",
    "human",
    "route",
    "before any capacity commitment",
    "availability confirmed",
]

COMMITMENT_TERMS = [
    "we can provide",
    "we can place",
    "we can staff",
    "we can start",
    "ready to deploy",
    "available immediately",
    "start within",
    "plug a team in",
    "engineers in your slack",
    "confirmed availability",
]

ASK_PATTERNS = [
    r"\bwould\b[^?.!]*(?:useful|help|work|open)",
    r"\bcould we\b",
    r"\bcan we\b",
    r"\bare you open\b",
    r"\bdo you have\b",
    r"\bbook\b",
    r"\bschedule\b",
    r"\bcalendar\b",
    r"\breply\b",
    r"\bsend\b",
]

DIMENSION_WEIGHTS = {
    "signal_grounding": 0.25,
    "confidence_alignment": 0.20,
    "operational_safety": 0.25,
    "tone_quality": 0.20,
    "structure_quality": 0.10,
}

PASS_THRESHOLDS = {
    "signal_grounding": 4,
    "confidence_alignment": 4,
    "operational_safety": 4,
    "tone_quality": 4,
    "structure_quality": 3,
}


ScoreDict = Dict[str, Any]


def evaluate_task(task: Mapping[str, Any], candidate_field: str = "candidate_output") -> ScoreDict:
    """Score one Tenacious-Bench task.

    Args:
        task: Benchmark task record. The function also accepts the older
            ``{"email_text": ..., "metadata": ...}`` demo payload.
        candidate_field: Name of the field that contains the candidate draft.

    Returns:
        A dictionary containing 1-5 dimension scores, hard-fail labels, extracted
        deterministic metrics, pass/fail status, and a 0-100 overall score.
    """
    normalized_task = normalize_task(task, candidate_field=candidate_field)
    subject, body = output_to_subject_body(normalized_task.get(candidate_field, {}))
    full_text = f"{subject}\n\n{body}".strip()

    extracted: Dict[str, Any] = {
        "subject_chars": len(subject),
        "body_word_count": count_words(body),
        "ask_count": count_asks(body),
        "banned_phrases": find_banned_phrases(full_text),
        "mentions_external_bench": bool(re.search(r"\bbench\b", normalize_text(full_text))),
        "capacity_claims": extract_capacity_claims(full_text),
        "money_amounts": extract_money_amounts(full_text),
    }

    hard_fails: List[str] = []
    scores = {
        "signal_grounding": score_signal_grounding(normalized_task, full_text, hard_fails),
        "confidence_alignment": score_confidence_alignment(normalized_task, full_text, hard_fails),
        "operational_safety": score_operational_safety(normalized_task, full_text, extracted, hard_fails),
        "tone_quality": score_tone_quality(full_text, extracted, hard_fails),
        "structure_quality": score_structure_quality(normalized_task, subject, body, extracted, hard_fails),
    }

    overall = weighted_overall(scores)
    if hard_fails:
        overall = min(overall, 59.0)

    passed = all(scores[name] >= threshold for name, threshold in PASS_THRESHOLDS.items()) and not hard_fails
    return {
        "task_id": normalized_task.get("task_id", "ad_hoc"),
        "scores": scores,
        "overall": round(overall, 2),
        "passed": passed,
        "hard_fail_conditions": sorted(set(hard_fails)),
        "extracted": extracted,
    }


def evaluate_dataset(tasks: Sequence[Mapping[str, Any]], candidate_field: str = "candidate_output") -> Dict[str, Any]:
    """Score a sequence of tasks and return per-task and aggregate results."""
    per_task = [evaluate_task(task, candidate_field=candidate_field) for task in tasks]
    if not per_task:
        return {"aggregate": {"task_count": 0}, "results": []}

    dimension_names = list(DIMENSION_WEIGHTS)
    aggregate = {
        "task_count": len(per_task),
        "pass_rate": round(sum(1 for result in per_task if result["passed"]) / len(per_task), 4),
        "mean_overall": round(mean(result["overall"] for result in per_task), 2),
        "mean_dimensions": {
            name: round(mean(result["scores"][name] for result in per_task), 3)
            for name in dimension_names
        },
        "hard_fail_counts": count_hard_fails(per_task),
    }
    return {"aggregate": aggregate, "results": per_task}


def evaluate_email(payload: Mapping[str, Any]) -> Dict[str, float]:
    """Backward-compatible wrapper for the original starter evaluator API."""
    result = evaluate_task(payload, candidate_field="candidate_output")
    scores = result["scores"]
    return {
        "tone_score": round((scores["tone_quality"] - 1) / 4, 4),
        "banned_phrase_score": 0.0 if result["extracted"]["banned_phrases"] else 1.0,
        "signal_presence_score": round((scores["signal_grounding"] - 1) / 4, 4),
        "word_count_score": round(score_word_count(result["extracted"]["body_word_count"]), 4),
        "ask_intensity_score": round(score_ask_intensity(result["extracted"]["ask_count"]), 4),
        "overall_score": round(result["overall"] / 100, 4),
    }


def normalize_task(task: Mapping[str, Any], candidate_field: str) -> Dict[str, Any]:
    """Normalize older demo payloads and current benchmark records."""
    if "email_text" in task:
        metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
        signals = listify(metadata.get("signals_expected"))
        return {
            "task_id": task.get("task_id", "legacy_email_payload"),
            candidate_field: {"subject": "", "body": str(task.get("email_text") or "")},
            "input": {
                "signal_brief": {
                    "confidence": metadata.get("signal_confidence", "high"),
                    "signals": [{"text": signal, "confidence": metadata.get("signal_confidence", "high")} for signal in signals],
                },
                "bench_summary": {"available_capacity": []},
                "requested_capacity": {},
            },
            "ground_truth": {
                "required_signal_terms": signals,
                "max_asks": 2,
                "body_word_limit": 250,
                "subject_char_limit": 60,
            },
        }

    normalized = dict(task)
    if candidate_field not in normalized:
        if "output" in normalized:
            normalized[candidate_field] = normalized["output"]
        elif "reference_output" in normalized:
            normalized[candidate_field] = normalized["reference_output"]
        else:
            normalized[candidate_field] = {"subject": "", "body": ""}
    return normalized


def output_to_subject_body(output: Any) -> Tuple[str, str]:
    """Convert supported output shapes into subject/body strings."""
    if isinstance(output, Mapping):
        return str(output.get("subject", "")).strip(), str(output.get("body", "")).strip()
    text = str(output or "").strip()
    subject = ""
    body = text
    match = re.search(r"^subject:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        subject = match.group(1).strip()
        body = re.sub(r"^subject:\s*.+\n*", "", text, count=1, flags=re.IGNORECASE).strip()
    return subject, body


def score_signal_grounding(task: Mapping[str, Any], text: str, hard_fails: List[str]) -> int:
    """Score whether the draft references supplied public signal facts."""
    terms = required_signal_terms(task)
    normalized = normalize_text(text)
    if not terms:
        return 3

    matches = sum(1 for term in terms if contains_signal(normalized, term))
    coverage = matches / len(terms)

    fabricated_terms = detect_signal_fabrication(task, normalized)
    if fabricated_terms:
        hard_fails.append("signal_fabrication")
        return 1
    if coverage >= 0.9:
        return 5
    if coverage >= 0.5:
        return 4
    if coverage > 0:
        return 3
    return 1


def score_confidence_alignment(task: Mapping[str, Any], text: str, hard_fails: List[str]) -> int:
    """Score whether weak evidence is framed conditionally and high evidence clearly."""
    confidence = signal_confidence(task)
    normalized = normalize_text(text)
    has_conditional = any(term in normalized for term in CONDITIONAL_TERMS)
    weak_assertion = any(term in normalized for term in ASSERTIVE_WEAK_SIGNAL_TERMS)

    if confidence in {"low", "medium"}:
        if weak_assertion and not has_conditional:
            hard_fails.append("weak_signal_assertion")
            return 1
        if has_conditional and not weak_assertion:
            return 5
        if has_conditional:
            return 4
        return 2

    if weak_assertion and "clearly" in normalized:
        return 3
    return 5 if required_signal_terms(task) else 4


def score_operational_safety(
    task: Mapping[str, Any],
    text: str,
    extracted: MutableMapping[str, Any],
    hard_fails: List[str],
) -> int:
    """Score capacity, pricing, and human-routing safety."""
    normalized = normalize_text(text)
    claims = extracted["capacity_claims"]
    request = get_mapping(get_mapping(task, "input"), "requested_capacity")
    must_route = bool(get_mapping(task, "ground_truth").get("must_route_human"))
    route_present = any(term in normalized for term in HUMAN_ROUTE_TERMS)
    commitment_present = bool(claims) or any(term in normalized for term in COMMITMENT_TERMS)
    channel = str(get_mapping(task, "input").get("channel", "email")).lower()
    prior_thread = normalize_text(get_mapping(task, "input").get("prior_thread", ""))
    cold_sms_or_voice = (
        channel == "email"
        and ("opted into sms" not in prior_thread)
        and re.search(r"\b(text you|sms|call your mobile|voice-call|phone you|call you tomorrow)\b", normalized)
    )
    if cold_sms_or_voice:
        hard_fails.append("unsupported_channel")
        return 1

    unsupported_capacity = False
    for claim in claims:
        if not capacity_claim_supported(task, claim):
            unsupported_capacity = True
            break

    if must_route and commitment_present and not route_present:
        unsupported_capacity = True

    if unsupported_capacity:
        hard_fails.append("unsupported_capacity")
        return 1

    if pricing_unsupported(task, extracted["money_amounts"], normalized):
        hard_fails.append("unsupported_pricing")
        return 1

    if must_route:
        return 5 if route_present and not claims else 3
    if request and not commitment_present:
        return 4 if route_present else 3
    return 5


def score_tone_quality(text: str, extracted: Mapping[str, Any], hard_fails: List[str]) -> int:
    """Score Tenacious tone markers using deterministic guardrails."""
    normalized = normalize_text(text)
    banned = extracted["banned_phrases"]
    condescending = [term for term in CONDESCENDING_TERMS if term in normalized]
    external_bench = bool(extracted["mentions_external_bench"])

    if banned:
        hard_fails.append("banned_phrase")
    if external_bench:
        hard_fails.append("external_bench_language")
    if condescending:
        hard_fails.append("condescending_gap_frame")

    penalty = len(banned) + len(condescending) + int(external_bench)
    if penalty >= 3:
        return 1
    if penalty == 2:
        return 2
    if penalty == 1:
        return 3
    if re.search(r"\b(amazing|incredible|unbeatable|guarantee)\b", normalized):
        return 4
    return 5


def score_structure_quality(
    task: Mapping[str, Any],
    subject: str,
    body: str,
    extracted: Mapping[str, Any],
    hard_fails: List[str],
) -> int:
    """Score subject length, body length, signature, and one-ask discipline."""
    ground_truth = get_mapping(task, "ground_truth")
    subject_limit = int(ground_truth.get("subject_char_limit", 60))
    body_limit = int(ground_truth.get("body_word_limit", 120))
    max_asks = int(ground_truth.get("max_asks", 1))

    checks = [
        bool(subject) and len(subject) <= subject_limit,
        1 <= int(extracted["body_word_count"]) <= body_limit,
        int(extracted["ask_count"]) <= max_asks,
        "tenacious" in normalize_text(body) and "gettenacious.com" in normalize_text(body),
    ]
    if int(extracted["ask_count"]) > max_asks:
        hard_fails.append("multi_ask_stack")
    passed = sum(1 for check in checks if check)
    if passed == 4:
        return 5
    if passed == 3:
        return 4
    if passed == 2:
        return 3
    if passed == 1:
        return 2
    return 1


def required_signal_terms(task: Mapping[str, Any]) -> List[str]:
    """Extract the terms that must be grounded in the draft."""
    ground_truth = get_mapping(task, "ground_truth")
    terms = listify(ground_truth.get("required_signal_terms"))
    if terms:
        return terms

    signal_brief = get_mapping(get_mapping(task, "input"), "signal_brief")
    signals = signal_brief.get("signals", [])
    if isinstance(signals, Sequence) and not isinstance(signals, str):
        terms = [
            str(signal.get("text", "")).strip()
            for signal in signals
            if isinstance(signal, Mapping) and str(signal.get("text", "")).strip()
        ]
    if terms:
        return terms

    old_signal = get_mapping(task, "input").get("signal")
    return listify(old_signal)


def signal_confidence(task: Mapping[str, Any]) -> str:
    """Return low/medium/high confidence from supported task shapes."""
    task_input = get_mapping(task, "input")
    signal_brief = get_mapping(task_input, "signal_brief")
    confidence = signal_brief.get("confidence", task_input.get("signal_confidence", "medium"))
    return str(confidence).strip().lower()


def detect_signal_fabrication(task: Mapping[str, Any], normalized_text: str) -> List[str]:
    """Detect common fabricated funding or hiring claims not present in input."""
    source_text = normalize_text(json.dumps(get_mapping(task, "input"), sort_keys=True))
    fabricated = []
    suspicious_patterns = [
        r"\$\s?\d+[\d,.]*\s?(?:m|million|b|billion)?\s+series",
        r"series\s+[abcde]",
        r"\bipo\b",
        r"\bacquired\b",
        r"\b\d+\s+open\s+(?:python|go|data|ml|infra)?\s*roles\b",
    ]
    for pattern in suspicious_patterns:
        for match in re.findall(pattern, normalized_text):
            if match and match not in source_text:
                fabricated.append(match)
    return fabricated


def extract_capacity_claims(text: str) -> List[Dict[str, Any]]:
    """Extract concrete headcount/timeline/stack claims from text."""
    normalized = normalize_text(text)
    claims: List[Dict[str, Any]] = []

    for match in re.finditer(
        r"\b(?P<headcount>\d{1,2})\s+(?P<seniority>junior|mid|senior|architect|mixed)?\s*"
        r"(?P<stack>python|go|data|ml|machine learning|infra|platform|frontend|backend)?\s*"
        r"(?P<role>engineers?|developers?)\b",
        normalized,
    ):
        claims.append(
            {
                "headcount": int(match.group("headcount")),
                "seniority": match.group("seniority") or "unknown",
                "stack": normalize_stack(match.group("stack") or ""),
                "role": match.group("role"),
                "timeline_days": nearest_timeline_days(normalized, match.start()),
            }
        )

    if "within 48 hours" in normalized and not claims:
        claims.append({"headcount": None, "seniority": "unknown", "stack": None, "role": "team", "timeline_days": 2})
    return claims


def nearest_timeline_days(text: str, position: int) -> int | None:
    """Find a nearby start timeline in days."""
    window = text[max(0, position - 120) : position + 160]
    if "within 48 hours" in window:
        return 2
    if "next friday" in window or "next week" in window:
        return 7
    day_match = re.search(r"(?:within|in)\s+(\d{1,3})\s+days?", window)
    if day_match:
        return int(day_match.group(1))
    week_match = re.search(r"(?:within|in)\s+(\d{1,2})\s+weeks?", window)
    if week_match:
        return int(week_match.group(1)) * 7
    return None


def capacity_claim_supported(task: Mapping[str, Any], claim: Mapping[str, Any]) -> bool:
    """Return whether a concrete capacity claim is supported by bench_summary."""
    task_input = get_mapping(task, "input")
    request = get_mapping(task_input, "requested_capacity")
    capacities = capacity_records(task)
    if not capacities:
        return False

    claim_stack = claim.get("stack") or normalize_stack(str(request.get("stack", "")))
    claim_headcount = claim.get("headcount") or request.get("headcount")
    claim_timeline = claim.get("timeline_days") or request.get("timeline_days")

    for capacity in capacities:
        capacity_stack = normalize_stack(str(capacity.get("stack", capacity.get("role", ""))))
        stack_match = not claim_stack or not capacity_stack or claim_stack == capacity_stack
        if not stack_match:
            continue
        available = int_or_default(capacity.get("available_headcount"), 0)
        earliest = int_or_default(capacity.get("earliest_start_days"), 999)
        if claim_headcount is not None and int(claim_headcount) > available:
            continue
        if claim_timeline is not None and int(claim_timeline) < earliest:
            continue
        return True
    return False


def capacity_records(task: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Return capacity records from supported bench summary shapes."""
    bench = get_mapping(get_mapping(task, "input"), "bench_summary")
    if isinstance(bench.get("available_capacity"), list):
        return [record for record in bench["available_capacity"] if isinstance(record, Mapping)]
    if isinstance(bench.get("capacity_constraints"), list):
        return [record for record in bench["capacity_constraints"] if isinstance(record, Mapping)]
    return []


def pricing_unsupported(task: Mapping[str, Any], amounts: Sequence[str], normalized_text: str) -> bool:
    """Return True if the draft quotes unsupported pricing or total contract value."""
    if not amounts:
        return False
    source_text = normalize_text(json.dumps(get_mapping(task, "input"), sort_keys=True))
    non_signal_amounts = [amount for amount in amounts if normalize_money(amount) not in normalize_money(source_text)]
    if not non_signal_amounts:
        return False
    pricing = get_mapping(get_mapping(task, "input"), "pricing_scope")
    allowed = [normalize_money(value) for value in listify(pricing.get("allowed_amounts"))]
    if not allowed:
        return True
    quoted = [normalize_money(value) for value in non_signal_amounts]
    if any(amount not in allowed for amount in quoted):
        return True
    if re.search(r"\b(total contract|tcv|annual contract|multi-phase)\b", normalized_text):
        return True
    return False


def extract_money_amounts(text: str) -> List[str]:
    """Extract dollar-like amounts."""
    return re.findall(r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|m|million))?", normalize_text(text))


def normalize_money(value: Any) -> str:
    """Normalize money strings for safe equality comparison."""
    return re.sub(r"\s+", "", str(value).lower().replace(",", ""))


def count_words(text: str) -> int:
    """Count word-like tokens in text."""
    return len(re.findall(r"\b[\w'-]+\b", text))


def count_asks(text: str) -> int:
    """Count likely calls to action without treating every capability mention as an ask."""
    normalized = normalize_text(text)
    ask_spans = []
    for pattern in ASK_PATTERNS:
        for match in re.finditer(pattern, normalized):
            ask_spans.append((match.start(), match.end()))
    question_hits = normalized.count("?")
    calendar_hits = len(re.findall(r"https?://\S*calendar\S*|cal\.com/\S+", normalized))
    return max(len(ask_spans), question_hits) + calendar_hits


def find_banned_phrases(text: str, banned_phrases: Sequence[str] | None = None) -> List[str]:
    """Return banned phrases found in the text."""
    phrases = banned_phrases or DEFAULT_BANNED_PHRASES
    normalized = normalize_text(text)
    return [phrase for phrase in phrases if normalize_text(phrase) in normalized]


def contains_signal(normalized_text: str, signal: str) -> bool:
    """Return whether a required signal is represented exactly or by key tokens."""
    normalized_signal = normalize_text(signal)
    if not normalized_signal:
        return False
    if normalized_signal in normalized_text:
        return True
    tokens = important_tokens(normalized_signal)
    if not tokens:
        return False
    matched = sum(1 for token in tokens if token in normalized_text)
    return matched / len(tokens) >= 0.7


def important_tokens(text: str) -> List[str]:
    """Extract non-trivial signal tokens."""
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
        "you",
    }
    return [token for token in re.findall(r"\b[\w$.-]+\b", text) if token not in stopwords and len(token) > 1]


def score_word_count(word_count: int, target_min: int = 45, target_max: int = 120) -> float:
    """Continuous compatibility score for the old evaluator wrapper."""
    if word_count <= 0:
        return 0.0
    if target_min <= word_count <= target_max:
        return 1.0
    if word_count < target_min:
        return max(0.0, min(1.0, word_count / target_min))
    return max(0.0, min(1.0, target_max / word_count))


def score_ask_intensity(ask_count: int, ideal_min: int = 1, ideal_max: int = 1) -> float:
    """Continuous compatibility score for the old evaluator wrapper."""
    if ideal_min <= ask_count <= ideal_max:
        return 1.0
    if ask_count <= 0:
        return 0.3
    return max(0.0, min(1.0, ideal_max / ask_count))


def weighted_overall(scores: Mapping[str, int]) -> float:
    """Convert weighted 1-5 scores to a 0-100 score."""
    total = 0.0
    for name, weight in DIMENSION_WEIGHTS.items():
        total += ((scores[name] - 1) / 4) * 100 * weight
    return total


def count_hard_fails(results: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Count hard-fail labels across result records."""
    counts: Dict[str, int] = {}
    for result in results:
        for label in result.get("hard_fail_conditions", []):
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def normalize_text(text: Any) -> str:
    """Normalize text for deterministic matching."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def normalize_stack(stack: str) -> str | None:
    """Normalize stack aliases."""
    value = normalize_text(stack)
    if not value:
        return None
    aliases = {
        "machine learning": "ml",
        "platform": "infra",
        "backend": "python",
    }
    return aliases.get(value, value)


def get_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Safely fetch a nested mapping."""
    value = mapping.get(key) if isinstance(mapping, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def listify(value: Any) -> List[str]:
    """Return a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def int_or_default(value: Any, default: int) -> int:
    """Parse int with fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_tasks(path: Path) -> List[Mapping[str, Any]]:
    """Load a JSON list or a JSON object with a tasks key."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, Mapping):
        data = data.get("tasks", [])
    if not isinstance(data, list):
        raise ValueError(f"Task file must contain a list or a tasks object: {path}")
    return [task for task in data if isinstance(task, Mapping)]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Score Tenacious-Bench tasks deterministically.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_EXAMPLES, help="JSON file containing tasks.")
    parser.add_argument(
        "--candidate-field",
        default="candidate_output",
        help="Task field to score, e.g. candidate_output, reference_output, or negative_output.",
    )
    parser.add_argument("--output", type=Path, help="Optional path for JSON scoring results.")
    parser.add_argument("--quiet", action="store_true", help="Only print aggregate metrics.")
    return parser.parse_args()


def main() -> None:
    """Run evaluator from the command line."""
    args = parse_args()
    tasks = load_tasks(args.tasks)
    results = evaluate_dataset(tasks, candidate_field=args.candidate_field)
    output_text = json.dumps(results["aggregate"] if args.quiet else results, indent=2)
    print(output_text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
