"""Convert train split tasks into DPO JSONL preference rows.

Each output row has:
    {"prompt": "...", "chosen": "...", "rejected": "..."}
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


DEFAULT_INPUT = Path("data/splits/train.json")
DEFAULT_OUTPUT = Path("training_data/dpo_dataset.jsonl")
SYSTEM_MESSAGE = (
    "You are Yabi, a concise Tenacious sales research partner. Write grounded B2B outreach emails. "
    "Never invent signals, funding, pricing, availability, or capacity. Do not use hype, spam phrases, "
    "or the word bench in prospect-facing copy."
)


Task = Mapping[str, Any]


def load_tasks(path: Path) -> List[Task]:
    """Load a JSON array of tasks."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array at {path}")
    return [task for task in data if isinstance(task, Mapping)]


def build_prompt(task: Task) -> str:
    """Combine input fields into a Qwen chat-template prompt string."""
    instruction = build_task_instruction(task)
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_MESSAGE}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{instruction}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Subject: "
    )


def build_task_instruction(task: Task) -> str:
    """Combine input fields into a readable user instruction."""
    task_input = get_mapping(task, "input")
    prospect = get_mapping(task_input, "prospect")
    signal_brief = get_mapping(task_input, "signal_brief")
    requested_capacity = get_mapping(task_input, "requested_capacity")
    bench_summary = get_mapping(task_input, "bench_summary")
    pricing_scope = get_mapping(task_input, "pricing_scope")

    lines = [
        "Write a concise Tenacious outreach email using only the task input.",
        "",
        "Prospect:",
        f"- Company: {prospect.get('company', 'Unknown')}",
        f"- Contact: {prospect.get('contact_name', 'Unknown')}",
        f"- Title: {prospect.get('title', 'Unknown')}",
        f"- Segment: {prospect.get('segment', 'Unknown')}",
        "",
        "Signal:",
        f"- Summary: {signal_brief.get('summary', '')}",
        f"- Confidence: {signal_brief.get('confidence', '')}",
        f"- Source window: {signal_brief.get('source_window', '')}",
    ]

    signals = signal_brief.get("signals", [])
    if isinstance(signals, Sequence) and not isinstance(signals, str):
        for signal in signals:
            if isinstance(signal, Mapping):
                lines.append(f"- Signal detail: {signal.get('text', '')} ({signal.get('confidence', '')})")

    lines.extend(
        [
            "",
            "Requested capacity:",
            f"- Headcount: {requested_capacity.get('headcount', '')}",
            f"- Role: {requested_capacity.get('role', '')}",
            f"- Stack: {requested_capacity.get('stack', '')}",
            f"- Seniority: {requested_capacity.get('seniority', '')}",
            f"- Timeline days: {requested_capacity.get('timeline_days', '')}",
            f"- Timezone overlap hours: {requested_capacity.get('timezone_overlap_hours', '')}",
            "",
            "Available capacity summary:",
        ]
    )

    capacities = bench_summary.get("available_capacity", [])
    if isinstance(capacities, Sequence) and not isinstance(capacities, str):
        for capacity in capacities:
            if isinstance(capacity, Mapping):
                lines.append(
                    "- "
                    f"{capacity.get('available_headcount', '')} "
                    f"{capacity.get('seniority', '')} "
                    f"{capacity.get('stack', '')} "
                    f"{capacity.get('role', '')}, earliest start "
                    f"{capacity.get('earliest_start_days', '')} days, "
                    f"{capacity.get('timezone_overlap_hours', '')} hours overlap"
                )
    notes = bench_summary.get("notes")
    if notes:
        lines.append(f"- Notes: {notes}")

    lines.extend(
        [
            "",
            "Pricing scope:",
            f"- Allowed amounts: {pricing_scope.get('allowed_amounts', [])}",
            f"- Allowed phrases: {pricing_scope.get('allowed_phrases', [])}",
            f"- Can quote total contract value: {pricing_scope.get('can_quote_total_contract_value', False)}",
            "",
            f"Prior thread: {task_input.get('prior_thread', '')}",
            f"Channel: {task_input.get('channel', '')}",
            f"Goal: {task_input.get('outreach_goal', '')}",
            "",
            "Return only the email. Keep it under 120 words. Include exactly one subject line.",
            "Use this structure:",
            "Subject: <short subject>",
            "",
            "Hi <contact>,",
            "",
            "<2 short grounded paragraphs>",
            "",
            "<one clear 15-minute CTA>",
            "",
            "Best,",
            "Yabi",
            "Research Partner, Tenacious Intelligence Corporation",
            "gettenacious.com",
        ]
    )
    return "\n".join(line.rstrip() for line in lines if line is not None).strip()


def choose_output(task: Task) -> str:
    """Return the good email for DPO chosen output."""
    for field in ("reference_output", "candidate_output", "output"):
        if field in task:
            text = output_to_text(task[field])
            if text:
                return text
    raise ValueError(f"Task {task.get('task_id', '<missing>')} has no output field.")


def make_rejected(task: Task, chosen: str, index: int) -> Tuple[str, str]:
    """Create a weaker rejected output using one of three deterministic strategies."""
    strategies = [
        ("remove_signal", remove_signal_mentions),
        ("make_generic", make_generic_email),
        ("remove_cta", remove_cta),
    ]
    strategy_name, strategy = strategies[index % len(strategies)]
    rejected = strategy(task, chosen)
    if normalize_space(rejected) == normalize_space(chosen):
        rejected = make_generic_email(task, chosen)
        strategy_name = "make_generic_fallback"
    return rejected, strategy_name


def remove_signal_mentions(task: Task, chosen: str) -> str:
    """Remove paragraphs that mention required signal terms."""
    signal_terms = required_signal_terms(task)
    if not signal_terms:
        return make_generic_email(task, chosen)

    paragraphs = chosen.split("\n\n")
    kept = []
    removed_any = False
    for paragraph in paragraphs:
        normalized = normalize_text(paragraph)
        if any(normalize_text(term) and normalize_text(term) in normalized for term in signal_terms):
            removed_any = True
            continue
        kept.append(paragraph)

    if not removed_any:
        return make_generic_email(task, chosen)

    rejected = "\n\n".join(kept).strip()
    return insert_generic_signal_sentence(task, rejected)


def make_generic_email(task: Task, _chosen: str) -> str:
    """Return a generic outreach email with little signal grounding."""
    prospect = get_mapping(get_mapping(task, "input"), "prospect")
    contact = prospect.get("contact_name") or "there"
    company = prospect.get("company") or "your team"
    return (
        "Subject: Quick question\n\n"
        f"Hi {contact},\n\n"
        f"I wanted to reach out because Tenacious helps teams like {company} move faster with flexible "
        "engineering support. We can share how we usually support roadmap capacity when priorities change.\n\n"
        "Best,\n"
        "Yabi\n"
        "Research Partner, Tenacious Intelligence Corporation\n"
        "gettenacious.com"
    )


def remove_cta(_task: Task, chosen: str) -> str:
    """Remove paragraphs that look like calls to action."""
    cta_terms = [
        "would ",
        "could we",
        "can we",
        "are you open",
        "15 minutes",
        "15-minute",
        "call",
        "chat",
        "meeting",
        "calendar",
        "reply",
    ]
    paragraphs = chosen.split("\n\n")
    kept = []
    for paragraph in paragraphs:
        normalized = normalize_text(paragraph)
        if "?" in paragraph or any(term in normalized for term in cta_terms):
            continue
        kept.append(paragraph)
    return "\n\n".join(kept).strip() or make_generic_email(_task, chosen)


def insert_generic_signal_sentence(task: Task, text: str) -> str:
    """Insert a vague sentence so the rejected email remains readable."""
    prospect = get_mapping(get_mapping(task, "input"), "prospect")
    company = prospect.get("company") or "your team"
    generic_sentence = f"I saw a recent update from {company} and thought it might be relevant."

    paragraphs = text.split("\n\n")
    if len(paragraphs) >= 2:
        paragraphs.insert(2, generic_sentence)
        return "\n\n".join(paragraphs).strip()
    return f"{text}\n\n{generic_sentence}".strip()


def required_signal_terms(task: Task) -> List[str]:
    """Extract signal terms from ground truth or input."""
    ground_truth = get_mapping(task, "ground_truth")
    terms = listify(ground_truth.get("required_signal_terms"))
    if terms:
        return terms

    signal_brief = get_mapping(get_mapping(task, "input"), "signal_brief")
    terms = listify(signal_brief.get("summary"))
    signals = signal_brief.get("signals", [])
    if isinstance(signals, Sequence) and not isinstance(signals, str):
        for signal in signals:
            if isinstance(signal, Mapping):
                terms.extend(listify(signal.get("text")))
    return terms


def write_jsonl(rows: Sequence[Mapping[str, str]], path: Path) -> None:
    """Write DPO rows as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def output_to_text(output: Any) -> str:
    """Convert string or {subject, body} output into one email string."""
    if isinstance(output, Mapping):
        subject = str(output.get("subject", "")).strip()
        body = str(output.get("body", "")).strip()
        if subject:
            return f"Subject: {subject}\n\n{body}".strip()
        return body
    return str(output or "").strip()


def strip_subject_prefix(text: str) -> str:
    """Remove the literal Subject: prefix because the prompt already provides it."""
    text = text.strip()
    if text.lower().startswith("subject:"):
        return text.split(":", 1)[1].lstrip()
    return text


def get_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Safely fetch a nested mapping."""
    value = mapping.get(key) if isinstance(mapping, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def listify(value: Any) -> List[str]:
    """Return a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def normalize_text(text: Any) -> str:
    """Normalize text for matching."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def normalize_space(text: str) -> str:
    """Normalize whitespace without changing case."""
    return re.sub(r"\s+", " ", text).strip()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Create DPO JSONL from train tasks."""
    args = parse_args()
    tasks = load_tasks(args.input)
    rows: List[Dict[str, str]] = []
    strategy_counts: Counter[str] = Counter()

    for index, task in enumerate(tasks):
        chosen = strip_subject_prefix(choose_output(task))
        rejected, strategy = make_rejected(task, chosen, index)
        rejected = strip_subject_prefix(rejected)
        rows.append(
            {
                "prompt": build_prompt(task),
                "chosen": chosen,
                "rejected": rejected,
            }
        )
        strategy_counts[strategy] += 1

    write_jsonl(rows, args.output)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "examples": len(rows),
                "rejected_strategies": dict(sorted(strategy_counts.items())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
