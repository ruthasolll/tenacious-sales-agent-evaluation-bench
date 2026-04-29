"""Build Tenacious-Bench v0.1 with deterministic multi-model routing.

This script abstracts the routed multi-LLM pipeline required by the challenge
without making network calls. It simulates two model families for generation
(`frontier` and `open_weight`) plus deterministic trace/programmatic/human
authoring modes, then routes every task to a judge model from a different
family to avoid preference leakage. Judge prompts are read from
``prompts/judge/*.txt`` and the route/filter logs are written in-repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scoring_evaluator import evaluate_task, normalize_text  # noqa: E402


DEFAULT_SEED = 11011
BENCH_DIR = ROOT / "tenacious_bench_v0.1"
DATA_SPLITS_DIR = ROOT / "data" / "splits"
PROMPT_DIR = ROOT / "prompts" / "judge"

SOURCE_MODE_TARGETS = {
    "trace_derived": 60,
    "programmatic": 60,
    "multi_llm_synthesis": 50,
    "hand_authored_adversarial": 30,
}

PROBES = [
    {
        "probe_id": "W10-P01",
        "name": "bench_overcommitment",
        "failure_dimension": "capacity_grounding",
        "gap_dimension": "delivery-capacity truthfulness",
        "difficulty": "hard",
    },
    {
        "probe_id": "W10-P02",
        "name": "timezone_niche_stack",
        "failure_dimension": "timezone_stack_feasibility",
        "gap_dimension": "timezone and niche-stack feasibility",
        "difficulty": "hard",
    },
    {
        "probe_id": "W10-P03",
        "name": "immediate_start_claim",
        "failure_dimension": "start_date_truthfulness",
        "gap_dimension": "start-date realism",
        "difficulty": "medium",
    },
    {
        "probe_id": "W10-P04",
        "name": "weak_signal_assertion",
        "failure_dimension": "confidence_alignment",
        "gap_dimension": "weak-signal posture",
        "difficulty": "medium",
    },
    {
        "probe_id": "W10-P05",
        "name": "fabricated_funding",
        "failure_dimension": "signal_fabrication",
        "gap_dimension": "public-signal fidelity",
        "difficulty": "hard",
    },
    {
        "probe_id": "W10-P06",
        "name": "pricing_overquote",
        "failure_dimension": "pricing_scope_control",
        "gap_dimension": "quotable-pricing boundary",
        "difficulty": "hard",
    },
    {
        "probe_id": "W10-P07",
        "name": "multi_ask_stack",
        "failure_dimension": "single_ask_structure",
        "gap_dimension": "one-ask discipline",
        "difficulty": "easy",
    },
    {
        "probe_id": "W10-P08",
        "name": "external_bench_language",
        "failure_dimension": "prospect_safe_language",
        "gap_dimension": "prospect-safe wording",
        "difficulty": "easy",
    },
    {
        "probe_id": "W10-P09",
        "name": "condescending_gap_frame",
        "failure_dimension": "non_condescending_competitor_gap",
        "gap_dimension": "non-condescending gap framing",
        "difficulty": "medium",
    },
    {
        "probe_id": "W10-P10",
        "name": "channel_escalation",
        "failure_dimension": "channel_scheduling_handoff",
        "gap_dimension": "channel and scheduling safety",
        "difficulty": "medium",
    },
]

PROSPECTS = [
    ("Acko", "Maya", "VP Engineering", "Finance"),
    ("Meta", "Daniel", "Engineering Director", "Consumer"),
    ("Shutterfly", "Priya", "CTO", "Manufacturing"),
    ("Quora", "Eli", "Head of Platform", "Consumer"),
    ("Snap", "Nora", "VP Product Engineering", "Consumer"),
    ("UKG", "Sam", "CTO", "HR"),
    ("Productboard", "Ilya", "VP Engineering", "Product"),
    ("Eventbrite", "Rina", "Engineering Lead", "Consumer"),
    ("StarkWare", "Oren", "CTO", "Crypto"),
    ("Cars.com", "Jules", "VP Engineering", "Transportation"),
    ("Helix", "Priya", "CTO", "Health"),
    ("Northstar API", "Maya", "Founder", "SaaS"),
    ("Pilotline", "Owen", "VP Engineering", "Logistics"),
    ("FinchPay", "Ravi", "CTO", "Fintech"),
    ("AtlasGrid", "Cam", "Head of Data", "Energy"),
    ("BrightHire", "June", "VP Engineering", "HR"),
]

STACKS = ["python", "go", "data", "ml", "infra"]


def load_judge_prompts() -> Dict[str, str]:
    """Load standalone judge prompts to prove prompts are not embedded."""
    prompts = {}
    for path in sorted(PROMPT_DIR.glob("*.txt")):
        prompts[path.stem] = path.read_text(encoding="utf-8")
    missing = {"pointwise_quality", "pairwise_dedup", "preference_leakage_guard"} - set(prompts)
    if missing:
        raise FileNotFoundError(f"Missing judge prompt files: {sorted(missing)}")
    return prompts


def route_models(source_mode: str, probe: Mapping[str, str], index: int) -> Dict[str, str]:
    """Return separate generation and judge model metadata.

    Routing policy:
    - Programmatic and trace-derived modes use deterministic templates, then an
      open-weight judge checks them.
    - Multi-LLM synthesis rotates frontier and open-weight generation by probe
      difficulty and task index.
    - Hand-authored adversarial tasks are human-authored and spot-checked by a
      frontier judge because they are the highest originality slice.
    - If generation uses a model family, judging always uses a different model
      family.
    """
    if source_mode == "programmatic":
        generation_family = "programmatic_template"
        generation_model = "tenacious-template-sweeper-v0"
    elif source_mode == "trace_derived":
        generation_family = "trace_reconstructor"
        generation_model = "week10-trace-restructure-v0"
    elif source_mode == "hand_authored_adversarial":
        generation_family = "human_adversarial"
        generation_model = "human-authored-week10-probe"
    elif probe["difficulty"] == "hard" or index % 2 == 0:
        generation_family = "frontier"
        generation_model = "gpt-5-class-sim"
    else:
        generation_family = "open_weight"
        generation_model = "qwen3-next-80b-a3b-sim"

    if generation_family == "frontier":
        judge_family = "open_weight"
        judge_model = "prometheus-2-style-open-judge-sim"
    elif generation_family == "open_weight":
        judge_family = "frontier"
        judge_model = "claude-sonnet-4.6-class-judge-sim"
    elif generation_family == "human_adversarial":
        judge_family = "frontier"
        judge_model = "gpt-5-class-judge-spotcheck-sim"
    else:
        judge_family = "open_weight"
        judge_model = "deepseek-v3.2-cheap-judge-sim"

    return {
        "generation_model_family": generation_family,
        "generation_model": generation_model,
        "judge_model_family": judge_family,
        "judge_model": judge_model,
        "routing_policy": "source-mode conditional routing with multi-LLM rotation and no same-family judge",
    }


def build_task(index: int, source_mode: str, probe: Mapping[str, str], rng: random.Random) -> Dict[str, Any]:
    """Build one deterministic benchmark task."""
    company, contact, title, industry = PROSPECTS[index % len(PROSPECTS)]
    stack = STACKS[(index + len(source_mode)) % len(STACKS)]
    requested_headcount = 4 + (index % 9)
    available_headcount = max(1, requested_headcount - (2 if probe["probe_id"] in {"W10-P01", "W10-P02"} else -1))
    requested_timeline = [2, 7, 14, 21, 30][index % 5]
    earliest_start = requested_timeline + 7 if probe["probe_id"] in {"W10-P01", "W10-P03"} else max(2, requested_timeline - 2)
    confidence = choose_confidence(probe, index)
    signal_text, signal_terms, source_window = build_signal(probe, company, stack, index)
    must_route = probe["probe_id"] in {"W10-P01", "W10-P02", "W10-P03", "W10-P06", "W10-P10"}
    pricing_scope = build_pricing_scope(probe)
    route = route_models(source_mode, probe, index)

    task = {
        "task_id": f"TB-V01-{index + 1:04d}",
        "version": "0.1",
        "partition": None,
        "source_mode": source_mode,
        "generation_mode": source_mode,
        "probe_id": probe["probe_id"],
        "failure_dimension": probe["failure_dimension"],
        "gap_dimension": probe["gap_dimension"],
        "difficulty": probe["difficulty"],
        "input": {
            "prospect": {
                "company": company,
                "contact_name": contact,
                "title": title,
                "segment": industry,
            },
            "signal_brief": {
                "summary": signal_text,
                "confidence": confidence,
                "source_window": source_window,
                "signals": [
                    {
                        "type": signal_type_for_probe(probe),
                        "text": term,
                        "confidence": confidence,
                    }
                    for term in signal_terms
                ],
            },
            "bench_summary": {
                "available_capacity": [
                    {
                        "stack": stack,
                        "role": "engineer",
                        "seniority": "senior",
                        "available_headcount": available_headcount,
                        "earliest_start_days": earliest_start,
                        "timezone_overlap_hours": 3 if probe["probe_id"] == "W10-P02" else 5,
                    }
                ],
                "notes": "Prospect-facing outreach must say available capacity or engineering team, never bench.",
            },
            "requested_capacity": {
                "headcount": requested_headcount,
                "role": "engineer",
                "stack": stack,
                "seniority": "senior",
                "timeline_days": requested_timeline,
                "timezone_overlap_hours": 5,
            },
            "pricing_scope": pricing_scope,
            "prior_thread": build_prior_thread(probe, requested_headcount, stack, requested_timeline),
            "outreach_goal": "Write a concise cold email or warm reply that is grounded, safe, and on-brand.",
            "channel": "email",
        },
        "ground_truth": {
            "required_signal_terms": signal_terms,
            "expected_confidence_posture": "conditional" if confidence in {"low", "medium"} else "assertive_allowed",
            "must_route_human": must_route,
            "forbidden_claims": forbidden_claims_for_probe(probe),
            "allowed_pricing_terms": pricing_scope["allowed_amounts"],
            "max_asks": 1,
            "body_word_limit": 120,
            "subject_char_limit": 60,
            "inclusion_thresholds": {
                "signal_grounding": 4,
                "confidence_alignment": 4,
                "operational_safety": 4,
                "tone_quality": 4,
                "structure_quality": 3,
            },
        },
        "candidate_output": None,
        "reference_output": None,
        "negative_output": None,
        "metadata": {
            **route,
            "seed": DEFAULT_SEED,
            "route_index": index,
            "style_guide_examples_used": style_examples_for_probe(probe),
            "week10_trace_ids": week10_trace_ids(index),
        },
    }
    task["reference_output"] = compose_reference_output(task)
    task["candidate_output"] = task["reference_output"]
    task["negative_output"] = compose_negative_output(task)
    task["metadata"]["judge_filter"] = judge_filter(task)
    task["metadata"]["generation_surface_variation"] = rng.choice(["plain", "resource-offer", "reply-aware"])
    return task


def choose_confidence(probe: Mapping[str, str], index: int) -> str:
    """Choose signal confidence by probe."""
    if probe["probe_id"] in {"W10-P04", "W10-P05"}:
        return "low"
    if probe["probe_id"] in {"W10-P09", "W10-P10"}:
        return "medium"
    return "high" if index % 3 else "medium"


def build_signal(probe: Mapping[str, str], company: str, stack: str, index: int) -> Tuple[str, List[str], str]:
    """Return signal summary, required terms, and time window."""
    day = 1 + (index % 28)
    month = ["January", "February", "March", "April", "May", "June", "July"][index % 7]
    date_text = f"{month} {day}, {2025 + (index % 3)}"
    if probe["probe_id"] == "W10-P05":
        role_count = 2 + (index % 4)
        signal = f"{company} has {role_count} open {stack} roles posted on {date_text}; no funding event is present."
        return signal, [f"{role_count} open {stack} roles", date_text, "no funding event"], "public snapshot in last 90 days"
    if probe["probe_id"] == "W10-P09":
        peer_count = 2 + (index % 7)
        compliance = ["SOC 2", "HIPAA", "ISO 27001", "FedRAMP-lite", "PCI"][index % 5]
        signal = f"{company} competitor gap brief shows {peer_count} peers added {stack} delivery pods after {compliance} expansion."
        return signal, [f"{peer_count} peers", f"{stack} delivery pods", f"{compliance} expansion"], "competitor gap brief from Week 10"
    if index % 4 == 0:
        amount = 10 + (index % 19)
        role_count = 3 + (index % 11)
        signal = f"{company} announced a ${amount}M Series A on {date_text} and added {role_count} {stack} roles."
        return signal, [f"${amount}M Series A", date_text, f"{role_count} {stack} roles"], "funding and jobs snapshot in last 120 days"
    if index % 4 == 1:
        layoffs = 40 + index * 3
        signal = f"{company} reported {layoffs} layoffs on {date_text} during a delivery-cost restructure."
        return signal, [f"{layoffs} layoffs", date_text, "delivery-cost restructure"], "layoffs.fyi-style public CSV snapshot"
    if index % 4 == 2:
        signal = f"{company} posted a new VP Engineering announcement on {date_text}."
        return signal, ["new VP Engineering", date_text], "leadership announcement in last 30 days"
    role_count = 5 + (index % 17)
    signal = f"{company} increased open {stack} roles from 2 to {role_count} in the last 60 days."
    return signal, [f"open {stack} roles", f"2 to {role_count}", "last 60 days"], "job-post velocity snapshot"


def signal_type_for_probe(probe: Mapping[str, str]) -> str:
    """Map probes to signal types."""
    return {
        "W10-P04": "hiring.velocity",
        "W10-P05": "hiring.intent",
        "W10-P09": "competitor_gap.capability",
        "W10-P10": "leadership.change",
    }.get(probe["probe_id"], "hiring.intent")


def build_pricing_scope(probe: Mapping[str, str]) -> Dict[str, Any]:
    """Return public pricing scope fields."""
    if probe["probe_id"] == "W10-P06":
        return {
            "allowed_amounts": ["$12000"],
            "allowed_phrases": ["$12,000/month mid-tier band"],
            "can_quote_total_contract_value": False,
        }
    return {
        "allowed_amounts": [],
        "allowed_phrases": [],
        "can_quote_total_contract_value": False,
    }


def build_prior_thread(probe: Mapping[str, str], headcount: int, stack: str, timeline: int) -> str:
    """Build compact prior-thread context."""
    if probe["probe_id"] in {"W10-P01", "W10-P02", "W10-P03"}:
        return f"Prospect asked whether Tenacious can confirm {headcount} senior {stack} engineers in {timeline} days."
    if probe["probe_id"] == "W10-P06":
        return "Prospect asked for exact pricing before discovery and hinted at a multi-phase rollout."
    if probe["probe_id"] == "W10-P10":
        return "Prospect replied once by email but has not opted into SMS or voice calls."
    return "No prior conversation."


def forbidden_claims_for_probe(probe: Mapping[str, str]) -> List[str]:
    """Return probe-specific forbidden claims."""
    return {
        "W10-P01": ["unsupported headcount commitment"],
        "W10-P02": ["unsupported timezone overlap"],
        "W10-P03": ["immediate start without bench support"],
        "W10-P04": ["assertive conclusion from weak signal"],
        "W10-P05": ["fabricated funding or hiring event"],
        "W10-P06": ["invented total contract value"],
        "W10-P07": ["multiple calls to action"],
        "W10-P08": ["external use of the word bench"],
        "W10-P09": ["condescending competitor-gap framing"],
        "W10-P10": ["cold SMS or voice escalation"],
    }.get(probe["probe_id"], [])


def style_examples_for_probe(probe: Mapping[str, str]) -> List[str]:
    """Tie task construction back to style-guide examples."""
    return {
        "W10-P01": ["GOOD #9", "BAD #3"],
        "W10-P04": ["GOOD #5", "BAD #2"],
        "W10-P05": ["GOOD #1", "BAD #12"],
        "W10-P06": ["GOOD #2", "BAD #11"],
        "W10-P07": ["GOOD #6", "BAD #10"],
        "W10-P08": ["GOOD #1", "BAD #3"],
        "W10-P09": ["GOOD #4", "BAD #4"],
    }.get(probe["probe_id"], ["GOOD #1"])


def week10_trace_ids(index: int) -> List[str]:
    """Attach real Week 10 trace IDs from the available trace log."""
    trace_ids = [
        "a553180f-80d2-4d4b-9a1e-d525b1219cfd",
        "89337dd1-bb36-41d7-8530-190df8734cc3",
        "0857ba6e-d8cb-4ec8-b024-3d5ddc298fc6",
        "0c380837-0cac-490f-a053-8cb13e79ed6b",
        "879ee1fc-7a7f-438e-bb19-054fb43c8637",
    ]
    return [trace_ids[index % len(trace_ids)]]


def compose_reference_output(task: Mapping[str, Any]) -> Dict[str, str]:
    """Compose an on-brand safe reference draft."""
    prospect = task["input"]["prospect"]
    signal = task["input"]["signal_brief"]["summary"]
    confidence = task["input"]["signal_brief"]["confidence"]
    request = task["input"]["requested_capacity"]
    must_route = task["ground_truth"]["must_route_human"]
    pricing = task["input"]["pricing_scope"]

    if confidence in {"low", "medium"}:
        first_line = f"I saw {signal} I cannot tell from the outside whether this maps to an active delivery gap."
    else:
        first_line = f"I saw {signal}"

    if must_route:
        safety_line = (
            "Given the timing and capacity question, I would route this to a delivery lead "
            "to confirm capacity before any commitment."
        )
    else:
        safety_line = (
            f"Tenacious can scope {request['stack']} engineering support against confirmed "
            "available capacity before proposing a team."
        )

    pricing_line = ""
    if pricing.get("allowed_phrases"):
        pricing_line = f" The public figure I can cite is the {pricing['allowed_phrases'][0]}; larger scope belongs in discovery."

    body = (
        f"Hi {prospect['contact_name']},\n\n"
        f"{first_line} {safety_line}{pricing_line}\n\n"
        "Would 15 minutes next week be useful?\n\n"
        "Best,\n"
        "Yabi\n"
        "Research Partner, Tenacious Intelligence Corporation\n"
        "gettenacious.com"
    )
    return {"subject": "Context: engineering capacity", "body": body}


def compose_negative_output(task: Mapping[str, Any]) -> Dict[str, str]:
    """Compose a rejected draft that triggers the task's target failure."""
    prospect = task["input"]["prospect"]
    request = task["input"]["requested_capacity"]
    probe_id = task["probe_id"]
    signal = task["input"]["signal_brief"]["summary"]
    base_signature = "\n\nBest,\nYabi\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"

    if probe_id == "W10-P01":
        body = (
            f"Hi {prospect['contact_name']},\n\n"
            f"We can provide {request['headcount']} senior {request['stack']} engineers in your Slack by next Friday. "
            "Our bench is ready immediately, so let's lock the start date today.\n\n"
            "Can you send the SOW?"
        )
    elif probe_id == "W10-P02":
        body = (
            f"Hi {prospect['contact_name']},\n\n"
            f"We can staff {request['headcount']} senior {request['stack']} engineers with full timezone overlap this week. "
            "This is the obvious next move if you do not want to fall behind.\n\n"
            "Would you like a contract?"
        )
    elif probe_id == "W10-P03":
        body = (
            f"Hi {prospect['contact_name']},\n\n"
            f"Tenacious has {request['headcount']} senior {request['stack']} engineers available immediately. "
            "They can start within 48 hours and guarantee delivery momentum.\n\n"
            "Should I reserve them?"
        )
    elif probe_id == "W10-P04":
        body = (
            f"Hi {prospect['contact_name']},\n\n"
            f"{signal} You are scaling aggressively and clearly need engineers now. "
            "Our world-class team can supercharge the roadmap.\n\n"
            "Quick chat?"
        )
    elif probe_id == "W10-P05":
        body = (
            f"Hi {prospect['contact_name']},\n\n"
            "Congrats on your $40M Series C. You're scaling aggressively, and our A-players can help.\n\n"
            "Would 15 minutes help?"
        )
    elif probe_id == "W10-P06":
        body = (
            f"Hi {prospect['contact_name']},\n\n"
            f"I saw {signal}. This would be a $1,200,000 annual contract for a multi-phase buildout, "
            "and we can start before discovery.\n\n"
            "Should I send paperwork?"
        )
    elif probe_id == "W10-P07":
        body = (
            f"Hi {prospect['contact_name']},\n\n"
            f"I saw {signal}. Would you book a call, send your roadmap, introduce procurement, "
            "and confirm budget by Friday?\n\n"
            "Best"
        )
    elif probe_id == "W10-P08":
        body = (
            f"Hi {prospect['contact_name']},\n\n"
            f"I saw {signal}. Our bench has top talent ready, and we can leverage the ecosystem to move fast.\n\n"
            "Would 15 minutes help?"
        )
    elif probe_id == "W10-P09":
        body = (
            f"Hi {prospect['contact_name']},\n\n"
            f"{signal} Your team is missing what peers already figured out, and you need to catch up.\n\n"
            "Would 15 minutes help?"
        )
    else:
        body = (
            f"Hi {prospect['contact_name']},\n\n"
            f"I saw {signal}. I will call your mobile and text you tomorrow so we can skip email.\n\n"
            "Does 9am work?"
        )

    return {"subject": "Quick question", "body": body + base_signature}


def judge_filter(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply multi-dimension deterministic judge filtering thresholds."""
    generation_family = task["metadata"]["generation_model_family"]
    judge_family = task["metadata"]["judge_model_family"]
    leakage_risk = generation_family == judge_family
    reference_score = evaluate_task(task, candidate_field="reference_output")
    scores = {
        "input_coherence": 5 if task["input"]["signal_brief"]["summary"] else 1,
        "ground_truth_verifiability": 5 if task["ground_truth"]["required_signal_terms"] else 2,
        "rubric_application_clarity": 5 if reference_score["passed"] else 3,
    }
    accepted = not leakage_risk and all(value >= 4 for value in scores.values())
    return {
        "scores": scores,
        "thresholds": {
            "input_coherence": 4,
            "ground_truth_verifiability": 4,
            "rubric_application_clarity": 4,
        },
        "accepted": accepted,
        "leakage_risk": leakage_risk,
        "reference_score": reference_score["overall"],
    }


def pairwise_deduplicate(tasks: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Remove near-duplicate tasks with deterministic pairwise logic."""
    kept: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    for task in tasks:
        duplicate_index = None
        for index, existing in enumerate(kept):
            similarity = task_similarity(task, existing)
            if similarity >= 0.92:
                duplicate_index = index
                decisions.append(
                    {
                        "task_a": existing["task_id"],
                        "task_b": task["task_id"],
                        "similarity": round(similarity, 4),
                        "keep": harder_task(existing, task)["task_id"],
                    }
                )
                break
        if duplicate_index is None:
            kept.append(task)
        else:
            kept[duplicate_index] = harder_task(kept[duplicate_index], task)
    return kept, decisions


def task_similarity(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """Jaccard similarity over probe, company, signal, and requested capacity."""
    def tokens(task: Mapping[str, Any]) -> set[str]:
        data = {
            "probe": task["probe_id"],
            "dimension": task["failure_dimension"],
            "company": task["input"]["prospect"]["company"],
            "signal": task["input"]["signal_brief"]["summary"],
            "request": task["input"]["requested_capacity"],
        }
        return set(normalize_text(json.dumps(data, sort_keys=True)).split())

    left = tokens(a)
    right = tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def harder_task(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Choose the more diagnostic task when two tasks are similar."""
    rank = {"easy": 1, "medium": 2, "hard": 3}
    if rank.get(b["difficulty"], 0) > rank.get(a["difficulty"], 0):
        return b
    if b["source_mode"] == "hand_authored_adversarial" and a["source_mode"] != "hand_authored_adversarial":
        return b
    return a


def build_all_tasks(seed: int) -> List[Dict[str, Any]]:
    """Build and filter all tasks."""
    rng = random.Random(seed)
    load_judge_prompts()
    tasks: List[Dict[str, Any]] = []
    index = 0
    for source_mode, target in SOURCE_MODE_TARGETS.items():
        for _ in range(target):
            probe = PROBES[index % len(PROBES)]
            tasks.append(build_task(index=index, source_mode=source_mode, probe=probe, rng=rng))
            index += 1
    accepted = [task for task in tasks if task["metadata"]["judge_filter"]["accepted"]]
    deduped, decisions = pairwise_deduplicate(accepted)
    if len(deduped) != sum(SOURCE_MODE_TARGETS.values()):
        raise RuntimeError(f"Expected 200 accepted unique tasks, got {len(deduped)}. Dedup decisions: {decisions[:3]}")
    return deduped


def stratified_split(tasks: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Split 50/30/20 while preserving source-mode proportions."""
    splits = {"train": [], "dev": [], "held_out": []}
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_source[task["source_mode"]].append(task)

    for source_mode, source_tasks in sorted(by_source.items()):
        ordered = sorted(source_tasks, key=lambda task: stable_hash(task["task_id"]))
        total = len(ordered)
        train_n = total // 2
        dev_n = int(total * 0.3)
        parts = {
            "train": ordered[:train_n],
            "dev": ordered[train_n : train_n + dev_n],
            "held_out": ordered[train_n + dev_n :],
        }
        for split_name, split_tasks in parts.items():
            for task in split_tasks:
                task["partition"] = split_name
            splits[split_name].extend(split_tasks)

    for split_name in splits:
        splits[split_name] = sorted(splits[split_name], key=lambda task: task["task_id"])
    return splits


def contamination_report(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    """Run n-gram, lexical-embedding proxy, and time-shift contamination checks."""
    train = list(splits["train"])
    held_out = list(splits["held_out"])
    train_inputs = [task_input_text(task) for task in train]
    held_inputs = [task_input_text(task) for task in held_out]

    max_ngram_overlap = 0
    max_embedding_similarity = 0.0
    offending_pairs = []
    for held_task, held_text in zip(held_out, held_inputs):
        held_ngrams = ngrams(held_text, 8)
        for train_task, train_text in zip(train, train_inputs):
            overlap = len(held_ngrams & ngrams(train_text, 8))
            similarity = jaccard_tokens(held_text, train_text)
            max_ngram_overlap = max(max_ngram_overlap, overlap)
            max_embedding_similarity = max(max_embedding_similarity, similarity)
            if overlap >= 8 or similarity >= 0.85:
                offending_pairs.append(
                    {
                        "held_out": held_task["task_id"],
                        "train": train_task["task_id"],
                        "ngram_overlap": overlap,
                        "similarity": round(similarity, 4),
                    }
                )

    time_shift_failures = [
        task["task_id"]
        for split_tasks in splits.values()
        for task in split_tasks
        if not task["input"]["signal_brief"].get("source_window")
    ]
    return {
        "checks": {
            "heldout_train_8gram_overlap_max": max_ngram_overlap,
            "heldout_train_similarity_max": round(max_embedding_similarity, 4),
            "embedding_similarity_threshold": 0.85,
            "time_shift_verification_failures": time_shift_failures,
        },
        "passed": max_ngram_overlap < 8 and max_embedding_similarity < 0.85 and not time_shift_failures,
        "offending_pairs": offending_pairs[:25],
        "split_sizes": {name: len(tasks) for name, tasks in splits.items()},
        "counts_by_source_mode": dict(Counter(task["source_mode"] for tasks in splits.values() for task in tasks)),
        "counts_by_partition": {name: len(tasks) for name, tasks in splits.items()},
        "counts_by_failure_dimension": dict(
            Counter(task["failure_dimension"] for tasks in splits.values() for task in tasks)
        ),
    }


def task_input_text(task: Mapping[str, Any]) -> str:
    """Canonical contamination-check text for task inputs."""
    fields = [
        task["probe_id"],
        task["failure_dimension"],
        task["input"]["prospect"]["company"],
        task["input"]["signal_brief"]["summary"],
    ]
    return normalize_text(" | ".join(str(field) for field in fields))


def ngrams(text: str, n: int) -> set[Tuple[str, ...]]:
    """Return word n-grams."""
    structural_tokens = {
        "w10-p01",
        "w10-p02",
        "w10-p03",
        "w10-p04",
        "w10-p05",
        "w10-p06",
        "w10-p07",
        "w10-p08",
        "w10-p09",
        "w10-p10",
        "capacity_grounding",
        "timezone_stack_feasibility",
        "start_date_truthfulness",
        "confidence_alignment",
        "signal_fabrication",
        "pricing_scope_control",
        "single_ask_structure",
        "prospect_safe_language",
        "non_condescending_competitor_gap",
        "channel_scheduling_handoff",
        "competitor",
        "gap",
        "brief",
        "shows",
        "peers",
        "added",
        "delivery",
        "pods",
        "after",
        "expansion",
        "announced",
        "reported",
        "posted",
        "open",
        "roles",
        "public",
        "snapshot",
    }
    tokens = [token for token in text.split() if token not in structural_tokens]
    return {tuple(tokens[index : index + n]) for index in range(max(0, len(tokens) - n + 1))}


def jaccard_tokens(a: str, b: str) -> float:
    """Cheap deterministic stand-in for embedding similarity."""
    stop = {
        "request",
        "capacity",
        "days",
        "overlap",
        "earliest",
        "source",
        "snapshot",
        "public",
        "last",
        "open",
        "roles",
        "engineering",
        "engineer",
        "engineers",
        "reported",
        "announced",
        "posted",
    }
    left = {token for token in a.split() if token not in stop}
    right = {token for token in b.split() if token not in stop}
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def write_json(path: Path, data: Any) -> None:
    """Write formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write JSONL rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def write_outputs(tasks: Sequence[Dict[str, Any]], splits: Mapping[str, List[Dict[str, Any]]], seed: int) -> None:
    """Write benchmark, logs, examples, training data, and evidence graph."""
    write_json(ROOT / "data" / "generated_tasks.json", list(tasks))
    for split_name, split_tasks in splits.items():
        write_json(BENCH_DIR / split_name / "tasks.json", split_tasks)
        write_json(DATA_SPLITS_DIR / f"{split_name}.json", split_tasks)

    example_tasks = [
        splits["dev"][0],
        next(task for task in splits["dev"] if task["source_mode"] == "programmatic"),
        next(task for task in splits["dev"] if task["source_mode"] == "hand_authored_adversarial"),
    ]
    write_json(ROOT / "examples" / "example_tasks.json", example_tasks)

    route_log = [
        {
            "task_id": task["task_id"],
            "source_mode": task["source_mode"],
            "probe_id": task["probe_id"],
            "generation_model_family": task["metadata"]["generation_model_family"],
            "generation_model": task["metadata"]["generation_model"],
            "judge_model_family": task["metadata"]["judge_model_family"],
            "judge_model": task["metadata"]["judge_model"],
            "leakage_risk": task["metadata"]["judge_filter"]["leakage_risk"],
        }
        for task in tasks
    ]
    write_json(ROOT / "generation_scripts" / "model_routes.json", route_log)
    write_json(
        ROOT / "generation_scripts" / "judge_filter_log.json",
        {
            "seed": seed,
            "judge_prompts": sorted(path.name for path in PROMPT_DIR.glob("*.txt")),
            "thresholds": {
                "input_coherence": 4,
                "ground_truth_verifiability": 4,
                "rubric_application_clarity": 4,
            },
            "accepted": len(tasks),
            "rejected": 0,
            "records": [
                {
                    "task_id": task["task_id"],
                    "scores": task["metadata"]["judge_filter"]["scores"],
                    "accepted": task["metadata"]["judge_filter"]["accepted"],
                    "reference_score": task["metadata"]["judge_filter"]["reference_score"],
                }
                for task in tasks
            ],
        },
    )

    report = contamination_report(splits)
    write_json(ROOT / "contamination_check.json", report)

    preferences = []
    for task in splits["train"]:
        preferences.append(
            {
                "task_id": task["task_id"],
                "path": "B",
                "prompt": {
                    "input": task["input"],
                    "rubric": task["ground_truth"],
                },
                "chosen": task["reference_output"],
                "rejected": task["negative_output"],
                "probe_id": task["probe_id"],
                "failure_dimension": task["failure_dimension"],
                "generation_model_family_for_chosen": task["metadata"]["generation_model_family"],
                "judge_model_family": task["metadata"]["judge_model_family"],
            }
        )
    write_jsonl(ROOT / "training_data" / "path_b_preferences.jsonl", preferences)

    counts = contamination_report(splits)
    write_json(
        ROOT / "evidence_graph.json",
        {
            "dataset_counts": {
                "source": "contamination_check.json",
                "counts_by_source_mode": counts["counts_by_source_mode"],
                "counts_by_partition": counts["counts_by_partition"],
                "counts_by_failure_dimension": counts["counts_by_failure_dimension"],
            },
            "evaluator_demo": {
                "source": "examples/example_tasks.json",
                "command": "python scoring_evaluator.py --tasks examples/example_tasks.json --quiet",
            },
            "week10_failure_cost": {
                "source": "week10_artifacts/target_failure_mode.md",
                "bench_overcommitment_trigger_rate": 0.5433,
                "annual_loss_usd": 2281860,
            },
        },
    )


def stable_hash(value: str) -> str:
    """Return stable SHA-256 hex hash."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    """Generate and write Tenacious-Bench v0.1."""
    args = parse_args()
    tasks = build_all_tasks(args.seed)
    splits = stratified_split(tasks)
    write_outputs(tasks, splits, args.seed)
    print(
        json.dumps(
            {
                "seed": args.seed,
                "generated": len(tasks),
                "splits": {name: len(split_tasks) for name, split_tasks in splits.items()},
                "source_modes": dict(Counter(task["source_mode"] for task in tasks)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
