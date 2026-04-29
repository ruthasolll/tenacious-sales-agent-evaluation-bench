"""Generate Tenacious-style outreach tasks from layoffs data and seed examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from split_dataset import save_splits, split_tasks


DEFAULT_LAYOFFS = Path("data/raw/layoffs_fyi.csv")
DEFAULT_SEEDS = Path("data/seed/seed_examples.json")
DEFAULT_GENERATED = Path("data/generated_tasks.json")
DEFAULT_SPLITS = Path("data/splits")
DEFAULT_TASK_COUNT = 200


Task = Dict[str, Any]


def load_seed_examples(path: Path) -> List[Dict[str, Any]]:
    """Load seed examples from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing seed examples file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        examples = data.get("examples", [])
    else:
        examples = data
    if not isinstance(examples, list):
        raise ValueError("seed_examples.json must contain a list or an object with an examples list.")
    return [example for example in examples if isinstance(example, dict)]


def extract_ctas(seed_examples: Sequence[Dict[str, Any]]) -> List[str]:
    """Extract reusable CTAs from seed examples with safe fallbacks."""
    ctas = [str(example.get("cta")).strip() for example in seed_examples if example.get("cta")]
    if ctas:
        return ctas
    return [
        "Would 15 minutes next week be useful to compare options?",
        "If you are reviewing delivery capacity, I can send a one-page comparison of engagement models.",
        "If this is active, would a 15-minute scoping call help? If not, no reply needed.",
    ]


def load_layoffs(path: Path) -> pd.DataFrame:
    """Load layoffs.fyi data."""
    if not path.exists():
        raise FileNotFoundError(f"Missing layoffs CSV: {path}")
    return pd.read_csv(path)


def generate_tasks(df: pd.DataFrame, seed_examples: Sequence[Dict[str, Any]], count: int) -> List[Task]:
    """Generate deterministic outreach tasks from layoffs records."""
    ctas = extract_ctas(seed_examples)
    tasks: List[Task] = []
    seen_pairs = set()

    for _, row in df.iterrows():
        company = clean_text(row.get("Company"))
        if not company:
            continue

        signal = build_signal(row)
        pair_key = (company.lower(), signal.lower())
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        context = build_context(row)
        cta = ctas[len(tasks) % len(ctas)]
        output = compose_email(company=company, signal=signal, context=context, cta=cta)
        tasks.append(
            {
                "input": {
                    "company": company,
                    "signal": signal,
                    "context": context,
                },
                "output": output,
            }
        )
        if len(tasks) == count:
            return tasks

    raise ValueError(f"Only generated {len(tasks)} unique tasks; requested {count}.")


def build_signal(row: pd.Series) -> str:
    """Build a grounded layoff/restructure signal from one CSV row."""
    date = clean_text(row.get("Date"))
    count = numeric_or_none(row.get("Laid_Off_Count"))
    percentage = numeric_or_none(row.get("Percentage"))

    date_part = f"on {date}" if date else "recently"
    if count and percentage:
        return f"{int(count):,} layoffs ({percentage:.0%} workforce reduction) reported {date_part}"
    if count:
        return f"{int(count):,} layoffs reported {date_part}"
    if percentage:
        return f"{percentage:.0%} workforce reduction reported {date_part}"
    return f"the layoffs.fyi restructure entry dated {date}" if date else "a layoffs.fyi restructure entry"


def build_context(row: pd.Series) -> str:
    """Build non-hallucinated context from CSV columns."""
    parts = [
        ("industry", clean_text(row.get("Industry"))),
        ("location", clean_text(row.get("Location_HQ"))),
        ("country", clean_text(row.get("Country"))),
        ("stage", clean_text(row.get("Stage"))),
        ("source", clean_text(row.get("Source"))),
    ]
    return "; ".join(f"{key}: {value}" for key, value in parts if value)


def compose_email(company: str, signal: str, context: str, cta: str) -> str:
    """Compose a Tenacious-style cold outreach email from grounded inputs."""
    industry = normalize_industry(extract_context_value(context, "industry"))
    return (
        "Subject: Context: delivery capacity after restructure\n\n"
        f"Hi {company} team,\n\n"
        f"I saw {signal}. I cannot tell from the outside whether delivery capacity "
        f"is actually constrained, but teams in {industry} often use this moment "
        "to separate critical roadmap work from work that can pause.\n\n"
        "Tenacious supports managed engineering teams when there is a real delivery "
        "gap, with scope confirmed before any capacity commitment. "
        f"{cta}\n\n"
        "Best,\n"
        "Yabi\n"
        "Research Partner, Tenacious Intelligence Corporation\n"
        "gettenacious.com"
    )


def extract_context_value(context: str, key: str) -> str:
    """Extract one key from a semicolon-delimited context string."""
    prefix = f"{key}: "
    for part in context.split("; "):
        if part.startswith(prefix):
            return part[len(prefix) :]
    return ""


def normalize_industry(industry: str) -> str:
    """Return prospect-facing industry wording."""
    if not industry or industry.lower() in {"other", "unknown"}:
        return "your market"
    return industry


def clean_text(value: Any) -> str:
    """Convert a dataframe value into a clean string."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def numeric_or_none(value: Any) -> float | None:
    """Convert a dataframe value into a float or None."""
    if value is None or pd.isna(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def save_tasks(tasks: Sequence[Task], path: Path) -> None:
    """Save generated tasks to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(tasks), indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layoffs", type=Path, default=DEFAULT_LAYOFFS)
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--generated-output", type=Path, default=DEFAULT_GENERATED)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--count", type=int, default=DEFAULT_TASK_COUNT)
    return parser.parse_args()


def main() -> None:
    """Generate tasks and write train/dev/held-out splits."""
    args = parse_args()
    df = load_layoffs(args.layoffs)
    seed_examples = load_seed_examples(args.seeds)
    tasks = generate_tasks(df=df, seed_examples=seed_examples, count=args.count)
    save_tasks(tasks, args.generated_output)

    train, dev, held_out = split_tasks(tasks)
    save_splits(train, dev, held_out, args.splits_dir)
    print(
        json.dumps(
            {
                "generated": len(tasks),
                "train": len(train),
                "dev": len(dev),
                "held_out": len(held_out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
