"""Summarize the raw layoffs.fyi CSV for Tenacious-Bench dataset authoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


DEFAULT_INPUT = Path("data/raw/layoffs_fyi.csv")
DEFAULT_OUTPUT = Path("data/raw/layoffs_summary.json")


def load_layoffs_csv(path: Path) -> pd.DataFrame:
    """Load the layoffs CSV from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Missing layoffs CSV: {path}")
    return pd.read_csv(path)


def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Return the first matching column name from a candidate list."""
    normalized = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise KeyError(f"None of these columns were found: {candidates}")


def summarize_layoffs(df: pd.DataFrame) -> Dict[str, Any]:
    """Build a summary dictionary for layoffs data."""
    company_col = find_column(df, ["Company", "company"])
    layoffs_col = find_column(df, ["Laid_Off_Count", "laid_off_count", "Total Laid Off"])
    industry_col = find_column(df, ["Industry", "industry"])

    working = df.copy()
    working[layoffs_col] = pd.to_numeric(working[layoffs_col], errors="coerce").fillna(0)

    top_companies = (
        working.groupby(company_col, dropna=True)[layoffs_col]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    industries = (
        working[industry_col]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .head(10)
        .reset_index()
    )

    return {
        "number_of_rows": int(len(df)),
        "top_5_companies_by_layoffs": [
            {
                "company": str(row[company_col]),
                "layoffs": int(row[layoffs_col]),
            }
            for _, row in top_companies.iterrows()
        ],
        "most_common_industries": [
            {
                "industry": str(row[industry_col]),
                "count": int(row["count"]),
            }
            for _, row in industries.iterrows()
        ],
        "missing_values_summary": {
            str(column): int(count) for column, count in df.isna().sum().to_dict().items()
        },
    }


def save_summary(summary: Dict[str, Any], path: Path) -> None:
    """Save the summary dictionary as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Load layoffs data, print the summary, and save it to disk."""
    args = parse_args()
    df = load_layoffs_csv(args.input)
    summary = summarize_layoffs(df)
    save_summary(summary, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
