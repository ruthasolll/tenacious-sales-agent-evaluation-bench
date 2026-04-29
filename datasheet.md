# Tenacious-Bench v0.1 Datasheet

## Motivation

Tenacious-Bench v0.1 evaluates B2B outreach quality under operational constraints. The initial task slice focuses on layoffs/restructure signals because these are public, auditable, and relevant to late-stage delivery-capacity risk.

## Composition

The current dataset contains 200 generated outreach tasks created from `data/raw/layoffs_fyi.csv` and `data/seed/seed_examples.json`.

- Train: 100 tasks
- Dev: 60 tasks
- Held-out: 40 tasks

Each task has:

```json
{
  "input": {
    "company": "...",
    "signal": "...",
    "context": "..."
  },
  "output": "email text"
}
```

## Collection

The raw source is a local layoffs.fyi-style CSV snapshot. The generation script uses only CSV-backed layoff/restructure signals and does not invent funding or hiring events.

## Processing

`scripts/build_dataset.py` summarizes the raw CSV into `data/raw/layoffs_summary.json`. `scripts/generate_tasks.py` creates 200 tasks and writes split files. `scripts/check_contamination.py` checks duplicate inputs, duplicate outputs, and company-signal overlap across splits.

## Intended Use

This dataset is intended for evaluator development, training-data filtering, and early Tenacious-Bench scoring experiments.

## Limitations

The current slice covers layoffs/restructure outreach only. It does not yet cover funding signals, job-post velocity, leadership changes, competitor capability gaps, or explicit bench-overcommitment stress cases.
