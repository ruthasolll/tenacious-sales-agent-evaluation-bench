# Tenacious-Bench v0.1

Tenacious-Bench is a sales-agent evaluation benchmark for Tenacious-style B2B outreach. It tests whether an agent can ground outreach in public signals, respect confidence limits, avoid delivery-capacity overcommitment, quote only safe pricing, preserve the Tenacious voice, and use one clear ask.

## Current Status

Implemented:

- 200 benchmark tasks in `tenacious_bench_v0.1/`
- 50/30/20 split: 100 train, 60 dev, 40 held-out
- Four authoring modes: trace-derived, programmatic, multi-LLM synthesis, hand-authored adversarial
- Multi-model routing abstraction with separate generation and judge model families
- Standalone judge prompts in `prompts/judge/`
- Deterministic evaluator in `scoring_evaluator.py`
- Three runnable example tasks in `examples/example_tasks.json`
- Path B preference data in `training_data/path_b_preferences.jsonl`
- Executable local Path B critic trainer in `training/train_path_b_critic.py`
- Passing contamination report in `contamination_check.json`

## Setup

Use Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The core evaluator and generator use only the Python standard library. `pandas` is kept for the older raw CSV helper scripts under `scripts/`.

## Run The Evaluator

Score the three in-repo demo tasks:

```bash
python scoring_evaluator.py --tasks examples/example_tasks.json --quiet
```

Score a full partition:

```bash
python scoring_evaluator.py --tasks tenacious_bench_v0.1/dev/tasks.json --quiet
```

Show the rejected examples fail:

```bash
python scoring_evaluator.py --tasks examples/example_tasks.json --candidate-field negative_output --quiet
```

## Regenerate The Dataset

```bash
python generation_scripts/generate_benchmark.py --seed 11011
python scripts/check_contamination.py --splits-dir data/splits
```

Regeneration rewrites the benchmark partitions, examples, route logs, judge-filter log, contamination report, evidence graph, and Path B preference pairs.

Train the local Path B critic baseline:

```bash
python training/train_path_b_critic.py
```

## Directory Guide

- `audit_memo.md`: probe-to-gap audit memo with Week 10 trace evidence
- `datasheet.md`: Gebru + Pushkarna-style datasheet
- `methodology.md`: Path B rationale, split logic, contamination procedure, design decisions
- `schema.json` and `schemas/schema.json`: task schema
- `scoring_evaluator.py`: deterministic scoring evaluator with calibration comments
- `tenacious_bench_v0.1/`: train/dev/held-out task partitions
- `generation_scripts/generate_benchmark.py`: routed generation and judge-filter pipeline
- `prompts/judge/`: standalone judge prompts
- `synthesis_memos/`: analytical reading memos
- `training_data/`: Path B preference pairs
- `training/`: local critic trainer, weights, and training log
- `contamination_check.json`: n-gram, similarity, and time-shift contamination report

## What Is Next

- Replace simulated model-family routing with actual OpenRouter calls and logged costs.
- Seal or delayed-release the held-out split before public leaderboard use.
- Train a small Path B critic on `training_data/path_b_preferences.jsonl`.
- Add discovery-call and multi-turn trajectory tasks for Tenacious-Bench v0.2.
- Publish the dataset card and public artifact links after staff sign-off.
