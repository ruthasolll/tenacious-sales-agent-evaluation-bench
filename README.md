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
python generation_scripts/generate_benchmark.py --seed 11011 --judge-tier dev
python scripts/check_contamination.py --splits-dir data/splits
```

Regeneration rewrites the benchmark partitions, examples, route logs, judge-filter log, contamination report, evidence graph, and Path B preference pairs.

For final reportable filtering, use the reserved eval-tier route:

```bash
python generation_scripts/generate_benchmark.py --seed 11011 --judge-tier eval
```

Judge model IDs are configured in `generation_scripts/judge_routing_config.json`. Each run writes `generation_scripts/judge_filter_log.json` with per-task pass/fail status, scores, thresholds, and reasons.

Train the local Path B critic baseline:

```bash
python training/train_path_b_critic.py
```

Train the DPO LoRA adapter used for the Colab Path-B experiment:

```bash
python scripts/create_dpo_dataset.py
python training/train_dpo.py
```

The DPO script prints and saves one visible config: pinned backbone revision, LoRA-only adapter settings, epochs, batch size, gradient accumulation, warmup, scheduler, beta, and learning rate.

Run the ablation/statistics report after model comparison:

```bash
python training/evaluate_trained_model.py --tasks data/splits/held_out.json --limit 0
python ablations/run_ablation_analysis.py
```

The ablation artifact reports Delta A with paired bootstrap 95% CI and paired sign-flip p value, the same-backbone prompt-only baseline for Delta B, and per-task token/cost/latency fields.

## Directory Guide

- `audit_memo.md`: probe-to-gap audit memo with Week 10 trace evidence
- `datasheet.md`: Gebru + Pushkarna-style datasheet
- `methodology.md`: Path B rationale, split logic, contamination procedure, design decisions
- `schema.json` and `schemas/schema.json`: task schema
- `scoring_evaluator.py`: deterministic scoring evaluator with calibration comments
- `tenacious_bench_v0.1/`: train/dev/held-out task partitions
- `generation_scripts/generate_benchmark.py`: routed generation and judge-filter pipeline
- `generation_scripts/judge_routing_config.json`: explicit dev-tier and eval-tier judge model IDs
- `prompts/judge/`: standalone judge prompts
- `synthesis_memos/`: analytical reading memos
- `training_data/`: Path B preference pairs
- `training/`: local critic trainer, weights, and training log
- `ablations/run_ablation_analysis.py`: paired statistics and cost-Pareto artifact builder
- `contamination_check.json`: n-gram, similarity, and time-shift contamination report

## Public Artifact References

Current repository: <https://github.com/ruthasolll/tenacious-sales-agent-evaluation-bench>

Planned public release targets to fill after upload:

| Artifact | URL / status |
|---|---|
| Hugging Face dataset | [ruthasolll/tenacious-bench-v0.1](https://huggingface.co/datasets/ruthasolll/tenacious-bench-v0.1) |
| Hugging Face model / LoRA adapter | [ruthasolll/tenacious-qwen2.5-1.5b-dpo-lora](https://huggingface.co/ruthasolll/tenacious-qwen2.5-1.5b-dpo-lora) |
| Technical blog post | [Evaluating Tenacious-Bench: Synthetic Data and DPO on Substack](https://ruthasolll.substack.com/p/evaluating-tenacious-bench) |
| Community engagement | [Issue #42 on Allen AI's open-instruct repo: Evaluation Gap for Sales Scenarios](https://github.com/allenai/open-instruct/issues/42) |

## License And Credits

This dataset and its documentation are released under the [CC-BY-4.0 License](LICENSE). Code is provided for coursework reproducibility under the MIT License (or open use). 

Credit: 
- Ruth Asoll, Tenacious-Bench coursework artifacts, Week 10 trace-derived failure taxonomy.
- The public papers cited in `methodology.md` and `methodology_rationale.md` including *Best Practices and Lessons Learned on Synthetic Data* (Liu et al., COLM 2024), *Datasheets for Datasets* (Gebru et al., 2021), *Data Cards* (Pushkarna et al., FAccT 2022), *A Survey on LLM-as-a-Judge* (Gu et al., 2024-2025), and *Preference Leakage* (Li et al., 2025).

## What Is Next

- Replace simulated model-family routing with actual OpenRouter calls and logged costs.
- Seal or delayed-release the held-out split before public leaderboard use.
- Train a small Path B critic on `training_data/path_b_preferences.jsonl`.
- Add discovery-call and multi-turn trajectory tasks for Tenacious-Bench v0.2.
- Publish the dataset card and public artifact links after staff sign-off.
