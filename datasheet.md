# Datasheet: Tenacious-Bench v0.1

## Summary Layer

Tenacious-Bench v0.1 is a 200-task benchmark for evaluating B2B sales-agent outreach under Tenacious-specific constraints: public-signal grounding, weak-signal humility, capacity safety, pricing boundaries, one-ask structure, and the Tenacious style guide. It follows Gebru et al.'s Datasheets sections and Pushkarna et al.'s layered data-card pattern: summary, structured overview, and schema-level notes.

## Structured Overview

### Motivation

The dataset exists because generic task-completion or retail benchmarks do not grade the central Tenacious failure: a fluent sales agent can overpromise engineering capacity, fabricate a signal, or quote unsafe pricing while still sounding professional. Week 10 traces recorded retail reward/cost but not `signal_brief`, `bench_summary`, or prospect-facing outreach. v0.1 converts those gaps into machine-verifiable tasks.

### Composition

Each task contains a prospect brief, public signal, confidence level, requested capacity, available-capacity summary, pricing scope, prior thread, reference output, rejected output, probe ID, source mode, generation/judge route, and ground-truth evaluator constraints.

Counts by source mode:

| Source mode | Count | Share |
|---|---:|---:|
| Trace-derived | 60 | 30% |
| Programmatic | 60 | 30% |
| Multi-LLM synthesis | 50 | 25% |
| Hand-authored adversarial | 30 | 15% |

Counts by partition:

| Partition | Count | Share |
|---|---:|---:|
| Train | 100 | 50% |
| Dev | 60 | 30% |
| Held-out | 40 | 20% |

Counts by failure dimension:

| Failure dimension | Count |
|---|---:|
| `capacity_grounding` | 20 |
| `timezone_stack_feasibility` | 20 |
| `start_date_truthfulness` | 20 |
| `confidence_alignment` | 20 |
| `signal_fabrication` | 20 |
| `pricing_scope_control` | 20 |
| `single_ask_structure` | 20 |
| `prospect_safe_language` | 20 |
| `non_condescending_competitor_gap` | 20 |
| `channel_scheduling_handoff` | 20 |

Short example tasks by generation mode:

| Mode | Example |
|---|---|
| Trace-derived | A Week 10 trace is restructured into a capacity-check task where a prospect asks for senior Go engineers faster than availability supports. |
| Programmatic | Template sweeps vary stack, headcount, source signal, start date, and signal confidence to test the same rubric under controlled changes. |
| Multi-LLM synthesis | A simulated frontier/open-weight router creates hard weak-signal and pricing cases, then a separate judge family filters them. |
| Hand-authored adversarial | A competitor-gap email tempts the agent to shame the prospect or claim peer behavior beyond the supplied brief. |

### Collection

Inputs were derived from Week 10 artifacts (`trace_log.jsonl`, `probe_library.md`, `target_failure_mode.md`), the Tenacious style guide v2 examples, and public-signal templates such as funding events, job-post velocity, layoffs/restructure notices, leadership changes, and competitor-gap briefs. No private prospect data is included.

### Preprocessing

`generation_scripts/generate_benchmark.py` builds the dataset from a fixed seed (`11011`). It routes task creation across source modes and simulated model families, applies a judge filter, performs pairwise deduplication, writes splits, and emits `contamination_check.json`. Held-out contamination checks passed: max train/held-out 8-gram overlap count was 6, lexical embedding-proxy similarity was 0.7895 below the 0.85 threshold, and time-shift verification had no failures.

### Uses

Intended uses: deterministic evaluator development, Path-B preference-pair preparation, sales-agent regression testing, prompt guardrail comparison, and public benchmark discussion. Out-of-scope uses: production lead scoring, claims about individual companies, or replacing human delivery-capacity review.

### Distribution

The public development and training partitions are in `tenacious_bench_v0.1/dev/tasks.json` and `tenacious_bench_v0.1/train/tasks.json`. `held_out/tasks.json` is present for this coursework repository so the evaluator can be run end to end; a public leaderboard release should seal or delay-release held-out tasks.

### Maintenance

Versioning uses `TB-V01-*` task IDs and a `version` field. v0.2 should add real discovery-call traces, more pricing bands, multilingual outreach, and stronger embedding-based contamination checks. Any new source mode must update this datasheet, `schema.json`, and `generation_scripts/model_routes.json`.

### License + Rationale

Recommended license: CC-BY-4.0 for dataset records and documentation. Rationale: the dataset is synthetic/derived from public-style signals and educational Week 10 artifacts; attribution encourages reuse while preventing ambiguity about provenance. Raw third-party CSV snapshots should retain their original terms and not be redistributed outside the coursework context unless licensing is confirmed.

## Schema-Level Notes

- `probe_id` maps every task to one Week 10 probe.
- `failure_dimension` is the primary stratification label used for counts and filtering.
- `input.signal_brief.confidence` controls confidence-aware phrasing.
- `input.bench_summary.available_capacity` and `input.requested_capacity` are compared by `scoring_evaluator.py`.
- `input.pricing_scope` defines which dollar amounts are safe to quote.
- `ground_truth.inclusion_thresholds` records the 1-5 score thresholds.
- `metadata.generation_model_family` and `metadata.judge_model_family` must differ for model-generated tasks to avoid preference leakage.
