# Datasheet: Tenacious-Bench v0.1

## Summary Layer

Tenacious-Bench v0.1 is a 200-task benchmark for evaluating B2B sales-agent outreach under Tenacious-specific constraints: public-signal grounding, weak-signal humility, capacity safety, pricing boundaries, one-ask structure, and the Tenacious style guide. It follows Gebru et al.'s Datasheets sections and Pushkarna et al.'s layered data-card pattern: summary, structured overview, and schema-level notes.

The core object is not a generic sales-email dataset. Each record binds an outreach prompt to operational fields that a real Tenacious workflow would have to respect: the prospect-facing signal, the confidence level of that signal, available delivery capacity, requested capacity, pricing boundaries, and a probe-specific failure dimension. This design makes the benchmark useful for testing whether a model can stay faithful to business constraints even when the unsafe answer would sound fluent.

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

The trace-derived slice preserves the structure of observed Week 10 failure modes but removes private operational context. The programmatic slice sweeps controlled variables such as stack, headcount, start-date pressure, signal confidence, and pricing scope. The multi-LLM synthesis slice simulates a routed generation process with different model families for generation and judging. The hand-authored adversarial slice targets failure cases that template sweeps tend to miss, especially condescending competitor framing and unsafe handoff language.

### Preprocessing

`generation_scripts/generate_benchmark.py` builds the dataset from a fixed seed (`11011`). It routes task creation across source modes and simulated model families, applies a judge filter, performs pairwise deduplication, writes splits, and emits `contamination_check.json`. Held-out contamination checks passed: max train/held-out 8-gram overlap count was 6, lexical embedding-proxy similarity was 0.7895 below the 0.85 threshold, and time-shift verification had no failures.

### Uses

Intended uses: deterministic evaluator development, Path-B preference-pair preparation, sales-agent regression testing, prompt guardrail comparison, and public benchmark discussion. Out-of-scope uses: production lead scoring, claims about individual companies, or replacing human delivery-capacity review.

The safest use is comparative: run the same candidate system across train/dev/held-out partitions and inspect per-dimension changes. The benchmark should not be used to infer that a model is ready to send live outreach without human review. A high score means the draft obeyed encoded constraints, not that the prospect actually has buying intent or that Tenacious has confirmed delivery capacity.

### Limitations

Tenacious-Bench v0.1 measures a narrow slice of sales-agent behavior: single-turn email or reply drafting under public-signal and delivery-capacity constraints. It does not include live CRM state, private account notes, warm-intro permission chains, real-time calendar state, sales-owner assignment, or actual delivery staffing systems. As a result, it can grade whether an email avoids overclaiming capacity, but it cannot decide whether the account should be contacted today. 

The public-signal fields are lossy proxies. A hiring post, layoff article, or leadership-change signal can lag the real account need by weeks, and redacted case-study context removes named-account nuance that a real seller might use. The benchmark may therefore over-reward outputs that confidently restate stale public facts and under-reward cautious outputs that ask for confirmation before assuming the signal maps to an active pain.

The dataset also uses synthetic and reconstructed examples. This is necessary for privacy and coverage, but it creates **synthetic artifacts**. For example, company and contact combinations are often simplified, and the surface wording can end up being much more regular than the messy unpredictability of a real inbound thread. Adversarial negatives are also explicitly formulated to be cleaner than production failures for evaluation consistency. Consequently, model gains on v0.1 should strictly be evaluated as evidence of constraint-following improvement rather than guaranteed proof of an overall reply-rate lift in production outbound campaigns.

### Biases

The dataset exhibits **domain skew** in that it is heavily skewed toward English-language B2B technology outreach in North American-style sales motions. Industries such as SaaS, fintech, HR, logistics, internet infrastructure, and consumer technology are represented far more heavily than public sector, enterprise hardware, healthcare procurement, small local services, or general non-English enterprise sales. A model tuned exclusively on this benchmark will likely learn a concise Tenacious voice that works well for technical buyers (like CTOs or VP Engineering profiles), but will sound critically under-contextualized for relationship-heavy, highly-regulated, or international accounts where direct outreach may be seen as inappropriate.

There is also a prominent **failure-mode bias**. v0.1 intentionally over-samples operational safety risks—such as capacity overcommitment, pricing overquote, fabricated signals, and channel handoff errors—because those were the highest-cost Week 10 gaps and are most fatal to the firm's brand. That makes the benchmark incredibly robust for safety alignment and critic model training, but it systematically under-measures the agent's creative value proposition quality, account prioritization strategy, multi-thread account maneuvering, and long-horizon sequence planning.

Finally, the source-mode mixture can imprint **synthetic artifacts and model style preferences**. Programmatic tasks make controlled comparisons easier but repeat underlying schema phrasing in ways real reps wouldn't. Multi-LLM synthesis enhances variety but still occasionally imports the judge or generator model-families' specific phrasing quirks (e.g., repeating specific transition phrases like "However" or "Furthermore"). Hand-authored adversarial tasks supply excellent diagnostics but ultimately reflect the authors' limited scope of what a "hard" Tenacious failure looks like in an idealized world. v0.2 intends to alleviate this by incorporating real redacted discovery-call traces and post-send human labels.

### Distribution

The public development and training partitions are in `tenacious_bench_v0.1/dev/tasks.json` and `tenacious_bench_v0.1/train/tasks.json`. `held_out/tasks.json` is present for this coursework repository so the evaluator can be run end to end; a public leaderboard release should seal or delay-release held-out tasks.

### Maintenance

Versioning uses `TB-V01-*` task IDs and a `version` field. v0.2 should add real discovery-call traces, more pricing bands, multilingual outreach, and stronger embedding-based contamination checks. Any new source mode must update this datasheet, `schema.json`, and `generation_scripts/model_routes.json`.

Maintenance should include a standing release checklist: recompute source-mode and failure-dimension counts, rerun contamination checks, regenerate the judge-filter log, update inter-rater agreement on a 30-task subset, and record any changes to pricing, capacity, or style-guide assumptions. Any production-derived additions should be redacted before entry and should preserve enough metadata to explain why the task is diagnostic.

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
