# Methodology

## Chosen Path

I choose Path B: a preference-tuned judge/critic. Week 10 evidence points to inconsistency rather than pure generation inability. The highest-risk failure, bench overcommitment, had a 0.5433 trigger rate in `week10_artifacts/target_failure_mode.md`, while tone drift and scheduling were materially lower. The available Week 10 traces (`a553180f-80d2-4d4b-9a1e-d525b1219cfd`, `89337dd1-bb36-41d7-8530-190df8734cc3`, `0857ba6e-d8cb-4ec8-b024-3d5ddc298fc6`, `0c380837-0cac-490f-a053-8cb13e79ed6b`, `879ee1fc-7a7f-438e-bb19-054fb43c8637`) show that the old benchmark logged reward/cost but did not expose signal, capacity, or message fields. The right intervention is therefore a critic that can reject unsafe drafts before they reach a prospect.

## Paper-Grounded Justification

Liu et al. (COLM 2024), §3.2 and §4, argue that synthetic datasets need source anchoring, filtering, and diversity controls; Tenacious-Bench uses Week 10 probes as anchors, four source modes, judge filtering, and pairwise deduplication. Gu et al.'s LLM-as-a-Judge survey, §2.2 and §5, motivates multi-dimension scoring and calibration; the evaluator uses five dimensions and explicit 1-5 thresholds. For Path B, DPO (Rafailov et al., §3) explains preference-pair training, while SimPO (§3) and ORPO (§3) motivate lower-cost reference-free critics. Li et al.'s preference-leakage warning, §4, is implemented directly: generation and judge model families differ in `metadata`.

Path A was rejected because the observed failure is not only bad email generation; it is unsafe acceptance of plausible drafts that violate capacity, pricing, or signal truth. Path C was rejected because the Week 10 artifacts do not expose enough step-level trajectories to label process rewards. Path B fits the data shape: one input, one preferred draft, one rejected draft, and a deterministic rubric explaining the preference.

## Dataset Construction

The generation pipeline is `generation_scripts/generate_benchmark.py` with seed `11011`. It creates 200 tasks:

| Mode | Count | Reason |
|---|---:|---|
| Trace-derived | 60 | Highest fidelity to Week 10 behavior |
| Programmatic | 60 | Controlled sweeps over stack, headcount, confidence, dates |
| Multi-LLM synthesis | 50 | Hard cases routed across simulated frontier/open-weight families |
| Hand-authored adversarial | 30 | Original edge cases aimed at the Week 10 failure taxonomy |

Judge prompts live in `prompts/judge/*.txt`. The pointwise judge requires `input_coherence >= 4`, `ground_truth_verifiability >= 4`, and `rubric_application_clarity >= 4`. Pairwise dedup keeps the more diagnostic task when similarity is high.

The judge filter now exposes an explicit tier flag:

```bash
python generation_scripts/generate_benchmark.py --judge-tier dev
python generation_scripts/generate_benchmark.py --judge-tier eval
```

Model IDs for both tiers live in `generation_scripts/judge_routing_config.json`. The generated `generation_scripts/judge_filter_log.json` records, for every task, the judge tier, judge model ID, pass/fail status, scores, thresholds, and structured reasons.

## Split and Stratification

The split is 50/30/20 because the challenge requires enough training data for Path B while preserving a substantial public dev set and a held-out slice. The generator stratifies by source mode, yielding 100 train, 60 dev, and 40 held-out tasks. Failure dimensions are balanced at 20 tasks each, preventing the critic from improving only on the dominant bench-overcommitment class.

## Contamination Procedure and Results

Three checks run in `contamination_check.json`:

| Check | Threshold | Result |
|---|---:|---:|
| Train/held-out 8-gram overlap | `< 8` | 6 |
| Train/held-out lexical embedding proxy | `< 0.85` | 0.7895 |
| Time-shift verification failures | 0 | 0 |

The report passed. `scripts/check_contamination.py` also reports no duplicate inputs, no duplicate outputs, and no train/held-out company-signal overlap.

## Inter-Rater Agreement

A 30-task subset was labeled twice using the five evaluator dimensions. The simulated agreement matrix is committed in `inter_rater_agreement.md`; all dimensions meet or exceed 0.83 exact agreement, above the 0.80 revision trigger.

## Design Decisions

Operational safety is weighted highest with signal grounding because Week 10 cost arithmetic showed overcommitment dominates revenue risk. Structure is lower-weighted because word count and subject length are repairable, but multi-ask stacking remains a hard fail. The evaluator is deterministic by design so repeated runs on the same agent output are stable within automated grading. The held-out slice is included in this coursework repo for reproducibility; a public leaderboard should seal it.
