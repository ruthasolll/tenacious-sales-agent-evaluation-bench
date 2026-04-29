# Synthesis Memo 1: Synthetic Data Best Practices

## Paper Design Choice

Liu et al., *Best Practices and Lessons Learned on Synthetic Data for Language Models* (COLM 2024), emphasize scaling synthetic data through diverse generation sources plus aggressive filtering. I accept the filtering argument, but I disagree with treating diversity volume as the main lever for this project.

## Critique

For Tenacious, the highest-risk failures are not underrepresented phrasings; they are operational boundary violations. Week 10 named `bench_overcommitment` as a 0.5433 trigger-rate failure with a projected annual risk of $2,281,860. A larger synthetic pool that varies surface wording would not necessarily create more diagnostic tasks unless it binds each draft to `bench_summary`, `requested_capacity`, and `signal_confidence`.

## Tenacious Evidence

The Week 10 trace format records retail reward and duration, but trace IDs such as `a553180f-80d2-4d4b-9a1e-d525b1219cfd` and `879ee1fc-7a7f-438e-bb19-054fb43c8637` contain no prospect signal or capacity fields. Week 11 therefore needs fewer generic examples and more schema pressure. Tenacious-Bench v0.1 uses 200 tasks, but the important design is not count; it is the probe-to-field map in `audit_memo.md`.

## Resulting Design Decision

I used four source modes, but capped the dataset at 200 tasks and balanced ten failure dimensions at 20 each. Every generated task must pass pointwise judge thresholds and pairwise dedup. This is a deliberate disagreement with a volume-first reading of synthetic-data best practice: for Tenacious, high-quality constraint coverage beats extra paraphrases.
