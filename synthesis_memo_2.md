# Synthesis Memo 2: LLM-as-a-Judge and Preference Leakage

## Paper Design Choice

Gu et al.'s LLM-as-a-Judge survey treats stronger judge models as a path to better evaluation reliability. Li et al.'s preference-leakage paper adds an important warning: if the same family generates and judges, the benchmark can reward family-specific preferences rather than task truth.

## Critique

I disagree with making a single strongest judge the default for Tenacious-Bench v0.1. A powerful judge might improve nuance, but the central Tenacious errors are often concrete: a draft either claims 12 engineers when only 4 are available, says "bench" to a prospect, invents a funding round, or stacks four asks. Those checks should be deterministic first and model-judged second.

## Tenacious Evidence

The style guide's BAD #3, BAD #10, BAD #11, and BAD #12 are failures because they violate known fields, not because they require subtle literary taste. Week 11 tasks encode those fields directly: `bench_summary.available_capacity`, `pricing_scope.allowed_amounts`, `ground_truth.max_asks`, and `signal_brief.signals`. The evaluator catches these failures without depending on a single judge model's preference distribution.

## Resulting Design Decision

The generation pipeline rotates simulated `frontier` and `open_weight` families for generation, then uses a different judge family for filtering. The final scoring script remains deterministic and rubric-bound. This critiques judge-first benchmark construction: Tenacious needs judges for quality filtering and calibration, but the public score must be reproducible from fields a grader can inspect.
