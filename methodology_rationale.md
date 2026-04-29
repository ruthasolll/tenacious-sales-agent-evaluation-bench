# Methodology Rationale: Path B Critic

Path B is the best fit because the target failure is not "cannot write a good email"; the style guide already gives good drafts. The failure is that the system cannot reliably detect when a plausible draft makes an unsafe commitment. Week 10 trace IDs `a553180f-80d2-4d4b-9a1e-d525b1219cfd`, `89337dd1-bb36-41d7-8530-190df8734cc3`, and `879ee1fc-7a7f-438e-bb19-054fb43c8637` are useful precisely because they show what the older trace format omits: no capacity fields, no signal confidence, and no prospect-facing draft to critique.

The training data in `training_data/path_b_preferences.jsonl` follows the preference-pair pattern from DPO, but the intended lightweight implementation is closer to SimPO/ORPO because a reference-free critic is cheaper and sufficient for rejection sampling. Prometheus 2 motivates rubric-specific judge behavior; Li et al.'s preference-leakage concern is addressed by storing separate `generation_model_family` and `judge_model_family` values on each task.

The critic should learn to prefer `reference_output` over `negative_output` for the same task input. The chosen outputs pass `scoring_evaluator.py`; rejected outputs trigger the probe-specific hard fail, such as unsupported capacity, fabricated funding, external "bench" wording, unsupported pricing, or multi-ask stacking.
