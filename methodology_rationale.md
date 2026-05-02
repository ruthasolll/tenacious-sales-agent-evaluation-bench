# Methodology Rationale: Path B Critic

Path B is the best fit because the target failure is not "cannot write a good email"; the style guide already gives good drafts. The failure is that the system cannot reliably detect when a plausible draft makes an unsafe commitment. Week 10 trace IDs `a553180f-80d2-4d4b-9a1e-d525b1219cfd`, `89337dd1-bb36-41d7-8530-190df8734cc3`, and `879ee1fc-7a7f-438e-bb19-054fb43c8637` are useful precisely because they show what the older trace format omits: no capacity fields, no signal confidence, and no prospect-facing draft to critique.

The training data in `training_data/path_b_preferences.jsonl` follows the preference-pair pattern from DPO, but the intended lightweight implementation is closer to SimPO/ORPO because a reference-free critic is cheaper and sufficient for rejection sampling. *Direct Preference Optimization* (Rafailov et al., NeurIPS 2023), §3 motivates direct optimization over preference pairs without a separate reward-model training loop. *SimPO* (Meng et al., NeurIPS 2024), §3 motivates reference-free preference optimization when the evaluation signal is pairwise and cost-sensitive. *ORPO* (Hong et al., EMNLP 2024), §3 motivates adding preference pressure without a full RLHF stack. *Prometheus 2* (Kim et al., 2024), §3 motivates rubric-specific judge behavior; *Preference Leakage* (Li et al., 2025), §4 motivates avoiding same-family generation and judging because preference leakage can make synthetic benchmarks reward model-family style rather than task truth.

The critic should learn to prefer `reference_output` over `negative_output` for the same task input. The chosen outputs pass `scoring_evaluator.py`; rejected outputs trigger the probe-specific hard fail, such as unsupported capacity, fabricated funding, external "bench" wording, unsupported pricing, or multi-ask stacking.

## Paper-Grounded Design Choices

*Best Practices and Lessons Learned on Synthetic Data for Language Models* (Liu et al., COLM 2024), §3.2 and §4 are the reason the dataset uses four source modes plus filtering rather than one large homogeneous synthetic dump. For this project, "diversity" means failure-dimension and source-mode coverage, not simply many paraphrases. The judge filter in `generation_scripts/generate_benchmark.py` turns that into code: every task receives pointwise quality scores, leakage checks, and structured pass/fail reasons before it enters a split.

*A Survey on LLM-as-a-Judge* (Gu et al., 2024–2025), §2.2 and §5 motivate treating LLM-as-judge outputs as calibrated evidence rather than ground truth. Tenacious-Bench therefore uses judge routing for filtering and inter-rater-style calibration, while the public score remains deterministic through `scoring_evaluator.py`. This keeps the benchmark reproducible and makes every hard fail inspectable.

*Data Cards* (Pushkarna et al., FAccT 2022), §3 motivates layered dataset documentation. The repository follows that pattern with `datasheet.md` for high-level and schema-level context, `methodology.md` for generation and split logic, and `evidence_graph.json` for artifact-to-claim traceability.

## Why Path A Was Rejected

Path A, an SFT generator, would directly optimize email writing quality. That is attractive, but it does not target the observed Week 10 failure tightly enough. As the trace IDs prove, the dominant failure was not style drift but operational misjudgment: saying capacity, pricing, or timing was safe when the structured fields inherently did not support that claim. A generator SFT run on its own can easily learn a nicer, more polite house style while still making aggressively unsafe commitments (e.g. committing 12 engineers when only 4 are available). To actually fix the root problem, Path A would require a separate safety filter anyway, making it an obsolete primary path compared to fixing the filter mechanism directly.

Path A also radically increases deployment risk because a core generation model changes the full surface of all outbound emails in unpredictable ways. In contrast, a Path B critic can sit passively behind an existing generator, reject specifically unsafe drafts, and provide per-dimension rationale that map directly back to the Tenacious rubric without touching the underlying prose on the drafts that operate perfectly fine.

## Why Path C Was Rejected

Path C, a process reward model (PRM), would make sense if the available traces contained deep multi-step agent trajectories: research action, source selection, capacity lookup, draft, revision, handoff steps all independently scorable. The Week 10 artifacts did not expose a granular process map. They logged final conversation outcomes and high-level failure classifications, but simply didn't provide enough intermediate causal state logic to label step-level rewards reliably. 

Because the evidence is strictly single-turn and final-draft-centered, enforcing process supervision using Path C would be almost completely invented and hallucinated rather than trace-derived. Path B aligns much closer to the ground truth of the dataset: each task provides one concrete input, one preferred final output, one decisively rejected final output, and a straightforward rubric that acts directly on that preference.
