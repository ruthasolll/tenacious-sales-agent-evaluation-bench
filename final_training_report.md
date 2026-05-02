# Tenacious-Bench Path B Training Memo

## Page 1: Executive Decision Memo

### Executive Summary

Path B DPO fine-tuning on `unsloth/qwen2.5-1.5b` produced a completed 60-task Tenacious-Bench validation lift of +3.250 overall-score points over the same-backbone prompt-engineered baseline (base 58.862 -> fine-tuned 62.112), with a paired bootstrap 95% CI of [+0.804, +6.021] and a paired sign-flip permutation p value of 0.0153. This is a real but small learning signal: 18 tasks improved, 6 regressed, 36 were unchanged, and the pass rate remained 0.0% for both systems. Recommendation: do not deploy yet, because the current artifact is a dev-split run rather than final held-out evidence and the trained model still produces occasional template/recipient artifacts; move to deploy-with-caveat only after a held-out rerun clears +5.0 points with lower CI above +1.0, pass rate above 20%, and no cost or QA-regression trigger.

### Headline Delta A: Trained Component Lift

The completed comparison artifact is `outputs/evaluation_comparison.json`, generated from `data/splits/dev.json` with 60 tasks. The grading target asks for held-out reporting; therefore the table below is the current validation result and should be replaced by the same run on `data/splits/held_out.json` before this is treated as the final held-out claim.

| Metric | Result |
|---|---:|
| Evaluation file in current artifact | `data/splits/dev.json` |
| Evaluation tasks | 60 |
| Prompt-engineered baseline mean overall | 58.862 |
| DPO fine-tuned mean overall | 62.112 |
| Delta A point estimate | +3.250 |
| 95% confidence interval | [+0.804, +6.021] |
| Paired statistical test | Paired bootstrap over per-task deltas, 10,000 resamples |
| Additional test | Paired sign-flip permutation test, p = 0.0153 |
| Improved / regressed / unchanged tasks | 18 / 6 / 36 |
| Pass rate | 0.0% base, 0.0% fine-tuned |

Final held-out command to run in Colab:

```bash
python training/evaluate_trained_model.py --tasks data/splits/held_out.json --limit 0
```

### Delta B: Prompt-Engineered Baseline Honesty

The prompt-engineered baseline is not a weak strawman. It uses the same backbone, `unsloth/qwen2.5-1.5b`, the same task fields, the same Qwen chat-template shape, the same `Subject:` generation anchor, and the same deterministic decoding settings as the fine-tuned model; the only removed intervention is the LoRA adapter in `outputs/`.

| Comparison | Mean overall | Interpretation |
|---|---:|---|
| Prompt-engineered baseline, no adapter | 58.862 | Stronger than the original unanchored generation path because prompt anchoring alone reduced instruction-copying. |
| DPO adapter on same prompt shape | 62.112 | +3.250 points over prompt-only, but still not deployable because pass rate is 0.0%. |

This is a positive but modest result for training. It supports the claim that the DPO adapter changed behavior, but it also shows that prompt engineering did a large share of the format repair and that the trained component has not yet converted the benchmark into passing outputs.

### Cost Per Task and Production Implication

| Pipeline | Direct API cost per task | Token/capacity shape | Latency |
|---|---:|---|---|
| Prompt-engineered baseline | $0.00 in the current Colab/local run | Same 1.5B backbone, same max generation cap | Not logged in current evaluator |
| DPO fine-tuned adapter | $0.00 in the current Colab/local run | Same 1.5B backbone plus a 73.9 MB LoRA adapter | Not logged in current evaluator |
| Incremental trained-component delta | $0.00 direct API cost per task | No additional external model call | Latency omission is a known instrumentation gap |

The cost result is favorable but not decisive. Because the trained component uses the same backbone and does not add an extra judge call at inference time, the production blocker is quality, not direct model spend. Before any deployment decision, `training/evaluate_trained_model.py` should add wall-clock timing per generation so latency can be reported beside score lift.

### Production Recommendation

Recommendation: do not deploy.

The evidence is not strong enough for production: the current completed run shows only a +3.250 validation lift, both systems have a 0.0% pass rate, and the run is on `dev.json` rather than the required held-out split. Move to deploy with caveat only if the held-out evaluation shows mean lift of at least +5.0 overall points, the lower bound of the 95% paired bootstrap CI is above +1.0, pass rate reaches at least 20%, and manual QA finds fewer than 8 template/recipient or unsupported-claim failures per 100 generated emails.

## Page 2: Risk, Coverage, and Controls

### Tenacious-Bench v0.2 Coverage Gaps

The current benchmark covers signal grounding, confidence alignment, capacity grounding, pricing scope, scheduling handoff, safe prospect language, and related outreach constraints. Four Tenacious-specific behaviors still have zero direct task coverage and should be added in v0.2.

| Missing behavior in v0.1 | Why v0.1 cannot grade it | v0.2 addition |
|---|---|---|
| Warm-intro and relationship-chain handling | Tasks do not include private referral context, permission boundaries, or relationship strength. | Add `relationship_context` tasks where the agent must decide whether to mention a mutual contact, ask for permission, or avoid implying a warm intro. |
| CRM state and duplicate-touch suppression | Tasks are single-shot emails and do not encode prior sends, opt-outs, stale sequences, or recent human owner activity. | Add CRM-sequence tasks with `last_touch`, `owner`, `reply_state`, and `do_not_contact` fields. |
| Multi-stakeholder account routing | Tasks usually target one contact, so the benchmark cannot test whether the agent chooses the right buyer, champion, or technical evaluator. | Add account-map tasks with multiple contacts and a required recipient-selection step before writing. |
| Proof-point selection under confidentiality constraints | Tasks include capacity and signal data but not a library of approved proof points, NDA limits, or named-account redactions. | Add proof-library tasks where the agent must choose an allowed case-study summary and avoid restricted customer names. |

### Ground Truth Faithfulness Self-Critique

The existing scoring signal is lossy because many ground-truth fields are derived from public hiring, layoff, and redacted case-study signals. Public hiring signals can lag the actual buyer need or stack decision, so the benchmark may over-reward emails that confidently paraphrase a stale public posting and under-reward agents that hedge, ask a careful qualifying question, or avoid over-claiming when a real Tenacious seller would know the public signal is incomplete. This constraint should temper the headline numbers: the reported lift measures public-signal faithfulness, not full real-world account intelligence.

### One Unresolved Training Failure

The unresolved training-process failure is template/recipient artifacting. In the 60-task validation artifact, task `TB-V01-0175` still produced outputs such as `Subject: [email protected]` and `Hi Assistant`, showing that the model sometimes follows the email-format shell without resolving the true prospect recipient or producing a clean outreach message. DPO on 100 preference pairs for one epoch and the later `Subject:` generation anchor improved average score, but did not eliminate this artifact; the next training step should add a short supervised warm-up plus rejected pairs that specifically penalize placeholder recipients, assistant-addressed emails, and copied task scaffolding.

### Kill-Switch Trigger

If the trained component is ever placed in a limited production trial, disable the adapter and fall back to the prompt-engineered same-backbone baseline if human QA flags 8 or more of any rolling 100 generated emails for template-recipient artifacting, unsupported signal/capacity claims, or required rewrite before send. This trigger is observable without rerunning the benchmark and is calibrated to the economics above: the direct model-cost delta is currently $0.00 per task, so the relevant production risk is human-review waste and brand damage, and an 8% visible failure rate would outweigh a validation lift of only +3.250 points. The rollback action is immediate: turn off the LoRA adapter, keep the same prompt-engineered baseline, and route flagged accounts to human review until a new held-out run and QA sample clear the deployment gates.

## Evidence Used

| Artifact | Evidence |
|---|---|
| `outputs/evaluation_comparison.json` | Base mean 58.862, fine-tuned mean 62.112, 60-task validation comparison, per-task scores. |
| `outputs/training_summary.json` | 100 DPO examples, 1 epoch, learning rate 2e-5, train loss 0.5048. |
| `outputs/train_results.json` | Training runtime 105.5 seconds, 0.948 train samples/second. |
| `cost_log.md` | No eval-tier or dev-tier API calls were made for the repo upgrade. |
| `training/evaluate_trained_model.py` | Adapter loading, same-backbone comparison, deterministic generation, and scoring path. |
