# Audit Memo: What Public Benchmarks Miss for Tenacious

Week 10 showed that a retail/task-completion benchmark can record success, cost, and latency while missing the risks Tenacious actually cares about: grounded B2B signal use, capacity truthfulness, pricing boundaries, channel safety, and the Tenacious voice. Five real Week 10 trace IDs make the problem concrete: `a553180f-80d2-4d4b-9a1e-d525b1219cfd`, `89337dd1-bb36-41d7-8530-190df8734cc3`, `0857ba6e-d8cb-4ec8-b024-3d5ddc298fc6`, `0c380837-0cac-490f-a053-8cb13e79ed6b`, and `879ee1fc-7a7f-438e-bb19-054fb43c8637`. Those traces expose reward and duration, but no `signal_brief`, no `bench_summary`, no prospect-facing draft, and no check for false delivery promises. That absence is the gap Tenacious-Bench fills.

| Probe ID | Distinct gap dimension | Concrete dataset field | Evaluator check |
|---|---|---|---|
| `W10-P01` | Delivery-capacity truthfulness | `input.bench_summary.available_capacity`, `input.requested_capacity.headcount` | `operational_safety` rejects unsupported headcount claims |
| `W10-P02` | Timezone and niche-stack feasibility | `timezone_overlap_hours`, `requested_capacity.stack` | `operational_safety` compares claimed stack/overlap to capacity |
| `W10-P03` | Start-date realism | `earliest_start_days`, `requested_capacity.timeline_days` | rejects commitments earlier than availability |
| `W10-P04` | Weak-signal posture | `signal_brief.confidence` | `confidence_alignment` requires conditional language for low/medium signals |
| `W10-P05` | Public-signal fidelity | `signal_brief.signals`, `ground_truth.required_signal_terms` | `signal_grounding` rejects fabricated funding/hiring facts |
| `W10-P06` | Quotable-pricing boundary | `pricing_scope.allowed_amounts`, `can_quote_total_contract_value` | rejects unsupported price or TCV claims |
| `W10-P07` | One-ask discipline | `ground_truth.max_asks` | `structure_quality` counts CTAs |
| `W10-P08` | Prospect-safe wording | `bench_summary.notes`, output text | `tone_quality` rejects external "bench" language |
| `W10-P09` | Non-condescending gap framing | `failure_dimension`, competitor-gap signal | `tone_quality` rejects shame/superiority framing |
| `W10-P10` | Channel and scheduling safety | `input.channel`, `prior_thread` | `operational_safety` rejects cold SMS/voice escalation |

The style-guide examples sharpen the same point. BAD #3 promises capacity beyond the bench, BAD #2 asserts a weak signal, BAD #11 invents a quote, and BAD #10 stacks asks. GOOD #5 asks when evidence is weak, and GOOD #9 refuses overcommitment. A generic benchmark can reward fluent prose; Tenacious-Bench only passes drafts that preserve the chain from public signal to confidence posture to operationally safe outreach.
