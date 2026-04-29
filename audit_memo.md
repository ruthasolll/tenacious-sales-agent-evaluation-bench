# Week 11 Audit Memo: Tenacious-Bench Day 1

## Overview

Tenacious outreach is a constraint-aware decision system, not a tone optimization problem. The style guide's good examples work because they connect tone to operational truth: grounded signal, confidence-aware phrasing, one ask, clean formatting, and human routing when capacity or pricing cannot be verified. This matters because `bench_overcommitment` -- promising engineers beyond real delivery capacity -- is the highest-revenue-loss failure mode, occurs late in the sales cycle, and can sound professional while being false.

## Supporting Trace Evidence

The available Week 10 `trace_log.jsonl` contains 150 tau2 retail traces, not B2B outreach traces. Using task/simulation as the trace key, five real examples show the gap: task 11 / `a553180f-80d2-4d4b-9a1e-d525b1219cfd` failed with reward 0; task 34 / `89337dd1-bb36-41d7-8530-190df8734cc3` failed; task 76 / `0857ba6e-d8cb-4ec8-b024-3d5ddc298fc6` failed; task 104 / `0c380837-0cac-490f-a053-8cb13e79ed6b` failed; task 105 / `879ee1fc-7a7f-438e-bb19-054fb43c8637` failed after 1192 seconds. These traces record retail reward, cost, and duration, but no signal brief, bench summary, capacity claim, confidence flag, or prospect-facing message. That absence is itself evidence: the existing benchmark cannot grade Tenacious's core risk.

## Key Failure Patterns

Tone failures include vendor cliches, fake urgency, and condescension: BAD #7 invents scarcity; BAD #4 says the prospect is "behind the curve." Signal failures include weak-signal assertion and fabrication: BAD #2 turns 2 open roles into "scaling aggressively," while BAD #12 fabricates a "$40M Series C." Structural failures include stacked asks and premature contracts: BAD #10 asks for four decisions; BAD #11 invents a $1.2M quote. The most severe failure is operational: BAD #3 promises "12 senior Go engineers in two weeks" when the bench has only 4. This is the pattern generic tone checks miss.

## Key Success Patterns

GOOD #1 grounds personalization in a "$14M Series A" and Python roles rising from 2 to 7. GOOD #5 handles weak evidence correctly: "I cannot tell from the outside" and asks rather than asserts. GOOD #4 frames a capability gap as "two readings," not a leadership failure. GOOD #6 uses a low-friction one-ask resource touch. Most important, GOOD #9 refuses overcommitment: 15 engineers in 30 days is at the edge of capacity, so the agent confirms only 6 to 8 in 21 days and offers a 60-day ramp or referral.

## Benchmark Gaps

tau2-Bench or generic outreach benchmarks would reward fluency, politeness, and task completion, but not delivery truth. They would not know that "bench" is banned externally, that Go capacity is 4, that weak signals require interrogative language, or that multi-phase total contract values route to humans. They also under-detect implicit guarantees. "Engineers in your Slack by next Friday" is not just optimistic copy; it is an operational commitment requiring bench-summary support.

## Evaluation Implications

Tenacious-Bench should make these checks machine-verifiable: at least one grounded signal is cited; signal confidence matches assertive versus conditional wording; capacity claims extract headcount, stack, seniority, overlap, and timeline; extracted claims are compared against `bench_summary`; unsupported capacity is a hard failure; pricing is limited to public bands; banned phrases, external "bench" language, fabricated events, fake urgency, and multi-ask stacking are penalized. Dataset fields must therefore include `signal_brief`, `signal_confidence`, `bench_summary`, `requested_capacity`, `pricing_scope`, `prior_thread`, `candidate_output`, and hard-fail labels.

## Risky Edge Cases and Conclusion

Weak signals, high-pressure hiring asks, and implicit overpromising are the edge cases v0.1 must target. A passing agent should sound Direct, Grounded, Honest, Professional, and Non-condescending, but tone is insufficient. Tenacious-Bench must measure the chain: signal evidence -> confidence posture -> capacity/pricing gate -> safe outreach. Overcommitment should fail even when the prose is concise and persuasive.
