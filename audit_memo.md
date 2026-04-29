# Week 11 Audit Memo: Tenacious-Bench Day 1

## Overview

Tenacious outreach is not just "good sales tone." The style guide defines a decision system: ground claims, match wording to confidence, keep one ask, avoid vendor cliches, and route unsupported capacity or pricing to humans. This matters because `bench_overcommitment` -- promising engineers beyond real delivery capacity -- is the highest-revenue-loss failure mode, appears late in the sales cycle, and can sound polished while being operationally false. Tenacious-Bench must measure evidence discipline, not tone alone.

## Key Failure Patterns

- **Unsupported capacity claim:** The agent commits headcount, stack, or timeline without checking availability. Evidence: BAD #3 promises "12 senior Go engineers in two weeks" when the bench has only 4 senior Go engineers; BAD #12 promises a 15-engineer team in 30 days.
- **Weak-signal assertion:** Ambiguous signals become confident claims. Evidence: BAD #2 turns 2 open roles into "scaling aggressively" and assumed recruiting pain. This is a tone failure and a grounding failure.
- **Fabricated context:** Public facts or buying windows are invented, then used to justify outreach. Evidence: BAD #12 fabricates a "$40M Series C."
- **Pressure and scope stacking:** The agent creates urgency, discounts, contracts, or multiple asks to preserve momentum. Evidence: BAD #7 invents a Q1 slot and discount; BAD #10 stacks four asks; BAD #11 invents a $1.2M quote.

These patterns create late-stage trust collapse after buying intent already exists.

## Key Success Patterns

- **Grounded specificity:** Good drafts name verifiable signals, not generic personalization. Evidence: GOOD #1 cites the "$14M Series A" and Python roles moving from 2 to 7.
- **Confidence-aware language:** Weak signals trigger questions. Evidence: GOOD #5 says "I cannot tell from the outside" and asks whether demand is larger than postings suggest.
- **Bench-gated honesty:** The agent refuses unsupported delivery. Evidence: GOOD #9 says 15 engineers in 30 days is at the edge of capacity, confirms 6 to 8 in 21 days, and offers a 60-day ramp or referral.
- **Disciplined structure and tone:** Good drafts stay short, use one ask, and frame gaps as research rather than executive failure. Evidence: GOOD #6 only asks whether to send a PDF; GOOD #4 offers "two readings" for absent MLOps roles.

## Benchmark Gaps

tau2-Bench or generic outreach benchmarks would likely reward fluency, politeness, and task completion, but miss the difference between "well-written" and "deliverable." They would not know that Go capacity is 4, that "bench" is banned externally, or that multi-phase total contract values require human handoff. Generic personalization checks might accept any company-specific phrase; Tenacious requires signal grounding with counts, dates, named peers, confidence, and segment fit.

The hardest gap is implicit overcommitment. "Engineers in your Slack by next Friday" is not merely optimistic phrasing; it is an operational guarantee requiring bench-summary support. Existing benchmarks often score explicit wording, not the hidden business claim.

## Risky Edge Cases

Weak signals are risky because fluent models convert ambiguity into authority. High-pressure hiring asks are worse: requests for 12 to 15 engineers invite the agent to keep the deal alive by overpromising. Implicit overpromising is hardest: "we move fast," "contracts by Wednesday," or "plug a team in" may imply capacity without naming a number. The style guide guards against this through honesty and bench-to-brief checks, but it does not fully specify how to score implied guarantees.

## Conclusion

Tenacious-Bench must measure the chain: signal evidence -> confidence posture -> capacity/pricing gate -> safe outreach. A passing agent should sound Direct, Grounded, Honest, Professional, and Non-condescending, but that is insufficient. It must refuse unsupported commitments, detect implicit guarantees, route risky asks to humans, and treat overcommitment as disqualifying even when the prose is concise.
