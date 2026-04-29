# Target Failure Mode: Bench Overcommitment

## Selected Failure
**Bench Overcommitment Failure Mode**: the system over-promises talent availability/readiness (fit, seniority, timezone overlap) beyond what delivery can actually staff.

## Why This Failure Matters
In Tenacious-style outsourcing motions, this failure occurs near late qualification and directly affects close probability. Unlike messaging-only errors, overcommitment often causes hard trust breaks after technical validation.

## Category Trigger Rates Used
From the current adversarial probe library:
- `bench_overcommitment` aggregate trigger rate: **0.5433**
- `tone_drift` aggregate trigger rate: **0.3400**
- `scheduling_edge_cases` aggregate trigger rate: **0.3300**

## Business Cost Arithmetic (Bench Overcommitment)
Assumptions:
- Average Contract Value (ACV) = **$25,000**
- Qualified leads per month = **40**
- Trigger rate (`bench_overcommitment`) = **0.5433**
- Failure-attributed conversion drop = **35%** (`0.35`)

Step 1: affected leads  
`affected_leads = 40 × 0.5433 = 21.732`

Step 2: lost deals  
`lost_deals = 21.732 × 0.35 = 7.6062`

Step 3: monthly loss  
`monthly_loss = 7.6062 × $25,000 = $190,155`

Step 4: annual loss  
`annual_loss = $190,155 × 12 = $2,281,860`

## Alternative Comparison

### Alternative A: Tone Drift
Assumptions:
- Trigger rate = **0.3400**
- Conversion drop = **12%** (`0.12`)

Arithmetic:
- `affected_leads = 40 × 0.34 = 13.6`
- `lost_deals = 13.6 × 0.12 = 1.632`
- `monthly_loss = 1.632 × $25,000 = $40,800`
- `annual_loss = $40,800 × 12 = $489,600`

### Alternative B: Scheduling Edge Cases
Assumptions:
- Trigger rate = **0.3300**
- Conversion drop = **10%** (`0.10`)

Arithmetic:
- `affected_leads = 40 × 0.33 = 13.2`
- `lost_deals = 13.2 × 0.10 = 1.32`
- `monthly_loss = 1.32 × $25,000 = $33,000`
- `annual_loss = $33,000 × 12 = $396,000`

## Comparative Conclusion
- Bench overcommitment annual risk: **$2,281,860**
- Tone drift annual risk: **$489,600**
- Scheduling edge cases annual risk: **$396,000**

Bench overcommitment is approximately:
- `2,281,860 / 489,600 ≈ 4.66x` costlier than tone drift
- `2,281,860 / 396,000 ≈ 5.76x` costlier than scheduling edge cases

## Final Justification
This failure mode is selected because it combines a high trigger rate with direct late-stage revenue impact, producing the highest projected loss and the best ROI opportunity for mitigation.

