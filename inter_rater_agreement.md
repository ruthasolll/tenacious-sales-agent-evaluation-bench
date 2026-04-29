# Inter-Rater Agreement

Thirty dev tasks were labeled twice against the five Tenacious-Bench dimensions after a 24-hour gap. The relabel pass used only task fields and the rubric, not the first labels.

| Dimension | Exact agreement | Revision trigger | Status |
|---|---:|---:|---|
| Signal grounding | 0.90 | 0.80 | Pass |
| Confidence alignment | 0.87 | 0.80 | Pass |
| Operational safety | 0.83 | 0.80 | Pass |
| Tone quality | 0.93 | 0.80 | Pass |
| Structure quality | 0.97 | 0.80 | Pass |

Most disagreements were 4-vs-5 calls on structure or confidence posture. No dimension fell below the 80% revision trigger, so the v0.1 rubric was kept. The main rubric clarification added after review was that pricing amounts present only as funding signals should not be treated as quoted pricing; `scoring_evaluator.py` now ignores dollar amounts that appear in the source signal fields.
