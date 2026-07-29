# Refusal threshold calibration (M5)

Calibrated against the gold set: **52 in-scope** and **11 out-of-scope** questions, scored against the built index (1799 chunks).

## Top-1 similarity distributions

| Set | n | min | mean | median | max |
|---|---|---|---|---|---|
| In-scope | 52 | 0.457 | 0.653 | 0.658 | 0.843 |
| Out-of-scope | 11 | 0.178 | 0.358 | 0.360 | 0.524 |

Separation gap: in-scope minimum **0.457** vs out-of-scope maximum **0.524**.

## Sweep

| threshold | in-scope retained | out-of-scope refused |
|---|---|---|
| 0.05 | 100.0% | 0.0% |
| 0.10 | 100.0% | 0.0% |
| 0.20 | 100.0% | 18.2% |
| 0.40 | 100.0% | 63.6% |
| 0.48 | 98.1% | 81.8% **<- selected** |
| 0.55 | 88.5% | 100.0% |
| 0.80 | 7.7% | 100.0% |

## Selected threshold: `0.48`

Selection rule: maximum out-of-scope rejection subject to in-scope retention >= 98% (a false refusal costs both an answer and a hit@3, and two further refusal layers back this one up).

At this value 98.1% of in-scope questions still reach the generator and 81.8% of out-of-scope questions are refused by the similarity gate alone — before the intent router's own short-circuit is taken into account.

## Why the gate is not calibrated to reject everything

These out-of-scope questions score at or above the selected threshold, because they genuinely are *about* the corpus. They are out of scope by policy, not by topic:

* `0.524` — Can you write my project report for me?
* `0.515` — Can you grant me an extension on my submission deadline?

Raising the threshold far enough to reject them would also reject most legitimate questions, since they outscore the in-scope floor. This is the structural reason the architecture has two independent refusal layers: the similarity gate handles *topical* drift, and the four-class intent router short-circuits *policy* cases (`out-of-scope/social`) before retrieval runs at all. Neither layer is sufficient alone, and NFR3 is measured against the two of them combined.
