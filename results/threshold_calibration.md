# Refusal threshold calibration (M5)

Calibrated against the gold set: **38 in-scope** and **11 out-of-scope** questions, scored against the built index (698 chunks).

## Top-1 similarity distributions

| Set | n | min | mean | median | max |
|---|---|---|---|---|---|
| In-scope | 38 | 0.386 | 0.650 | 0.664 | 0.843 |
| Out-of-scope | 11 | 0.122 | 0.293 | 0.239 | 0.697 |

Separation gap: in-scope minimum **0.386** vs out-of-scope maximum **0.697**.

## Sweep

| threshold | in-scope retained | out-of-scope refused |
|---|---|---|
| 0.05 | 100.0% | 0.0% |
| 0.10 | 100.0% | 0.0% |
| 0.20 | 100.0% | 36.4% |
| 0.35 | 100.0% | 81.8% **<- selected** |
| 0.40 | 97.4% | 81.8% |
| 0.55 | 84.2% | 90.9% |
| 0.80 | 7.9% | 100.0% |

## Selected threshold: `0.35`

Selection rule: maximum Youden's J (in-scope retention + out-of-scope rejection - 1), tie-broken to the centre of the optimal plateau.

At this value 100.0% of in-scope questions still reach the generator and 81.8% of out-of-scope questions are refused by the similarity gate alone — before the intent router's own short-circuit is taken into account.

## Why the gate is not calibrated to reject everything

These out-of-scope questions score at or above the selected threshold, because they genuinely are *about* the corpus. They are out of scope by policy, not by topic:

* `0.697` — Can you write my project report for me?
* `0.538` — Can you grant me an extension on my submission deadline?

Raising the threshold far enough to reject them would also reject most legitimate questions, since they outscore the in-scope floor. This is the structural reason the architecture has two independent refusal layers: the similarity gate handles *topical* drift, and the four-class intent router short-circuits *policy* cases (`out-of-scope/social`) before retrieval runs at all. Neither layer is sufficient alone, and NFR3 is measured against the two of them combined.
