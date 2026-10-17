# ModuleMate 2.3 evidence learning validation

These experiments were run on an Apple M4 Pro with 24 GiB memory. They use original
fictional text, not private uploads or university material. The conversational
model is the existing pinned local Qwen3.5 9B 4-bit model. No hosted inference,
external GPU or LLM fine-tuning was used.

## Supervised models

The [model/data card](EVIDENCE_MODEL_CARD.md) describes 22 documents, 154 questions,
30 features and three trained logistic heads, with lexical/embedding/reader
ablations. Twelve documents are fitted, five calibrate probabilities, and five
are reserved for testing. Derived negatives and whole-document bundles remain
inside their document partition. The test models were frozen before inspection;
later transfer conversations were not fitted. This small authored diagnostic
does not estimate representative user accuracy.

| Frozen test, 315 candidate/bundle examples | Answerability AP | Brier | ECE, 10 bins |
|---|---:|---:|---:|
| Original lexical-feature head | 0.9028 | 0.0451 | 0.0430 |
| With pretrained embedding signal | 0.8962 | 0.0426 | 0.0261 |
| With embedding and reader signals | 0.9100 | 0.0420 | 0.0318 |

AP means average precision, not percentage of correct chatbot answers. Calibration
measurements are imperfect on so few document groups. At threshold 0.85, the reader
variant accepts 36 examples with four errors; the embedding variant accepts 26
with one error. More coverage is a tradeoff, not unconditional improvement.
The reader has 18 mistakes at threshold 0.5, including confident false positives
on missing salt concentration and incomplete compound questions. The runtime
therefore never uses these scores to force an answer or refusal.

The candidate-ranking test uses 25 answerable questions against all 20 passages in
the five test documents. The implemented lexical/dense **candidate rank-fusion
diagnostic** has a first relevant hit at rank one on 20/25 questions. A 0.25 learned
embedding-feature vote gives 22/25; the reader-feature variant gives 23/25.
All required gold passages remain within three for the fusion variants. This is
not an end-to-end SQL retrieval or chatbot accuracy result. Pure lexical learned
ranking reaches 24/25 here but performed worse on calibration; it was not promoted
based on the test result. On the separate historical 46-query actual retrieval
fixture, the embedding learned vote trades 42 first hits for 41, retaining 46/46
hits at three. These negative results remain visible.

## Same-generator evidence-policy comparison

The transfer fixture has six original documents and eight turns, including a
reference to another file, a follow-up, conflicting versions, partial exercise
help, a reserved book, an unnamed sponsor and rewriting supplied text.

All three retained runs use the same inference-code/model-artifact fingerprint
`778f63e7762c65bafe3eef204d56bdaedb6992c3eed9960701d7208dc4289bd7`.
Neither code nor fixture changed during these runs. The full policy provides a
larger context allowance; these are policy/cost comparisons, not equal-token
experiments. Medians include first-request loading and are machine/run specific.

| Policy | Constraint checks | Median seconds |
|---|---:|---:|
| Existing evidence selection | 5/8 | 10.29 |
| Full selected documents | 7/8 | 17.34 |
| Adaptive evidence | 7/8 | 11.60 |

Read the [retained answers and actual source chunks](../evaluation/observations/evidence-v23.json).
Every retained chunk was checked against a fresh parse of the original fixture,
and the record also contains test errors and source/data/model fingerprints.

Manual reading matters more than the table:

- The baseline misses the first footwear answer because keyword narrowing
  excludes the explicitly referenced equipment file. Adaptive selection follows
  that uploaded-file reference; the full-document baseline also finds the rule.
- The baseline's sandals follow-up states the correct prohibition but labels it
  insufficient. That constraint failure is not a wrong factual answer.
- Both new policies correctly distinguish the current seven places, the meeting
  point/time and price, the reservation condition and the absent sponsor. The
  supplied-text rewrite avoids library retrieval entirely.
- Partial weight help remains conservatively labeled insufficient. It identifies
  missing measurements and mentions assembly, but awkwardly reads the heading
  'Limits' as part of the assembly description. A writer-only heading-removal
  trial removes that wording without changing stored source evidence; its
  separate observation is retained in the presentation follow-up.
- A strict quantity request still produces a decorative bracketed label, and
  a source-evidence footer duplicates the source panel. These are presentation
  failures, not fixed by a good retrieval score.

The transfer fixture was authored after calibration and is now a known development
diagnostic. Constraint passes do not prove complete support or general intelligence.

## Rejected verifier and other limits

The optional independent NLI model uses a pinned safetensors revision and offline
inference. On 15 short original calibration claims it gets 12 labels correct.
Actual paragraph answers produce many false flags, including valid dates,
definitions and source-attribution phrasing. The final retrospective diagnostic
flags 33 of 47 historical original-fixture answers; that count is **not** the number
of wrong answers. Manual inspection confirms both false flags and real unsupported
details. It is disabled by default and never automatically rejects/rewrites answers.

Reproduce that experiment with `scripts/prepare_support_model.py`,
`scripts/evaluate_support.py` and `scripts/evaluate_recorded_support.py` using the
ML extras. The public pretrained verifier is integrated, not trained here.

The generation model can still infer an unstated detail or over-refuse. The earlier
ten-turn exploratory comparison removed a story misunderstanding but still added
a dark-cupboard setting to a source-derived seedling example. No claim is made that
these semantic problems have all been solved. Citation binding proves source
identity, not entailment. Local feedback is a review label, not verified truth;
private corrections are not automatically shared or fitted.

## Engineering checks

At the evidence-policy checkpoint, 219 offline tests pass and eight model tests
are skipped deliberately. Those eight real-weight checks pass separately,
including scoped batch retrieval, learned reader/source-span binding and offline
inference with network connections denied. Lint/formatting, dependency audit and
the release-tree guard pass. Neural files remain outside Git.

Tests cover split leakage/provenance, reproducible numerical artifacts, corruption,
source deletion and scope changes, model identity changes, query batching, bounded
recovery, document links, source-number remapping, missing models, feedback storage
and export, fresh input transformations and preserved lesson continuation.
Streamlit AppTest exercises the feedback flow. Additional presentation and final
UI checks are recorded below when completed; this is not a live-browser inspection.
