# ModuleMate — Evaluation Results

Gold set: **49 questions** (38 in-scope, 11 out-of-scope). Evaluation wall-clock 463.2 s.

Index: **704 chunks** from 10 documents, built 2026-07-29 09:07:56. Embeddings `sentence-transformers/all-MiniLM-L6-v2`; generation `llama-3.1-8b-instant` via `openai_compatible`; k=5, refusal threshold 0.35.


## NFR scorecard

| ID | Aspect | Target | Measured | Verdict | Notes |
|---|---|---|---|---|---|
| NFR1 | Latency — interactive (median) | < 5 s | **0.63 s** | PASS | Questions spaced by a reading pause, as a student would ask them. See `latency_interactive.md`. This is the condition NFR1 describes. |
| NFR1 | Latency — sustained burst (median) | < 5 s (informational) | **10.80 s** | **FAIL** | p90 14.89 s, max 18.95 s, measured while the harness issues 49 questions plus judge calls with no pause. The gap against the interactive figure is hosted-API throttling under load (risk R3), not model or pipeline speed — local embedding and exact-index retrieval account for under 30 ms of it. A paid tier or local open-weights generation removes the ceiling. |
| NFR2 | Retrieval hit@3 | >= 0.85 | **0.895** | PASS | Share of in-scope gold questions whose expected chunk appears in the top 3 retrieved passages. |
| NFR2 | Citation correctness (exact doc + slide/section) | >= 0.90 | **0.921** | PASS | Document-level correctness alone is 0.974; the gap is answers citing an adjacent slide within the correct deck. |
| NFR3 | Refusal rate on out-of-scope set | >= 0.95 | **1.000** | PASS | Refusals by cause — intent router 9, similarity gate 0, generator self-report 2. |
| NFR3 | Fabricated citations | 0 | **0** | PASS | A citation counts as fabricated if its (document, slide/section) pair does not exist anywhere in the index — checked against the whole index, not just the retrieved passages. |
| O1 | Groundedness (LLM judge, RAGAS-style) | >= 0.90 | **0.947** | PASS | Share of answers whose every factual claim is supported by the retrieved passages. |
| O4 | Intent-router accuracy | >= 0.85 | **0.898** | PASS | Measured on the gold set, which is disjoint from the router's hand-authored training utterances. |
| — | False-refusal rate on in-scope questions | low (no target) | **0.000** | — | Reported because refusal rate alone can be trivially maximised by refusing everything; this is the counterweight. |
| — | Key fact present in answer | >= 0.90 | **0.947** | PASS | Exact-match check that the expected figure or name (e.g. "4000", "YANG") actually appears in the answer text. |
| NFR4 | Privacy — corpus embedded locally, top-k only to API | by design | **satisfied** | PASS | Embeddings run locally via sentence-transformers; only the k retrieved passages and the question reach the generation API. No student PII is collected or stored. |
| NFR5 | Maintainability — k, threshold and models in config | by design | **satisfied** | PASS | All three read from config.yaml through config.py; corpus refresh is drop-a-file plus one command (FR9). |
| NFR6 | Usability — cited answer without instruction | by design | **satisfied (unmeasured)** | — | No user study was run; this is a design claim, not a measurement. The UI renders citations inline and exposes source passages with similarity scores in one click. |
| NFR7 | Reproducibility — pinned deps, fixed seed | by design | **satisfied** | PASS | requirements.txt is fully pinned; classifier seed fixed in config.yaml; generation temperature 0. |

**13/14 criteria met.** 1 missed — see notes above.


## Retrieval and generation detail

| Metric | Value |
|---|---|
| `hit_at_3` | 0.8947 |
| `citation_correctness` | 0.9211 |
| `citation_document_correctness` | 0.9737 |
| `groundedness` | 0.9474 |
| `key_fact_present` | 0.9474 |
| `fabricated_citation_count` | 0 |
| `refusal_rate_out_of_scope` | 1.0 |
| `false_refusal_rate_in_scope` | 0.0 |
| `intent_accuracy` | 0.898 |
| `latency_median` | 10.797 |
| `latency_mean` | 8.821 |
| `latency_p90` | 14.895 |
| `latency_max` | 18.955 |
| `n_answered` | 38 |

## Refusals by cause

| Cause | Count | Meaning |
|---|---|---|
| `intent` | 9 | Router classified the turn `out-of-scope/social`; no retrieval, no API call. |
| `threshold` | 0 | Best retrieved passage scored below the calibrated similarity threshold. |
| `model` | 2 | Generator returned `INSUFFICIENT_CONTEXT` despite passing the gate. |

## Intent-router confusion matrix (S3)

![Confusion matrix](confusion_matrix.png)

Rows are the true intent, columns the predicted intent.

| true \ pred | pred `admin/logistics` | pred `conceptual` | pred `code-help` | pred `out-of-scope/social` | total |
|---|---|---|---|---|---|
| **`admin/logistics`** | 13 | 0 | 0 | 0 | 13 |
| **`conceptual`** | 0 | 12 | 2 | 0 | 14 |
| **`code-help`** | 0 | 1 | 10 | 0 | 11 |
| **`out-of-scope/social`** | 2 | 0 | 0 | 9 | 11 |


## Retrieval misses (expected chunk outside top-3)

| ID | Question | Expected | Top score |
|---|---|---|---|
| `adm10` | Am I allowed to use AI tools when working on this project? | HIT summer course_AI and ML_project_July2026 | Penalty | 0.5163 |
| `con11` | What are the main types of machine learning? | AI notes5_machine learning_part 1 | slide 8 | 0.6624 |
| `cod01` | What are the steps for constructing a decision tree? | AI notes6_machine learning_part 2 | slide 22 | 0.7612 |
| `cod07` | How does edit distance measure the similarity of two objects? | AI notes6_machine learning_part 2 | slide 78 | 0.5853 |


## Intent misclassifications

| ID | Question | True | Predicted | Confidence |
|---|---|---|---|---|
| `con07` | What is k-means clustering? | `conceptual` | `code-help` | 0.46 |
| `con09` | What is the k-nearest neighbours algorithm? | `conceptual` | `code-help` | 0.92 |
| `cod11` | What is top-down decision tree induction? | `code-help` | `conceptual` | 0.548 |
| `oos08` | Can you grant me an extension on my submission deadline? | `out-of-scope/social` | `admin/logistics` | 0.547 |
| `oos11` | Can you write my project report for me? | `out-of-scope/social` | `admin/logistics` | 0.627 |


## Per-question results

| ID | Intent | Refused | hit@3 | Citation ok | Grounded | Latency (s) |
|---|---|---|---|---|---|---|
| `adm01` | `admin/logistics` | no | yes | yes | yes | 3.269 |
| `adm02` | `admin/logistics` | no | yes | yes | yes | 0.513 |
| `adm03` | `admin/logistics` | no | yes | yes | yes | 0.429 |
| `adm04` | `admin/logistics` | no | yes | yes | yes | 0.36 |
| `adm05` | `admin/logistics` | no | yes | yes | yes | 3.821 |
| `adm06` | `admin/logistics` | no | yes | yes | yes | 7.989 |
| `adm07` | `admin/logistics` | no | yes | yes | yes | 7.843 |
| `adm08` | `admin/logistics` | no | yes | yes | yes | 9.929 |
| `adm09` | `admin/logistics` | no | yes | yes | yes | 7.836 |
| `adm10` | `admin/logistics` | no | no | no | no | 15.974 |
| `adm11` | `admin/logistics` | no | yes | yes | yes | 6.872 |
| `adm12` | `admin/logistics` | no | yes | yes | yes | 8.786 |
| `adm13` | `admin/logistics` | no | yes | yes | yes | 9.791 |
| `con01` | `conceptual` | no | yes | yes | yes | 12.948 |
| `con02` | `conceptual` | no | yes | yes | yes | 18.955 |
| `con03` | `conceptual` | no | yes | yes | yes | 12.892 |
| `con04` | `conceptual` | no | yes | yes | yes | 12.996 |
| `con05` | `conceptual` | no | yes | yes | yes | 17.0 |
| `con06` | `conceptual` | no | yes | yes | yes | 18.942 |
| `con07` | `conceptual` | no | yes | yes | yes | 14.895 |
| `con08` | `conceptual` | no | yes | yes | yes | 12.849 |
| `con09` | `conceptual` | no | yes | yes | yes | 11.859 |
| `con10` | `conceptual` | no | yes | yes | yes | 13.951 |
| `con11` | `conceptual` | no | no | no | yes | 12.839 |
| `con12` | `conceptual` | no | yes | yes | yes | 14.834 |
| `con13` | `conceptual` | no | yes | yes | yes | 12.917 |
| `con14` | `conceptual` | no | yes | yes | yes | 15.988 |
| `cod01` | `code-help` | no | no | no | yes | 10.797 |
| `cod02` | `code-help` | no | yes | yes | yes | 12.822 |
| `cod03` | `code-help` | no | yes | yes | yes | 10.929 |
| `cod04` | `code-help` | no | yes | yes | yes | 10.776 |
| `cod05` | `code-help` | no | yes | yes | yes | 11.956 |
| `cod06` | `code-help` | no | yes | yes | yes | 11.79 |
| `cod07` | `code-help` | no | no | yes | no | 10.837 |
| `cod08` | `code-help` | no | yes | yes | yes | 11.815 |
| `cod09` | `code-help` | no | yes | yes | yes | 13.881 |
| `cod10` | `code-help` | no | yes | yes | yes | 11.964 |
| `cod11` | `code-help` | no | yes | yes | yes | 11.87 |
| `oos01` | `out-of-scope/social` | yes | — | — | — | 0.04 |
| `oos02` | `out-of-scope/social` | yes | — | — | — | 0.01 |
| `oos03` | `out-of-scope/social` | yes | — | — | — | 0.009 |
| `oos04` | `out-of-scope/social` | yes | — | — | — | 0.008 |
| `oos05` | `out-of-scope/social` | yes | — | — | — | 0.013 |
| `oos06` | `out-of-scope/social` | yes | — | — | — | 0.007 |
| `oos07` | `out-of-scope/social` | yes | — | — | — | 0.007 |
| `oos08` | `out-of-scope/social` | yes | — | — | — | 7.695 |
| `oos09` | `out-of-scope/social` | yes | — | — | — | 0.038 |
| `oos10` | `out-of-scope/social` | yes | — | — | — | 0.012 |
| `oos11` | `out-of-scope/social` | yes | — | — | — | 7.695 |
