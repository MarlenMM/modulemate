# HIT Summer School origin and current implementation

ModuleMate originated as the Smart Chatbot project for HIT Summer School 2026's
Artificial Intelligence and Machine Learning course. The current app includes
subsequent portfolio engineering; it is not presented as an unchanged July submission.

## Source review

The 15 October review used the original two-page instructor project brief,
`HIT summer course_AI and ML_project_July2026.pdf`, and the submitted project
report in the user's private education archive. They are not redistributed here.

The instructor brief requires an AI/ML-related project and a written report,
with assessment of literature/background, project objectives and technologies,
design/implementation/testing, and report quality. Smart Chatbot is one suggested
topic. It explicitly permits AI assistance. It does **not** mandate a specific
embedding model, dataset, four-class router or database. Those were choices in
the submitted ModuleMate design.

## Traceability to the original design

| Original design objective | Version 2.1 implementation | Qualification |
|---|---|---|
| O1: ground substantive answers in evidence | Local sentence selection; optional structured generation with quote and support checks | A relevant sentence is not proof that it answers every implication |
| O2: cite document/location | Stable passage IDs, document names and original format locators | Local markers resolve in the source panel |
| O3: decline weak evidence | Combined retrieval filters, missing-detail notices, insufficient-evidence responses | Thresholds are development heuristics, not calibrated confidence |
| O4: classify intent | TF-IDF word/character features plus logistic regression; four original intent categories | Predictions guide style; only narrow social rules bypass evidence, so classifier mistakes do not block legitimate topics |
| O5: conversational continuity | Resolved topics, comparison-side references, clarification and scope-aware history | Deterministic dialogue state, not unrestricted coreference understanding |
| O6: measurable maintenance | Incremental indexing, preparation UI, ablations, separate training/evaluation fixtures and CI | Synthetic development results are not inherited course metrics |
| FR1/FR8: ingestion and source visibility | Six supported file formats, bounded parsers, expandable passages | Scanned diagrams still need OCR or text descriptions |
| FR2/FR5: persistent semantic retrieval | Local sentence-transformer vectors in SQLite, exact cosine search, word/dense rank fusion, sentence reranking and extractive reading | No managed vector service needed at the bounded course size |
| FR6: generation | Optional provider adapter with consent and validation | Local mode remains extractive and is labeled accordingly |
| FR9: refresh index | New uploads indexed incrementally; Prepare this course builds missing vectors | Current user library replaces the legacy fixed filesystem corpus |
| FR10: evaluation | Retrieval ablations, intent evaluation, conversation regressions, real-model integration tests | No current LLM-judged groundedness percentage is claimed |

## Changes since the summer submission

The project now supports private accounts, multiple user-owned courses, document
uploads, study cards and scheduling. The original single-course/no-account scope
therefore no longer applies to the product. It still demonstrates NLP, supervised
classification, pretrained neural representations, retrieval/reranking,
conversation state, grounded response construction and empirical evaluation.

The current work **does not fine-tune an LLM or embedding model on student data**.
The classifier learns from 120 original synthetic utterances. Encoder, reranker and reader
weights come from named pretrained models with pinned revisions. This distinction
is essential when explaining what was trained versus integrated.

Historical report scores remain historical. Current scores are computed from the
new original benchmark and published with denominators, failures and limitations.
