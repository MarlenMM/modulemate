# HIT Summer School origin and current implementation

ModuleMate began as the Smart Chatbot project for HIT Summer School 2026's
Artificial Intelligence and Machine Learning course. Version 2.2 is subsequent
portfolio development, not an unchanged July submission.

## Requirements checked against the source

The original two-page `HIT summer course_AI and ML_project_July2026.pdf` was read
again on 16 October. The submitted project report was reviewed privately for its
chosen objectives. Neither personal submissions nor instructor PDFs are bundled.

The instructor requires an AI/ML-related project and a 4000-word report (around
25 pages, with appropriate diagrams). Assessment covers literature/background,
objectives/significance/technologies, design/implementation/testing, and report
quality. Smart Chatbot and AI-powered search engine are listed as separate topic
suggestions. The brief permits AI assistance; it does not require a particular
model, fixed corpus, four-class router, account system, or GPU training.

The four intent categories, retrieval/generation design, and numbered objectives
below came from the student's proposed design, not additional instructor rules.

## Traceability

| Original design objective | Current implementation | Qualification |
|---|---|---|
| O1: ground substantive answers | Question-aware context, supporting citations, exact source-span selection, optional factual review and bounded correction | Explanations connect supplied material; missing information is disclosed. Labeled illustrations may demonstrate a provided concept. Citation validity is not proof of truth |
| O2: cite documents and locations | Cited passage IDs, original names and page/slide/section/cell locators | Only cited evidence is attached to a generated answer |
| O3: decline weak evidence | Explicit insufficient/clarification results survive the entire pipeline; invalid drafts are not replaced with unrelated answers | The model may still misjudge answerability; evaluation includes distractor/missing-policy cases |
| O4: classify intent | Original four-label TF-IDF/logistic-regression experiment remains reproducible | The generative tutor follows the actual question instead of letting the tiny classifier block or misroute it |
| O5: conversational continuity | Recent scoped turns, comparison references, topic changes, learner feedback and ordered lesson continuation | Context is bounded; new document selection drops incompatible history |
| O6: measurable maintenance | Offline regression tests, real-weight checks, original conversation challenge and recorded model outputs/timing | Constraint checks are diagnostics, not a semantic accuracy percentage |
| FR1/FR8: ingest and inspect sources | Six formats, isolated parsers, heading-aware chunks and source panels | Image-only pages/diagrams still need supplied OCR/text |
| FR2/FR5: local semantic retrieval | Persistent local vectors, word/dense rank fusion, optional reranker/reader and context expansion | Pretrained weights are integrated, not trained on coursework |
| FR6: generation | Offline MLX instruction model or explicitly enabled compatible provider | Excerpt mode remains available but is clearly a limited fallback |
| FR9: refresh material | Incremental document/vector ingestion and deletion cascades | No inherited fixed university corpus is imported |
| FR10: evaluate | Adversarial original fixtures, numerical routines, scope/migration/UI tests and manual reading of actual answers | Old report and 2.1 results remain historical |

## What was trained and what was integrated

The small intent classifier is trained on 120 original synthetic utterances.
The encoder, reranker, extractive reader and instruction models use named pretrained
weights with pinned revisions. No LLM or embedding model is fine-tuned on personal
student data. Bounded numeric tools compute statistics/arithmetic directly; they
are not a claim that a language model learned exact arithmetic.

The current single-user local scope removes the later account system. Courses,
uploads, conversation, cards and planning remain useful portfolio extensions.
Documentation describes both the original course identity and the engineering
changes since the submission. A project report can explain the retrieval/generation
tradeoff, experiments, observed failures and limitations using the current source;
new measurements must not be confused with historical submission results.
