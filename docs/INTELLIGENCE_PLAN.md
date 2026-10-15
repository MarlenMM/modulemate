# ModuleMate intelligence improvement cycle — 15 October 2026

## Evidence and diagnosis

Reviewed the original two-page HIT AI/ML project brief in the local Education
archive and the submitted ModuleMate report (especially objectives O1–O6 and
FR1–FR10). The official brief asks for an AI/ML project, literature survey,
objectives, design, implementation and testing; Smart Chatbot is a suggested
topic. It permits AI assistance. The detailed router/RAG requirements belong to
the student's proposed design, not a prescribed instructor architecture.

Version 2's private library is worth preserving. Its default FTS keyword lookup,
whole-chunk dumps, shallow follow-up concatenation and 12 easy lexical checks
are insufficient evidence of a smart chatbot. The former fixed university corpus
and its reported metrics must not be restored or reused as new performance claims.
The course documents and submitted report remain private reference material.

## Design decisions

- Keep private accounts, source ownership, consent for hosted generation and all
  study workflows. Keep an empty public library; use original evaluation content.
- Make local pretrained ML useful without a hosted API: sentence embeddings,
  persistent per-chunk vectors, hybrid lexical/dense retrieval and cross-encoder
  reranking. Model files are public weights; documents never go to model hosting.
- Install model dependencies as a reproducible optional `ml` extra, with an explicit
  one-command model preparation step and visible local-model status in the UI.
  The lightweight installation still gives focused extractive answers. Never
  silently download weights during a question, or call a hosted model implicitly.
- Train a small reproducible TF-IDF/logistic intent model on original utterances.
  Its uncertain predictions guide response style, not blanket domain rejection.
  Social turns use a narrow guard; evidence decides whether a question is covered.
- Compose short source-grounded answers locally with sentence selection,
  redundancy reduction, comparison coverage, extractive outlines and recall
  prompts. Label extraction honestly. Do not pretend it is free-form reasoning.
- Resolve follow-ups from validated current-scope source context; clarify ambiguous
  references. A new topic must replace, rather than accumulate, old context.
- Preserve provenance and source quotes, filter weak matches, and keep independent
  hosted answer support verification. Distinguish relevance from entailment.

## Iteration 1 — retrieval and ML foundation

1. Record lexical baseline and write original multi-domain evaluation cases.
2. Add pinned local encoder/reranker setup, lazy loading and bounded inference.
3. Migrate SQLite non-destructively for model-versioned vectors, with deletion
   cascades, finite-value/shape validation and authorization at read/write time.
4. Fuse lexical and semantic candidates, rerank, deduplicate, expose retrieval
   diagnostics and preserve filters. Test incremental indexing and cache freshness.
5. Add reproducible intent training and separate evaluation utterances.

## Iteration 2 — conversational answers

1. Route greetings/help, explanation, comparison, overview and guided practice.
2. Resolve source/first/second and topic follow-ups; handle missing context safely.
3. Rank complete source sentences, favor question-specific evidence and avoid
   whole-page dumps. Keep both comparison sides and report missing sides.
4. Offer source-grounded follow-up actions. Persist notices and retrieval metadata
   so a rerender does not hide limitations or change meaning.
5. Render readable, safely escaped prose with citations and expandable sources.
   Keep source text untrusted and avoid externally loaded images/links.

## Iteration 3 — evaluation and hardening

1. Run lexical vs dense vs hybrid/reranked ablations on the same original corpus:
   hit@1, hit@3, MRR, weak-evidence rejection and warm/cold latency separately.
2. Evaluate conversation continuity, concise answer coverage, summary diversity,
   comparison completeness, unsupported questions, multilingual lexical fallback,
   malicious source text and document deletion/account isolation.
3. Test live local models, the configured hosted provider on synthetic notes, and
   Streamlit user journeys. Record failures and repair them before reporting.
4. Update setup, architecture, HIT provenance, limitations and measured results.
   Run lint, formatting, tests, dependency audit and release checks; commit work.

## Acceptance and limits

The no-key experience must answer a focused question with concise cited source
sentences, handle a contextual follow-up, compare two supported concepts, guide
recall and decline unsupported details. Local ML must demonstrably retrieve
paraphrases missed by lexical search, without weakening ownership or deletion.
Model/setup failures must produce an actionable local fallback. No download or
model load may hold a database write lock. All quality figures identify datasets,
backend and scope; neither tiny synthetic sets nor self-verification establish
general academic accuracy. The historical report stays historical: later portfolio
engineering is documented separately rather than represented as summer work.


## Completed cycle — 16 October 2026

Implemented the three planned iterations, then extended answer selection after
reviewing complete responses. Added a pinned local SQuAD2 reader constrained to
an already selected source, English stemming, detail/ordering checks, antecedent
context, CJK sentence boundaries, recall-attempt continuity and exact-identifier
constraints. The larger benchmark exposed an identifier regression, which was
repaired before release. Passage reranking is available for ablation and disabled
by default because the measured hybrid recall was better.

The current retrieval set improved hit@1 from 38/46 to 42/46 and hit@3 from 43/46
to 46/46. Complete-answer reference-fact coverage is 45/46 development questions
and 18/20 additional challenge questions. These are limited synthetic checks;
misses and weak classifier results remain reported. See `VALIDATION.md` for
reproduction, live-provider failures and remaining limits.
