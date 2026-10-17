# Evidence learning development

This work builds on the 2.2 conversational assistant. The pretrained generator
remains the language engine. Original supervised evidence decisions, controlled
experiments, failure analysis and reproducible artifacts are the contribution
being developed here. A new model or extra processing stage must earn its place.

## Working plan

1. Record the existing test/retrieval baseline. Keep a selectable baseline path
   for controlled comparisons with the same generator, documents and scope.
2. Build an original, English, multi-subject evidence dataset. Group examples
   and derived negatives by source document; keep training, calibration and test
   groups disjoint. Include answerable, incomplete, absent and misleading cases.
   Never fit on the existing conversation development set or private uploads.
3. Train small supervised passage relevance and answerability heads. Record
   features, labels, source hashes, split membership and hyperparameters. Store
   inspectable numerical parameters rather than executable pickle artifacts.
4. Evaluate relevance rankings, answerability probabilities and selective
   decisions against simple baselines. Report counts, misses and calibration;
   the generated dataset is a limited diagnostic, not general intelligence.
5. Use learned scores to prioritize evidence without allowing a small classifier
   to veto a useful model answer. Evaluate bounded evidence recovery and
   per-question-part coverage. Preserve explicit user document boundaries.
6. Test an independent pretrained entailment checker on original supported,
   contradicted and unsupported claims, including actual 2.2 failure patterns.
   Pin weights and label mapping. Use it only where validation supports its use;
   never represent its predictions as a truth guarantee.
7. Simplify or repair context decisions exposed by the comparisons. Read whole
   short documents where useful, preserve both sides of multi-part requests,
   and disclose incomplete context. Prevent keyword rules from excluding a
   semantically relevant document without evidence that this helps.
8. Inspect actual local-model answers, not just phrase matches. Keep latency,
   evidence and model/corpus/code fingerprints. Revisit errors and repeat the
   experiments as new bugs or useful ideas emerge.
9. Review caching, inference cost, model failures, malformed artifacts, source
   deletion and user scope. Update the UI only where it helps understand answers.
10. Update architecture, model/data cards, HIT alignment and measured limitations;
    run lint, release checks, meaningful regression tests and real-model checks.

## Validation strategy

- Fast tests: source boundaries, numerical artifact validation, feature stability,
  split integrity, deterministic inference, missing-model behavior and recovery.
- Experiments: fixed document-group splits, ranking hit@k/MRR, answerability
  precision/recall and probability calibration, selective error versus coverage.
- Integration: actual retrieved passages, isolated SQLite fixtures, local pretrained
  model inference and a small number of complete original conversations.
- Failure analysis: read every misclassified test case and representative actual
  answers; distinguish retrieval, evidence judgment, generation and presentation.
- Keep the final test split frozen. Once inspected it is a known diagnostic;
  later design changes require fresh transfer examples before accuracy claims.

No external GPU is needed for the initial small supervised models. University
documents, personal chats, runtime databases and model weights remain untracked.
This log will record rejected ideas as well as adopted improvements. There is no
claim that a finite test suite can establish perfect chatbot intelligence.

## Development cycles, 18 October 2026

1. **Baseline and original supervision.** Recorded 174 passing offline tests and
   the existing 46-query neural retrieval results. Authored 22 multi-subject
   documents/154 questions with disjoint document-group splits and explicit gold
   and partial evidence. Built pairwise ranking, relevance and answerability heads
   with disjoint probability calibration and inspectable JSON artifacts.
2. **Evidence policy and comparison.** Kept baseline/full/adaptive policies in the
   real conversation runner. Added per-part evidence estimates and one bounded
   deeper search. Found that keyword narrowing incorrectly claimed full-library
   coverage; corrected it. Complete short libraries now retain potentially linked
   material instead of being narrowed to one keyword-matching file.
3. **Independent support experiment.** Prepared pinned offline NLI weights and
   labeled original supported/contradicted/unknown claims. The checker caught
   documented unsupported expansions, but also flagged valid paragraphs. Tested
   source-heading normalization, attribution removal, adjacent/sentence windows
   and combined cited evidence. Retained reproducible diagnostics, disabled the
   production experiment by default, and did not adopt automatic rejection/repair.
4. **Reader-assisted learning.** The encoder-only head often confused relevance
   with sufficient evidence. Added validated SQuAD2 span signals and retrained
   the same supervised heads without changing test groups. Calibration AP rose
   from 0.8598 to 0.9464. Added combined-context scoring after a cross-document
   case showed that neither individual passage could establish the whole answer.
   The frozen test later showed AP 0.9100, with confident errors still present.
5. **Reference following and cost.** A new transfer case exercised useful recovery;
   paraphrases also exposed confident missing-context errors. Added a bounded
   graph over explicit uploaded filenames/sections, preserving selection limits
   and refusing to guess duplicate filenames. Batched query embeddings and reused
   scoped vectors within one request. Preserved source freshness checks and lazy
   identifier-specific embedding rather than embedding unrelated files eagerly.
6. **Local human review.** Added scoped durable answer ratings and a downloadable
   queue ordered by feedback and estimated uncertainty. Labels require review;
   private user feedback is never automatically fitted or shared. Added input-only
   transformation routing, omitted irrelevant history for fresh quoted tasks,
   reset stale comparison state and preserved lesson continuation separately.
7. **Adversarial review and release checks.** Reviewed source/reference scope,
   malformed artifacts, nonfinite outputs, inference failures, query batching,
   deleted snapshots, model identity changes, split leakage and dataset provenance.
   The checker no longer displays an inference status when disabled. Training
   validation no longer depends on Python assertions. Fixed fixture fingerprinting
   so a changed file cannot be mislabeled as the data originally loaded.

The separate validation report records final model results, actual conversation
outputs, regression checks and limits. The additional transfer fixture is a known
development diagnostic after inspection. Models were not refitted on its outputs
or on the fixed test documents. Negative ranking/verification results remain visible.
