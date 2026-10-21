# Original evaluation material

`synthesis_challenge.json` adds original fictional presentation, workflow, short
fact and story probes for main ideas, follow-ups, analogies, connections and
unknown details. Run `scripts/evaluate_conversations.py --suite synthesis`.
These are inspected development diagnostics, not a blinded intelligence score.
`tests/test_ocr.py` constructs original raster, mixed PDF, DOCX and PPTX fixtures;
`MODULEMATE_TEST_OCR=1` enables actual local recognition in the isolated parser.
`MODULEMATE_TEST_OCR_STRESS=1` additionally checks a 65-page original scanned PDF.
`understanding_transfer.json` contains new preparation, interpretation, conditional
rule and implicit-companion cases. Run `--suite understanding`. Phrase flags can
be false positives: the model's word “identical” does not match a list containing
only “same/unchanged/constant”. Keep raw results and read the complete answers.
`source_check_transfer.json` probes incidental topic words, conditional facts that
need more than 50 words, explicit missing prices and conceptual reframing. Run
`--suite source-check`. Its expected phrases are also diagnostics, not a semantic
verdict. It is original material used in development, with no fitting or retraining.
`source_scope_transfer.json` adds unknown-versus-negative distinctions and direct
preparation feasibility after a real missing-name error was observed. Run
`--suite source-scope`. For example, “not possible to determine if admission is
free” can trigger the literal forbidden phrase “admission is free” while correctly
withholding the price. Retain that flag as a diagnostic limitation, not an error
label for the answer. The scenario's review criteria explain the semantic issue.
`overview_transfer.json` adds a 72-section original presentation with three major
parts and a closing connection. It checks an unqualified main-concept question,
including an ordinary misspelling, and an unsupported measured-outcome question.
Run `--suite overview`. It exercises sampled breadth rather than claiming every
long-document detail was read. These are development cases, with no fitting.
`notation_transfer.json` embeds the bytes of an original native-text PDF containing
a drawn fraction bar. Run `--suite notation`. It checks useful conceptual help and
grouping of a complete two-term numerator when applying the displayed rule.
The original PDF bytes are covered by the JSON fixture fingerprint. A controlled
writer-annotation comparison is retained; both conditions gave the correct result.

[The 2.4 observations](observations/understanding-v24.json) retain full original
fixture answers, negative candidates, timing, source identity and code/data/model
fingerprints. They distinguish a frozen baseline, development trials and a complete
46-turn checkpoint. Required-word flags are not semantic answer correctness;
manual observations retain known wrong conclusions and entity substitutions too.

The 2.3 evidence work adds `evidence_learning.json` (original document-grouped
training/calibration/test annotations), `claim_support.json` (independent NLI
diagnostics), and `evidence_transfer.json` (new multi-document conversations).
The [model/data card](../docs/EVIDENCE_MODEL_CARD.md) explains labels, exact splits,
training and limits. Run `scripts/train_evidence.py` and
`scripts/evaluate_evidence.py --split test`; use `--reader` with the prepared ML
extras for the neural-feature variant. `scripts/evaluate_conversations.py --suite
evidence --policy baseline|full|adaptive` compares evidence policies with the same
generator. The [2.3 retained observations](observations/evidence-v23.json) preserve
actual answers and model errors. Constraint checks are not semantic accuracy.

`partial_help.json` adds four original fictional documents and six checks of
preparatory help, missing results and exact facts. One sculpture example reuses
the known transfer material; these are development cases, not a new blinded test.
Run `scripts/evaluate_conversations.py --suite partial`. The retained 2.3 log
includes regressions and rejected planning experiments as well as final outputs.

The independent support classifier is explicitly prepared with
`scripts/prepare_support_model.py`; `scripts/evaluate_support.py` and
`scripts/evaluate_recorded_support.py` reproduce its failures on original material.
It is disabled by default. No private review export or personal upload is training
data in this repository.

`general_documents.json` is the primary 2.2 conversation development set. Its nine
original fictional documents cover events, history, biology, project requirements,
practical notices, code and fiction. Requests and expected responses are English.
It checks precise facts, conflicting versions, negative/conditional requirements,
partial exercise help, missing evidence, follow-ups, text rewriting and hostile
source instructions. Run `scripts/evaluate_conversations.py --suite general`.
The runner records actual prose, source context, model calls, latency and code/data
fingerprints in ignored `results/`. Phrase checks alone do not establish correctness.
[The 2.2 observations](observations/v2.2-conversations.json) retain 47 original-fixture
responses from the full run, focused recovery and new-fact probe, with source chunks
and provenance. The full run predates the final presentation cleanup; the other
two match the recorded QA checkpoint before the later card-only correction.

`conversation_challenge.json` and `conversation_transfer.json` retain earlier
development cases. Some predate the clarified source-only answer policy and are
historical experiments, rather than the current product acceptance criteria.

The following retrieval/extraction observations describe the earlier 2.1 pipeline.

These synthetic documents and utterances were written for this upgrade, not
copied from HIT materials, user libraries, or the old gold set. Facts are simple
teaching examples and invented course logistics. Intent training lives separately
in `modulemate/intent_training.json`; evaluation utterances are not fitted.

The retrieval set is an engineering development benchmark, not a blinded test or
estimate of general accuracy. It deliberately includes paraphrases, lexical
matches, near-topic distractors, missing topics and multilingual word matches.
The conversation regressions check observable behavior rather than an LLM judge.
Run both light and real-model suites; only the real-model run supports neural
retrieval claims. Reports include all misses and denominator counts.


`answer_checks.json` defines reference facts for the 46 development questions.
`answer_challenge.json` contains 20 additional questions written after that pass;
no thresholds were fitted to the challenge results. `answers-local.json` and
`answers-challenge-local.json` retain every answer and miss. Passing a fact check
only means all specified phrases occur in the answer; it does not prove that
all wording, emphasis or implied reasoning is correct. Citation-marker checks
verify resolvability, not semantic citation correctness.

`results-initial.json` records the first attempt where a hard passage-reranker
threshold harmed recall. `results-local.json` is the final retrieval ablation;
`results-multiqa-candidate.json` is the rejected alternative encoder experiment.
`answers-without-reader.json` records the sentence pipeline ablation before the
final within-source reader constraint; it is historical development evidence,
not a current apples-to-apples causal estimate of reader benefit.
## Material-based Planner events (2.6)

`planner_materials.json`, `planner_transfer.json`, `planner_acceptance.json` and
`planner_dense.json`
contain original fictional CC0 schedules and adversarial non-events. They cover
source/date/time role binding, ambiguities, cancellations, conflicting notices,
relative dates, numeric order, term years, time zones/DST, lists, weekly recurrence,
metadata/history, source instructions, long selections and source-shaped identifiers.
Transfer and later acceptance cases were authored after earlier implementation
cycles; all become development cases once inspected.

Run the prepared local generator against isolated temporary databases:

```sh
uv run --locked --extra ml --extra local python scripts/evaluate_planner.py \
  --fixture evaluation/planner_dense.json evaluation/planner_materials.json evaluation/planner_transfer.json evaluation/planner_acceptance.json \
  --output results/planner.json
```

`--approach free_json` compares free model-written ISO fields with literal temporal
binding and source review. Reports preserve actual model calls, events, held
drafts, ignored entries, source membership, transactional save counts and calendar
round trips. Expected-field diagnostics and minimum uncertainty counts do not
establish semantic accuracy; read actual events and evidence. Fixture, module and
app identities are recorded in [the validation](../docs/VALIDATION_V26.md) and
[retained observations](observations/planner-v26.json). No learner library is used.

## Material-based cards (2.5)

`card_materials.json` and `card_transfer.json` contain original fictional CC0
material for card generation across concepts, constraints, code, distinct source
values, short facts, literary interpretation, tables, formulas, source instructions,
and topic focus. `card_long.json` is a 72-section coverage/deduplication stress
case with repeated fictional concepts and a unique closing principle. These are
authored development cases, not a blind or representative accuracy benchmark.

Run against the explicitly prepared local generator in isolated temporary storage:

```sh
uv run --locked --extra ml --extra local python scripts/evaluate_cards.py \
  --fixture evaluation/card_materials.json evaluation/card_transfer.json evaluation/card_long.json \
  --semantic --output results/cards.json
```

Use `--style "Question and answer"` or `--style "Fill in the blank"` for focused
style checks. `--approach legacy` compares the older whole-excerpt API. The
`--semantic` flag uses prepared local semantic selection, independent QA reading
and conservative duplicate checks; it never downloads models or fits user data.

The report retains actual cards, per-call outputs/stats, source/answer membership,
transactional save counts and diagnostic phrase mentions. Read the questions and
answers: a matching phrase in evidence is not semantic correctness, and a forbidden
phrase may occur in a valid negation. Results from different fingerprints must not
be relabeled as one release. See [the output records](observations/cards-v25.json),
[development notes](../docs/CARDS_DEVELOPMENT.md) and
[validation](../docs/VALIDATION_V25.md) for the retained evidence.
