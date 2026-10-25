# Material-based learning cards

Development started 21 October 2026 from `a3ea4a8`. This record separates actual
observations from intended behavior. Only original fictional material is used in
the public evaluation; the learner's library and saved cards are preserved.

The previous Cards page supported manual cards, longest-word cloze drafts and a
six-card AI path. The AI chose an excerpt ID and copied its full text as the answer.
This protected source membership but often coupled a focused question to an
unfocused answer. One whole-batch support verdict could also discard good cards.

## Approaches considered

| Approach | Useful property | Main limitation | Decision |
|---|---|---|---|
| Longest-word deletion | Fast, no model | Often blanks an incidental word; weak learning objective | Replace with intentional source-based clozes |
| Whole-excerpt answers | Exact source membership | Answers can contain multiple facts or reveal a different target | Keep the legacy API for compatibility; compare actual output |
| Freely generated Q/A | Natural, concise and conceptual | Can omit conditions or invent explanations | Tested, then rejected as the default answer-writing path |
| Full planner/writer/critic chain | Flexible | Added calls, latency and opportunities for error | Start with direct structured batches and selective repair |
| Fine-tuning / external GPU | Could help a measured model failure | No evidence yet that it is necessary | Evaluate prepared models first |

Card formulation follows the practical guidance in
[SuperMemo's twenty rules](https://www.super-memory.org/archive/articles/20rules.htm):
understand the material, keep individual retrieval targets small, and supply enough
context to disambiguate them. These are design guidelines, not a measured learning
benefit for this application.

Quality priorities: supported answers and question premises; preserved numbers,
conditions and exceptions; clear context; focused recall; useful section coverage;
few duplicates; inspectable source evidence. Requested counts are upper targets,
never a reason to manufacture material.

## Improvement record

### Initial baseline

The current implementation and actual source-binding path were inspected. Seven
original cases cover concepts, conditional measurements, code, conflicting named
trials, a six-character source, boilerplate and model-directed source instructions.
The legacy and new pipelines were run against the same material, with actual
output retained. Phrase/concept checks are diagnostics and require manual reading.

## Adopted design

The generator chooses a learning target and writes its question. The application
offers literal answer spans from the selected material, binds the chosen answer
ID, and owns the answer and its supporting excerpt. This is more precise than
copying an entire source paragraph, and more reliable than asking the writer to
reproduce or paraphrase the answer. Conceptual relationships can still be tested
when the material actually states them. Cards do not manufacture a cross-source
synthesis or a missing explanation.

For a cloze, the app masks the selected phrase in the actual source sentence.
It preserves glyphs, units, inequalities, compound bounds, modal words and
negation. Recognized two-column Markdown table rows get readable app-built
questions. The exact table remains inspectable. A short labelled numerical fact
can take a deterministic path with no model request. Offline Source cloze mode
chooses definitions, subjects and meaningful quantities rather than a longest word.

Material selection distributes passages across documents and their beginning,
middle and end. Prepared local embeddings can rank topic paraphrases. The UI
reports read/available passage and document counts. A repair can fill a useful
missed section instead of repeatedly rephrasing a fact already covered. A topic
request does not force this collection-wide coverage step into unrelated sections.

Exact answer and source membership are necessary but insufficient. Individual
model review checks the question premise, completeness, attribution and conditions.
The prepared extractive QA reader provides an independent round-trip check: a
strong, source-offset-validated alternative answer can request a repair. Missing
or weak reader spans never veto a card. The logit-margin cutoff of 10 and the
low-overlap rule are conservative development heuristics, not calibrated correctness
probabilities. Reader failure leaves ordinary source/model review available.
When that reader points to a different offered fact, the app first proposes an
answer re-binding in the same validated passage, preserving the question. The
replacement still passes ordinary validation and model review. Rewriting the
question alone was rejected after it retained the wrong audit fact in a new form.
If a cloze targets an entire statement, the app tries smaller offered spans in
that same evidence statement; it never masks the entire sentence or guesses text.

Duplicate checks distinguish values, inequalities, modality, subjects and table-row
targets. Fuzzy/embedding merging also requires a common literal fact context;
similar questions with different supported facts must survive. New previews have
independent widget IDs. Saved-card context excludes unselected source documents,
secondary-source cards outside the selection, and unlinked personal cards.
Review requests use small request-local source IDs, filenames, locations and
quotes; database identifiers are omitted. Repeated filenames retain their import
labels in saved-card review and quizzes as well as in document selection.

Schema 4 stores supporting excerpts separately from card answers. Saving a preview
is transactional: it rechecks course, document, passage, locator and quotation
inside the write transaction. Existing databases receive a private backup before
migration. Old cards/reviews retain their IDs and schedules. Editing a card retains
its excerpts while changing the review label. Deleting any supporting document
removes the derived card and its reviews. No learner data are training data.

## Observed improvements and rejected trials

These are recorded investigations, not a preset cycle target or a quality score.

| Investigation | Actual observation | Adopted response |
|---|---|---|
| Legacy baseline | Whole-excerpt answers bundled multiple facts and included contact/footer text; the injection case lost all drafts | Separate the learning answer from its evidence; keep independently useful cards |
| Free answer writing | A source's possible behavior became a categorical claim; one result changed `can` to `will` | Preserve source wording rather than depend on a critic to repair paraphrases |
| Model-written clozes | Good targets had missing/wrong blanks, added context, or rewritten sentence wording | App-constructed literal clozes |
| Exact text reproduction | The writer still inserted words or selected the whole short fact as a blank | App-owned answer options selected by ID; deterministic short-fact path |
| Model-directed source text | A critic treated a neighboring instruction as making ordinary facts untrustworthy | Omit only directly model-addressed instructions from card input; retain exact useful statements |
| Fresh transfer | Flattened table clozes and page/footer cards were accepted | Recognize standalone metadata; bind table rows to readable questions |
| Source roles | A question swapped the reader with the narrator while its answer quote stayed real | Narrow actor consistency check plus factual review; omit failed repairs |
| Compound bounds | A substring could lose `at or` or the unit/operator beside a value | Expand attached qualifications from the original source; preserve strict/inclusive distinctions |
| Unicode | NFKC normalization could change a quotation between generation and durable save | Use whitespace-only normalization for provenance; preserve ligatures and full-width digits |
| Deck coverage | A collision fact appeared as both QA and cloze while stable keys were missing | Exact completed-cloze deduplication and focused coverage repair |
| Long-material QA | An audit's lack of a guarantee was paired with a question asking what it records | Independently compare the question with a prepared extractive reader |
| Reader diagnostic | At margin 6, two of 48 inspected QA cards disagreed: one had a wrong answer, one valid broad question had two source-supported answers | Use a stronger margin-10 disagreement only for targeted repair; never veto no-answer/weak cases |
| Reader repair | Rewording the audit question retained the wrong fact | Preserve the question and propose a source-bound answer re-binding before review |
| Whole-statement cloze | Fill-in-the-blank mode lost useful concepts/formulas after repeatedly choosing an entire sentence | Refine to a meaningful smaller offered span of the same fact, without another writer call |
| Equal answers | Different subjects/table rows could share an answer and be wrongly merged | Preserve target identity and require shared fact context for fuzzy merging |
| UI/state/storage | Preview editors could otherwise retain old values; a stale source could otherwise partially save a selection | Fresh preview IDs, selection invalidation, atomic saving, source reauthorization and migration regressions |
| Final context audit | Database identifiers added irrelevant critic input; repeated filenames were ambiguous in saved recall views | Request-local review IDs and consistent import labels |
| Final dependency check | `certifi` 2026.7.22 had a newly reported advisory | Update only that transitive package to 2026.10.19 and repeat the audit |
| Repeated release audit | Eight Pillow advisories now required 13.0.0, while Streamlit's metadata still capped Pillow below 13 | Upgrade Pillow with an explicit uv override, inspect the affected APIs, then repeat the full image/OCR/model/UI checks |

Stored checkpoint outputs, including rejected and superseded trials, are retained in
`evaluation/observations/cards-v25.json`. Use each record's status and fingerprint:
some older filenames contain "final" but represent superseded implementations.
The original material is CC0 and contains no learner uploads. Source IDs and
database IDs are local to each isolated run.

## Runtime and resource choices

Development cycles were driven by observed failures. Interactive requests are
bounded: 1–24 target cards, small output batches, bounded source context and one
combined repair/coverage pass per batch. This prevents an ordinary generation
request from becoming an endless background job. Fewer cards are preferable to
padding. Models are explicitly prepared and reused; there are no runtime downloads.
The reader probe took 0.62 seconds for 48 QA cards on this Mac after model loading;
that is a single diagnostic timing, not a performance guarantee.

No hosted inference, Kaggle job or language-model fine-tuning was needed. The
observed failures were chiefly answer binding, question alignment, selection and
interface behavior. These observations favored fixing application contracts
before trying external training.
The generator remains the pinned local Qwen3.5 9B; semantic selection/duplicates
and question reading use the existing prepared pretrained models. The original
supervised evidence artifacts and their frozen data partitions are unchanged.

## Limits

Model-generated question premises and reviews remain fallible. A strong reader
can also be wrong; its margin is not confidence in a card's truth. Exact quotations
cannot correct OCR mistakes or ambiguous source notation. Literal answers can be
longer than an ideal human-written paraphrase. Bounded context samples long selections,
and the table shortcut is for recognized two-column Markdown tables, not arbitrary
table/diagram understanding. Users can edit, select or discard every preview.
The authored development cases are not blind or representative accuracy evidence.

See [validation and reproduction](VALIDATION_V25.md) for final results and their
code identities.
