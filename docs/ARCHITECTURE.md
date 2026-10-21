# ModuleMate architecture

`app.py` is the Streamlit interface. There is one local workspace; `db.py` provides
transactional course/document scoping and cascades. Schema 3 migrates previous
account databases after an online backup, without rebuilding source IDs or losing
study records. No runtime authentication code remains.

`ingest.py` runs format extraction in an isolated bounded subprocess, then chunks
text along Markdown/heading boundaries while carrying the actual heading hierarchy.
PDF pages and presentation slides preserve their locators. No notebook code runs.
Documents expose text previews so extraction failures can be inspected.

`ocr.py` adds local Apple Vision recognition on macOS and an installed Tesseract
fallback on other systems. PDFium renders raster-containing PDF pages, including
mixed native/image pages, and pages with no native text (including outlined
letters without image objects); native text is retained separately. PPTX group pictures,
DOCX paragraph/table images and PNG/JPEG/WebP uploads also enter OCR. Recovered
text keeps an explicit OCR page/slide/image locator. No fuzzy correction edits
numbers or names. Low recognition scores retain text with an inspection notice;
they never decide source credibility. Recognition has image/pixel/time limits,
request-local duplicate reuse and no downloads or hosted calls. The parser's
resource boundary contains PDFium/native image processing. Unread locations are
reported; original images are not stored.

`document_parts.py` reads referenced Word header/footer and note parts, text boxes,
native PowerPoint chart caches and referenced SmartArt labels. It never fetches
external relationships or runs a workbook. Chart rows retain category/value or
X/Y alignment, series names and raw number formats; section headings carry series
identity across chunks. Notebook saved text, raster outputs and attachments are
read statically with cell/output locators. Rich executable/HTML-only displays are
omitted with notices; saved output is never recomputed or represented as a result
of running code in ModuleMate.

`search.py` provides Unicode FTS/BM25 matching and short-topic heading preference.
`retrieval.py` adds normalized local embeddings, persistent digest-checked vectors,
rank fusion and optional reranking. It validates scope before inference and source
identity again before storing/returning results. Missing models retain word matching.
`RetrievalIndex` keeps scoped source vectors and batched query embeddings only for
one request. It checks store, course, document scope and model identity at reuse;
final source identity checks still run. Identifier-specific queries do not cause
unrelated documents to be embedded eagerly.

`learning.py` loads three original supervised heads from bounded, schema-checked
JSON numerical artifacts. Lexical, embedding and reader variants share explicit
feature order/version and record fit/calibration source identities. No fitting
or downloads occur on a user request. `adaptive.py` assesses individual passages
and combined context for the whole request and explicit subquestions. Reader
spans must match source offsets. An uncertain estimate can buy one deeper scoped
search; a replacement must improve a part without substantially reducing another.
These are advisory estimates, not probabilities of final answer correctness.

`context.py` assembles evidence for the actual question, comparison sides and
subquestions. It adds adjacent passages from the same document, resolves named
files, reads complete short selections, and budgets longer contexts by characters
and estimated tokens. Lessons use contiguous ranges with a saved continuation
cursor and document signature. Ambiguous lecture requests ask for the document.
The cursor survives questions between sections. Filename/lecture-number aliases
and document codes retain later sections that do not repeat a title-page code.
Lesson parts have a few explicit topic targets. Framing/title pages and obvious
model-directed instruction packets are not teaching targets. Each current target
needs cited evidence before a generated lesson can advance; a roadmap mentioning
later topics is not sufficient. Catalog heading previews sample across a long
document, including its end, rather than truncating the outline at the first pages.
Distinctive query terms focus factual context on relevant documents, retaining
both sides of comparisons and collection-wide requests. History has a bounded
token budget and retains the final check question when a long answer is shortened.
The interface and answer instructions are English only; there is no query
translation stage or language selector.
Short allowed libraries bypass keyword narrowing. Explicit uploaded-file
references can add relevant linked sections from the already allowed document
set, with at most two targets and one hop. Duplicate filenames are not guessed.
No paths, links or source commands are executed. Soft document filtering no
longer incorrectly establishes complete-library coverage.

`conversation.py` instructs a generative tutor to answer the latest request, using
recent turns for context rather than treating previous answers as evidence. Course
claims require supplied sources. Explanations can paraphrase and connect those
sources; missing information is disclosed rather than filled from model knowledge.
Quoted text/code supplied by the user can be transformed with an explicit input
basis. Labeled illustrations may demonstrate a provided concept.
`calculations.py` executes bounded arithmetic and descriptive statistics through a
small AST grammar and Python numeric routines, never `eval` or uploaded code.
Numeric tools use explicitly supplied data. A narrow check can also catch explicit
numeric assertions; it does not verify arbitrary prose maths. Lesson planning uses
document headings and passage order across subjects.

The local tutor writes natural Markdown first, then a separate labeling pass
attaches evidence and conversation state. The evidence pass owns final citation
numbers; incidental writer markers are removed without changing code literals.
This keeps explanation writing separate from JSON formatting. The writer supplies
a small response-type marker, removed
before rendering; the evidence labeler cannot turn an ordinary answer into a
clarification or override a missing-information response. A missing-information
answer may still cite a source's explicit gap or useful preliminary facts.
Unsolicited model-written source footers are removed before labeling, preventing
invented bibliography quotations from competing with the app's actual source panel.
Explicit source/quote requests and fenced code are preserved. Dangling closing
reasoning tags are normalized; actual reasoning blocks are rejected.
Text transformations request only the transformed result, rather than the Explain
mode's commentary. Explicit short formats also constrain the final task payload.
Explicit comparison subjects belong to the user's parsed request, not a model's
labels for comparison aspects. Polite requests to rewrite/summarize are not
clamped as short yes/no facts. Fact lookup, checklists and why-questions carry
tasks suited to that request. Elliptical requests such
as “an everyday example” explicitly carry the previous subject into the final
request. Unrelated document outlines are omitted when focused retrieval finds
no usable passages. Narrow “only the current fact” requests reject unsolicited
archived comparisons and long digressions.
Current-only requests exclude files explicitly named as retired/archived; historical
comparisons retain them. This does not infer an unmarked document's publication status.
`evidence.py` constructs contiguous source excerpts. The
local labeler selects their IDs instead of regenerating quote text; the application
binds those IDs to the original source. This prevents dropped or rewritten source
sentences, but selecting an irrelevant excerpt is still possible.
`schemas.py` constrains local metadata, review and card
generation using Outlines Core. A small byte-preserving MLX adapter avoids the
larger framework's unused disk/pickle cache dependency. Compatible hosted providers
retain a one-call JSON path; local OpenAI-compatible servers can use the separated
writing path.

Responses have a typed envelope, real source IDs and exact supporting quotes.
Presentation variants such as typographic citation brackets are normalized. Code
subscripts are not citations. Only cited passages are saved as answer evidence,
with compact numbering. A targeted review of precise course facts/code can trigger
one correction;
a short exact source statement has a deterministic path. Negative reviews must
quote both the actual disputed draft and supporting source evidence (or reference
computed numeric results). Negative verdicts without matching draft/evidence quotes
are ignored. Missing-evidence responses without factual citations bypass the critic;
incidental cited claims can be corrected without inventing the absent answer.
The instructions distinguish an unknown identity from the false claim that nobody
acted; actual model answers can still confuse them, as the 2.4 observations show.
Review outages
are labeled and do not erase a valid primary answer. General conceptual lessons
bypass the model critic after repeated false objections in development challenges.
Review is not a correctness oracle. Invalid primary drafts produce a clear
error with related source passages, never an unrelated sentence as an answer.

`generation.py` manages an explicitly prepared MLX instruction model in one bounded
worker process. The worker loads pinned local safetensors without remote code or
network access, uses bounded prefill/cache/output, and restarts after a timeout.
It uses seeded temperature/top-p/top-k sampling and permits ordinary JSON
whitespace. Generic grammar indexes are cached; learner prompts and KV state are
not. The worker releases per-request frames before waiting for the next request.
Model role delimiters inside task data are JSON-escaped without changing the
decoded text. Numeric tools, grammar/citation checks and the model remain distinct
components; none gives a general guarantee of semantic correctness. If focused
retrieval finds no context and the user has supplied no text to transform, the app
reports missing material without generating an outside answer.
`AIBackend` supports explicitly enabled compatible local/hosted servers and reports
provider limits without exposing exceptions. A local failure never invokes hosting.

The old `dialogue.py` extractive engine remains a clearly labeled Excerpt mode for
installations without a generator. Its intent classifier and reader/retrieval
ablations remain educational diagnostics, not evidence of generative accuracy.

`study.py` handles draft cloze cards, spaced scheduling, practice and calendar
export. All saved artifacts remain course scoped. The public conversation fixture
and offline tests exercise filters, migration, continuation, numeric routines,
citation binding, errors and UI flows. Real-weight checks are separate from
network-free CI. We do not fine-tune a language model on private coursework.

`cards.py` implements the 2.5 learning-card path separately from conversation.
It samples scoped material across document positions, with optional local semantic
topic ranking. It offers app-owned literal answer spans inside exact evidence;
the generator chooses an answer ID and writes a focused question. The app owns
answer wording, source identity and cloze reconstruction. Direct model-addressed
instructions and standalone footer/metadata lines are omitted from the card input,
while retained text must still match its original passage. Tables use question/
answer cards. Numeric bounds, negation and adjacent modal words remain attached
to selected answers. A small labelled numerical fact takes a deterministic path.

Individual model verdicts can request a targeted repair without discarding other
cards. They remain fallible; phrase binding proves source membership rather than
entailment. A prepared extractive reader can independently point to a different
answer to the generated question, using validated source offsets and a conservative
margin/overlap heuristic. Strong disagreement requests a repair; weak or absent
spans never veto a card. A proposed answer re-binding preserves the question and
still passes source validation and model review. A whole-statement cloze choice
can be refined to a smaller offered span in the identical evidence statement.
Lexical and optional pretrained embedding checks reduce duplicates
while protecting numbers, operators, qualifications and named subjects. Context,
outputs and repair attempts are bounded for interactive use. No user material is
used to fit a card-quality model or fine-tune a language model.

Previews are course scoped, editable and invalidated when the document selection
changes. Fresh generation IDs isolate their widget state. `Store.add_cards` saves
the selected batch atomically after rechecking current passage identity, locator,
course and exact quotation. Schema 4 stores evidence in `card_sources`, with a
backup before upgrading an existing database. Deleting any supporting document
removes its derived cards and reviews. Editing a saved card retains evidence but
changes the review label to user-edited. Supporting material is hidden until the
answer is revealed in recall/quiz views. The older `assistant.generate_cards` API
retains its exact-excerpt behavior for compatibility; the Cards UI uses `draft_deck`.

`support.py` is an explicitly enabled independent NLI experiment, disabled by
default after false-positive trials. It checks bounded source/claim pairs with
pinned offline safetensors, preserves claim text during token truncation, and can
consider multiple cited sources. Its metadata is diagnostic; it never silently
rejects or rewrites an answer. Disabled checks do not display an inference status.

Pure transformations of newly quoted text omit library retrieval and unrelated
history. A follow-up transformation retains the relevant input conversation;
document-dependent requests retain retrieval. Fresh quoted tasks reset stale
comparison subjects, while saved lesson continuation is preserved separately.

Answer ratings are stored inside the existing message metadata with course/message
scope validation, so no new database migration is needed. `feedback.py` prioritizes
local human-review cases using explicit feedback and estimated uncertainty.
Export requires a deliberate download; user feedback is not a verified gold label
and is never fitted or uploaded automatically.

The writer and evidence catalog omit only leading headings already represented
in the locator; stored source text and quote membership remain unchanged. Strict
quantity requests can drop a decorative label without changing its numeric value.
Nested unrequested source footers are omitted, while requested quotations and code
stay intact. Two uncommon answer terms in an uncited excerpt can produce a local
citation-review hint. The hint is not an entailment verdict, an automatic citation,
or an extra generation request. A real relabeling trial failed and was not retained.

For a declarative prelude followed by one explicit question, request assembly
keeps the prelude as background and the question as the current action. Retrieval,
validation and saved history retain the original text. Compound questions,
quotations and code are preserved. Ordinary `How can I…` method questions do not
become institutional permission requests; explicit policy terms still do.
Direct preparation questions such as `Can I prepare … before I have …?` also
retain method treatment. Their literal step inventory is a reminder, not an
exhaustive replacement for the source's action passages. Pure omission-only
notes can appear verbatim beside the answer without becoming false prohibitions.
Procedures retain concrete supported actions and constraints without requiring
an unavailable final result. These controls do not add model calls or certify
semantic correctness. Experimental planning/reasoning stages were rejected.

Literal source-information gaps and direct negative statements can also be
repeated as separate writer hints with source IDs. Their wording is copied, not
generated, and all ordinary passages stay available. The hints are neither truth
labels nor answerability gates. Pure omission notes displayed beside preparation
are excluded from both the action writer's passages and its information-gap hints;
physical negatives remain in the writer context. Unrecognized phrasing has no hint.

Card fronts/backs are rendered as plain text or escaped labels, so source URLs,
array indices and angle brackets remain valid literal data. Backs still require
exact source membership. Chat code is literal; active image/media markup and
reasoning envelopes outside code are rejected. Displaying a source URL never
causes a fetch, tool execution or external action.

Inferred fact-finding is a writing hint, not an implicit 50-word user limit.
Incidental words such as `example` and `calculation` do not relabel source content
as an outside addition. Citation identity/quotation validation remains mandatory;
the metadata basis label alone is not an entailment verdict. Explicit sentence
counts and requested current-version scope remain checked separately.

Single-topic adaptive requests prefer distinctive terms in uploaded filenames
over incidental body words. This also applies to short libraries, so a generic
word such as 'measurements' cannot pull a field guide into an answer about a
sculpture. Comparisons and compound requests keep their union. Explicit links
can still add sections from any allowed file, and broader recovery can search
beyond the preferred titles. Filename matching is a precision heuristic, not
proof that every relevant source has been found.

Main-idea, takeaway and whole-presentation explanations use structural coverage
instead of treating an extractive answerability head as a judge of synthesis.
Short selections are read completely; longer ones sample actual passages across
the beginning, middle and end. Scoped illustrative follow-ups reuse the prior
answer's evidence. A shared subject in a preferred title can recover relevant
body text in differently named allowed files without importing an unrelated
event's rules. Ambiguous document-level requests ask which file is intended.

The writer may synthesize supported details and mark interpretations/analogies.
A failed evidence-labeling call retries labels using the same prose. If it still
fails, the app reports an evidence-attachment error and shows related passages;
it does not claim that the document lacks the information. Auxiliary evidence
footers/JSON packets are omitted while requested code and quotations survive.
Full short source bodies remain citable even below the normal quotation length.
Explicit prose attributions such as `the text states “…”` and `according to … “…”`
receive a narrow literal wording check, with terminal sentence punctuation ignored.
One correction uses the draft and actual source passages, without earlier assistant
prose. Its original missing-result status survives an incidental marker change.
An unresolved wording mismatch is disclosed in the notice/source panel rather than
discarding the useful answer; this check does not judge ordinary synthesis.

Possible stacked PDF equations receive one advisory notation notice per document.
A narrow short-expression/denominator pattern excludes matching final page numbers
and ordinary prose; it is not comprehensive math recognition. The unchanged source
text remains available. The writer receives a grouping reminder and can still
explain the supported concept while identifying a notation gap. It cannot use
that notice as a source-credibility or answerability veto.

Independent material requests do not receive prior assistant prose as writer input.
Actual follow-ups keep the bounded conversation segment starting with the latest
independent material answer. The source-selection planner still has the conversation
needed to resolve references, and source permissions remain separate from history.

After an actual missing answer, orchestration can use a previously rejected
deeper-search candidate, or perform the one wider search if none occurred.
Only additional allowed text earns this second generation. A learned score
cannot forbid it. If the second answer is still missing or fails validation, the
original missing answer survives. This is bounded recovery, never a forced answer.

For preparation with missing inputs, a small verbatim source-step inventory helps
the writer preserve all prescribed components. Pure omission-only notes from the
same documents can be attached literally after the prose with their original
source binding. They are displayed to the learner, not discarded or treated as
new prohibitions. Mixed passages, policy conditions and strict short-format
requests retain the ordinary path. Metadata must still cite the written answer
before a literal omission note is added; an empty labeling result cannot gain
apparent support just from that note. Procedural `can I` questions no longer
trigger a policy critic solely because of their modal wording.
