# ModuleMate 2.3 architecture

`app.py` is the Streamlit interface. There is one local workspace; `db.py` provides
transactional course/document scoping and cascades. Schema 3 migrates previous
account databases after an online backup, without rebuilding source IDs or losing
study records. No runtime authentication code remains.

`ingest.py` runs format extraction in an isolated bounded subprocess, then chunks
text along Markdown/heading boundaries while carrying the actual heading hierarchy.
PDF pages and presentation slides preserve their locators. No notebook code runs.
Documents expose text previews so extraction failures can be inspected.

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
This distinguishes an unknown identity from the false claim that nobody acted.
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

Generated cards select an evidence-span ID. The app copies the actual source
substring into the back, strips leading heading boilerplate from candidates and
deduplicates identical backs. It validates each selected ID and its original source
again before checking the question's support. Legacy provider responses with an
explicit back still require an exact source match. Cards remain drafts for review.

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
Procedures retain concrete supported actions and constraints without requiring
an unavailable final result. These controls do not add model calls or certify
semantic correctness. Experimental planning/reasoning stages were rejected.

Single-topic adaptive requests prefer distinctive terms in uploaded filenames
over incidental body words. This also applies to short libraries, so a generic
word such as 'measurements' cannot pull a field guide into an answer about a
sculpture. Comparisons and compound requests keep their union. Explicit links
can still add sections from any allowed file, and broader recovery can search
beyond the preferred titles. Filename matching is a precision heuristic, not
proof that every relevant source has been found.
