# ModuleMate 2.2 architecture

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
