# Architecture

The UI is a thin authenticated client of Python service modules. No component
loads an existing `corpus/` or `index_store/` directory.

```text
Streamlit session (opaque token; AI consent defaults off)
   -> Store (validates session + ownership on every operation)
      -> SQLite users -> courses -> documents/chunks, chats, cards, tasks
      -> local intent model + dialogue planner -> resolved question
      -> FTS5 + local embeddings -> fused candidates
      -> contextual sentence relevance + constrained extractive reader
      -> extractive answer/comparison/outline/recall composer
      -> optional AI provider -> JSON / citation / quote validation
      -> local response fallback when provider output is missing or invalid
```

| Module | Responsibility |
|---|---|
| `app.py` | Sign-in, course navigation and study workflows |
| `modulemate/db.py` | Versioned schema, auth, transactions, ownership, deletion, export |
| `modulemate/ingest.py` | Type/size validation, extraction and deterministic chunking |
| `modulemate/extract_worker.py` | Separate parser process, time/CPU/memory limits |
| `modulemate/search.py` | Unicode tokens, CJK bigrams, BM25 and coverage ranking |
| `modulemate/local_models.py` | Pinned local CPU inference, no implicit downloads |
| `modulemate/retrieval.py` | Model-versioned vector persistence, fusion, reranking, scope revalidation |
| `modulemate/reader.py` | Bounded SQuAD2 span selection with source offset validation |
| `modulemate/intent.py` | Original-data TF-IDF/logistic classification |
| `modulemate/dialogue.py` | Follow-up resolution, mode selection, sentence ranking and composition |
| `modulemate/assistant.py` | Orchestration, provider boundary, validation and fallback |
| `modulemate/study.py` | Cloze drafts, due dates, task allocation, calendar export |
| `modulemate/settings.py` | Operator-only environment configuration |

## Data and transactions

SQLite schema version 2 is created/migrated atomically under `BEGIN IMMEDIATE`.
The v1-to-v2 migration adds embeddings and message metadata without replacing
accounts, sources or conversations; old messages receive empty metadata. Concurrent
initialization handles the initial WAL transition's busy response. Foreign keys
are enabled for each connection. Unknown newer schemas fail closed. Future
schema changes require an explicit versioned migration and a backup/restore test.
Connections are short lived; no user-content cache is shared between sessions.

Every public Store operation revalidates an opaque session token (stored hashed
in the database), resolves the owner, and checks the requested course/document/
card/task. SQL parameters carry all user values; FTS expressions are quoted,
normalized tokens. A document and all its chunks/index updates commit together.
The extracted content hash deduplicates within a course, independently of its
filename. Cross-course imports are allowed and remain independent.

FTS5 maintains a content-external index via insert/delete triggers. Retrieval
first filters to an authorized course and optional document IDs, then ranks up
to 100 candidates using term coverage and BM25. Returned scores are not labeled
as confidence or probability. No inference is made from the old model's calibrated
similarity threshold. CJK bigrams improve substring matching; they are not a full
multilingual semantic model.

Deleting a document cascades to chunks, vectors, source-linked cards and their reviews.
It also clears the course's entire chat history because paraphrases can contain
removed information. Manually authored unlinked cards remain the user's own
content. Course/account deletion cascades through all descendants. SQLite secure
delete is enabled, but no forensic-erasure promise is made for WAL, backups,
provider logs or downloaded copies.

## AI boundary

Local tools run local ML and never call a hosted generator. When the user opts in, evidence and a bounded recent
history are sent to the operator-configured provider. Documents are JSON data,
never system instructions. The model cannot invoke tools, execute code, access
files or retrieve URLs. Source text and generated text render without HTML or
remote-image execution in the chat.

AI card answers must themselves be exact source excerpts (whitespace normalized); invalid drafts are dropped individually before the support check. The response validator rejects unknown IDs, fabricated quotes (whitespace differences are normalized),
missing section evidence, excessive output and suspicious citation/link markup.
A separate second request to the same provider checks semantic support and rejects unsupported answers/card batches. Both calls are bounded and failures fall back to local tools. A model can still attach a real quote to a bad inference; this is not a semantic
entailment guarantee. The UI says so. Summaries explicitly disclose sampling.
Provider timeouts/errors and invalid responses preserve local evidence access.
There are no automatic paid retries or cross-provider fallbacks.

## Scheduling

Review intervals are deterministic, capped at one year. Again resets progress
and schedules tomorrow; Hard grows slowly, Good uses 1/3/doubling day intervals,
Easy starts at four days and grows faster. Due dates use the timezone selected
in the UI. Practice quizzes do not alter scheduling. Ratings are self-assessment,
not grades or a validated mastery score.

Task planning sorts incomplete tasks by deadline and fills daily capacity in
blocks of at most 50 minutes. Overdue tasks can use today's capacity; unallocated
work is reported. Plans are recalculated from current tasks and estimates, not
silently persisted as appointments. Calendar exports contain all-day deadline
events with escaped, UTF-8-folded RFC 5545 content.

## Local intelligence pipeline

The initial stage normalizes Unicode and retrieves lexical candidates. A local
384-dimensional encoder represents short windows of each authorized passage;
normalized window vectors are averaged and normalized again. This reduces the
risk of losing the tail of long passages at the encoder token limit, though very
long words or dense non-English tokens can still truncate. The cache key includes
the model revision and pooling version; the content digest includes document
identity. Float32 vectors are shape/finite-value validated on read and write.
Corrupt or obsolete vectors are rebuilt. No pickle or remote model code is loaded.

Embedding reads/writes are short authenticated transactions. Inference happens
outside database write locks. Each batch rechecks source identity/text before
persisting vectors, and returned passages are revalidated after inference, so a
deleted/replaced row cannot be resurrected. Only public model objects and the
original-data classifier are cached globally. User text and query vectors are
request-local; passage vectors persist only in the owning database.

Search fuses lexical and dense ranks (reciprocal-rank constant 30), considers at
most 32 candidates, and defaults to hybrid retrieval. Dense admission uses cosine
0.20; final hybrid candidates require cosine 0.32 or a lexical match. An optional
passage reranker is retained for reproducible ablations but is disabled by default:
it reduced top-three development recall from 100% to 97.8%. Its alternate gates
combine cosine 0.32, logit -3.5 and token coverage 0.6. These are engineering
heuristics, **not calibrated probabilities or entailment tests**. Duplicate passage
text is removed before context construction. Explicit alphanumeric identifiers
(e.g. a course code) constrain candidates; unknown identifiers return no evidence.

For comparisons, each side gets its own retrieval budget. Source sentence ranking
combines lexical relevance, embedding similarity and a cross-encoder signal;
local answers copy complete source sentences rather than paraphrasing them.
English sentence relevance includes Snowball stemming, question detail hints
(e.g. word count versus duration), ordering checks, a passage prior and a bounded
cross-encoder contribution. Pronoun sentences are ranked with source context.

A local MiniLM SQuAD2 reader then searches only the chosen source for a supporting
span. It masks question/special/padding tokens, uses overlapping 384-token
windows, and bounds spans to 40 tokens. Positive span-vs-null logit margins can
provide sentence-location hints; margins are never compared across unrelated
sources or shown as correctness probabilities. Every hint is revalidated by
source ID and exact character offsets. The answer retains the whole supporting
sentence and an antecedent when needed; a reader failure preserves lexical
source access. Reader rejection alone cannot veto strong relevant material.
Summary extraction uses TF-IDF centrality, heading relevance and source/section
coverage. Detailed why/example/inventor/date requests have conservative missing-
detail guards; these are English heuristics, not a complete factuality verifier.

Auto mode chooses the task from explicit phrasing. A separate supervised intent
classifier (120 original utterances, TF-IDF word/character n-grams and logistic
regression, fixed seed 42) guides answer length and hosted style hints. Low-score
predictions use conceptual defaults. Its social category never indiscriminately
blocks retrieval. Exact greetings/help use conversational responses; evidence
availability governs whether other questions can be answered.

Dialogue metadata stores the resolved topic, comparison subjects, source context,
response kind, notice and suggested follow-ups. Only grounded history in the
current document filter can inform retrieval or reach the optional provider.
Pronouns with no topic, or an ambiguous referent after a comparison, produce a
clarification. Specific new-topic questions do not inherit the old subject.
Recall mode asks the learner to explain before revealing source content; it does
not grade mastery. A following learner explanation triggers source comparison
for self-assessment. Source-driven prompts are not claims of free-form reasoning.

## Model and method references

- [Sentence-BERT paper](https://arxiv.org/abs/1908.10084): sentence representation
  learning for similarity and retrieval.
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2):
  deployed encoder; exact revision is pinned in `local_models.py`.
- [MiniLM extractive QA model card](https://huggingface.co/deepset/minilm-uncased-squad2):
  reader by deepset, based on Microsoft MiniLM and trained on SQuAD2; CC BY 4.0
  model license. This software downloads the unmodified pinned weights separately;
  the repository does not redistribute them.
- [Transformers question-answering documentation](https://huggingface.co/docs/transformers/tasks/question_answering):
  token-offset-based extractive inference. The application uses the model directly,
  because the old question-answering pipeline is unavailable in Transformers v5.
- [Retrieve and rerank documentation](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html):
  efficient candidate retrieval followed by pairwise relevance scoring.
- [Logistic regression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html):
  supervised linear classifier used for intent hints.
- [Multi-QA candidate model card](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1):
  evaluated as an alternative and rejected after worse development results.

See `evaluation/results-local.json` for the deployed configuration and
`evaluation/results-multiqa-candidate.json` for the rejected alternative.
