# Architecture

The UI is a thin authenticated client of Python service modules. No component
loads an existing `corpus/` or `index_store/` directory.

```text
Streamlit session (opaque token; AI consent defaults off)
   -> Store (validates session + ownership on every operation)
      -> SQLite users -> courses -> documents/chunks, chats, cards, tasks
      -> FTS5 local retrieval -> source passages
      -> optional AI provider -> JSON / citation / quote validation
      -> evidence-only fallback when missing or invalid
```

| Module | Responsibility |
|---|---|
| `app.py` | Sign-in, course navigation and study workflows |
| `modulemate/db.py` | Versioned schema, auth, transactions, ownership, deletion, export |
| `modulemate/ingest.py` | Type/size validation, extraction and deterministic chunking |
| `modulemate/extract_worker.py` | Separate parser process, time/CPU/memory limits |
| `modulemate/search.py` | Unicode tokens, CJK bigrams, BM25 and coverage ranking |
| `modulemate/assistant.py` | Provider boundary, study prompts, source validation, fallback |
| `modulemate/study.py` | Cloze drafts, due dates, task allocation, calendar export |
| `modulemate/settings.py` | Operator-only environment configuration |

## Data and transactions

SQLite schema version 1 is created atomically under `BEGIN IMMEDIATE`. Concurrent
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

Deleting a document cascades to chunks, source-linked cards and their reviews.
It also clears the course's entire chat history because paraphrases can contain
removed information. Manually authored unlinked cards remain the user's own
content. Course/account deletion cascades through all descendants. SQLite secure
delete is enabled, but no forensic-erasure promise is made for WAL, backups,
provider logs or downloaded copies.

## AI boundary

Local tools never call AI. When the user opts in, retrieval and a bounded recent
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
