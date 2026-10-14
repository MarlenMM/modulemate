# ModuleMate: personal study workspace

## Product decision
Replace the fixed university corpus and course-specific classifier with a general,
user-owned study library. Start empty. Materials are indexed for retrieval, not
used to train a model. Keep the ModuleMate name and a lightweight Streamlit UI.
The release target is a complete local/self-hosted app, not a public SaaS launch.

## 1. Remove the old corpus safely
- Preserve the existing implementation and course artifacts in a private local
  archive outside this repository before removal.
- Remove bundled source text, OCR caches, corpus manifests, course-specific
  evaluation data and misleading historical performance reports from HEAD.
- Never import an old index automatically. Ignore all runtime/user data.
- Document that existing Git history/remote copies still contain removed files;
  history rewriting and collaborator coordination are a separate release action.

## 2. Private, durable storage
- SQLite with schema versioning, foreign keys, transactional ingestion and FTS5.
- Accounts with salted scrypt password hashes, opaque expiring sessions and login
  throttling. Every course/document/chat/card/task operation checks ownership.
- Local-only server binding by default. No global cache of user content.
- Course CRUD, document deduplication, deletion cascades, private account export,
  password change and account deletion. Never store provider keys in the database.

## 3. User-provided materials
- Upload PDF, DOCX, PPTX, TXT, Markdown and notebook files; paste notes directly.
- Validate size/type, archive expansion and parser resource limits; never execute
  notebooks, macros, links, uploaded code or embedded instructions.
- Extract page/slide/section/cell locators, retain short and multilingual notes,
  chunk deterministically, and report unreadable/scanned pages honestly.
- Store extracted text only. Provide source previews and download of extracted
  text; originals need not remain on disk. Limit documents/chunks per course.

## 4. Evidence-first assistant
- Local full-text search with token normalization, course/document filters,
  diversity and bounded context. No model downloads or API key for core use.
- Offline evidence mode returns clearly labeled excerpts, not invented answers.
- Optional OpenAI-compatible generation configured by the operator, activated
  only with per-session user consent. Disclose exactly what is sent.
- Explain, summarize, compare and Socratic modes with prior conversation support.
- Treat source text as untrusted; validate citation IDs and verbatim evidence
  quotes in structured model output. Fail back to evidence on provider failure or
  invalid output. Citations prove provenance, not factual correctness.
- Persist course-scoped conversations; deleting sources clears derived chats.

## 5. Study workflows
- Create source-linked flashcards manually or from grounded AI output; offline
  cloze practice works without a provider. Deduplicate and allow edits/deletion.
- Due queue, reveal-before-grade, spaced review scheduling and progress counts.
- Practice quizzes from saved cards with self-assessment and cited answers.
- Tasks with dates, estimates and completion; deterministic daily study plan
  with capacity/overdue warnings and calendar export.

## 6. Product quality and delivery
- Clear onboarding, original fictional demo notes, empty/error/loading states,
  mobile-friendly navigation, accessible standard widgets and no HTML rendering
  of user-controlled content.
- Portable locked dependencies; CI, lint, unit/integration and Streamlit tests.
- Test cross-account isolation, deletion/index freshness, malicious uploads,
  citation validation, provider outages, duplicate imports, review dates and UI.
- Run real extraction/retrieval checks using original synthetic materials and
  inspect the running UI. No university material is used as a new test fixture.
- Write README, architecture, privacy/deployment and migration notes. Record
  actual validation and limits. Commit the verified implementation locally.

## Acceptance criteria
A fresh install starts without an index or API key. A new user can create an
account/course, upload or paste material, search it with locators, optionally ask
AI questions, save/review cards, manage tasks, export and delete their data.
Another account cannot read or modify it, including by guessing identifiers.
Removing a source removes its search results and derived cards/chat history.
The tracked tree contains no inherited university content or personal runtime data.

## Follow-on improvements, after core acceptance
Evaluate retrieval on original multi-subject fixtures, tighten failure handling,
run dependency/security checks, test backup/restore instructions and polish UI.
Do not claim a semantic-search quality score, production security certification,
or live-provider validation without actually measuring it.

## Completed implementation and verification

All six implementation phases are implemented. Acceptance flows pass in the
62-test suite, including cross-account denial and deletion/index freshness.
Follow-on work completed: multilingual lexical regression, 100-document timing,
concurrent initialization repair, stronger password hashing, semantic support
checks, exact-answer AI card validation, useful cloze drafts, task editing,
manual source-linked cards, release-tree checks, backup restore, and desktop/
phone UI inspection. See `docs/VALIDATION.md` for measurements and limitations.

The public release must use the existing remote history as its parent, because
the local legacy checkout and public repository have distinct histories. Preserve
the local history privately; do not merge or push its old corpus-bearing commits.
Earlier remote history cleanup remains separate from latest-tree removal.
