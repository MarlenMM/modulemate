# Validation — 14 October 2026

Historical 2.0 results. The account flow described here was removed in 2.2.
Use `VALIDATION_V22.md` for the current conversational engine and workspace checks.

All fixtures are original synthetic text or programmatically generated files.
No former university corpus, student submissions or group chats were used for
these checks. Measurements below are local observations, not product guarantees.

## Automated checks

- **62 tests passed** on Python 3.12.13 / macOS. Final application test run:
  `uv run pytest -q`, 25.59 seconds.
- Ruff lint and format checks passed.
- Locked dependencies installed successfully. `pip-audit` reported **no known
  vulnerabilities** in the installed environment. This is an advisory-database
  check, not a security certification.
- CI runs lint, formatting, release-tree checks, tests, retrieval regression and
  dependency audit on Linux with Python 3.12 and 3.13. Remote execution results
  are separate from the local results reported here.

The suite exercises PDF text/page extraction and blank-page warnings, encrypted
PDF rejection, DOCX paragraph/table order, PPTX text/notes, UTF-8 and CJK text,
notebook non-execution/output exclusion, ZIP expansion/entity rejection, bounded
chunking, duplicate imports, atomic rollback, concurrent first start, hashed
passwords/tokens, login throttling, expired sessions, password rotation, backup
restore, and account/course/document/card/task isolation.

Deletion tests check FTS integrity and removal of source-linked cards, reviews
and conversations. Assistant tests reject invented source IDs/quotes, tracking
markup, malformed JSON, incorrect verifier types and unsupported drafts. Local
mode never invokes the provider. Review tests reject duplicate grading before
the next due date. Planning tests check daily capacity, overdue work, completed
tasks, unmet estimates, and UTF-8 calendar line folding.

Streamlit AppTest covers registration, course creation, example import, evidence
chat, card drafting/saving/review, quiz practice, task completion, export UI,
source deletion and sign-out. A separate browser session stays signed out.

## Retrieval and performance

`uv run python scripts/evaluate.py`:

| Check | Observation |
|---|---:|
| Original fixture documents | 6 |
| Questions (multiple subjects, English/Korean/Chinese) | 12 |
| Expected document ranked first | 12 / 12 |
| Unrelated-query no-match check | Passed |
| Median local search | 0.85 ms |

This intentionally small lexical regression includes shared query/source words.
It does not measure paraphrase recall, general academic accuracy or AI quality.

`uv run python scripts/benchmark.py` (final chunking implementation):

| Check | Observation |
|---|---:|
| Documents / passages | 100 / 3,700 |
| Synthetic ingestion | 0.57 seconds |
| Median / p95 search | 18.39 / 22.75 ms |
| Exact synthetic topic hits | 100 / 100 |

The benchmark generates text locally and excludes parser subprocess startup,
real-world file complexity, network traffic and AI latency. Different machines,
content and concurrent users will produce different results.

## Live AI checks

The existing operator Groq key was used only with the original Learning Lab
example. No key was printed, committed or included in an export.

- Explain, compare, summarize and Socratic generation were exercised against live
  available models. Early checks exposed whitespace-sensitive quote validation
  and unsupported explanatory claims; both led to implementation changes.
- Quotes now normalize whitespace but still require matching words/punctuation.
- Answers require a second support-check request to the same provider. Failures
  return clearly labeled source excerpts. This can also reject usable answers.
- With `openai/gpt-oss-120b`, the four-mode smoke run accepted Explain, Summarize
  and Socratic answers; Compare fell back to evidence. An earlier 20b run
  accepted Explain and Compare while the other modes fell back. These are small,
  nondeterministic smoke observations, not a measured accuracy score.
- After changing card validation to anchor the **actual answer** in the source,
  a live 120b test produced **six cards** that passed exact-answer membership and
  semantic support checks. Invalid individual drafts no longer discard valid
  cards from the same response.
- The local installation is configured for the 120b model. The public source has
  no mandatory provider/model. AI remains off until a user opts in.

Reproduce intentionally (uses the configured API and may incur charges):

```bash
uv run python scripts/smoke_ai.py --allow-provider
```

A second model pass is still an AI judgment, not proof of correctness. The same
model can repeat an error. All answers should remain checkable against sources.

## Live interface

The running app was inspected in the Codex browser: sign-in, course creation,
original example import, evidence answer and source controls. The library was
visually checked at the normal desktop viewport and at **390 × 844**. The phone
layout collapses the sidebar and stacks metrics without horizontal overflow.
Screenshots show synthetic test content and a temporary test account, not user
course materials. The displayed provider model reflects the preview configuration
at capture time.

- [Desktop library](screenshots/library.jpg)
- [Phone library](screenshots/mobile-library.jpg)

## Material limits

- Local lexical search can miss paraphrases. There is no measured semantic
  retrieval claim and no recycled metric from the former corpus.
- Images, diagrams and scans need user-supplied text/OCR. Summaries sample up to
  eight passages; they are not whole-document coverage guarantees.
- AI provider quotas/outages, malformed output and conservative support checks
  can trigger local fallback. No automatic paid retry loop is used.
- Account isolation is tested, but public multi-tenant hosting requires further
  infrastructure and security work documented in the deployment guide.
- JSON account exports are readable portable copies; automatic re-import is not
  supported. Full operator DB backup/restore is tested.
- Latest-tree cleanup does not erase earlier Git commits, hosted caches or forks.
