# Validation — 16 October 2026

Historical version 2.1 measurements. These do not measure the 2.2 generator or account-free application; see [the current rebuild](INTELLIGENCE_REBUILD.md).

This upgrade was developed across 15–16 October using original synthetic
material. No private HIT documents, submissions, chats or user library contents
are test fixtures. The instructor brief and original project report were reviewed
privately for requirement alignment. The [v2.0 validation](VALIDATION_V2.md)
remains a historical record and does not describe the new implementation.

## Automated and live checks

The complete suite covers extraction, ownership/authentication, schema migration,
backup/restore, deletion, study scheduling, provider failure/validation, vector
cache persistence and repair, concurrent indexing/deletion, conversation state,
exact identifiers, reader span membership and Streamlit user journeys.

Real-model checks are explicitly enabled; ordinary tests do not download weights.
The encoder, reranker and reader successfully loaded and inferred with socket
connections denied. Lint, formatting, the locked environment and package advisory
checks are run as release checks. Results are local macOS/Python 3.12 observations;
Linux/Python 3.13 CI results are separate.

Reproduce:

```bash
uv sync --locked --extra ml
uv run --extra ml python scripts/prepare_models.py
MODULEMATE_TEST_LOCAL_ML=1 uv run --extra ml pytest -q
uv run --extra ml ruff check .
uv run --extra ml ruff format --check .
uv run --extra ml pip-audit --progress-spinner off
uv run --extra ml python scripts/check_release.py
```

Release checks: **88 tests** with real models enabled, or **83 passed / 5 skipped**
in the lightweight installation. See the checked-in scripts for reproduction.

## Retrieval ablation

The [retrieval set](../evaluation/retrieval.json) contains **46 questions**,
**23 passages in seven documents**, **21 paraphrase questions**, 25 lexical
questions (including Korean/Chinese), and eight unrelated questions. It is a
**development benchmark**, not a blinded test or general academic accuracy
estimate. Thresholds and model choices were evaluated on it.

| Method | Passage hit@1 | Passage hit@3 | MRR@6 | Unrelated queries rejected |
|---|---:|---:|---:|---:|
| Word matching | 38/46 (82.6%) | 43/46 (93.5%) | 0.866 | 8/8 |
| Dense only | 42/46 (91.3%) | 45/46 (97.8%) | 0.946 | 8/8 |
| Hybrid, deployed default | 42/46 (91.3%) | 46/46 (100%) | 0.953 | 8/8 |
| Hybrid + passage reranker | 42/46 (91.3%) | 45/46 (97.8%) | 0.946 | 8/8 |

Paraphrase hit@1 improved from **14/21 to 18/21**. The small intent evaluation
scored **25/32 (78.1%)**, with all misses retained in the report. The classifier
therefore guides style and does not indiscriminately refuse questions.

An early cross-encoder-only relevance threshold discarded useful paraphrases.
Rank fusion improved that behavior, but passage reranking still reduced recall;
the final default keeps hybrid passage retrieval and uses the cross-encoder for
sentence selection. The Multi-QA encoder alternative performed worse in this
configuration (82.6% reranked hit@1; 7/8 unrelated questions rejected) and was not
selected. The first attempt and alternative are retained as development evidence.

Machine-readable results and all retrieval misses:
[final](../evaluation/results-local.json),
[first attempt](../evaluation/results-initial.json),
[alternative](../evaluation/results-multiqa-candidate.json).

```bash
uv run --extra ml python scripts/evaluate_intelligence.py --local-ml
uv run --extra ml python scripts/prepare_models.py --include-candidate
uv run --extra ml python scripts/evaluate_intelligence.py --local-ml --encoder multiqa
```

## Complete answers

Retrieval hits do not establish answer quality. The composed answers were also
checked against explicit reference facts, including required conditions, names,
units and numbers. Every answer is saved for inspection.

| Check | Word matching assistant | Local neural assistant |
|---|---:|---:|
| All specified facts present, development questions | 37/46 | 45/46 |
| All specified facts present, additional challenge questions | Not measured | 18/20 |
| Citation markers resolve, development questions | 46/46 | 46/46 |
| Citation markers resolve, challenge questions | Not measured | 20/20 |

The 20 challenge questions were written after the initial development pass; no
thresholds were fitted to those results. These checks are **phrase-based fact
coverage**, not semantic answer correctness, citation entailment, graded learner
understanding, or LLM-judged groundedness. A sentence can contain expected facts
while still being incomplete or badly emphasized. Citation resolvability means
only that the marker points to an available passage.

The final development miss is a recall-practice paraphrase that selects an
unhelpful model-evaluation statement. Two challenge misses concern naming the
structure with push/pop and explaining how unbounded recursion is avoided.
These failures remain visible; no general accuracy or perfect refusal claim is
made. Very indirect paraphrases, missing details and conflicting sources still
need user judgment or a stronger optional generator.

The local reader improved detail selection but made errors when its raw scores
were compared across unrelated paragraphs. The final version selects a relevant
source first, uses the reader only within that source, validates every offset,
and displays whole source sentences with necessary antecedents. No-answer scores
are hints, not correctness probabilities. Korean/Chinese sentence selection uses
word matching rather than the English reader.

```bash
uv run --extra ml python scripts/evaluate_answers.py
uv run --extra ml python scripts/evaluate_answers.py --local-ml
uv run --extra ml python scripts/evaluate_answers.py --local-ml --challenge
```

[All local answers](../evaluation/answers-local.json),
[challenge answers](../evaluation/answers-challenge-local.json),
[word-matching answers](../evaluation/answers-lexical.json).

## Performance

The small answer evaluation reports median/p95 end-to-end local answer latency
in its JSON files. These include local retrieval, composition and persistence;
the first request also includes lazy reader loading. They exclude initial model
installation and use a small synthetic corpus. The 100-document benchmark reports
index preparation separately from warm retrieval and excludes answer composition
and hosted generation.

| Scale | Observation |
|---|---:|
| Synthetic documents / passages | 100 / 3,700 |
| Complete neural index preparation + first search | 67.72 seconds |
| Warm retrieval median / p95 | 281.12 / 316.15 ms |
| Exact topic identifier hits | 100/100 |
| Small development corpus answer median / p95 | 120.71 / 183.81 ms |
| Additional challenge answer median / p95 | 125.77 / 418.22 ms |

The large run occurred while other local evaluations were active; CPU contention
contributes to its timings. These are observations, not latency guarantees.
[Raw large-library measurement](../evaluation/benchmark-local.json).

```bash
uv run --extra ml python scripts/benchmark.py --local-ml
```

An intermediate large-library run caught exact identifiers losing to semantically
similar text. Alphanumeric identifiers now constrain candidates; an unknown
identifier returns no evidence. This is tested separately from ordinary semantic
paraphrases. Inference does not hold database write locks, and stale/deleted
source identities are rechecked before caching/display.

## Hosted provider and interface

The configured Groq `openai/gpt-oss-120b` provider was exercised using only the
original fictional Learning Lab example. In that smoke run, Explain and Summarize
passed source/support validation; Compare and Socratic generation fell back to
local responses, and card generation failed with a provider error. These are
availability/validation observations, not measured AI accuracy. Fake-provider
regressions separately cover outages, fabricated quotes, invalid IDs and
unsupported claims. No automatic paid retries or provider switches were added.

The live browser was checked with a temporary synthetic account: comparison,
“Explain the second one,” source controls and follow-up buttons. The final
screenshot uses only original demo material. Streamlit tests additionally cover
recall practice, card review, planning, export, deletion and browser isolation.

![Local conversation](screenshots/local-conversation.jpg)

## Remaining limits

Local answers are extractive, not unrestricted language-model reasoning. Summary
sampling, ambiguous questions, numerical/negation subtleties and source conflicts
remain limitations. English model quality does not establish multilingual
semantic retrieval quality. Hosted generation can still fail validation or be
unavailable. No private content is bundled, trained into models or used to claim
these results; old public Git history is not erased by this upgrade.
