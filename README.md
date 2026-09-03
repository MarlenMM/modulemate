# ModuleMate

[![CI](https://github.com/MarlenMM/modulemate/actions/workflows/ci.yml/badge.svg)](https://github.com/MarlenMM/modulemate/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-750014.svg)](LICENSE)
[![Fabricated citations: 0](https://img.shields.io/badge/fabricated%20citations-0-1a7f37.svg)](results/eval_results.md)

A retrieval-grounded course assistant for the **HIT Global Summer School 2026**.

ModuleMate answers student questions using **only** material issued by the
programme. Every substantive answer cites the document and slide, page, section
or cell it came from, and when the corpus does not support an answer the system
says so and redirects to the tutor rather than guessing.

### Measured, on a 63-question gold set

| | | | |
|---|---|---|---|
| Retrieval hit@3 | **0.923** | Groundedness (LLM judge) | **0.960** |
| Citation correctness | **0.940** | Intent-router accuracy | **0.857** |
| Fabricated citations | **0** | Refusal rate, out-of-scope | **1.000** |

**14 of 15 criteria met.** The one that is not — sustained-burst latency, 10.8 s
median against a < 5 s target — is recorded as a **FAIL** in
[the scorecard](results/eval_results.md), with its cause identified (hosted-API
throttling; under 30 ms of it is retrieval) rather than the target being moved.
The false-refusal rate on in-scope questions (**0.038**) is published next to
the refusal rate, because a refusal rate on its own can be maximised by
refusing everything.

---

### The interface

![ModuleMate's opening screen: a sidebar showing 1,799 indexed passages from 51
source documents, and a main panel explaining how a question is routed, searched
and answered, with three suggested questions to start
from.](docs/empty-state.png)

Built around the evidence rather than around the chat. The answer is the only
element allowed to be large; everything the system measured — intent, similarity
scores, model, latency — is set in monospace beneath it, so prose and evidence
are never mistaken for each other. The rule down the left of a turn carries its
state: grey for the question, teal for an answer that cleared the retrieval
gate, amber for a refusal, brick for a generation backend that could not be
reached.

![A refused turn: the question "What are the opening hours of the campus
library?" answered with a refusal, above a meter showing the best passage
scoring 0.441 against a gate of 0.48 — short by 0.039.](docs/refusal.png)

Every turn draws the decision that produced it. The meter puts the best
passage's cosine on a scale with the calibrated threshold notched on it, so a
refusal can be read rather than taken on trust: here a reasonable question was
routed correctly, retrieved the five closest passages, and still fell 0.039
short of the gate — refused in 0.3 s, because no model is called once retrieval
has failed. The same meter appears under answers, showing the margin they
cleared by.

---

**Corpus** — 51 staff-issued documents across five formats, 1,799 indexed passages:

| Source | Contents |
|---|---|
| AI & Machine Learning module | 8 lecture decks, teaching schedule, assignment brief, project brief |
| Week 1 Machine Learning | 8 lecture PDFs, 4 assignment briefs, 8 lab notebooks, readmes |
| Intelligent Software Engineering | code-review lab guide, course-introduction lecture |
| Programme level | Global Summer School brochure and academic theme list |
| Announcements | 5 course group-chat exports |

> **The corpus itself is not in this repository.** Every document in it is
> teaching material issued by HIT staff, and redistributing it is not mine to
> do. What is committed is the *manifest* naming those files and the pipeline
> that ingests them: point `corpus_sources.yaml` at your own copy of the
> material and `index.py` rebuilds everything. This is also why the measured
> numbers above, rather than a hosted demo, are the reproducible artifact here.

The corpus is declared in [`corpus_sources.yaml`](corpus_sources.yaml), not in
code. It contains **course-issued material only**: the student's own reports,
plans, answered assignments and projects are excluded by explicit rule *and* by
an authorship guard that rejects any file whose opening pages carry the
student's name or ID.

```
question
   │
   ├─▶ intent router (4 classes) ──▶ out-of-scope AND confident ─────▶ refusal
   │
   ├─▶ query rewriter (follow-ups → standalone)
   │
   ├─▶ retrieval
   │     ├─ dense (FAISS, MiniLM)      ┐
   │     ├─ lexical (BM25)             ├─ reciprocal rank fusion
   │     ├─ cross-encoder re-rank      ┘
   │     └─ source-authority prior
   │        └─ best cosine < threshold ───────────────────────────▶ refusal
   │
   └─▶ grounded generation ──▶ cited answer + suggested follow-ups
          └─ INSUFFICIENT_CONTEXT / out-of-scope request ─────────▶ refusal
```

Three independent refusal layers, because none of them catches everything:
the router stops social and policy turns before any network call, the
similarity gate stops topical drift before generation, and the generation
prompt stops requests to act as the tutor. Measured refusal rate is 1.000
*because* they cover each other's failures.

## Setup

Requires **Python 3.12** and **LibreOffice** (for the legacy `.ppt`/`.doc`
conversion). On macOS the OCR stage additionally uses the system Vision
framework via `pyobjc`. First run downloads two models (~350 MB): the
sentence-transformer used for embeddings and the cross-encoder used for
re-ranking.

```bash
python3.12 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set the API key for your generation backend:

```bash
cp .env.example .env   # then edit and add your key
```

Which variable is read is controlled by `generation.api_key_env` in
`config.yaml`. The default backend is Groq via its OpenAI-compatible endpoint;
OpenAI, DeepSeek, a local vLLM server, or Anthropic can be substituted by
editing `config.yaml` alone — no code change.

## Build the index

Sources arrive in six formats, several of them legacy binaries, and the Week 2
decks are largely images — so ingestion runs in four stages. Run them in order:

```bash
.venv/bin/python scripts/collect_corpus.py   # apply corpus_sources.yaml -> corpus/raw
```

```bash
./scripts/convert_corpus.sh                  # .ppt/.doc/.rtf -> .pptx/.docx/.txt
```

```bash
.venv/bin/python scripts/ocr_slides.py       # OCR image-only slides (macOS Vision)
```

```bash
.venv/bin/python index.py                    # chunk -> embed -> FAISS index
```

`index.py` is the single-command rebuild required by FR9 — idempotent, and it
regenerates everything downstream of the converted corpus. To change what is
indexed, edit `corpus_sources.yaml` and re-run the four commands; no code
change is needed. `collect_corpus.py` prints a full audit of what it included,
skipped, de-duplicated and rejected.

## Run the app

```bash
.venv/bin/python -m streamlit run app.py
```

The interface is built around the evidence rather than around the chat. The
answer is the only element allowed to be large; anything the system measured —
intent, similarity scores, model, latency — is set in monospace beneath it so
prose and evidence are never mistaken for each other. The rule down the left
of each turn carries its state: grey for the question, teal for an answer that
cleared the retrieval gate, amber for a refusal, brick for a generation backend
that could not be reached. Under every answered *and* refused turn,
`ui.gate_meter` draws the decision itself — the best passage's cosine on a
scale with the calibrated threshold notched on it — because a refusal is only
worth trusting if the reason for it can be seen.

## Run the evaluation

```bash
.venv/bin/python eval/harness.py     # runs the gold set, writes results/eval_raw.json
.venv/bin/python eval/report.py      # writes results/eval_results.md + confusion_matrix.png
```

Useful flags: `--limit N` to run a subset, `--no-judge` to skip the LLM
groundedness judge and use the lexical fallback (halves token spend).

Supporting measurements:

```bash
.venv/bin/python scripts/ablation.py              # what each retrieval stage contributes
.venv/bin/python scripts/analyse_retrieval.py     # exact vs document-level hit@k (no API cost)
.venv/bin/python scripts/calibrate_threshold.py   # re-derive the refusal threshold
.venv/bin/python scripts/measure_latency.py       # interactive (non-burst) latency
.venv/bin/python classify.py                      # retrain router + generalisation probes
.venv/bin/python feedback.py                      # summarise logged thumbs up/down
.venv/bin/python -m pytest tests/ -q              # sanity tests
```

## Configuration

Everything tunable lives in `config.yaml` — retrieval `k`, the refusal
similarity threshold, source-authority weights, and every model name. No module
hardcodes these (NFR5). *What* is indexed lives separately in
`corpus_sources.yaml`. The refusal threshold is not a guess: it is derived from measured
in-scope versus out-of-scope similarity distributions by
`scripts/calibrate_threshold.py`, and the derivation is written to
`results/threshold_calibration.md`.

## Layout

| Path | Purpose |
|---|---|
| `corpus_sources.yaml` | Declares which source files enter the corpus, and why |
| `scripts/collect_corpus.py` | Applies the manifest; de-dupes; authorship guard |
| `ingest.py` | M1 — extraction (pptx/docx/pdf/ipynb/txt), OCR merge, chunking |
| `index.py` | M2 — embedding + persistent FAISS index (also the FR9 rebuild) |
| `retrieve.py` | M3/M5/C1/C2 — hybrid retrieval, re-ranking, refusal gate |
| `feedback.py` | C3 — thumbs up/down logging for post-hoc error analysis |
| `classify.py` | M6 — four-class intent router (the sole trained artefact) |
| `generate.py` | M4/S1 — swappable LLM backend, grounded prompt, query rewriting |
| `pipeline.py` | end-to-end turn handling shared by the UI and the harness |
| `app.py` | M7/S2 — chat UI: turn flow, source panel, ratings, follow-ups |
| `ui.py` | the interface's design system — palette, type scale, mark, gate meter |
| `.streamlit/config.toml` | theme and client settings the stylesheet cannot set |
| `config.py` / `config.yaml` | S4 — configuration loader and values |
| `eval/` | M8 — gold set, harness, report generator |
| `scripts/` | corpus conversion, OCR, threshold calibration, latency probe |
| `results/` | measured outputs (scorecard, confusion matrix, calibration) |
| `docs/` | interface screenshots used above |

## Results

See `results/eval_results.md` for the full NFR scorecard, and
`IMPLEMENTATION_NOTES.md` for the build narrative, key decisions
and known limitations.
