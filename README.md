# ModuleMate

A retrieval-grounded course assistant for the HIT Summer Course 2026
"Artificial Intelligence and Machine Learning" module.

ModuleMate answers student questions using **only** the module's own teaching
materials — the eight lecture decks, the teaching schedule and the assessment
brief. Every substantive answer cites the document and slide (or section) it
came from, and when the corpus does not support an answer the system says so
and redirects to the tutor rather than guessing.

```
question
   │
   ├─▶ intent router (4 classes)  ──▶ out-of-scope + confident ──▶ refusal
   │
   ├─▶ query rewriter (follow-ups → standalone)
   │
   ├─▶ top-k dense retrieval (FAISS)
   │        └─ best similarity < threshold ──────────────────────▶ refusal
   │
   └─▶ grounded generation ──▶ cited answer   (or INSUFFICIENT_CONTEXT ─▶ refusal)
```

## Setup

Requires **Python 3.12** and **LibreOffice** (for the legacy `.ppt`/`.doc`
conversion). On macOS the OCR stage additionally uses the system Vision
framework via `pyobjc`.

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

The corpus ships as legacy binary Office files and the decks are largely
images, so ingestion has three stages. Run them in order:

```bash
./scripts/convert_corpus.sh          # .ppt/.doc -> .pptx/.docx  (LibreOffice)
.venv/bin/python scripts/ocr_slides.py   # OCR the image-only slides (macOS Vision)
.venv/bin/python index.py                # chunk -> embed -> FAISS index
```

`index.py` is the single-command rebuild required by FR9 — it is idempotent and
regenerates everything downstream of the converted corpus. To refresh the
corpus, drop a new file into `corpus/raw/` and re-run the three commands. No
code change is needed.

## Run the app

```bash
.venv/bin/python -m streamlit run app.py
```

## Run the evaluation

```bash
.venv/bin/python eval/harness.py     # runs the gold set, writes results/eval_raw.json
.venv/bin/python eval/report.py      # writes results/eval_results.md + confusion_matrix.png
```

Useful flags: `--limit N` to run a subset, `--no-judge` to skip the LLM
groundedness judge and use the lexical fallback (halves token spend).

Supporting measurements:

```bash
.venv/bin/python scripts/calibrate_threshold.py   # re-derive the refusal threshold
.venv/bin/python scripts/measure_latency.py       # interactive (non-burst) latency
.venv/bin/python classify.py                      # retrain router + generalisation probes
.venv/bin/python -m pytest tests/ -q              # sanity tests
```

## Configuration

Everything tunable lives in `config.yaml` — retrieval `k`, the refusal
similarity threshold, and every model name. No module hardcodes these
(NFR5). The refusal threshold is not a guess: it is derived from measured
in-scope versus out-of-scope similarity distributions by
`scripts/calibrate_threshold.py`, and the derivation is written to
`results/threshold_calibration.md`.

## Layout

| Path | Purpose |
|---|---|
| `ingest.py` | M1 — extraction, OCR merge, metadata-preserving chunking |
| `index.py` | M2 — embedding + persistent FAISS index (also the FR9 rebuild) |
| `retrieve.py` | M3/M5 — top-k retrieval and the similarity refusal gate |
| `classify.py` | M6 — four-class intent router (the sole trained artefact) |
| `generate.py` | M4/S1 — swappable LLM backend, grounded prompt, query rewriting |
| `pipeline.py` | end-to-end turn handling shared by the UI and the harness |
| `app.py` | M7/S2 — Streamlit chat UI with the expandable source panel |
| `config.py` / `config.yaml` | S4 — configuration loader and values |
| `eval/` | M8 — gold set, harness, report generator |
| `scripts/` | corpus conversion, OCR, threshold calibration, latency probe |
| `results/` | measured outputs (scorecard, confusion matrix, calibration) |

## Results

See `results/eval_results.md` for the full NFR scorecard, and
`IMPLEMENTATION_SUMMARY_FOR_REPORT.md` for the build narrative, key decisions
and known limitations.
