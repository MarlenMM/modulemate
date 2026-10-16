# ModuleMate

A local chatbot for understanding and working with your own materials. Built from the
HIT Summer School 2026 Smart Chatbot project, then extended as a portfolio project.

Ask questions, explain a passage, summarize, compare, work through a method, or
rewrite text you provide. The same conversation system works across subjects and
document types. Answers use your selected materials; if the information is missing,
ModuleMate says so. It can explain the supported part of an exercise without
inventing a complete solution. Use **Check sources** to inspect
cited passages. The interface and answers are in English. No login, registration,
or hosted deployment is required.

## Start on Apple Silicon

Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/) are required.

```sh
uv sync --locked --extra ml --extra local
uv run --extra ml --extra local python scripts/prepare_models.py
uv run --extra ml --extra local python scripts/prepare_chat_model.py
uv run --extra ml --extra local streamlit run app.py
```

Open <http://127.0.0.1:8501>, create a course, and upload notes or add the original
Learning Lab example. Model setup downloads pinned public weights into your home
cache; it never uploads your study material. Subsequent local inference works offline.
The 9B conversational model uses about 5.5 GiB of downloaded weights. A smaller
candidate can be prepared with `MODULEMATE_CHAT_VARIANT=4b` (see validation for results).
Keep that variable set when running the app to select it. No external GPU is needed.

The `ml` extra provides semantic retrieval, reranking and an extractive reader.
The `local` extra provides MLX generation on Apple Silicon. Use both extras in
`uv run` commands so uv preserves the installed runtimes.

## Other computers and optional providers

All library, cards and planning tools also run with `uv sync` and
`uv run streamlit run app.py`. Without a conversational model, the assistant
explicitly shows **Excerpt mode**; it is a limited source-reading fallback.

For a compatible local server (for example, an already installed Ollama server),
copy `.env.example` to `.env`, set its base URL (such as
`http://127.0.0.1:11434/v1`) and an installed model name. A loopback server needs no
API key. Hosted providers require HTTPS, a model name and a key. Select the
configured provider in the sidebar and explicitly enable provider requests.
These send selected text and recent conversation to that provider. Local MLX
never silently falls back to a hosted API.

## What changed in 2.2

- Genuine instruction-model conversations replace sentence selection as the main
  prepared local experience. Short topics, examples, code explanations, learner
  feedback and topic changes use the actual latest question and recent context.
  Factual answers need supporting material. Text supplied in a question can also
  be rewritten or organized without inventing a document citation.
- Section-aware ingestion preserves headings; evidence assembly handles multiple
  question parts, adjacent passages, complete short documents and ordered lessons
  that continue through long material without claiming unseen coverage. Distinctive
  topic terms keep supplies, dates and rules attached to the right document.
- Bounded numeric tools handle explicitly supplied numbers. Examples that
  illustrate a provided concept are distinguished from statements in the notes.
- Source IDs and exact quotes are checked. An additional factual review can request
  one correction supported by real evidence; its outage does not erase an otherwise valid primary response.
  A model refusal remains a refusal, and rejected drafts are never disguised as
  answers by substituting an unrelated quotation.
- Local answers are written separately from their evidence metadata. Constrained
  decoding keeps protocol fields out of the explanation; short-format requests,
  quiz actions and learner replies have explicit conversation state.
- One local workspace replaces accounts. Existing account databases are backed up
  and migrated, preserving every course and its documents, chats and study tools.
- Cards support spaced review and practice quizzes; a capacity-aware planner exports
  deadlines to a calendar. Documents, courses and conversations can be managed locally.

## Supported materials

PDF, DOCX, PPTX, TXT, Markdown and IPYNB (15 MB per file). Notebook code is read as
text and never executed. Image-only scans and diagrams require OCR or a text
description supplied by you. No university corpus, personal student submission,
credentials, runtime database or model weights are bundled in this repository.

## Quality and project evidence

```sh
uv run --extra ml --extra local ruff check .
uv run --extra ml --extra local pytest -q
uv run --extra ml --extra local python scripts/evaluate_conversations.py
```

The default conversation challenge uses original fictional material spanning
events, history, biology, project requirements, practical notices, code and fiction.
It covers conflicting versions, precise questions, explanations, exercise help,
follow-ups, transformations and missing evidence. It records
actual prose, model calls and timing. Its automated phrase/length constraints are
**diagnostics, not a claim of semantic accuracy**. Read the answers as well.
Historical 2.1 retrieval and extraction scores do not measure this new generator.

- [Rebuild plan and cycle log](docs/INTELLIGENCE_REBUILD.md)
- [2.2 validation and observed limits](docs/VALIDATION_V22.md)
- [HIT project alignment](docs/HIT_PROJECT_ALIGNMENT.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Local data and provider boundaries](docs/PRIVACY_AND_DEPLOYMENT.md)
- [Migration](docs/MIGRATION.md)

The model can still make mistakes. A matching quote or model review does not prove
that an explanation is correct; the source panel is there to help you inspect it.

## Model and implementation references

[MLX LM](https://github.com/ml-explore/mlx-lm),
[Outlines Core](https://github.com/dottxt-ai/outlines-core),
[Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-9B), and
[the pinned MLX conversion](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit)
provide the local generative runtime and weights. The app integrates pretrained
models; it does not claim to have trained an LLM on student work.

MIT license for the application. Model licenses remain with their respective authors.
