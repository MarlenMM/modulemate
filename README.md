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
Use the uv setup: the project explicitly overrides Streamlit's Pillow upper bound
to install Pillow 13 security fixes. See [compatibility validation](docs/VALIDATION_V25.md).

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

## Learning cards from your materials

Open **Cards → Create cards → Generate from materials**. Select documents, add an
optional topic, choose 1–24 cards and a recall style, then click **Generate learning
cards**. AI selects learning targets and writes questions; the application copies
concise answer spans from the actual source. It constructs cloze sentences itself
so blanks cannot silently rewrite a condition, number or formula.

The selection samples useful sections across documents, including their endings.
Topic selection and conservative paraphrase checks can use the prepared local
embedding model. Each generated question/answer receives an individual source
review; a prepared independent QA reader can also request a repair for a strong
question/answer mismatch. Targeted repairs retain independently useful cards and
can fill a useful missed section. Sparse material can
produce fewer cards. Short labelled numerical facts need no model call.

Inspect the preview, edit any question or answer, select the cards to keep, and
save them together. **Supporting material** stays available after saving and
revealing an answer. Your edits retain the excerpts but are marked as edited.
**Write your own** remains available, as does explicitly labelled **Source cloze**
practice without a generator. A failed AI run preserves an earlier preview.

Model checks are fallible. Exact source membership does not prove that a question
has the right premise, or that OCR correctly read the original. Long selections
show their actual passage coverage; concentrate on a smaller selection for depth.
See [card design and experiments](docs/CARDS_DEVELOPMENT.md) and
[card validation](docs/VALIDATION_V25.md).

## Understanding and evidence

Qwen supplies the pretrained language capability. ModuleMate now also trains its
own passage-ranking, relevance and answerability heads on original labeled data.
The models combine text features, pretrained embeddings and source-bound reader
signals. Training, calibration and test documents are separate; inspectable JSON
weights and reproduction commands are included.

The assistant checks evidence for each requested part and their combined context.
Uncertain coverage can trigger one deeper search, without letting a small model
force a refusal. Explicit references to another uploaded file can bring in its
relevant sections within your selection. Short libraries are read together.
Request-local caches batch searches; input-only rewrites omit unnecessary library
retrieval and unrelated history.

Main ideas, themes, interpretations and connections are supported learning tasks;
they need not appear as a ready-made sentence in a slide. Broad questions read
across the material. Long selections include passages from their beginning,
middle and end. If a document reference is ambiguous, the assistant asks which
one you mean. Source checks bind citations to real text; they do not require an
explanation to be a quotation. Short facts and formulas remain citable.
If an answer reports missing information, one wider look can reconsider additional
allowed passages even when a small evidence model was confidently wrong.
Document focus survives navigation between Assistant and Cards, separately for
each course and tool. Sampled replies show how many available passages were read.
An explicit source quotation whose wording cannot be verified receives one focused
correction attempt; an unresolved wording mismatch is marked for inspection rather
than turning the whole answer into a refusal.
Fresh material questions use current source text without copying prior assistant
claims. Follow-ups retain the conversation segment needed to resolve their references.
Possible stacked PDF equations get a layout notice. The source stays available;
unclear fraction grouping can be checked without blocking the conceptual explanation.

Rate an answer to save a local correction. **Conversation options → Export answer
review cases** prepares a queue ordered by feedback and estimated uncertainty.
These are cases for human review; feedback never automatically retrains the model.

```sh
uv run python scripts/train_evidence.py
uv run python scripts/evaluate_evidence.py --split test
uv run --extra ml --extra local python scripts/train_evidence.py --reader
uv run --extra ml --extra local python scripts/evaluate_evidence.py --reader --split test
uv run --extra ml --extra local python scripts/evaluate_conversations.py --suite evidence --policy adaptive
```

See the [evidence model/data card](docs/EVIDENCE_MODEL_CARD.md) and
[2.3 experiments and limitations](docs/VALIDATION_V23.md). Small synthetic results
do not establish general chatbot accuracy. An independent NLI experiment remains
disabled by default after falsely flagging valid answers; adding another model
was not automatically an improvement.

## Conversational foundation from 2.2

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
  deadlines to a calendar. Generated cards select actual source spans for their
  backs and remove duplicates. Documents, courses and conversations can be managed locally.

## Supported materials

PDF, DOCX, PPTX, TXT, Markdown, IPYNB, PNG, JPEG and WebP (15 MB per file).
Notebook code and saved text/image outputs are read as static material, never
executed or recomputed. Saved results may be stale. Local OCR recovers printed text
from scanned and mixed PDF pages, embedded slide/document pictures and uploaded
images, including outlined PDF letters that have no native text or image objects.
Apple Vision is installed on macOS with the normal setup. Linux/Windows
use an installed [Tesseract engine with English data](https://tesseract-ocr.github.io/tessdoc/Installation.html)
(Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-eng`). Native text remains
usable when OCR is unavailable. No hosted OCR service is called.

Word headers, footers, referenced notes and text boxes are also searchable.
Presentations retain native chart categories and cached values with their number
formats, plus SmartArt text labels. Cached values are not recalculated workbook
results, and text labels do not establish a diagram's visual relationships.

OCR locations and import notices identify recovered and unread content. Inspect
the text preview, especially numbers, names and formulas. Diagram relationships,
chart values and handwriting may still need a description. Imports are bounded
to 500 OCR images and a 35-second recognition budget; remaining locations are
reported. Original uploads are discarded, so reimport an original file to recover
images omitted by an older version. No university corpus, personal student submission,
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
- [ML development plan and cycles](docs/ML_DEVELOPMENT.md)
- [Evidence model and original dataset](docs/EVIDENCE_MODEL_CARD.md)
- [2.3 validation](docs/VALIDATION_V23.md)
- [Current intelligence decisions and cycles](docs/INTELLIGENCE_V24.md)
- [2.4 validation and real-app checks](docs/VALIDATION_V24.md)
- [2.5 learning-card design, experiments and validation](docs/VALIDATION_V25.md)
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
