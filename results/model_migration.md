# Generation model migration

**2026-09-03.** The provider retired every Llama model this system was built on.
`llama-3.3-70b-versatile` (generation) and `llama-3.1-8b-instant` (query
rewriting, follow-up suggestions) both return `404 model_not_found`. Because a
404 is a bad request rather than a capacity error, the R3 fallback chain
deliberately re-raises it instead of retrying elsewhere — correct behaviour,
but it meant every question failed outright until the config was changed.

Seven chat-capable models remained on the provider. This is how the
replacements were chosen.


## Candidates, on the three calls the app actually makes

One admin question ("What is the word limit for the individual project
report?"), the real prompts, and the real token caps each call site uses:
600 for an answer, 120 for a rewrite, 120 for follow-ups.

| Model | answer | rewrite | follow-ups | verdict |
|---|---|---|---|---|
| `qwen/qwen3.8-27b` | 0.5s, 42 tok, cited | 0.3s, 13 tok | 0.5s, 34 tok, 3 lines | **selected** |
| `openai/gpt-oss-120b` | 0.9s, 139 tok, cited | 0.7s, 107 tok | 0.7s, hits cap, 2 truncated lines | fallback |
| `openai/gpt-oss-20b` | 0.8s, 155 tok, cited | 0.6s, 97 tok | 0.6s, hits cap, **0 lines** | fallback |
| `qwen/qwen3.6-27b` | 1.2s, 477 tok | leaks `<think>` | leaks `<think>` | rejected |
| `groq/compound-mini` | 1.1s, 221 tok, cited | 0.8s, 132 tok | 1.3s, 339 tok | rejected |
| `allam-2-7b` | 0.3s, 83 tok | prefixes "Rewritten question:" | 0.3s, 82 tok | rejected |

Three rejections were on correctness, not speed:

- **`qwen3.6-27b`** emits its raw `<think>` reasoning block into the answer
  text. A student would read the model's deliberation as the answer.
- **`allam-2-7b`** reintroduces exactly the failure the 8B model was dropped
  for: "Based on the provided CONTEXT, the word limit is…". Retrieval
  scaffolding leaking into the answer tells the student how the machine works
  instead of answering the question.
- **`groq/compound-*`** are agentic models that can reach the web. A system
  whose entire guarantee is that it answers only from staff-issued material
  cannot be built on a model that can go and look something up.

The two gpt-oss models are reasoning models and spend the 120-token budget the
auxiliary calls allow on reasoning before emitting anything, so they return a
truncated or empty completion. Follow-up suggestions (C4) fail silently by
design — a missing suggestion is a non-event — which means this would have
degraded invisibly rather than raising anything. They remain sound as
generation fallbacks, where the budget is 600.


## The two real contenders, on citation fidelity

Ten in-scope gold questions, identical retrieved passages for both models, so
the only variable is the generator.

| Model | cited | citation matches a retrieved passage | scaffolding leaks | median latency |
|---|---|---|---|---|
| `qwen/qwen3.8-27b` | 9/10 | 9/10 | 0/10 | **0.55s** |
| `openai/gpt-oss-120b` | 9/10 | 9/10 | 0/10 | 1.32s |

They are indistinguishable on grounding and identical on citation accuracy.
Latency broke the tie: sustained-burst latency is the one criterion this
system fails (10.8s median against a < 5s target, almost entirely hosted-API
time), so where two models are equal on correctness the faster one is the
right default.


## Consequence for the published numbers

**The scorecard in `eval_results.md` was measured on `llama-3.3-70b-versatile`
and has not been re-measured on the replacement.** Retrieval metrics are
unaffected — hit@3, MRR and the refusal-gate calibration involve no generator
at all — but groundedness, citation correctness and every latency figure are
properties of a model that no longer exists. The table above is a ten-question
spot check, not a substitute for the 63-question harness. Re-running
`eval/harness.py` is what would make the published numbers describe the
system as it now stands.
