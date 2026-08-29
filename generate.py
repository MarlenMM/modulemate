"""M4 — Grounded generation with citation, and S1 query rewriting (FR6, FR4).

The generator sits behind a thin `LLMBackend` interface (the R3/R4 mitigation
in the locked risk register): retrieval, refusal and the UI never import a
vendor SDK, so the backend can be swapped — Groq, OpenAI, DeepSeek, Anthropic,
or a local open-weights server — by editing config.yaml alone.

Grounding is enforced in four places, not one:
  1. the corpus manifest decides what may be cited at all;
  2. the system prompt confines the model to the supplied passages, and names
     the exact citation label it must copy;
  3. the passages are the only corpus text in the context window;
  4. the refusal gate (retrieve.py) can stop generation before it starts.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import config
from retrieve import Retrieved

# One [Source: ...] bracket, whose contents are parsed by parse_citations.
CITATION_BLOCK_RE = re.compile(r"\[Source:\s*([^\]]+?)\s*\]", re.I)

# A locator is "slide 12" / "page 7" / "cell 3" / "part 9", or a section title.
_LOCATOR_RE = re.compile(r"^(slide|page|cell|part)\s+\d+$", re.I)
# The same, but with a document name run on after it — the shape produced when
# a model space-joins two citations inside one bracket.
_LOCATOR_THEN_DOC_RE = re.compile(
    r"^((?:slide|page|cell|part)\s+\d+)\s+(\S.*)$", re.I
)

SYSTEM_PROMPT = """\
You are ModuleMate, a course assistant for the HIT Global Summer School 2026.

Your knowledge covers the material issued by the programme: the Artificial \
Intelligence and Machine Learning module (lecture decks, teaching schedule, \
assignment and project briefs), the Week 1 machine-learning lectures and lab \
notebooks, the Intelligent Software Engineering code-review lab guide, \
programme-level information, and course announcement channels.

You answer ONLY from the numbered CONTEXT passages supplied with each question.

Rules, in order of priority:
1. Use only facts stated in the CONTEXT. Never use outside knowledge, and never \
guess at a fact that is not there — not a deadline, not a percentage, not a \
slide number, not a definition.
2. If the CONTEXT does not contain enough information to answer, reply with \
exactly: INSUFFICIENT_CONTEXT
3. End every substantive answer with a citation naming the passage you used, in \
exactly this format, on its own final line:
   [Source: <document>, <slide or section>]
   Copy the document name and the slide/section label verbatim from the \
"Cite this passage as:" line of the passage you actually used. That label \
is the only correct name for the document. Do NOT use a title, heading or \
running header that appears inside the passage text, even when it looks \
more like a real document name — a citation naming anything other than the \
supplied label cannot be resolved and counts as fabricated. Do not invent a \
slide or page number. If you drew on two passages, give both citations on \
that line separated by a space.
4. Be concise: two to five sentences unless the question needs a list.
5. Write for a student, in your own voice. The retrieval machinery is \
invisible to them, so never refer to it: no "the context", "the passages", \
"Passage 1", "the provided text", "according to the document". State the fact \
directly and let the citation line carry the provenance. Write "The report \
must be 4000 words", not "According to Passage 1, the report must be 4000 \
words".
6. You explain what the module materials say. You are not the tutor and hold \
no authority over the course. If the student asks you to act on their behalf \
or to exercise the tutor's judgement — granting an extension, waiving or \
reducing a penalty, excusing an absence, accepting extenuating circumstances, \
overturning a mark, predicting their grade, or writing/completing any part of \
their assessed work — reply with exactly: INSUFFICIENT_CONTEXT
Answering the factual question behind such a request is still fine when it is \
asked as a question ("what is the late penalty?"), but never act on the \
request itself.
"""

REWRITE_PROMPT = """\
You rewrite follow-up questions in a conversation into standalone search \
queries for a course-materials search engine.

Given the recent conversation and the newest user message, output ONE \
self-contained question that preserves the user's intent and resolves every \
pronoun and ellipsis ("it", "that one", "the second one") using the \
conversation.

Output only the rewritten question. No preamble, no quotes. If the newest \
message is already self-contained, output it unchanged.
"""


@dataclass
class Generation:
    """A generated answer plus what it cost and whether it was grounded."""

    text: str
    is_refusal: bool
    latency_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_used: str = ""      # which model actually served it (see R3 fallback)


# --------------------------------------------------------------------------
# Swappable backend interface (R3/R4)
# --------------------------------------------------------------------------

class LLMBackend(ABC):
    """Minimal contract every generation backend must satisfy."""

    @abstractmethod
    def complete(self, system: str, user: str, model: str,
                 temperature: float, max_tokens: int) -> tuple[str, int, int]:
        """Return (text, prompt_tokens, completion_tokens)."""


class OpenAICompatibleBackend(LLMBackend):
    """Any host exposing the OpenAI /v1/chat/completions schema.

    Covers OpenAI itself, Groq, DeepSeek, Together, and a local vLLM or
    llama.cpp server — the swap is a base_url change in config.yaml.
    """

    def __init__(self, api_key: str, base_url: str | None, timeout: float):
        from openai import OpenAI
        self._client = OpenAI(
            api_key=api_key, base_url=base_url or None, timeout=timeout
        )

    def complete(self, system, user, model, temperature, max_tokens):
        resp = self._client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = resp.usage
        return (
            (resp.choices[0].message.content or "").strip(),
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )


class AnthropicBackend(LLMBackend):
    """Claude models. Unused by default but kept wired so the locked
    'hosted LLM API, swappable' decision is demonstrably true."""

    def __init__(self, api_key: str, timeout: float):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

    def complete(self, system, user, model, temperature, max_tokens):
        resp = self._client.messages.create(
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text, resp.usage.input_tokens, resp.usage.output_tokens


class ModelUnavailable(RuntimeError):
    """Every model in the fallback chain refused the request."""


def is_capacity_error(exc: Exception) -> bool:
    """True for rate-limit / quota / overload errors, which a different model
    can serve, as opposed to bad-request errors, which it cannot."""
    name = type(exc).__name__
    if name in ("RateLimitError", "InternalServerError", "APITimeoutError",
                "APIConnectionError"):
        return True
    text = str(exc).lower()
    return any(t in text for t in
               ("rate limit", "quota", "429", "over capacity", "overloaded",
                "503", "502"))


def make_backend(cfg=None) -> LLMBackend:
    """Construct the backend named in config.yaml."""
    cfg = cfg or config.load()
    kind = cfg.require("generation.backend")
    timeout = cfg.get("generation.timeout_seconds", 30)
    if kind == "openai_compatible":
        return OpenAICompatibleBackend(
            cfg.api_key, cfg.get("generation.base_url"), timeout
        )
    if kind == "anthropic":
        return AnthropicBackend(cfg.api_key, timeout)
    raise ValueError(f"Unknown generation.backend: {kind!r}")


# --------------------------------------------------------------------------
# Prompt construction and generation
# --------------------------------------------------------------------------

def build_context_block(passages: list[Retrieved]) -> str:
    """Render retrieved passages with the exact citation label to copy.

    Each passage is prefixed with the literal bracket the model must emit,
    which is what makes citation correctness measurable rather than hoped for.
    """
    blocks = []
    for i, p in enumerate(passages, start=1):
        blocks.append(
            f"--- PASSAGE {i} ---\n"
            f"Cite this passage as: [Source: {p.source_document}, "
            f"{p.slide_or_section}]\n"
            f"{p.text}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, passages: list[Retrieved]) -> str:
    return (
        f"CONTEXT:\n{build_context_block(passages)}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using only the CONTEXT above, and end with the citation line."
    )


class Generator:
    """Grounded answer generation and conversational query rewriting.

    Calls go through `_complete_with_fallback`, which walks a chain of models
    on capacity errors. This is risk R3 in the register — "LLM API cost,
    latency or unavailability" — implemented rather than merely documented.
    It is not hypothetical: a full evaluation exhausted the primary model's
    100,000 tokens/day allowance twice during this build, and each provider
    model carries its own separate quota, so falling through to the next one
    keeps the assistant answering instead of failing the turn.
    """

    def __init__(self, cfg=None, backend: LLMBackend | None = None):
        self.cfg = cfg or config.load()
        self.backend = backend or make_backend(self.cfg)
        self.model = self.cfg.require("generation.model")
        self.rewrite_model = self.cfg.get(
            "generation.rewrite_model", self.model
        )
        self.followup_model = self.cfg.get(
            "generation.followup_model", self.rewrite_model
        )
        self.temperature = self.cfg.get("generation.temperature", 0.0)
        self.max_tokens = self.cfg.get("generation.max_tokens", 600)
        self.fallback_models = list(
            self.cfg.get("generation.fallback_models") or []
        )
        # Which model served the last call, so the harness and the UI can
        # report when the primary was unavailable rather than hiding it.
        self.last_model_used: str = self.model
        self.refusal_message = " ".join(
            self.cfg.require("refusal.message").split()
        )

    def _complete_with_fallback(self, system: str, user: str, model: str,
                                temperature: float, max_tokens: int
                                ) -> tuple[str, int, int, str]:
        """Try `model`, then each fallback, on capacity errors only.

        Returns (text, prompt_tokens, completion_tokens, model_actually_used).
        A bad-request error is re-raised immediately: no other model will
        handle a malformed request any better, and silently retrying would
        turn a bug into four bugs.
        """
        errors: list[str] = []
        for candidate in [model, *self.fallback_models]:
            if candidate is None:
                continue
            try:
                text, p_tok, c_tok = self.backend.complete(
                    system, user, candidate, temperature, max_tokens
                )
                self.last_model_used = candidate
                return text, p_tok, c_tok, candidate
            except Exception as exc:
                if not is_capacity_error(exc):
                    raise
                errors.append(f"{candidate}: {type(exc).__name__}")
                continue
        raise ModelUnavailable(
            "all generation models unavailable — " + "; ".join(errors)
        )

    def answer(self, question: str, passages: list[Retrieved]) -> Generation:
        """Generate a grounded, cited answer from retrieved passages."""
        import time
        started = time.time()

        if not passages:
            return Generation(self.refusal_message, True, time.time() - started)

        text, p_tok, c_tok, used = self._complete_with_fallback(
            SYSTEM_PROMPT,
            build_user_prompt(question, passages),
            self.model,
            self.temperature,
            self.max_tokens,
        )

        # The model's own escape hatch — treated as a refusal, not an answer.
        if "INSUFFICIENT_CONTEXT" in text.upper():
            return Generation(
                self.refusal_message, True, time.time() - started,
                p_tok, c_tok, used
            )

        return Generation(
            text, False, time.time() - started, p_tok, c_tok, used
        )

    def suggest_followups(self, question: str,
                          passages: list[Retrieved]) -> list[str]:
        """C4 — up to three follow-ups the corpus can actually answer.

        Grounded in the retrieved passages rather than generated freely, so a
        suggestion never sends the student into a refusal. Runs on the cheap
        model with a tight token cap; failures are swallowed because a missing
        suggestion is a non-event, and returns [] when the turn was refused.
        """
        if not passages:
            return []
        excerpts = "\n\n".join(
            f"[{p.source_document} — {p.slide_or_section}]\n{p.text[:400]}"
            for p in passages[:4]
        )
        user = (
            f"QUESTION JUST ASKED: {question}\n\n"
            f"RETRIEVED EXCERPTS:\n{excerpts}\n\nFollow-up questions:"
        )
        try:
            text, _, _ = self.backend.complete(
                FOLLOWUP_PROMPT, user, self.followup_model, 0.3, 120
            )
        except Exception:
            return []
        out: list[str] = []
        for line in text.splitlines():
            line = line.strip().lstrip("-*0123456789. ").strip()
            if len(line) > 8 and line.lower() != question.strip().lower():
                out.append(line)
        return out[:3]

    def rewrite_query(self, question: str,
                      history: list[tuple[str, str]]) -> str:
        """S1/FR4 — resolve a context-dependent follow-up into a standalone query.

        `history` is [(user_text, assistant_text), ...] most recent last.
        Returns `question` unchanged when there is no history to resolve against.
        """
        if not history:
            return question
        turns = self.cfg.get("conversation.history_turns", 4)
        recent = history[-turns:]
        transcript = "\n".join(
            f"Student: {u}\nModuleMate: {a}" for u, a in recent
        )
        user = (
            f"CONVERSATION SO FAR:\n{transcript}\n\n"
            f"NEWEST STUDENT MESSAGE: {question}\n\n"
            f"Rewritten standalone question:"
        )
        try:
            text, _, _ = self.backend.complete(
                REWRITE_PROMPT, user, self.rewrite_model, 0.0, 120
            )
        except Exception:
            return question  # rewriting is an enhancement, never a hard failure
        cleaned = text.strip().strip('"').split("\n")[0].strip()
        return cleaned or question


FOLLOWUP_PROMPT = """\
You suggest what a student might reasonably ask next.

You are given the question they just asked and excerpts from the course
materials that were retrieved for it. Propose up to three short follow-up
questions that those same materials can actually answer.

Rules:
* Each must be answerable from the excerpts shown. Never invent a topic the
  excerpts do not cover — a suggestion the assistant would then refuse is
  worse than no suggestion.
* Keep each under twelve words, phrased as a student would type it.
* Do not repeat the question already asked.
* Output one question per line, nothing else. No numbering, no preamble.
"""


def parse_citations(text: str) -> list[tuple[str, str]]:
    """Extract [(document, slide_or_section)] from an answer.

    The harness scores citation correctness against this, so the parser has to
    survive the formats models actually emit rather than only the one the
    prompt asks for. Measured on real output, a strong model reliably produced
    two citations inside a single bracket:

        [Source: AI notes6..., slide 62, Lab Guide..., page 7]
        [Source: AI notes6..., slide 18 AI notes6..., slide 22]

    A regex that treats a bracket as exactly one (document, locator) pair reads
    those as one citation with a nonsense locator — which the fabrication check
    then reports as an invented source. That is a measurement bug, not a model
    failure, and it cost four false NFR3 violations before it was found.

    So: split each bracket on commas, then walk the fields pairing document
    with locator, handling the case where a locator and the next document are
    run together without a comma.
    """
    out: list[tuple[str, str]] = []
    for block in CITATION_BLOCK_RE.findall(text):
        fields: list[str] = []
        for raw in block.split(","):
            field = raw.strip()
            if not field:
                continue
            # "slide 18 AI notes6_machine learning" -> ["slide 18", "AI notes6…"]
            run_on = _LOCATOR_THEN_DOC_RE.match(field)
            if run_on and fields:
                fields.append(run_on.group(1))
                fields.append(run_on.group(2).strip())
            else:
                fields.append(field)

        # Pair them up: document, locator, document, locator, ...
        i = 0
        while i + 1 < len(fields):
            doc, loc = fields[i], fields[i + 1]
            if doc and loc:
                out.append((doc, loc))
            i += 2
    return out


if __name__ == "__main__":
    from retrieve import Retriever

    r, g = Retriever(), Generator()
    q = "What is the word limit for the individual project, and how is it weighted?"
    res = r.retrieve(q)
    print(f"top score {res.top_score:.3f} (threshold {res.threshold})\n")
    if res.is_confident:
        out = g.answer(q, res.passages)
        print(out.text)
        print(f"\nlatency {out.latency_seconds:.2f}s  "
              f"tokens {out.prompt_tokens}+{out.completion_tokens}")
        print("parsed citations:", parse_citations(out.text))
    else:
        print("refused")
