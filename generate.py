"""M4 — Grounded generation with citation, and S1 query rewriting (FR6, FR4).

The generator sits behind a thin `LLMBackend` interface (the R3/R4 mitigation
in the locked risk register): retrieval, refusal and the UI never import a
vendor SDK, so the backend can be swapped — Groq, OpenAI, DeepSeek, Anthropic,
or a local open-weights server — by editing config.yaml alone.

Grounding is enforced in three places, not one:
  1. the system prompt confines the model to the supplied passages;
  2. the passages are the only corpus text in the context window;
  3. the refusal gate (retrieve.py) can stop generation before it starts.
"""
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import config
from retrieve import Retrieved

CITATION_RE = re.compile(r"\[Source:\s*([^,\]]+?)\s*,\s*([^\]]+?)\s*\]")

SYSTEM_PROMPT = """\
You are ModuleMate, a course assistant for the HIT Summer Course 2026 module \
"Artificial Intelligence and Machine Learning".

You answer ONLY from the numbered CONTEXT passages supplied with each question. \
These passages are extracted from the module's own lecture slides, teaching \
schedule and assessment brief.

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
passage header you actually used. Do not invent a slide number. If you drew on \
two passages, give both citations on that line separated by a space.
4. Be concise: two to five sentences unless the question needs a list.
5. Write for a student on the module. Do not mention "the context" or "the \
passages" — just answer.
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
    """Grounded answer generation and conversational query rewriting."""

    def __init__(self, cfg=None, backend: LLMBackend | None = None):
        self.cfg = cfg or config.load()
        self.backend = backend or make_backend(self.cfg)
        self.model = self.cfg.require("generation.model")
        self.rewrite_model = self.cfg.get(
            "generation.rewrite_model", self.model
        )
        self.temperature = self.cfg.get("generation.temperature", 0.0)
        self.max_tokens = self.cfg.get("generation.max_tokens", 600)
        self.refusal_message = " ".join(
            self.cfg.require("refusal.message").split()
        )

    def answer(self, question: str, passages: list[Retrieved]) -> Generation:
        """Generate a grounded, cited answer from retrieved passages."""
        import time
        started = time.time()

        if not passages:
            return Generation(self.refusal_message, True, time.time() - started)

        text, p_tok, c_tok = self.backend.complete(
            SYSTEM_PROMPT,
            build_user_prompt(question, passages),
            self.model,
            self.temperature,
            self.max_tokens,
        )

        # The model's own escape hatch — treated as a refusal, not an answer.
        if "INSUFFICIENT_CONTEXT" in text.upper():
            return Generation(
                self.refusal_message, True, time.time() - started, p_tok, c_tok
            )

        return Generation(
            text, False, time.time() - started, p_tok, c_tok
        )

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


def parse_citations(text: str) -> list[tuple[str, str]]:
    """Extract [(document, slide_or_section)] from an answer.

    The eval harness parses this exact bracket to score citation correctness.
    """
    return [(d.strip(), s.strip()) for d, s in CITATION_RE.findall(text)]


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
