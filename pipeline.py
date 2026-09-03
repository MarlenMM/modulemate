"""End-to-end turn handling — the join between the router, retriever,
refusal gate and generator.

The UI (app.py) and the evaluation harness (eval/harness.py) both drive this
module, so what is measured in testing is exactly what a student experiences;
there is no separate evaluation-only code path.

Order of operations for one turn:
    1. classify intent                                    (M6/FR3)
    2. out-of-scope/social -> canned refusal, no LLM call  (M5/FR7)
    3. rewrite follow-up into a standalone query           (S1/FR4)
    4. top-k dense retrieval                               (M3/FR5)
    5. below similarity threshold -> refusal, no LLM call  (M5/FR7)
    6. grounded generation with citation                   (M4/FR6)
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import config
from classify import IntentClassifier, IntentPrediction
from generate import Generation, Generator, parse_citations
from retrieve import Retrieved, Retriever


@dataclass
class TurnResult:
    """Everything one turn produced — answer, provenance and diagnostics."""

    question: str
    rewritten_query: str
    intent: IntentPrediction
    answer: str
    is_refusal: bool
    refusal_reason: str | None          # "intent" | "threshold" | "model" | None
    passages: list[Retrieved] = field(default_factory=list)
    top_score: float = 0.0
    threshold: float = 0.0
    citations: list[tuple[str, str]] = field(default_factory=list)
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    followups: list[str] = field(default_factory=list)   # C4
    model_used: str = ""      # which model served this turn (R3 fallback)
    # Set only when the generation backend itself failed and the caller asked
    # to degrade rather than raise. A backend outage is not a refusal and is
    # deliberately kept in its own field so it can never be counted as one.
    error: str | None = None

    @property
    def was_rewritten(self) -> bool:
        return self.rewritten_query.strip() != self.question.strip()


class ModuleMate:
    """The assembled assistant. Construct once, reuse across turns."""

    def __init__(self, cfg=None, generator: Generator | None = None):
        self.cfg = cfg or config.load()
        self.retriever = Retriever(self.cfg)
        self.classifier = IntentClassifier(self.cfg)
        self.generator = generator or Generator(self.cfg)
        self.refusal_message = " ".join(
            self.cfg.require("refusal.message").split()
        )
        # C4 is off during evaluation: it costs an extra call per turn and
        # would inflate the measured latency with work a metric never reads.
        self.suggest_followups = bool(
            self.cfg.get("ui.suggest_followups", False)
        )

    @property
    def manifest(self) -> dict:
        """Index build metadata — surfaced in the UI as the R1 staleness cue."""
        return self.retriever.manifest

    def ask(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
        rewrite: bool = True,
        suggest: bool | None = None,
        on_stage: Callable[[str], None] | None = None,
        degrade_on_error: bool = False,
    ) -> TurnResult:
        """Answer one question.

        `on_stage` is called with the name of each stage as it is entered
        ("routing", "searching", "writing"). The UI uses it to report what the
        pipeline is actually doing during the wait; the evaluation harness
        passes nothing and the calls become no-ops, so neither caller is
        running a different code path from the other.

        `degrade_on_error` decides what happens when the generation backend
        fails outright — a decommissioned model, a network outage, an expired
        key. The UI passes True and gets a TurnResult carrying `error` and the
        passages retrieval already found, so a student still sees the material
        even when no answer can be written. The harness leaves it False so a
        broken backend fails the run loudly instead of quietly scoring zero.
        """
        started = time.time()
        history = history or []
        stage = on_stage or (lambda _name: None)

        stage("routing")
        intent = self.classifier.predict(question)

        # --- Step 2: intent short-circuit, before any network call (FR7) ---
        if self.classifier.should_short_circuit(intent.label, intent.confidence):
            return TurnResult(
                question=question,
                rewritten_query=question,
                intent=intent,
                answer=self.refusal_message,
                is_refusal=True,
                refusal_reason="intent",
                latency_seconds=time.time() - started,
            )

        # --- Step 3: conversational query rewriting (S1/FR4) ---
        stage("searching")
        query = (
            self.generator.rewrite_query(question, history)
            if rewrite and history else question
        )

        # --- Step 4: retrieval (FR5) ---
        result = self.retriever.retrieve(query)

        # --- Step 5: confidence gate, before generation (M5/FR7) ---
        if not result.is_confident:
            return TurnResult(
                question=question,
                rewritten_query=query,
                intent=intent,
                answer=self.refusal_message,
                is_refusal=True,
                refusal_reason="threshold",
                passages=result.passages,
                top_score=result.top_score,
                threshold=result.threshold,
                latency_seconds=time.time() - started,
            )

        # --- Step 6: grounded generation (M4/FR6) ---
        stage("writing")
        try:
            gen: Generation = self.generator.answer(query, result.passages)
        except Exception as exc:
            if not degrade_on_error:
                raise
            return TurnResult(
                question=question,
                rewritten_query=query,
                intent=intent,
                answer="",
                is_refusal=False,
                refusal_reason=None,
                passages=result.passages,
                top_score=result.top_score,
                threshold=result.threshold,
                latency_seconds=time.time() - started,
                error=f"{type(exc).__name__}: {exc}",
            )

        # --- Step 7: suggested follow-ups (C4), never on a refusal ---
        want_suggestions = (
            self.suggest_followups if suggest is None else suggest
        )
        followups: list[str] = []
        if want_suggestions and not gen.is_refusal:
            followups = self.generator.suggest_followups(query, result.passages)

        return TurnResult(
            question=question,
            rewritten_query=query,
            intent=intent,
            answer=gen.text,
            is_refusal=gen.is_refusal,
            refusal_reason="model" if gen.is_refusal else None,
            passages=result.passages,
            top_score=result.top_score,
            threshold=result.threshold,
            citations=parse_citations(gen.text),
            latency_seconds=time.time() - started,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            followups=followups,
            model_used=gen.model_used,
        )
