"""Learned evidence coverage and one bounded, scope-preserving recovery search."""

from __future__ import annotations

import re

from modulemate.context import broad_request, build_context
from modulemate.learning import load_evidence_model, reader_signals, score_passages
from modulemate.local_models import ModelUnavailable


def question_parts(question):
    # Split explicit requests, not noun phrases such as 'salt and pepper' or
    # ordinary comparison subjects. The complete request is always retained.
    if "```" in question or '"' in question or "'" in question:
        return [question]
    parts = [
        p.strip(" ?.;")
        for p in re.split(
            r"[?;\n]+|\b(?:and|also|and then)\s+(?=(?:who|whose|where|when|why|how|what|which|give|state|name|describe|explain)\b)",
            question,
            flags=re.I,
        )
        if len(p.strip(" ?.;")) >= 10
    ]
    return list(dict.fromkeys([question, *parts]))[:5] if len(parts) > 1 else [question]


def assess_evidence(question, passages, *, engine=None):
    parts = question_parts(question)
    if not passages:
        return {
            "model": "none",
            "advisory": True,
            "parts": [{"question": q, "estimate": 0.0, "passage_ids": []} for q in parts],
        }
    cosines, encoder = {}, None
    bundle = "\n\n".join(p.text for p in passages)
    use_bundle = 1 < len(passages) and len(bundle) <= 9000
    if engine:
        try:
            texts = list(
                dict.fromkeys(parts + [p.text for p in passages] + ([bundle] if use_bundle else []))
            )
            vectors = dict(zip(texts, engine.encode(texts), strict=True))
            cosines = {
                q: {p.id: float(vectors[q] @ vectors[p.text]) for p in passages} for q in parts
            }
            encoder = engine.key
        except (ModelUnavailable, ValueError, TypeError):
            pass
    result = []
    identity = "unavailable"
    for q in parts:
        readers = None
        bundle_reader = None
        if engine and encoder:
            try:
                signals = reader_signals(
                    engine, q, [p.text for p in passages] + ([bundle] if use_bundle else [])
                )
                bundle_reader = signals.pop(len(passages), None)
                readers = {passages[i].id: value for i, value in signals.items()}
            except (ModelUnavailable, ValueError, TypeError, AttributeError):
                pass
        scores, identity = score_passages(
            q, passages, cosines=cosines.get(q), encoder=encoder, readers=readers
        )
        ranked = (
            sorted(passages, key=lambda p: (-scores[p.id].answerability, p.id)) if scores else []
        )
        best = ranked[:3]
        context_estimate = 0.0
        if use_bundle:
            model = load_evidence_model(encoder, reader=readers is not None)
            if model:
                context_estimate = model.predict(
                    q,
                    bundle,
                    " ".join(dict.fromkeys(p.name for p in passages)),
                    cosine=float(vectors[q] @ vectors[bundle]) if encoder else 0.0,
                    reader=bundle_reader,
                ).answerability
        estimate = scores[best[0].id].answerability if best else 0.0
        result.append(
            {
                "question": q,
                "estimate": round(max(estimate, context_estimate), 4),
                "passage_estimate": round(estimate, 4),
                "context_estimate": round(context_estimate, 4),
                "passage_ids": [p.id for p in best],
                "relevance": round(max((s.relevance for s in scores.values()), default=0), 4),
                "candidate_spans": [
                    readers[p.id]
                    | {"passage_id": p.id, "estimate": round(scores[p.id].answerability, 4)}
                    for p in best
                    if readers and p.id in readers
                ],
            }
        )
    return {"model": identity, "advisory": True, "parts": result}


def adaptive_context(
    store,
    course_id,
    question,
    plan,
    history,
    document_ids=None,
    *,
    engine=None,
    policy="adaptive",
    on_progress=None,
):
    context = build_context(
        store, course_id, question, plan, history, document_ids, engine=engine, policy=policy
    )
    if (
        policy != "adaptive"
        or context.clarification
        or context.lesson
        or broad_request(question, plan.mode)
        or plan.overview
        or (plan.context_ids and context.passages)
    ):
        # Factual answerability labels are not a useful verdict on a main idea
        # or a transformation of already sourced context. Structural coverage
        # supplies that evidence without scoring every page with an extractive QA head.
        return context
    resolved = plan.resolved_question or plan.query or question
    report = assess_evidence(resolved, context.passages, engine=engine)
    context.evidence = report | {"recovery_attempted": False, "recovery_applied": False}
    weak = [p for p in report["parts"] if p["estimate"] < 0.70]
    if not weak or context.complete or report["model"] == "unavailable":
        return context
    if on_progress:
        on_progress("Checking additional passages for the unanswered parts…")
    recovered = build_context(
        store,
        course_id,
        question,
        plan,
        history,
        document_ids,
        engine=engine,
        extra_queries=[p["question"] for p in weak],
        policy=policy,
        recover=True,
    )
    context.evidence["recovery_attempted"] = True
    if recovered.clarification or {p.id for p in recovered.passages} == {
        p.id for p in context.passages
    }:
        return context
    next_report = assess_evidence(resolved, recovered.passages, engine=engine)
    before, after = report["parts"], next_report["parts"]
    # A conservative adoption rule protects another already-covered part. The
    # score only buys a broader look; it cannot turn a guess into a refusal.
    improvements = [b["estimate"] - a["estimate"] for a, b in zip(before, after, strict=True)]
    if improvements and min(improvements) >= -0.08 and max(improvements) >= 0.10:
        recovered.evidence = next_report | {
            "recovery_attempted": True,
            "recovery_applied": True,
            "initial_estimates": [p["estimate"] for p in before],
        }
        return recovered
    # A small head may underrate useful new text. Keep it available for a
    # response-driven second look rather than letting that score hide it forever.
    context.recovery_candidate = recovered
    return context


def recover_missing_context(
    context, store, course_id, question, plan, history, document_ids=None, *, engine=None
):
    """One wider look after an actual missing answer, independent of confidence.

    Reuse an already performed search where possible. A second generation may
    still report a gap; this function neither forces an answer nor adds a citation.
    """
    if context.complete or context.lesson or context.policy != "adaptive":
        return None
    candidate = context.recovery_candidate
    if candidate is None and not context.evidence.get("recovery_attempted"):
        candidate = build_context(
            store,
            course_id,
            question,
            plan,
            history,
            document_ids,
            engine=engine,
            policy="adaptive",
            recover=True,
        )
    if candidate is None or candidate.clarification:
        return None
    before = {p.id for p in context.passages}
    if not ({p.id for p in candidate.passages} - before):
        return None
    candidate.evidence = context.evidence | {
        "response_recovery_attempted": True,
        "response_recovery_applied": False,
    }
    return candidate
