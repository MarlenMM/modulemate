"""Instruction-model answers, explicit evidence boundaries and bounded repair."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import get_close_matches

from modulemate.calculations import (
    numeric_claim_errors,
    verified_calculations,
)
from modulemate.context import estimate_tokens
from modulemate.evidence import bind_evidence, source_excerpts
from modulemate.generation import GenerationError
from modulemate.prompts import AUDIT_SYSTEM, METADATA_SYSTEM, SYSTEM, TEXT_SYSTEM
from modulemate.schemas import ANSWER_SCHEMA, AUDIT_SCHEMA, LESSON_SCHEMA, METADATA_SCHEMA
from modulemate.support import check_claims, get_support_model


@dataclass
class Generated:
    text: str
    kind: str
    basis: str
    cited_ids: list[int] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    audited: bool = False
    review_unavailable: bool = False
    topic: str = ""
    subjects: list[str] = field(default_factory=list)
    intent: str = ""
    history_omitted: int = 0
    support_check: dict = field(default_factory=dict)


def bounded_history(history, budget=2600):
    """Preserve recent meaning and final check questions within a token budget."""

    def excerpt(text, limit):
        cost = estimate_tokens(text)
        if cost <= limit:
            return text
        size = max(1, int(len(text) * max(1, limit - 30) / cost))
        head = max(1, size * 2 // 3)
        return text[:head] + "\n[Earlier text shortened]\n" + text[-max(1, size - head) :]

    result = []
    remaining = budget
    for m in reversed(history[-5:]):
        if remaining < 200:
            break
        question = excerpt(m["question"], min(400, remaining / 4))
        answer = excerpt(m["answer"], min(1300, remaining - estimate_tokens(question) - 80))
        turn = {
            "question": question,
            "answer": answer,
            "topic": m.get("metadata", {}).get("topic", "")[:200],
            "basis": m.get("metadata", {}).get(
                "basis", "historical; verify against current passages"
            ),
        }
        remaining -= estimate_tokens(json.dumps(turn, ensure_ascii=False))
        result.append(turn)
    return list(reversed(result))


def citation_ids(text):
    prose = re.sub(r"```[\s\S]*?```|`[^`]*`", "", text)
    return list(dict.fromkeys(int(n) for n in re.findall(r"(?<![\w])\[(\d+)\]", prose)))


def remap_citations(text, mapping):
    # A Python subscript or literal inside code is not a source reference.
    segments = re.split(r"(```[\s\S]*?```|`[^`]*`)", text)
    for i in range(0, len(segments), 2):
        segments[i] = re.sub(
            r"(?<![\w])\[(\d+)\]", lambda m: f"[{mapping[int(m[1])]}]", segments[i]
        )
    return "".join(segments)


def without_citations(text):
    """The local evidence pass owns numbering; preserve code and array literals."""
    segments = re.split(r"(```[\s\S]*?```|`[^`]*`)", text)
    for i in range(0, len(segments), 2):
        segments[i] = re.sub(r"(?<![\w])\[\d+\]", "", segments[i])
    return "".join(segments).strip()


def without_source_footer(text, question):
    """The source panel owns bibliographies unless the user requested one."""
    if re.search(
        r"\b(?:bibliograph\w*|references|sources?|quote|verbatim|exact sentence)\b|\b(?:list|show|give|include)\b.{0,40}\b(?:evidence)\b",
        question,
        re.I,
    ):
        return text
    offset, fence = 0, None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})(.*)$", line)
        if marker:
            if fence is None:
                fence = (marker[1][0], len(marker[1]))
            elif marker[1][0] == fence[0] and len(marker[1]) >= fence[1] and not marker[2].strip():
                fence = None
        elif fence is None:
            heading = line.strip().lstrip("#").strip().replace("**", "").replace("__", "")
            if (
                offset
                and text[:offset].strip()
                and re.match(
                    r"(?:sources?|references|source text|evidence)\s*(?::|$)", heading, re.I
                )
            ):
                return text[:offset].rstrip()
        offset += len(line)
    return text


def quote_matches(quote, source):
    quote, source = " ".join(quote.split()), " ".join(source.split())
    if quote in source:
        return True
    quote = " ".join(quote.replace("\\n", " ").split())
    if quote in source:
        return True
    # An ellipsis can omit intervening words, never change them or combine
    # unrelated sources. Each substantive fragment must occur in source order.
    fragments = re.split(r"\s*(?:\.{3}|…)\s*", quote)
    if len(fragments) < 2 or any(len(f) < 12 for f in fragments):
        return False
    offset = 0
    for fragment in fragments:
        found = source.find(fragment, offset)
        if found < 0:
            return False
        offset = found + len(fragment)
    return True


def check_visible_text(text):
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 20000:
        raise ValueError("answer_size")
    if re.search(r"https?://|!\[|</?(?:think|thinking|analysis)\b[^>]*>", text, re.I):
        raise ValueError("unsafe_markup")
    prose = re.sub(r"```[\s\S]*?```|`[^`]*`", "", text)
    keys = re.findall(
        r'(?:\\n|(?<!\w))["\']?(status|basis|citations|followups|subjects|intent)["\']?\s*:',
        prose,
        re.I,
    )
    if len(set(keys)) >= 3:
        raise ValueError("envelope_fields_inside_answer")


def parse_response(raw, passages, *, planned_lesson=False):
    if not isinstance(raw, str) or len(raw) > 40000:
        raise ValueError("response_size")
    # Accept the common harmless JSON wrapper; never fish an arbitrary object
    # out of reasoning, prose or a truncated response.
    raw = re.sub(r"^```(?:json)?\s*\n([\s\S]*?)\n```$", r"\1", raw.strip())
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("response_object")
    status, basis = obj.get("status"), obj.get("basis")
    if status not in {"answered", "insufficient", "clarify"} or basis not in {
        "materials",
        "general",
        "mixed",
        "none",
        "input",
    }:
        raise ValueError("response_status")
    text = obj.get("answer")
    check_visible_text(text)
    # Providers sometimes substitute typographic source brackets or omit an
    # inline marker while supplying a valid explicit evidence list. Normalize
    # presentation rather than throwing away an otherwise usable answer.
    text = re.sub(r"[【［]\s*(\d+)\s*[】］]", r"[\1]", text)
    text = re.sub(r"【(\d+)†[^】]*】", r"[\1]", text)
    text = re.sub(r"\[source\s+(\d+)\]", r"[\1]", text, flags=re.I)
    segments = re.split(r"(```[\s\S]*?```|`[^`]*`)", text)
    for i in range(0, len(segments), 2):
        segments[i] = segments[i].replace("\\n\\n", "\n\n")
    text = "".join(segments)
    refs = citation_ids(text)
    citations = obj.get("citations")
    if not isinstance(citations, list) or len(citations) > 12:
        raise ValueError("citations_shape")
    quoted = []
    for citation in citations:
        if not isinstance(citation, dict):
            raise ValueError("citation_shape")
        sid, quote = citation.get("source_id"), citation.get("quote", citation.get("text"))
        if type(sid) is not int or not 1 <= sid <= len(passages):
            raise ValueError("unknown_source")
        if (
            not isinstance(quote, str)
            or not 12 <= len(quote) <= 2000
            or not quote_matches(quote, passages[sid - 1].text)
        ):
            raise ValueError("quote_mismatch")
        quoted.append(sid)
    if not refs and quoted:
        refs = list(dict.fromkeys(quoted))
        text += " " + " ".join(f"[{i}]" for i in refs)
    if set(refs) - set(quoted):
        raise ValueError("citation_mismatch")
    # Valid but unused evidence is harmless. Keep the prose's actual references
    # instead of appending unrelated numbers to the end of the answer.
    if planned_lesson and status == "insufficient" and refs and basis in {"materials", "mixed"}:
        # A supported scheduled lesson part is an answer to that part. The
        # remaining sections are pending, not missing evidence. This does not
        # change refusals for ordinary questions or fabricate absent citations.
        status = "answered"
    if status == "answered" or (
        status in {"clarify", "insufficient"} and (refs or basis != "none")
    ):
        if not passages and not refs and basis in {"materials", "mixed"}:
            # An ordinary explanation without any supplied material is general
            # knowledge, not a reason to invent a missing-information result.
            # Source-specific questions are separately checked by generate().
            basis = "general"
        if basis == "materials" and re.search(
            r"\b(?:example|calculation|general explanation)\b", text, re.I
        ):
            basis = "mixed"
        if (
            basis == "none"
            or (basis in {"materials", "mixed"} and not refs)
            or (basis in {"general", "input"} and refs)
        ):
            raise ValueError("evidence_basis")
    elif basis != "none" or refs:
        raise ValueError("refusal_basis")
    suggestions = obj.get("followups", [])
    # Suggestions are optional UI affordances, not evidence. A fourth or
    # malformed suggestion must not erase a well-supported primary answer.
    suggestions = (
        [s.strip() for s in suggestions if isinstance(s, str) and 1 <= len(s.strip()) <= 200][:3]
        if isinstance(suggestions, list)
        else []
    )
    result = Generated(
        text.strip(),
        {"answered": "generated", "insufficient": "insufficient", "clarify": "clarification"}[
            status
        ],
        basis,
        refs,
        suggestions,
    )
    # Auxiliary state is useful, but malformed/missing labels must never erase
    # an otherwise valid answer. It cannot change document permissions.
    topic, subjects, intent = obj.get("topic"), obj.get("subjects"), obj.get("intent")
    if isinstance(topic, str) and len(topic) <= 200:
        result.topic = topic.strip()
    if (
        isinstance(subjects, list)
        and len(subjects) <= 3
        and all(isinstance(s, str) and 1 <= len(s) <= 100 for s in subjects)
    ):
        result.subjects = subjects
    if isinstance(intent, str) and intent in {"conceptual", "admin", "code_help", "social"}:
        result.intent = intent
    return result


def _audit(backend, question, answer, context, history):
    # Exact short source statements need no probabilistic veto. This prevents
    # a model critic rejecting a correct definition for containing a citation.
    plain = re.sub(r"\[\d+\]|[*_`#]", "", answer.text)
    plain = " ".join(plain.split())
    if (
        answer.basis == "materials"
        and len(plain) <= 500
        and any(plain in " ".join(p.text.split()) for p in context.passages)
    ):
        return True, []
    result = json.loads(
        backend.complete(
            AUDIT_SYSTEM,
            {
                "latest_question": question,
                "draft": answer.text,
                "basis": answer.basis,
                "lesson_progress": context.lesson,
                "context_complete": context.complete,
                "recent_conversation": bounded_history(history[-2:], budget=800),
                "verified_calculations": verified_calculations(question),
                "passages": [
                    {"source_id": i, "text": p.text}
                    for i, p in enumerate(context.passages, 1)
                    if not answer.cited_ids or i in answer.cited_ids
                ],
                "_max_tokens": 320,
                "_response_schema": AUDIT_SCHEMA,
            },
        )
    )
    if not isinstance(result, dict) or type(result.get("approved")) is not bool:
        raise ValueError("audit_shape")
    issues = result.get("issues")
    if (
        not isinstance(issues, list)
        or len(issues) > 3
        or any(
            not isinstance(s, dict)
            or not isinstance(s.get("quote"), str)
            or not 12 <= len(s["quote"]) <= 600
            or s["quote"] not in answer.text
            or any(
                not isinstance(s.get(k), str) or len(s[k]) > 600 for k in ("problem", "correction")
            )
            for s in issues
        )
    ):
        raise ValueError("audit_issues")
    if result["approved"]:
        if issues:
            raise ValueError("contradictory_audit")
        return True, []
    if not issues:
        raise ValueError("unsubstantiated_audit")
    for issue in issues:
        sid, quote = issue.get("source_id"), issue.get("source_quote")
        if sid == 0 and type(sid) is int and quote == "" and verified_calculations(question):
            continue
        if (
            type(sid) is not int
            or not 1 <= sid <= len(context.passages)
            or (answer.cited_ids and sid not in answer.cited_ids)
            or not isinstance(quote, str)
            or not 12 <= len(quote) <= 500
            or not quote_matches(quote, context.passages[sid - 1].text)
        ):
            raise ValueError("unsubstantiated_audit")
    return False, issues


def needs_review(question, mode):
    # Challenge runs repeatedly showed a general-purpose critic inventing
    # objections to correct lessons and definitions. Target review at precise
    # course facts and code requests; numeric requests use actual numeric tools.
    return bool(
        re.search(
            r"\b(?:deadline|due|submission|word count|how many words|report\s+[a-z]\d|extension|exam|grade|assessment|allowed|permission|approv\w*|excus\w*|according to|exact sentence|trace|code|function)\b|\b(?:can|may|should)\s+(?:I|we)\b|^\s*(?:who|when|where|why (?:did|was|were))\b",
            question,
            re.I,
        )
    )


def requires_materials(question):
    explicit = bool(
        re.search(
            r"\b(?:my|our|the|this) (?:exam|assignment|deadline|report|submission|lecturer)|\breport\s+[a-z]\d|\baccording to (?:my|these|this|the selected|the provided)|\bexact (?:sentence|line|quote)",
            question,
            re.I,
        )
    )
    identifier_fact = bool(
        re.search(r"\b[A-Z]{1,6}-?\d{1,4}\b", question, re.I)
        and re.search(
            r"\b(?:due|deadline|word count|exclud\w*|worth|submit|format|current)\b", question, re.I
        )
    )
    return explicit or identifier_fact or institutional_permission(question)


def institutional_permission(question):
    # Request shape, independent of the subject of a document.
    return bool(
        re.search(
            r"\b(?:can|may|should)\s+(?:I|we)\b|\b(?:allowed|permission|approv\w*|excus\w*)\b",
            question,
            re.I,
        )
    )


def narrow_fact_request(question):
    return bool(
        re.search(r"\b(?:only|just)\b", question, re.I)
        and re.search(
            r"\b(?:current|number|amount|cost|price|time|date|count|deadline|due|format|weight|exclud\w*)\b",
            question,
            re.I,
        )
    )


def source_fact_request(question):
    if re.match(
        r"\s*(?:please\s+)?(?:can|could|would|will|may) you\b", question, re.I
    ) and re.search(
        r"\b(?:summari[sz]e|rewrite|rephrase|proofread|organise|organize|format|extract|list|outline)\b",
        question,
        re.I,
    ):
        return narrow_fact_request(question)
    if re.search(
        r"\b(?:why|explain|example|compare|difference|in detail|fully|teach|tutor)\b",
        question,
        re.I,
    ):
        return narrow_fact_request(question)
    return (
        institutional_permission(question)
        or narrow_fact_request(question)
        or bool(
            re.match(
                r"\s*(?:please\s+)?(?:who|when|where|how many|how much|what date|is|are|do|does|did|can|may|will)\b",
                question,
                re.I,
            )
        )
    )


def check_fact_scope(text, question):
    detailed = bool(
        re.search(
            r"\b(?:all|fully|teach|tutor|in detail|detailed|thorough)\b|\b\d{2,3}\s+words\b",
            question,
            re.I,
        )
    )
    if source_fact_request(question) and (
        len(text.split()) > (300 if detailed else 50)
        or re.search(r"\b(?:general explanation|example|calculation)\b", text, re.I)
    ):
        raise ValueError(
            "course_specific_scope: answer only the requested source facts, in one or two short sentences. Remove outside explanations, examples, advice and conjectured restrictions."
        )


def parse_draft(raw):
    if not isinstance(raw, str):
        raise ValueError("writer_kind: expected a response type and Markdown")
    match = re.match(r"\s*\[\s*(ANSWER|MISSING|CLARIFY)\s*\]\s*", raw, re.I)
    kind = match[1].upper() if match else "ANSWER"
    if not match:
        tag = re.match(r"\s*\[([A-Z]{5,12})\]\s*", raw)
        close = (
            get_close_matches(tag[1], ["ANSWER", "MISSING", "CLARIFY"], n=1, cutoff=0.8)
            if tag
            else []
        )
        if close:
            match, kind = tag, close[0]
    # A short-format request can omit the marker. Do not discard valid prose
    # for a presentation detail; source requirements still apply below.
    text = (raw[match.end() :] if match else raw).strip()
    # Some instruct checkpoints emit a dangling closing tag even with thinking
    # disabled. Remove only that empty tail; opening reasoning blocks still fail.
    text = re.sub(r"\s*</(?:think|thinking|analysis)\s*>\s*$", "", text, flags=re.I)
    check_visible_text(text)
    return text, {"ANSWER": "answered", "MISSING": "insufficient", "CLARIFY": "clarify"}[kind]


def supplied_text(question):
    """Recognize explicit transformations of text/code actually in the request."""
    transform = bool(
        re.search(
            r"\b(?:rewrite|rephrase|proofread|summari[sz]e|format|shorten|organise|organize|extract)\b",
            question,
            re.I,
        )
    )
    content = bool(
        re.search(
            r"```[\s\S]{4,}```|[\"“][^\"”]{4,}[\"”]|(?<!\w)'[^'\n]{4,}'|:\s*\n?[\s\S]{20,}",
            question,
        )
    )
    explanation = bool(re.search(r"\b(?:explain|trace|debug)\b", question, re.I))
    substantial = bool(
        re.search(r"```[\s\S]{4,}```|:\s*\n?[\s\S]{60,}|[\"“][^\"”]{60,}[\"”]", question)
    )
    return (transform and content) or (explanation and substantial)


def check_requested_format(text, question, mode=None):
    """Enforce explicit short formats, not a guess at semantic correctness."""
    one = bool(re.search(r"\b(?:one|1)[- ]sentence\b", question, re.I))
    two = bool(re.search(r"\b(?:two|2)[- ]sentences?\b", question, re.I))
    plain = re.sub(r"\[\d+\]|[*_`#]", "", text).strip()
    plain = re.sub(r"\b(?:e\.g\.|i\.e\.|Dr\.|Mr\.|Mrs\.)", "abbreviation", plain)
    if narrow_fact_request(question):
        history_requested = re.search(
            r"\b(?:previous|archived|old|earlier|history|historical)\b", question, re.I
        )
        unrequested_history = (
            re.search(r"\bcurrent\b", question, re.I)
            and not history_requested
            and re.search(
                r"\b(?:previous|archived|formerly|earlier|superseded|old version)\b", plain, re.I
            )
        )
        if len(plain.split()) > 50 or "?" in plain or unrequested_history:
            raise ValueError(
                "requested_scope: give only the requested facts and qualifiers, in one short sentence if possible. No unrequested archived comparison, extra exclusions or follow-up question."
            )
    if mode == "Socratic tutor" and not re.search(r"\bhint\b", question, re.I):
        if len(re.findall(r"[?？]", plain)) != 1 or len(plain.split()) > 80:
            raise ValueError(
                "quiz_format: ask one focused question and stop; do not explain its answer"
            )
    if not (one or two):
        return
    endings = re.findall(r"[.!?]+(?=\s|$)|[。！？]", plain)
    count = len(endings) or 1
    target = 1 if one else 2
    if count != target or len(plain.split()) > 45 * target or "\n\n" in plain:
        raise ValueError(
            f"requested_format: write exactly {target} concise sentence(s), with no extra paragraph, example or note"
        )


def transformation_request(question, plan=None, history=()):
    input_allowed = supplied_text(question) or bool(
        plan
        and plan.used_context
        and history
        and history[-1].get("metadata", {}).get("basis") == "input"
    )
    input_transform = input_allowed and bool(
        re.search(
            r"\b(?:rewrite|rephrase|proofread|summari[sz]e|format|shorten|shorter|organise|organize|extract)\b",
            question,
            re.I,
        )
    )
    return input_allowed, input_transform


def generate(backend, question, mode, context, history, on_progress=None, *, plan=None):
    current_request = plan.resolved_question if plan and plan.resolved_question else question
    input_allowed, input_transform = transformation_request(question, plan, history)
    if not context.passages and not input_allowed:
        text = (
            "I couldn't find information answering that in the selected materials. Select the relevant document or add the missing text, and I can help work through it."
            if context.total_passages
            else "I don't have any materials to answer from yet. Add a document or paste the text you want me to work with."
        )
        return Generated(text, "insufficient", "none"), False
    detailed = mode == "Teach" or bool(
        re.search(r"\b(?:fully|in detail|step by step|thorough|detailed)\b", question, re.I)
    )
    payload = {
        "latest_question": current_request,
        "original_question": question,
        "study_mode": "Transform the supplied text" if input_transform else mode,
        "source_only_facts": source_fact_request(question),
        "user_supplied_text": input_allowed,
        "turn_task": (
            "Return ONLY the transformed text requested by the user. No introduction, commentary, explanation of edits, or follow-up question. Obey the requested length and format."
            if input_transform
            else "Evaluate the learner's answer to the preceding quiz. Briefly say what is right or wrong, explain any correction, then ask one useful next question without answering it."
            if plan and plan.review_attempt
            else (
                "Ask ONE focused quiz question and STOP. Do not include its answer, a definition, an explanatory paragraph, or a hint that reveals the answer. Wait for the learner."
                if mode == "Socratic tutor" and not re.search(r"\bhint\b", question, re.I)
                else (
                    "Teach the CURRENT scheduled lesson part. Remaining topics are scheduled for later, not missing evidence. Do not add generic source-availability disclaimers. Explain the supplied topics and end with one check question."
                    if context.lesson
                    else "Answer the latest request directly in the selected study mode."
                )
            )
        ),
        "resolved_topic": plan.topic if plan else "",
        "compared_subjects_in_order": plan.subjects if plan else [],
        "verified_calculations": verified_calculations(question),
        "response_length": (
            "Teach every current target in 200-350 words: at most one roadmap sentence, then explanations and examples, then one check question"
            if context.lesson and not context.lesson.get("complete")
            else "A structured lesson, up to 450 words"
        )
        if mode == "Teach"
        else (
            "One quiz question only, normally 20-50 words."
            if mode == "Socratic tutor"
            else (
                "Enough detail to answer every requested part, up to 300 words"
                if detailed
                else (
                    "Give only the requested fact(s) and essential qualifiers. If missing, one brief sentence naming the specific gap; no unrelated topics or outside substitute definition."
                    if requires_materials(question) or narrow_fact_request(question)
                    else "Concise: normally 30-70 words; specific facts can be much shorter. Obey an explicit user length instead."
                )
            )
        ),
        "_max_tokens": 1700 if mode == "Teach" else (1500 if detailed else 1100),
        "_response_schema": LESSON_SCHEMA if context.lesson else ANSWER_SCHEMA,
        "answer_language": "English",
        "recent_conversation": []
        if input_transform and supplied_text(question)
        else bounded_history(history),
        "context_complete": context.complete,
        "total_passages": context.total_passages,
        "document_catalog": context.catalog,
        "requested_parts": [p["question"] for p in context.evidence.get("parts", [])[1:]],
        "lesson_progress": context.lesson,
        "passages": [
            {"source_id": i, "document": p.name, "location": p.locator, "text": p.text}
            for i, p in enumerate(context.passages, 1)
        ],
        # Context and old assistant prose must not be the last instruction the
        # model sees. Restate the actual current request after all task data.
        "final_request": current_request,
        "final_response_constraint": "Follow this latest request's length and format. Earlier answers are context, not a template to copy.",
    }
    if source_fact_request(question) and not detailed and mode not in {"Teach", "Socratic tutor"}:
        payload["study_mode"] = "Find the requested source facts"
        payload["turn_task"] = (
            "Give only the requested facts and essential qualifiers. Apply stated conditions to the user's case. A supported rule or correction is an answer. If the fact is absent, say exactly what is missing. Do not discuss unrelated source content, embedded commands or the answering process."
        )
        payload["response_length"] = (
            "One concise answer sentence, or two if needed for multiple requested facts; at most 45 words. No outside explanation, examples, unasked historical details, advice or conjectured restrictions."
        )
    if re.search(r"\bchecklist\b", question, re.I) and not input_transform:
        payload["turn_task"] = (
            "Extract a checklist of all the requested mandatory items from the supplied material. Include operating constraints and limits as well as visible actions. Leave optional items out. Return the checklist itself; the app will show the supporting sources separately."
        )
    if re.match(r"\s*why\b", question, re.I) and not input_transform:
        payload["turn_task"] = (
            "Check the premise against the supplied text BEFORE explaining a cause. If the described event did not happen, correct the premise and state what actually happened instead. Do not invent a reason for a nonexistent event. For an actual event, use only supported reasons and say when its motivation is unstated."
        )
    requested_sentences = re.search(r"\b(one|1|two|2)[- ]sentences?\b", question, re.I)
    if requested_sentences:
        count = 1 if requested_sentences[1].lower() in {"one", "1"} else 2
        payload["response_length"] = (
            f"Exactly {count} concise sentence(s). No introduction, explanation of edits, extra paragraph or follow-up question."
        )
        payload["final_response_constraint"] = payload["response_length"]
    plain_output = bool(getattr(backend, "supports_plain_text", False))
    evidence, bindings = source_excerpts(context.passages)
    text_draft = None
    history_omitted = 0
    missing_locked = False
    for attempt in range(2):
        raw = ""
        try:
            if on_progress:
                on_progress(
                    "Refining the answer…" if attempt else "Writing an answer to your question…"
                )
            if plain_output:
                if text_draft is None:
                    raw = backend.complete(TEXT_SYSTEM, payload | {"_plain_text": True})
                    text_draft, draft_status = parse_draft(raw)
                    text_draft = without_citations(without_source_footer(text_draft, question))
                    history_omitted = getattr(backend, "last_stats", {}).get(
                        "history_turns_omitted", 0
                    )
                    check_visible_text(text_draft)
                    check_requested_format(text_draft, question, mode)
                    if draft_status == "answered":
                        check_fact_scope(text_draft, question)
                    errors = numeric_claim_errors(text_draft)
                    if errors:
                        payload["verified_draft_corrections"] = errors
                        raise ValueError(
                            "numeric_claim: correct the statistic using verified_draft_corrections"
                        )
                    if set(citation_ids(text_draft)) - set(range(1, len(context.passages) + 1)):
                        raise ValueError("unknown_inline_source")
                if on_progress:
                    on_progress("Attaching source evidence to the answer…")
                raw = backend.complete(
                    METADATA_SYSTEM,
                    {
                        "latest_question": current_request,
                        "answer_to_label": text_draft,
                        "response_kind": draft_status,
                        "source_only_facts": payload["source_only_facts"],
                        "scheduled_lesson": bool(context.lesson),
                        "source_evidence": evidence,
                        "repair": payload.get("repair"),
                        "_response_schema": METADATA_SCHEMA,
                        "_max_tokens": 1100,
                    },
                )
                labels = json.loads(raw)
                if not isinstance(labels, dict):
                    raise ValueError("metadata_object")
                labels["status"] = draft_status
                if draft_status == "insufficient":
                    labels["basis"] = "materials" if labels.get("citations") else "none"
                elif labels.get("citations") == []:
                    labels["basis"] = (
                        "input"
                        if input_allowed
                        else ("none" if draft_status == "clarify" else "general")
                    )
                elif labels.get("basis") == "none":
                    labels["basis"] = "materials"
                labels = bind_evidence(labels, bindings)
                raw = json.dumps(labels | {"answer": text_draft}, ensure_ascii=False)
            else:
                raw = backend.complete(SYSTEM, payload)
            answer = parse_response(raw, context.passages, planned_lesson=bool(context.lesson))
            if missing_locked and answer.kind != "insufficient":
                raise ValueError(
                    "missing_response: preserve the missing-information result; correct incidental claims without inventing the absent answer"
                )
            if input_allowed and answer.basis == "general" and not answer.cited_ids:
                answer.basis = "input"
            answer.history_omitted = history_omitted or getattr(backend, "last_stats", {}).get(
                "history_turns_omitted", 0
            )
            check_requested_format(answer.text, question, mode)
            if answer.kind == "generated":
                check_fact_scope(answer.text, question)
                if source_fact_request(question) and answer.basis not in {"materials", "input"}:
                    raise ValueError(
                        "course_specific_scope: course rules need only real source facts, without added general explanation; if missing, use [MISSING]."
                    )
            errors = numeric_claim_errors(answer.text)
            if errors:
                payload["verified_draft_corrections"] = errors
                raise ValueError(
                    "numeric_claim: correct the statistic using verified_draft_corrections"
                )
            if answer.kind == "generated" and context.lesson:
                missing = [
                    t["topic"]
                    for t in context.lesson.get("targets", [])
                    if not set(t["source_ids"]) & set(answer.cited_ids)
                ]
                if missing:
                    raise ValueError(
                        "lesson_coverage: explain and cite these current topics, not just a roadmap: "
                        + ", ".join(missing)
                    )
            if (
                answer.kind == "generated"
                and not answer.cited_ids
                and not (input_allowed and answer.basis == "input")
            ):
                raise ValueError(
                    "grounding_required: answer only from the supplied material. Cite genuinely supporting passages; if the requested information is absent use [MISSING] and say so. Do not invent an answer or substitute a merely related fact."
                )
            if answer.basis == "input" and not input_allowed:
                raise ValueError(
                    "grounding_required: the learner did not supply text for this answer; use real material or report the gap"
                )
            if (
                answer.kind == "generated"
                and answer.cited_ids
                and not input_transform
                and mode != "Socratic tutor"
            ):
                support_model = get_support_model()
                if support_model:
                    if on_progress:
                        on_progress("Checking statements against their sources…")
                    answer.support_check = check_claims(
                        answer.text, context.passages, answer.cited_ids, model=support_model
                    )
                else:
                    answer.support_check = {"available": False, "claims": [], "flags": []}
                # Independent NLI flags remain visible diagnostics until their
                # repair policy has been validated on real conversations.
            # A critic must not invent a fact to overturn missing evidence.
            # Refusals stay refusals throughout the pipeline.
            if (
                answer.kind == "clarification"
                or (answer.kind == "insufficient" and not answer.cited_ids)
                or not needs_review(question, mode)
            ):
                return answer, attempt > 0
            if on_progress:
                on_progress("Checking relevance, calculations and evidence…")
            try:
                approved, issues = _audit(backend, question, answer, context, history)
            except (GenerationError, ValueError, TypeError, KeyError):
                # Quote/citation checks have passed. A secondary model's outage
                # or malformed verdict must not erase a usable primary answer.
                # The UI records that the additional review was unavailable.
                answer.review_unavailable = True
                return answer, attempt > 0
            if approved:
                answer.audited = True
                return answer, attempt > 0
            payload["repair"] = {
                "issues": issues,
                "draft_to_correct": answer.text,
                "instruction": "Correct these evidenced issues. Preserve correct information and disclose actual gaps. "
                + (
                    "Write only the corrected learner-facing Markdown answer."
                    if plain_output
                    else "Return a fresh complete JSON answer."
                ),
            }
            if answer.kind == "insufficient":
                missing_locked = True
                payload["repair"]["instruction"] += (
                    " Keep response type [MISSING]; fix incidental claims while leaving the requested missing fact unknown."
                )
            text_draft = None
        except (ValueError, TypeError, KeyError) as exc:
            # One bounded regeneration. Outages are NOT retried here, and an
            # invalid answer is never presented as a generated answer.
            issue = str(exc)
            text_error = issue.startswith(
                (
                    "requested_format",
                    "requested_scope",
                    "writer_kind",
                    "quiz_format",
                    "envelope_fields",
                    "unsafe_markup",
                    "answer_size",
                    "unknown_inline_source",
                    "unknown_source",
                    "course_specific",
                    "grounding_required",
                    "lesson_coverage",
                    "numeric_claim",
                    "numeric_example",
                    "missing_response",
                )
            )
            if text_error:
                text_draft = None
            payload["repair"] = {
                "issue": str(exc)[:350],
                "rejected_draft": raw[:20000],
                "instruction": (
                    "Rewrite the learner-facing answer to correct the specified format/content error. No internal fields or JSON inside the answer. Obey the latest requested length and stop when answered."
                    if plain_output and text_error
                    else "Return the required JSON schema with exact source quotes. Missing evidence uses status=insufficient, basis=none, citations=[]. General explanations use basis=general. Materials/mixed require real source IDs and matching quotes. Do not invent evidence or put envelope fields inside answer."
                ),
            }
    raise ValueError("The model answer did not pass the bounded review")
