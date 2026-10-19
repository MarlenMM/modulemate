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
from modulemate.evidence import (
    NOTATION_CONSTRAINT,
    bind_evidence,
    citation_coverage_hints,
    preparation_omissions,
    preparation_steps,
    source_excerpts,
    source_information_limits,
    source_negative_statements,
    source_notation_limits,
    writer_text,
)
from modulemate.generation import GenerationError
from modulemate.prompts import AUDIT_SYSTEM, METADATA_SYSTEM, SYSTEM, TEXT_SYSTEM
from modulemate.requests import (
    conceptual_reframing,
    material_overview,
    preparatory_method,
    synthesis_request,
)
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
    citation_review: list[dict] = field(default_factory=list)


class AnswerValidationError(ValueError):
    def __init__(self, stage):
        self.stage = stage
        super().__init__("The model answer did not pass the bounded review")


def unsupported_attributed_quotes(text, passages):
    """Check claimed verbatim source words, not themes or ordinary paraphrases.

    Deliberately narrow attribution cues avoid treating a metaphor's scare quotes,
    an invented example, code, or a denied quotation as a source quotation.
    """
    sources = [" ".join(p.text.split()) for p in passages]
    unsupported = []
    for segment in re.split(r"(```[\s\S]*?```|`[^`]*`)", text)[::2]:
        for match in re.finditer(r'["“]([^"“”\n]{1,1200})["”]', segment):
            boundary = segment.rfind("\n\n", 0, match.start())
            start = boundary + 2 if boundary >= 0 else 0
            paragraph = segment[start : match.start()]
            prefix = paragraph[-200:]
            if re.match(r"\s*(?:\*\*)?(?:example|analogy)\b", paragraph, re.I):
                continue
            attributed = re.search(
                r"\b(?:text|sources?|materials?|notes?|document|passage|presentation)\b"
                r"[^.!?\n]{0,70}\b(?:states?|says?|reads?|calls?|names?)\b[^.!?\n]{0,70}$"
                r"|\baccording to\b[^.!?\n]{0,180}$",
                prefix,
                re.I,
            )
            if not attributed or re.search(
                r"\b(?:not|never|doesn't|don't|didn't)\b", attributed[0], re.I
            ):
                continue
            # English quotation style may put a sentence's comma/full stop
            # inside the closing mark. That is not changed source wording.
            quote = " ".join(match[1].split()).rstrip(",.;:!?")
            if quote and not any(quote in source for source in sources):
                unsupported.append(match[1])
    return list(dict.fromkeys(unsupported))


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


def request_history(history, plan):
    """Resolve follow-ups without teaching a fresh answer to copy older prose."""
    if plan is None:
        return bounded_history(history)
    if not (plan.used_context or plan.review_attempt):
        return []
    for index in range(len(history) - 1, -1, -1):
        metadata = history[index].get("metadata", {})
        if metadata.get("used_context") is False and metadata.get("kind") in {
            "generated",
            "insufficient",
        }:
            history = history[index:]
            break
    return bounded_history(history)


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
        r"\b(?:bibliograph\w*|references|quote|verbatim|exact sentence)\b|\b(?:list|show|give|include|cite|provide|add)\b.{0,40}\b(?:evidence|sources?|citations?)\b",
        question,
        re.I,
    ):
        return text
    if not re.search(r"\b(?:json|code|schema|program|syntax)\b", question, re.I):
        # The writer sometimes copies a source-binding packet into a fenced
        # block. Remove only a recognizable auxiliary packet, preserving real
        # code/arrays and any prose that follows it.
        def omit_attachment(match):
            try:
                packet = json.loads(match[2])
            except (ValueError, TypeError):
                return match[0]
            rows = packet if isinstance(packet, list) else [packet]
            fields = {"source_id", "evidence_id", "text", "quote", "document", "location"}
            if rows and all(
                isinstance(row, dict)
                and set(row).issubset(fields)
                and bool({"source_id", "evidence_id"} & set(row))
                and bool({"text", "quote"} & set(row))
                for row in rows
            ):
                return ""
            return match[0]

        text = re.sub(
            r"(?m)^[ \t]{0,3}(`{3,}|~{3,})(?:json)?[ \t]*\n([\s\S]*?)\n\1[ \t]*(?=\n|$)",
            omit_attachment,
            text,
        ).rstrip()
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
            heading = re.sub(r"^[-*]\s+", "", heading).strip()
            if (
                heading == line.strip()
                and re.match(r"sources?\s*:\s*\S", heading, re.I)
                and not re.search(r'["“]|\.(?:pdf|md|txt|pptx|docx)\b', heading, re.I)
            ):
                # "Sources: light and stored reserves" is ordinary content,
                # not a bibliography. Do not erase a real answer's labeled data.
                offset += len(line)
                continue
            if (
                offset
                and text[:offset].strip()
                and (
                    re.match(
                        r"(?:sources?|references|source text|source evidence|source support|supporting (?:evidence|sources?|material))\s*(?::|$)",
                        heading,
                        re.I,
                    )
                    or re.fullmatch(r"evidence\s*:?", heading, re.I)
                    or re.fullmatch(r"<evidence\s*>", heading, re.I)
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
    prose = re.sub(r"```[\s\S]*?```|`[^`]*`", "", text)
    # Fenced/inline code is rendered literally. URLs are also ordinary source
    # facts; displaying one never authorizes fetching it. Active image/media
    # syntax and reasoning envelopes remain disallowed outside literal code.
    if re.search(
        r"!\[|</?(?:think|thinking|analysis)\b[^>]*>|<\s*(?:img|iframe|script|audio|video|source)\b",
        prose,
        re.I,
    ):
        raise ValueError("unsafe_markup: put literal image or HTML syntax inside code")
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
        source_body = " ".join(writer_text(passages[sid - 1]).split())
        if (
            not isinstance(quote, str)
            or not min(12, len(source_body)) <= len(quote) <= 2000
            or not quote.strip()
            or (len(source_body) < 12 and " ".join(quote.split()) != source_body)
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
    question, _ = focus_request(question)
    if (
        synthesis_request(question)
        and not requires_materials(question)
        and not source_fact_request(question)
    ):
        return False
    return institutional_permission(question) or bool(
        re.search(
            r"\b(?:deadline|due|submission|word count|how many words|report\s+[a-z]\d|extension|exam|grade|assessment|allowed|permission|approv\w*|excus\w*|according to|exact sentence|trace|code|function)\b|^\s*(?:who|when|where|why (?:did|was|were))\b",
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
    # A method question such as 'How can I ...?' is not a request for
    # authorization. Explicit policy language still needs source-only facts.
    if re.search(r"\b(?:allowed|permission|approv\w*|excus\w*)\b", question, re.I):
        return True
    if conceptual_reframing(question) or preparatory_method(question):
        return False
    for modal in re.finditer(r"\b(?:can|may|should)\s+(?:I|we)\b", question, re.I):
        if not re.search(
            r"\b(?:how|what|where|when|why|which)(?:\s+[A-Za-z'-]+){0,3}\s*$",
            question[: modal.start()],
            re.I,
        ):
            return True
    return False


def narrow_fact_request(question):
    return bool(
        re.search(
            r"\b(?:only|just)\s+(?:the\s+)?(?:current\s+)?"
            r"(?:number|amount|cost|price|time|date|count|deadline|due date|format|weight|word count|exclusions?)\b",
            question,
            re.I,
        )
    )


def source_fact_request(question):
    if (
        synthesis_request(question) or preparatory_method(question)
    ) and not institutional_permission(question):
        return narrow_fact_request(question)
    if re.match(
        r"\s*(?:please\s+)?(?:can|could|would|will|may) you\b", question, re.I
    ) and re.search(
        r"\b(?:summari[sz]e|rewrite|rephrase|proofread|organise|organize|format|extract|list|outline)\b",
        question,
        re.I,
    ):
        return narrow_fact_request(question)
    if re.search(
        r"\b(?:why|explain|example|analogy|interpret|connect|relate|think of|compare|difference|in detail|fully|teach|tutor)\b",
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
    # Inferred fact-finding is a writing hint, not an implicit user word limit.
    # A source may itself discuss an example or calculation. Only reject an
    # explicitly added general section for a narrow fact / permission request;
    # actual length constraints are enforced by check_requested_format().
    if (narrow_fact_request(question) or institutional_permission(question)) and re.search(
        r"(?:^|[.!?]\s+|\n)\s*(?:\*{0,2})(?:general explanation|(?:illustrative |invented )?example)\s*(?:\*{0,2})\s*:",
        text,
        re.I,
    ):
        raise ValueError(
            "course_specific_scope: remove the unrequested general explanation or illustrative example. Preserve the requested source facts and their essential conditions."
        )


def parse_draft(raw, *, preserve_literal=False):
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
    # A model can emit a second, empty type marker after its actual answer.
    # Preserve the leading type; remove only an orphan tail, never quoted tags,
    # code, or a substantive second response.
    if not preserve_literal:
        text = re.sub(r"\n\s*\[(?:ANSWER|MISSING|CLARIFY)\]\s*$", "", text, flags=re.I).strip()
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


def normalize_scalar_label(text, question):
    """Shorten unambiguous scalar presentation without calculating or guessing."""
    if not re.search(
        r"\b(?:give|return|show|tell me)\s+only\s+(?:the\s+)?(?:current\s+)?(?:number|count|quantity)\b",
        question,
        re.I,
    ):
        return text
    match = re.fullmatch(
        r"\s*\[[A-Za-z][A-Za-z '\u2019-]{0,90}:\s*(-?\d+(?:\.\d+)?)\]\s*((?:\[\d+\]\s*)*)", text
    )
    if match:
        return (match[1] + " " + match[2].strip()).strip()
    # Keep conditions, bounds, uncertainty, code and multiple quantities intact.
    # This formats an existing answer; it cannot decide whether its fact is true.
    body = re.sub(r"(?:\s*\[\d+\])+\s*$", "", text).strip().strip("*")
    if len(body.split()) > 28 or re.search(r"[\n`\[\],;:?!]", body):
        return text
    if re.search(
        r"\b(?:not|no|unknown|unstated|missing|cannot|could|may|might|if|unless|when|"
        r"until|but|except|per|each|about|approximately|around|roughly|estimated|"
        r"minimum|maximum|least|most|up to|more|less|fewer|over|under|between|"
        r"times|twice|half|quarter|hundred|thousand|million|billion|trillion)\b",
        body,
        re.I,
    ):
        return text
    units = r"(?:one|two|three|four|five|six|seven|eight|nine)"
    cardinal = (
        rf"(?:zero|{units}|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
        rf"seventeen|eighteen|nineteen|(?:twenty|thirty|forty|fifty|sixty|seventy|"
        rf"eighty|ninety)(?:[- ]{units})?)"
    )
    scalar = rf"(?<![\w.-])(?:-?\d+(?:\.\d+)?|(?:minus |negative )?{cardinal})(?![\w-]|\.\d)"
    quantities = list(re.finditer(scalar, body, re.I))
    if len(quantities) != 1:
        return text
    value = quantities[0]
    prefix, suffix = body[: value.start()], body[value.end() :]
    predicate = re.fullmatch(
        r"(?:There\s+(?:is|are)|The\s+[A-Za-z '\u2019-]{1,100}\s+"
        r"(?:is|are|has|have|offers|provides))\s+(?:a total of\s+)?",
        prefix,
        re.I,
    )
    if not predicate or not re.fullmatch(r"(?:\s+[A-Za-z]+){0,4}\.?", suffix):
        return text
    citations = " ".join(re.findall(r"\[\d+\]", text))
    return (value[0] + " " + citations).strip()


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
        if unrequested_history:
            raise ValueError(
                "requested_scope: give only the requested current facts and qualifiers. Remove the unrequested archived comparison."
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
    if count != target or "\n\n" in plain:
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


def focus_request(question):
    """Separate a declarative prelude from one explicit final English question.

    Both are retained for the writer. Never split multiple questions, quotes or
    code; the original full request remains the source of retrieval and checks.
    """
    if (
        not question.rstrip().endswith("?")
        or question.count("?") != 1
        or re.search(r'["“”`]', question)
    ):
        return question, ""
    boundaries = list(
        re.finditer(
            r"(?<=[.!])\s+(?=(?:what|how|which|can|could|would|will|is|are|do|does|"
            r"did|where|when|why|should|may|must|who|whose)\b)",
            question,
            re.I,
        )
    )
    if not boundaries:
        return question, ""
    boundary = boundaries[-1]
    return question[boundary.end() :].strip(), question[: boundary.start()].strip()


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
        else request_history(history, plan),
        "context_complete": context.complete,
        "total_passages": context.total_passages,
        "document_catalog": context.catalog,
        "source_reading_limits": context.reading_limits,
        "source_information_limits": source_information_limits(context.passages),
        "source_negative_statements": source_negative_statements(context.passages),
        "requested_parts": [p["question"] for p in context.evidence.get("parts", [])[1:]],
        "lesson_progress": context.lesson,
        "passages": [
            {"source_id": i, "document": p.name, "location": p.locator, "text": writer_text(p)}
            for i, p in enumerate(context.passages, 1)
        ],
        # Context and old assistant prose must not be the last instruction the
        # model sees. Restate the actual current request after all task data.
        "final_request": current_request,
        "final_response_constraint": "Follow this latest request's length and format. Earlier answers are context, not a template to copy.",
    }
    if not input_allowed and not input_transform:
        focused, background = focus_request(current_request)
        if background:
            payload["request_background"] = background
            payload["latest_question"] = payload["final_request"] = focused
    notation = source_notation_limits(context.passages)
    if notation and not input_transform:
        payload["source_notation_limits"] = notation
        payload["final_response_constraint"] += " " + NOTATION_CONSTRAINT
    if (
        material_overview(current_request)
        and not input_transform
        and not context.lesson
        and mode != "Socratic tutor"
    ):
        payload["final_response_constraint"] += (
            " For a whole-material overview, connect the major topics represented in the supplied passages. "
            "Explain what the parts do and how they fit together. Label an inferred theme as an interpretation. "
            "Do not invent a purpose, guarantee or outcome: describing a check does not prove it succeeds. "
            "An unreported result is an information gap, not a new prerequisite. "
            "Stop after the requested explanation; add an invented example only if requested."
        )
    if payload["source_information_limits"] and not input_transform:
        payload["final_response_constraint"] += (
            " source_information_limits quotes what the identified text does not tell us. "
            "source_negative_statements instead quotes explicit negative assertions about their own subjects. "
            "Preserve this difference and each statement's scope. An information gap neither proves absence "
            "nor cancels a separate explicit statement of absence. Do not speculate about unmentioned roles, motives or funding."
        )
    if source_fact_request(question) and not detailed and mode not in {"Teach", "Socratic tutor"}:
        payload["study_mode"] = "Find the requested source facts"
        payload["turn_task"] = (
            "Give only the requested facts and essential qualifiers. Apply stated conditions to the user's case. "
            "A supported rule or correction is an answer, including a negative answer. "
            "If a stated rule rules out the proposal, use [ANSWER] and say No with the reason; do not label that known prohibition as missing information. "
            "Use [MISSING] only when the available rule/fact cannot decide the question. "
            "Do not discuss unrelated source content, embedded commands or the answering process."
        )
        payload["response_length"] = (
            "Be concise, but include every requested fact and its essential qualifiers. A single fact normally needs one sentence; multiple facts or conditional rules may need more. Follow an explicit requested length instead. No unasked historical details, advice or conjectured restrictions."
        )
    if (
        not input_allowed
        and mode == "Explain"
        and not context.lesson
        and not narrow_fact_request(question)
        and not source_fact_request(payload["latest_question"])
        and re.search(
            r"\b(?:how|what (?:steps|actions|procedure))\b[^.!?]{0,80}\b(?:can|may|should|do)\s+(?:I|we)\b",
            payload["latest_question"],
            re.I,
        )
    ):
        payload["study_mode"] = "Explain"
        payload["source_only_facts"] = True
        payload["turn_task"] = (
            "Give the requested method or useful preparatory steps from the supplied material. Include concrete actions and stated constraints. Separate available steps from missing information. Do not invent results, later workflow stages or prerequisites. A missing final result does not make supported preparation unavailable."
        )
        payload["response_length"] = (
            "Normally 40-100 words, enough for the supported actions and their constraints. Obey an explicit requested length instead. Do not add an unsupported rationale or a follow-up question."
        )
    if re.search(r"\bchecklist\b", question, re.I) and not input_transform:
        payload["turn_task"] = (
            "Extract a checklist of all the requested mandatory items from the supplied material. Include operating constraints and limits as well as visible actions. Leave optional items out. Return the checklist itself; the app will show the supporting sources separately."
        )
    gap_notes = []
    if (
        not input_transform
        and mode not in {"Teach", "Socratic tutor"}
        and not source_fact_request(question)
        and re.search(
            r"\b(?:prepare|preparatory|draft|outline|begin|start|set up|do now|do beforehand)\b",
            payload["latest_question"],
            re.I,
        )
        and re.search(
            r"\b(?:without|missing|unavailable|not (?:have|provided|supplied|available)|don't have|no\b.{0,35}\byet)\b"
            r"|\bbefore\s+(?:(?:I|we)\s+)?(?:have|receive|get|obtain)\b",
            question,
            re.I,
        )
    ):
        payload["turn_task"] = (
            "Explain what can be prepared BEFORE the unavailable inputs arrive. "
            "Use provided_preparation to track identified setup steps or draft components; also read the action passages for further instructions. "
            "Preserve all relevant source-stated components and their conditions. "
            "Use placeholders for unspecified values. State those open inputs separately after the available actions. "
            "The user is asking for preparation, not a completed final result. Follow explicit source requirements."
        )
        payload["provided_preparation"] = preparation_steps(context.passages)
        if getattr(backend, "supports_plain_text", False) and not re.search(
            r"\b(?:one|1|two|2)[- ]sentences?\b|\b\d+\s+words\b|\b(?:only|just)\b",
            question,
            re.I,
        ):
            gap_notes = preparation_omissions(context.passages, payload["provided_preparation"])
            if gap_notes:
                omitted = {gap["source_id"] for gap in gap_notes}
                payload["passages"] = [
                    p for p in payload["passages"] if p["source_id"] not in omitted
                ]
                payload["source_information_limits"] = [
                    gap
                    for gap in payload["source_information_limits"]
                    if gap["source_id"] not in omitted
                ]
                payload["context_complete"] = False
                payload["turn_task"] = (
                    "Explain the available preparation using provided_preparation and the action passages; the inventory may be incomplete. "
                    "Preserve every relevant prescribed component and its conditions; placeholders are fine for unknown values. "
                    "Return the useful preparation itself. The application separately appends exact source statements about unspecified inputs."
                )
        payload["response_length"] = (
            "Normally 50-110 words, focusing on available preparation and the specific remaining gap."
        )
    if (
        synthesis_request(current_request)
        and not input_transform
        and not context.lesson
        and mode != "Socratic tutor"
        and not narrow_fact_request(question)
    ):
        payload["turn_task"] = (
            "Explain the requested idea, theme, analogy or connection by bringing together the relevant material. "
            "Lead with the central insight, then show how concrete source details support it. "
            "The insight need not be stated verbatim. Mark an interpretation as an interpretation and "
            "a made-up illustration as Example or Analogy. Preserve factual limits; do not invent causes, "
            "private thoughts, measurements or policies. Do not refuse just because the source lacks a ready-made summary."
        )
        payload["response_length"] = (
            "Normally 70-120 words. State the insight and explain it with two or three relevant details, then stop. Obey the requested length instead."
        )
    illustration = bool(
        re.search(
            r"\b(?:give|show|provide|make|create)\b.{0,50}\b(?:analogy|metaphor)\b"
            r"|\b(?:explain|illustrate)\b.{0,80}\b(?:using|with|through) (?:an? )?(?:analogy|metaphor)\b"
            r"|^\s*(?:an? )?(?:(?:everyday|simple|practical) )?(?:analogy|metaphor)(?: please)?[.!? ]*$",
            question,
            re.I,
        )
        and not input_transform
        and not source_fact_request(question)
        and mode not in {"Teach", "Socratic tutor"}
    )
    if illustration:
        payload["turn_task"] = (
            "Give ONE clearly labeled Analogy illustrating the sourced concept. Briefly map its parts to the concept. "
            "It is an invented teaching illustration, not an event or fact from the source. "
            "Do not repeat the whole lesson, add a second example, or append quotations or a source-support section."
        )
        payload["response_length"] = (
            "Normally 40-80 words. Follow an explicit requested length instead."
        )
    if re.match(r"\s*why\b", question, re.I) and not input_transform:
        payload["turn_task"] = (
            "Explain the requested reason or implication using the supplied material. "
            "Check any factual premise first; correct a contradicted event instead of inventing its cause. "
            "For why an idea matters or how a method helps, explain the connection between its source details; "
            "a ready-made because sentence is not required. Distinguish a reasoned interpretation from "
            "a source-stated motivation. Do not invent private motives, causal findings or measured results."
        )
    requested_sentences = re.search(r"\b(one|1|two|2)[- ]sentences?\b", question, re.I)
    if requested_sentences:
        count = 1 if requested_sentences[1].lower() in {"one", "1"} else 2
        payload["response_length"] = (
            f"Exactly {count} concise sentence(s). No introduction, explanation of edits, extra paragraph or follow-up question."
        )
        payload["final_response_constraint"] += " " + payload["response_length"]
    plain_output = bool(getattr(backend, "supports_plain_text", False))
    evidence, bindings = source_excerpts(context.passages)
    text_draft = None
    history_omitted = 0
    missing_locked = False
    quote_missing_locked = False
    failure_stage = "review"
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
                    text_draft, draft_status = parse_draft(raw, preserve_literal=input_transform)
                    if quote_missing_locked:
                        draft_status = "insufficient"
                    text_draft = without_citations(
                        text_draft
                        if input_transform
                        else without_source_footer(text_draft, question)
                    )
                    if (
                        illustration
                        and draft_status == "answered"
                        and not re.match(r"\s*(?:\*{0,2})(?:analogy|example)\b", text_draft, re.I)
                    ):
                        text_draft = "**Analogy:** " + text_draft
                    if not input_transform:
                        text_draft = normalize_scalar_label(text_draft, question)
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
                if (
                    labels.get("citations") == []
                    and draft_status == "answered"
                    and context.passages
                    and not input_allowed
                ):
                    # A labeling failure is not proof that the writer must
                    # refuse. Retry evidence selection with the same prose.
                    raise ValueError(
                        "evidence_selection: select the excerpts underlying the answer's claims or synthesis; "
                        "paraphrases need not match a source sentence verbatim. If unsupported, leave citations empty."
                    )
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
                labeled_text = text_draft
                if draft_status == "answered":
                    for gap in gap_notes:
                        if len(labels["citations"]) < 12:
                            labels["citations"].append(dict(gap))
                            labeled_text += (
                                "\n\nOpen inputs — the material states: “" + gap["quote"] + "”"
                            )
                raw = json.dumps(labels | {"answer": labeled_text}, ensure_ascii=False)
            else:
                raw = backend.complete(SYSTEM, payload)
            answer = parse_response(raw, context.passages, planned_lesson=bool(context.lesson))
            if quote_missing_locked:
                answer.kind = "insufficient"
                answer.basis = "materials" if answer.cited_ids else "none"
            if illustration and answer.kind == "generated" and answer.cited_ids:
                answer.basis = "mixed"
            if not input_transform:
                answer.text = normalize_scalar_label(answer.text, question)
                unbound_quotes = (
                    unsupported_attributed_quotes(answer.text, context.passages)
                    if not input_allowed
                    else []
                )
                if unbound_quotes:
                    if not attempt:
                        quote_missing_locked = answer.kind == "insufficient"
                        payload["unmatched_source_quotations"] = unbound_quotes[:4]
                        payload["quoted_answer_to_correct"] = text_draft or answer.text
                        # Old assistant wording caused the observed false quote.
                        # Correct this draft from the actual passages alone.
                        payload["recent_conversation"] = []
                        payload["turn_task"] = (
                            "Correct quoted_answer_to_correct using the current source passages. "
                            "The phrases in unmatched_source_quotations were not found verbatim. "
                            "Use actual source words or a supported paraphrase without quotation marks. "
                            "Keep the useful answer and its missing-information status when applicable."
                        )
                        payload["final_response_constraint"] += (
                            " Correct the mismatched quotations now; do not copy the rejected attribution."
                        )
                        raise ValueError("quote_text: correct unmatched_source_quotations")
                    # Exact quote wording is inspectable; it is not a new
                    # semantic veto on an otherwise useful explanation or gap.
                    answer.citation_review.extend(
                        {
                            "type": "unmatched_attributed_quote",
                            "quote": quote,
                            "status": "Attributed wording could not be matched verbatim; inspect the original source.",
                        }
                        for quote in unbound_quotes
                    )
            for hint in citation_coverage_hints(
                answer.text, context.passages, answer.cited_ids, bindings
            ):
                p = context.passages[hint["source_id"] - 1]
                answer.citation_review.append(
                    {
                        "document_id": p.document_id,
                        "document": p.name,
                        "location": p.locator,
                        "quote": bindings[hint["evidence_id"]]["quote"],
                        "matched_terms": hint["details"],
                        "status": "Possible omitted evidence; human review required",
                    }
                )
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
                input_transform
                or answer.kind == "clarification"
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
            failure_stage = (
                "evidence"
                if issue.startswith(
                    (
                        "metadata",
                        "citation",
                        "quote",
                        "evidence",
                        "unknown_source",
                        "unknown_evidence",
                        "grounding_required",
                    )
                )
                else "format"
            )
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
                    "quote_text",
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
                    else (
                        "Return evidence/conversation metadata only. Select valid evidence_id values from source_evidence. "
                        "Cite the underlying material for supported paraphrases, synthesis and labeled analogies. "
                        "Do not rewrite the answer or invent evidence. Unsupported claims still cannot gain citations."
                        if plain_output
                        else "Return the required JSON schema with exact source quotes. Missing evidence uses status=insufficient, basis=none, citations=[]. General explanations use basis=general. Materials/mixed require real source IDs and matching quotes. Do not invent evidence or put envelope fields inside answer."
                    )
                ),
            }
    raise AnswerValidationError(failure_stage)
