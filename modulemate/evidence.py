"""Select source spans by ID instead of asking a model to reproduce quotations."""

from __future__ import annotations

import re

NOTATION_CONSTRAINT = (
    "Some source equations have layout-sensitive line breaks. Preserve fraction grouping; "
    "do not silently convert stacked terms into a differently grouped inline equation. "
    "If notation is unclear, explain the supported idea and identify the equation as needing "
    "the original layout. For a broad main-concept request, prioritize relationships between "
    "ideas instead of listing incidental equations or numerical examples."
)


def source_notation_limits(passages):
    """Flag possible stacked PDF notation, without reconstructing or rejecting it."""
    limits = []
    seen = set()
    for source_id, passage in enumerate(passages, 1):
        if not passage.name.casefold().endswith(".pdf") or passage.document_id in seen:
            continue
        lines = [line.strip() for line in passage.text.splitlines()]
        page_number = re.search(r"\bpage (\d+)\b", passage.locator, re.I)
        stacked = any(
            2 < len(top) <= 80
            and re.fullmatch(r"[A-Za-z0-9_+*/=().\[\]{}^`×÷− -]+", top)
            and re.search(r"[+*/^`×÷]", top)
            and re.fullmatch(r"\d{1,6}|[A-Za-z](?:\^\d)?", bottom)
            and not (index == len(lines) - 2 and page_number and bottom == page_number[1])
            for index, (top, bottom) in enumerate(zip(lines, lines[1:], strict=False))
        )
        if stacked:
            seen.add(passage.document_id)
            limits.append(
                {
                    "source_id": source_id,
                    "document": passage.name,
                    "location": passage.locator,
                    "notice": "Possible stacked equation: a short mathematical expression is followed by a separate number or symbol. Verify fraction grouping against the original layout; source text is retained unchanged.",
                }
            )
    return limits


def writer_text(passage):
    """Omit only duplicated leading headings already carried by the locator.

    Stored sources stay unchanged. Fenced code and headings not represented in
    the locator remain intact. The body is still a contiguous source substring.
    """
    match = re.match(r"\A(?:[ \t]{0,3}#{1,6}[ \t]+[^\n]*\n)+", passage.text)
    if not match:
        return passage.text
    headings = [line.lstrip("# \t").strip() for line in match[0].splitlines()]
    body = passage.text[match.end() :].strip()
    if body and all(h.casefold() in passage.locator.casefold() for h in headings):
        return body
    return passage.text


def source_excerpts(passages):
    """Return bounded, contiguous excerpts, with no model-written source text.

    Whitespace normalization is the same as the final quote validator's. Every
    excerpt remains a substring of its own source; omitted sentences are never
    joined. IDs are private to this request and carry no document permissions.
    """
    catalog, bindings = [], {}
    for source_id, passage in enumerate(passages, 1):
        source = " ".join(writer_text(passage).split())
        excerpts = []
        # A complete short fact ("pH = 7", a date, a formula) is still evidence.
        # Do not silently erase it because long quotations normally need 12 chars.
        while source:
            if len(source) <= 260:
                stop = len(source)
            else:
                ends = list(re.finditer(r"[.!?。！？](?=\s|$)", source[:261]))
                stop = next((m.end() for m in reversed(ends) if m.end() >= 12), 0)
                stop = stop or source.rfind(" ", 12, 261) or 260
                if stop < 12:
                    stop = 260
            quote = source[:stop].strip()
            # Do not drop a short tail that may contain a qualifier or unit.
            rest = source[stop:].strip()
            if rest and len(rest) < 12 and len(quote) + len(rest) < 280:
                quote += " " + rest
                rest = ""
            evidence_id = len(bindings) + 1
            bindings[evidence_id] = {"source_id": source_id, "quote": quote}
            excerpts.append({"evidence_id": evidence_id, "quote": quote})
            source = rest
        catalog.append(
            {
                "source_id": source_id,
                "document": passage.name,
                "location": passage.locator,
                "excerpts": excerpts,
            }
        )
    return catalog, bindings


def bind_evidence(labels, bindings):
    """Translate selected IDs to validated source bindings, never guess a match."""
    if not isinstance(labels, dict):
        raise ValueError("metadata_object")
    selected = labels.get("citations")
    if not isinstance(selected, list) or len(selected) > 12:
        raise ValueError("citations_shape")
    citations = []
    for entry in selected:
        evidence_id = entry.get("evidence_id") if isinstance(entry, dict) else None
        if type(evidence_id) is not int or evidence_id not in bindings:
            raise ValueError("unknown_evidence_id: select a real ID from source_evidence")
        citations.append(dict(bindings[evidence_id]))
    return labels | {"citations": citations}


def preparation_steps(passages):
    """Preserve explicit source instructions for a preparation request.

    This is a small source inventory, not a plan generator or an availability
    classifier. Conditional wording stays inside the verbatim step.
    """
    steps, seen = [], set()
    for source_id, passage in enumerate(passages, 1):
        prose = re.sub(r"```[\s\S]*?```", "", writer_text(passage))
        for sentence in re.split(r"(?<=[.!?])\s+|\n", prose):
            sentence = sentence.strip().lstrip("-* ")
            if (
                sentence not in seen
                and re.match(
                    r"^(?:start|write|finish|collect|list|import|exclude|plot|keep|fold|join|arrange|"
                    r"prepare|set|record|repeat|label|make|draft|create|add|choose|use|check|measure|"
                    r"mix|shape|bake|read|mark|select|gather|outline|describe|name|title|draw|reserve)\b",
                    sentence,
                    re.I,
                )
                and 12 <= len(sentence) <= 700
            ):
                steps.append({"source_id": source_id, "step": sentence})
                seen.add(sentence)
                if len(steps) == 16:
                    return steps
    return steps


def information_omission(sentence):
    """Recognize an explicit statement about the source's missing information.

    A physical object's lack of a property is different: 'shoes provide no
    protection' must remain an actual constraint, never a removable text gap.
    Unrecognized wording stays with the writer without a semantic annotation.
    """
    source = (
        r"(?:materials?|notes?|text|documents?|passages?|excerpts?|brief|instructions?|"
        r"directions?|notices?|handouts?|presentations?|slides?|sources?|cards?|files?|guides?|reports?)"
    )
    verbs = r"(?:specify|provide|state|give|include|contain|supply|name|identify|mention|list|record|report)"
    return bool(
        re.search(
            rf"\b{source}\s+(?:(?:does|do|did) not|doesn't|don't|didn't)\s+{verbs}\b"
            rf"|\b{source}\s+(?:provides?|contains?|suppl(?:y|ies)|gives?|lists?|names?|specif(?:y|ies)|mentions?) no\b"
            rf"|\bno\b[^.!?\n]{{0,100}}\b(?:provided|supplied|specified|given|stated|named|identified|listed)\b"
            rf"[^.!?\n]{{0,50}}\b(?:in|from|within)\s+(?:(?:the|this|these|available|provided)\s+)?{source}\b",
            sentence,
            re.I,
        )
    )


def source_information_limits(passages):
    """Repeat literal information gaps as writer hints, never answerability gates.

    All original passages remain available. These are scope reminders, not new
    facts, source credibility labels, citations, or proof that no other answer exists.
    """
    limits = []
    for source_id, passage in enumerate(passages, 1):
        prose = re.sub(r"```[\s\S]*?```", "", writer_text(passage))
        for sentence in re.split(r"(?<=[.!?])\s+|\n", prose):
            sentence = sentence.strip()
            if 12 <= len(sentence) <= 700 and information_omission(sentence):
                limits.append({"source_id": source_id, "quote": sentence})
                if len(limits) == 8:
                    return limits
    return limits


def source_negative_statements(passages):
    """Literal direct negatives for contrast with a source-information gap.

    These are source assertions, not certificates that the source is true.
    Unrecognized negatives remain in the original passages without this hint.
    """
    statements = []
    for source_id, passage in enumerate(passages, 1):
        prose = re.sub(r"```[\s\S]*?```", "", writer_text(passage))
        for sentence in re.split(r"(?<=[.!?])\s+|\n", prose):
            sentence = sentence.strip()
            if (
                12 <= len(sentence) <= 700
                and re.search(r"\b(?:has|have|had) no\b", sentence, re.I)
                and not information_omission(sentence)
            ):
                statements.append({"source_id": source_id, "quote": sentence})
                if len(statements) == 8:
                    return statements
    return statements


def preparation_omissions(passages, steps):
    """Select pure omission passages for literal display beside preparation.

    Mixed passages and conditions stay with the writer. These quotes are not
    evidence that another task is forbidden or requires an unspecified value.
    """
    documents = {passages[step["source_id"] - 1].document_id for step in steps}
    result = []
    for source_id, passage in enumerate(passages, 1):
        if passage.document_id not in documents:
            continue
        body = writer_text(passage).strip()
        sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n", body) if s.strip()]
        if (
            12 <= len(body) <= 1200
            and sentences
            and all(information_omission(sentence) for sentence in sentences)
            and "```" not in body
            and not re.search(
                r"https?://|!\[|<|(?<!\w)\[\d+\]"
                r"|\b(?:exceptions?|permissions?|prohibitions?|penalties|requirements?|approval|must|only|unless|before|after|allowed|permitted|forbidden|prohibited)\b",
                body,
                re.I,
            )
        ):
            result.append({"source_id": source_id, "quote": body})
            if len(result) == 2:
                break
    return result


def citation_coverage_hints(answer, passages, cited_ids, bindings):
    """Suggest omitted literal detail evidence; never certify semantic support.

    At least two uncommon answer terms must co-occur in an uncited excerpt.
    These are human-review hints, never automatic citations or model retries.
    Empty evidence/refusals are left alone, and code literals are not analyzed.
    """
    if not cited_ids:
        return []
    from modulemate.learning import _terms

    generic = _terms(
        "text source material document passage answer question request help provided states describes includes information follows gives useful methods example instructions however therefore following missing contain"
    )
    prose = re.sub(r"```[\s\S]*?```|`[^`]*`", "", answer)
    details = {t for t in _terms(prose) - generic if len(t) >= 3 and t.isalpha()}
    covered = set().union(*(_terms(writer_text(passages[sid - 1])) for sid in cited_ids))
    novel = details - covered
    if len(novel) < 2:
        return []
    source_terms = [_terms(writer_text(p)) for p in passages]
    rare = {
        t for t in novel if sum(t in terms for terms in source_terms) <= max(1, len(passages) // 3)
    }
    candidates = []
    seen_sources = set(cited_ids)
    for evidence_id, binding in bindings.items():
        if binding["source_id"] in seen_sources:
            continue
        matches = rare & _terms(binding["quote"])
        if len(matches) >= 2:
            candidates.append(
                {
                    "evidence_id": evidence_id,
                    "source_id": binding["source_id"],
                    "details": sorted(matches)[:6],
                }
            )
            seen_sources.add(binding["source_id"])
            if len(candidates) == 2:
                break
    return candidates
