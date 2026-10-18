"""Select source spans by ID instead of asking a model to reproduce quotations."""

from __future__ import annotations

import re


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
        while len(source) >= 12:
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
