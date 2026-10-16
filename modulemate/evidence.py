"""Select source spans by ID instead of asking a model to reproduce quotations."""

from __future__ import annotations

import re


def source_excerpts(passages):
    """Return bounded, contiguous excerpts, with no model-written source text.

    Whitespace normalization is the same as the final quote validator's. Every
    excerpt remains a substring of its own source; omitted sentences are never
    joined. IDs are private to this request and carry no document permissions.
    """
    catalog, bindings = [], {}
    for source_id, passage in enumerate(passages, 1):
        source = " ".join(passage.text.split())
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
