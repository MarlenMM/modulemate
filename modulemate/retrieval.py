"""Authorized hybrid retrieval with persistent vectors and bounded reranking."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from modulemate.ingest import split_text
from modulemate.local_models import ModelUnavailable, normalized_vectors
from modulemate.search import Passage, _scope, search, tokens


@dataclass
class Retrieval:
    passages: list[Passage] = field(default_factory=list)
    method: str = "lexical"
    notice: str = ""
    scores: dict[int, float] = field(default_factory=dict)
    indexed: int = 0


def scoped_passages(store, course_id, document_ids=None):
    with store.transaction() as (con, user):
        clause, params = _scope(con, store, user, course_id, document_ids)
        rows = con.execute(
            "SELECT c.id,c.document_id,d.name,c.locator,c.text FROM chunks c "
            "JOIN documents d ON d.id=c.document_id WHERE " + clause + " ORDER BY c.id",
            params,
        ).fetchall()
        return [Passage(**dict(row)) for row in rows]


def digest(passage):
    return hashlib.sha256((passage.document_id + "\0" + passage.text).encode()).hexdigest()


def _vectors(store, course_id, passages, engine):
    """Never hold a DB lock during inference; reauthorize writes and identity."""
    by_id = {p.id: p for p in passages}
    cached = {}
    with store.transaction() as (con, user):
        store._course(con, user, course_id)
        rows = con.execute(
            "SELECT e.* FROM embeddings e JOIN chunks c ON c.id=e.chunk_id "
            "JOIN documents d ON d.id=c.document_id WHERE d.course_id=? AND e.model=?",
            (course_id, engine.key),
        ).fetchall()
        for row in rows:
            p = by_id.get(row["chunk_id"])
            if p and row["digest"] == digest(p) and row["dimensions"] == engine.dimensions:
                try:
                    vec = np.frombuffer(row["vector"], dtype="<f4")
                    cached[p.id] = normalized_vectors([vec], 1, engine.dimensions)[0]
                except (ValueError, ModelUnavailable):
                    continue  # Repair a corrupt/incompatible cache from source text.
    missing = [p for p in passages if p.id not in cached]
    for start in range(0, len(missing), 64):
        batch = missing[start : start + 64]
        windows, owners = [], []
        for p in batch:
            pieces = split_text(p.text, size=600, overlap=80)
            windows.extend(pieces)
            owners.extend([p.id] * len(pieces))
        encoded = normalized_vectors(engine.encode(windows), len(windows), engine.dimensions)
        for p in batch:
            values = [encoded[i] for i, owner in enumerate(owners) if owner == p.id]
            cached[p.id] = normalized_vectors([np.mean(values, axis=0)], 1, engine.dimensions)[0]
        with store.transaction() as (con, user):
            store._course(con, user, course_id)
            for p in batch:
                # SQLite can reuse a deleted rowid: identity AND text must match.
                current = con.execute(
                    "SELECT c.text,c.document_id FROM chunks c JOIN documents d ON d.id=c.document_id "
                    "WHERE c.id=? AND d.course_id=?",
                    (p.id, course_id),
                ).fetchone()
                if (
                    current
                    and current["text"] == p.text
                    and current["document_id"] == p.document_id
                ):
                    con.execute(
                        "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?)",
                        (
                            p.id,
                            engine.key,
                            digest(p),
                            engine.dimensions,
                            cached[p.id].astype("<f4").tobytes(),
                        ),
                    )
    return np.asarray([cached[p.id] for p in passages]), len(missing)


def _fresh(store, course_id, passages, document_ids):
    # Revalidate only the selected identities, without rereading a whole course.
    with store.transaction() as (con, user):
        clause, params = _scope(con, store, user, course_id, document_ids)
        if not passages:
            return []
        ids = list(dict.fromkeys(p.id for p in passages))
        rows = con.execute(
            "SELECT c.id,c.document_id,d.name,c.locator,c.text FROM chunks c "
            "JOIN documents d ON d.id=c.document_id WHERE "
            + clause
            + " AND c.id IN ("
            + ",".join("?" for _ in ids)
            + ")",
            [*params, *ids],
        ).fetchall()
        current = {row["id"]: Passage(**dict(row)) for row in rows}
        return [p for p in passages if current.get(p.id) == p]


def _identifiers(query):
    return {
        term
        for term in tokens(query)
        if term.isascii() and any(c.isalpha() for c in term) and any(c.isdigit() for c in term)
    }


def retrieve(
    store,
    course_id,
    query,
    document_ids=None,
    *,
    engine=None,
    limit=6,
    rerank=False,
    strategy="hybrid",
):
    limit = max(1, min(limit, 12))
    lexical = search(store, course_id, query, document_ids, limit=12)
    identifiers = _identifiers(query)
    if identifiers:
        lexical = [p for p in lexical if identifiers & set(tokens(p.text))]
    if engine is None or not tokens(query):
        return Retrieval(lexical[:limit])
    passages = scoped_passages(store, course_id, document_ids)
    if identifiers:
        passages = [p for p in passages if identifiers & set(tokens(p.text))]
    if not passages:
        return Retrieval(method="hybrid")
    try:
        matrix, indexed = _vectors(store, course_id, passages, engine)
        query_vector = normalized_vectors(engine.encode([query[:2000]]), 1, engine.dimensions)[0]
        similarities = matrix @ query_vector
        dense_order = np.argsort(-similarities, kind="stable")[:32]
        by_id = {p.id: p for p in passages}
        dense = [passages[i] for i in dense_order if similarities[i] >= 0.20]
        dense_scores = {p.id: float(similarities[i]) for i, p in enumerate(passages)}
        fusion = {}
        lists = [dense] if strategy == "dense" else [lexical, dense]
        for ranking in lists:
            for rank, p in enumerate(ranking, 1):
                fusion[p.id] = fusion.get(p.id, 0.0) + 1 / (30 + rank)
        ordered = sorted(fusion, key=lambda pid: (-fusion[pid], -dense_scores.get(pid, 0), pid))[
            :32
        ]
        candidates = [by_id[pid] for pid in ordered if pid in by_id]
        method = "dense" if strategy == "dense" else "hybrid"
        scores = dense_scores
        if rerank and candidates:
            relevance = engine.rerank(query[:2000], [p.text for p in candidates])
            if len(relevance) != len(candidates) or not np.isfinite(relevance).all():
                raise ModelUnavailable("Invalid reranker scores.")
            scores = {p.id: float(score) for p, score in zip(candidates, relevance, strict=True)}
            # Cross-encoder logits are uncalibrated relevance signals, not confidence
            # probabilities. A weak logit must not veto a strong embedding or exact
            # match: this web-search model is brittle on academic paraphrases.
            ranking = sorted(candidates, key=lambda p: (-scores[p.id], p.id))
            rank_map = {p.id: i for i, p in enumerate(ranking, 1)}
            query_terms = set(tokens(query))
            candidates = [
                p
                for p in candidates
                if (
                    dense_scores[p.id] >= 0.32
                    or scores[p.id] >= -3.5
                    or len(query_terms & set(tokens(p.text))) / max(1, len(query_terms)) >= 0.6
                )
            ]
            candidates.sort(
                key=lambda p: (-(fusion[p.id] + 0.7 / (30 + rank_map[p.id])), -scores[p.id], p.id)
            )
            method += " + reranker"
        else:
            candidates = [
                p
                for p in candidates
                if dense_scores[p.id] >= 0.32 or (strategy != "dense" and p in lexical)
            ]
        chosen, seen = [], set()
        for p in candidates:
            key = " ".join(p.text.casefold().split())
            if key not in seen:
                chosen.append(p)
                seen.add(key)
            if len(chosen) >= limit:
                break
        return Retrieval(
            _fresh(store, course_id, chosen, document_ids), method, scores=scores, indexed=indexed
        )
    except (ModelUnavailable, ValueError, RuntimeError):
        return Retrieval(
            _fresh(store, course_id, lexical[:limit], document_ids),
            "lexical fallback",
            "Local semantic retrieval was unavailable. Using word matches for this turn.",
        )


def prepare_index(store, course_id, engine):
    """Prepare a course before chatting; the same incremental path serves queries."""
    passages = scoped_passages(store, course_id)
    if not passages:
        return 0
    _, indexed = _vectors(store, course_id, passages, engine)
    return indexed
