"""Local lexical retrieval with Unicode support and explicit ownership checks."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

STOPWORDS = frozenset(
    "a an the and or of to for in on is are was were be been what which how why when where who can could would should do does did about please explain tell me my these this that it its from with as at by i you we your summarize summary compare notes material course help learn study give make".split()
)
CJK = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+")


def tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    words = re.findall(r"[^\W_]+", normalized)
    result = []
    for word in words:
        if CJK.search(word):
            for piece in CJK.findall(word):
                result.extend(piece[i : i + 2] for i in range(max(1, len(piece) - 1)))
            latin = CJK.sub(" ", word).split()
            result.extend(latin)
        elif word not in STOPWORDS:
            result.append(word)
    return list(dict.fromkeys(result))


def index_text(text: str) -> str:
    # Preserve word frequencies for BM25; add CJK bigrams for non-space scripts.
    normalized = unicodedata.normalize("NFKC", text).casefold()
    additions = [
        run[i : i + 2] for run in CJK.findall(normalized) for i in range(max(1, len(run) - 1))
    ]
    return normalized + " " + " ".join(additions)


@dataclass(frozen=True)
class Passage:
    id: int
    document_id: str
    name: str
    locator: str
    text: str

    def to_dict(self):
        return asdict(self)


def _scope(con, store, user, course_id, document_ids):
    store._course(con, user, course_id)
    params = [course_id]
    clause = "d.course_id=?"
    if document_ids is not None:
        if not document_ids:
            return "0", []
        document_ids = list(dict.fromkeys(document_ids))[:100]
        for document_id in document_ids:
            doc = store._document(con, user, document_id)
            if doc["course_id"] != course_id:
                from modulemate.db import UserError

                raise UserError("Selected document is not in this course.")
        clause += " AND d.id IN (" + ",".join("?" for _ in document_ids) + ")"
        params.extend(document_ids)
    return clause, params


def search(store, course_id, query, document_ids=None, limit=6) -> list[Passage]:
    terms = tokens(query[:4000])[:24]
    with store.transaction() as (con, user):
        clause, params = _scope(con, store, user, course_id, document_ids)
        if not terms:
            return []
        expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        rows = con.execute(
            "SELECT c.id,c.document_id,d.name,c.locator,c.text,bm25(chunk_search) AS rank "
            "FROM chunk_search JOIN chunks c ON c.id=chunk_search.rowid "
            "JOIN documents d ON d.id=c.document_id WHERE chunk_search MATCH ? AND "
            + clause
            + " ORDER BY rank,c.id LIMIT 100",
            [expression, *params],
        ).fetchall()

        # Exact term coverage first; BM25 breaks ties. No fabricated probability score.
        def rank(row):
            heading = row["locator"] + " " + " ".join(row["text"].splitlines()[:2])
            # A short named concept should lead to its definition, rather than
            # a broad introductory paragraph that merely mentions the word.
            heading_hits = len(set(terms) & set(tokens(heading))) if len(terms) <= 3 else 0
            return (
                -heading_hits,
                -len(set(terms) & set(tokens(row["text"]))),
                row["rank"],
                row["id"],
            )

        rows = sorted(rows, key=rank)
        chosen, seen = [], set()
        for row in rows:
            key = (row["document_id"], row["locator"], row["text"])
            if key not in seen:
                chosen.append(Passage(**{key: row[key] for key in Passage.__dataclass_fields__}))
                seen.add(key)
            if len(chosen) >= max(1, min(limit, 12)):
                break
        return chosen


def overview(store, course_id, document_ids=None, limit=8) -> list[Passage]:
    """Evenly sample locations/documents. This is explicitly a sampled overview."""
    with store.transaction() as (con, user):
        clause, params = _scope(con, store, user, course_id, document_ids)
        rows = con.execute(
            "SELECT c.id,c.document_id,d.name,c.locator,c.text,c.position FROM chunks c JOIN documents d ON d.id=c.document_id WHERE "
            + clause
            + " ORDER BY d.created_at,d.id,c.position",
            params,
        ).fetchall()
        groups = {}
        for row in rows:
            groups.setdefault(row["document_id"], []).append(row)
        chosen = []
        # First/last/middle distributed across documents, without loading embeddings.
        for round_index in range(limit):
            for group in groups.values():
                order = [0, len(group) - 1, len(group) // 2] + list(range(1, len(group)))
                order = list(dict.fromkeys(order))
                if round_index < len(order):
                    row = group[order[round_index]]
                    chosen.append(
                        Passage(**{key: row[key] for key in Passage.__dataclass_fields__})
                    )
                    if len(chosen) == limit:
                        return chosen
        return chosen
