"""M3 — Top-k dense retrieval and the M5 refusal gate (FR5, FR7).

Retrieval and the confidence decision live together because the refusal gate
is a property of the retrieval result, not of the generator: if the best chunk
the index can offer is not similar enough to the question, no amount of careful
prompting makes an answer trustworthy, and generation is skipped entirely.

k and the similarity threshold are read from config.yaml (NFR5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
import index as index_mod


@dataclass
class Retrieved:
    """One retrieved passage with its cosine similarity to the query."""

    rank: int
    score: float
    chunk_id: str
    source_document: str
    slide_or_section: str
    text: str

    @property
    def citation(self) -> str:
        return f"[Source: {self.source_document}, {self.slide_or_section}]"


@dataclass
class RetrievalResult:
    """Outcome of one retrieval, including the refusal decision."""

    query: str
    passages: list[Retrieved]
    top_score: float
    threshold: float

    @property
    def is_confident(self) -> bool:
        """True when the best passage clears the calibrated threshold."""
        return self.top_score >= self.threshold


class Retriever:
    """Holds the FAISS index and chunk metadata for the process lifetime."""

    def __init__(self, cfg=None):
        self.cfg = cfg or config.load()
        self.index, self.chunks, self.manifest = index_mod.load(self.cfg)
        self.k = self.cfg.require("retrieval.k")
        self.threshold = self.cfg.require("retrieval.similarity_threshold")

    def retrieve(self, query: str, k: int | None = None) -> RetrievalResult:
        k = k or self.k
        vector = index_mod.embed_texts([query], self.cfg)
        scores, ids = self.index.search(vector, min(k, len(self.chunks)))

        passages: list[Retrieved] = []
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
            if idx < 0:  # FAISS pads with -1 when fewer than k results exist
                continue
            meta = self.chunks[int(idx)]
            passages.append(
                Retrieved(
                    rank=rank,
                    score=float(score),
                    chunk_id=meta["chunk_id"],
                    source_document=meta["source_document"],
                    slide_or_section=meta["slide_or_section"],
                    text=meta["text"],
                )
            )

        return RetrievalResult(
            query=query,
            passages=passages,
            top_score=passages[0].score if passages else 0.0,
            threshold=self.threshold,
        )


if __name__ == "__main__":
    r = Retriever()
    print(f"index: {r.manifest.get('n_chunks')} chunks, k={r.k}, "
          f"threshold={r.threshold}\n")
    for q in [
        "What is the word limit for the individual project?",
        "What is the difference between supervised and unsupervised learning?",
        "What is a convolutional neural network?",
        "What is the best pizza topping?",
    ]:
        res = r.retrieve(q)
        flag = "ANSWER" if res.is_confident else "REFUSE"
        print(f"[{flag}] top={res.top_score:.3f}  {q}")
        for p in res.passages[:2]:
            print(f"     {p.score:.3f}  {p.citation}  {p.text[:70]!r}")
        print()
