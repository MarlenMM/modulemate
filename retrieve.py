"""M3 — retrieval, and the M5 refusal gate (FR5, FR7), with C1 and C2.

Retrieval and the confidence decision live together because the refusal gate is
a property of the retrieval result, not of the generator: if the best chunk the
index can offer is not similar enough to the question, no amount of careful
prompting makes an answer trustworthy, and generation is skipped entirely.

The pipeline has four stages, each independently switchable in config.yaml so
its contribution can be ablated (see scripts/ablation.py):

  1. Dense retrieval    — FAISS inner product over normalised MiniLM vectors.
  2. Lexical retrieval  — BM25 (C1). Dense embeddings paraphrase well but blur
                          exact tokens, and a course corpus is full of tokens
                          that must match exactly: "ID3", "4000-word",
                          "AI_assignment_ID", "TF-IDF", "300 pull requests".
  3. Rank fusion        — Reciprocal Rank Fusion over the two rankers. RRF
                          combines orderings rather than scores, so BM25's
                          unbounded scores and cosine's [-1,1] never have to be
                          made commensurable.
  4. Cross-encoder      — re-ranks the fused shortlist (C2). Bi-encoders embed
                          query and passage independently; a cross-encoder
                          reads them together and is markedly better at
                          deciding *which* of several passages on the same
                          topic actually answers the question — precisely the
                          locator-level distinction hit@3 measures.

Authority weighting (see `authority_of`) is applied after ranking, and the
refusal gate always reads the dense cosine, so the calibrated threshold keeps
its meaning no matter which stages are enabled.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

import config
import index as index_mod

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_reranker_cache: dict[str, object] = {}


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens for BM25.

    Splitting on non-alphanumerics is deliberate: it turns "AI_assignment_ID"
    into ai/assignment/id and "4000-word" into 4000/word, so a student's
    phrasing can match the document's without either being quoted exactly.
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


@dataclass
class Retrieved:
    """One retrieved passage and every score that produced its rank.

    `score` is what ranking and display use; `dense_score` is the raw cosine
    the refusal gate reads. Keeping both means the gate stays interpretable
    even when a cross-encoder decided the ordering.
    """

    rank: int
    score: float
    chunk_id: str
    source_document: str
    slide_or_section: str
    text: str
    raw_score: float = 0.0        # dense cosine, before authority weighting
    authority: float = 1.0
    dense_score: float = 0.0      # authority-weighted cosine (gate signal)
    dense_rank: int | None = None
    lexical_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None
    rerank_rank: int | None = None

    @property
    def citation(self) -> str:
        return f"[Source: {self.source_document}, {self.slide_or_section}]"

    @property
    def found_by(self) -> str:
        """Which ranker(s) surfaced this passage — shown in the source panel."""
        parts = []
        if self.dense_rank is not None:
            parts.append(f"dense #{self.dense_rank}")
        if self.lexical_rank is not None:
            parts.append(f"BM25 #{self.lexical_rank}")
        if self.rerank_rank is not None:
            parts.append(f"reranker #{self.rerank_rank}")
        return " + ".join(parts) if parts else "—"


@dataclass
class RetrievalResult:
    """Outcome of one retrieval, including the refusal decision."""

    query: str
    passages: list[Retrieved]
    top_score: float
    threshold: float
    stages: dict = field(default_factory=dict)

    @property
    def is_confident(self) -> bool:
        """True when the best passage clears the calibrated threshold."""
        return self.top_score >= self.threshold


class Retriever:
    """Holds the index, BM25 model and re-ranker for the process lifetime."""

    def __init__(self, cfg=None):
        self.cfg = cfg or config.load()
        self.index, self.chunks, self.manifest = index_mod.load(self.cfg)
        self.k = self.cfg.require("retrieval.k")
        self.threshold = self.cfg.require("retrieval.similarity_threshold")
        self.authority_weights = self.cfg.get("retrieval.authority_weights") or {}
        self.default_authority = self.cfg.get("retrieval.default_authority", 1.0)
        self.oversample = self.cfg.get("retrieval.oversample", 4)

        self.hybrid = bool(self.cfg.get("retrieval.hybrid.enabled", False))
        self.rrf_k = self.cfg.get("retrieval.hybrid.rrf_k", 60)
        self.w_dense = self.cfg.get("retrieval.hybrid.weight_dense", 1.0)
        self.w_lexical = self.cfg.get("retrieval.hybrid.weight_lexical", 1.0)

        self.rerank = bool(self.cfg.get("retrieval.rerank.enabled", False))
        self.rerank_model_name = self.cfg.get("retrieval.rerank.model")
        self.rerank_candidates = self.cfg.get("retrieval.rerank.candidates", 24)
        self.w_rerank = self.cfg.get("retrieval.rerank.weight", 1.0)

        self._bm25 = None
        if self.hybrid:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    # ----------------------------------------------------------------------

    @property
    def reranker(self):
        """Cross-encoder, loaded lazily and cached across Retriever instances
        (Streamlit re-executes the script on every interaction)."""
        if not self.rerank:
            return None
        name = self.rerank_model_name
        if name not in _reranker_cache:
            from sentence_transformers import CrossEncoder
            _reranker_cache[name] = CrossEncoder(name, max_length=512)
        return _reranker_cache[name]

    def authority_of(self, source_document: str) -> float:
        """Prior on how much a source deserves to be believed.

        Semantic similarity alone cannot distinguish "the assessment brief
        states the word limit" from "a classmate speculated about the word
        limit" — measured, those two scored 0.552 and 0.567 for the same
        question, so the chat won. Weighting by source class restores the
        ordering a student would apply themselves, and protects the property
        the whole system exists for: never being confidently wrong about a
        deadline.
        """
        for prefix, weight in self.authority_weights.items():
            if source_document.startswith(prefix):
                return float(weight)
        return float(self.default_authority)

    # ----------------------------------------------------------------------

    def _dense(self, vector: np.ndarray, depth: int) -> list[tuple[int, float]]:
        scores, ids = self.index.search(vector, depth)
        return [(int(i), float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]

    def _lexical(self, query: str, depth: int) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        if not len(scores):
            return []
        top = np.argsort(scores)[::-1][:depth]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def _cosine_for(self, idx: int, vector: np.ndarray) -> float:
        """Cosine for a chunk BM25 found but dense retrieval did not return.

        Vectors are stored normalised, so reconstructing one and taking the
        inner product with the (also normalised) query gives cosine directly.
        """
        stored = self.index.reconstruct(int(idx))
        return float(np.dot(stored, vector[0]))

    # ----------------------------------------------------------------------

    def retrieve(self, query: str, k: int | None = None) -> RetrievalResult:
        k = k or self.k
        vector = index_mod.embed_texts([query], self.cfg)
        depth = min(max(k * self.oversample, k), len(self.chunks))

        dense = self._dense(vector, depth)
        lexical = self._lexical(query, depth) if self.hybrid else []

        dense_rank = {idx: r for r, (idx, _) in enumerate(dense, start=1)}
        lexical_rank = {idx: r for r, (idx, _) in enumerate(lexical, start=1)}
        cosine = {idx: s for idx, s in dense}

        # --- Reciprocal Rank Fusion over whichever rankers ran ---
        fused: dict[int, float] = {}
        for idx, r in dense_rank.items():
            fused[idx] = fused.get(idx, 0.0) + self.w_dense / (self.rrf_k + r)
        for idx, r in lexical_rank.items():
            fused[idx] = fused.get(idx, 0.0) + self.w_lexical / (self.rrf_k + r)
        if not fused:
            return RetrievalResult(query, [], 0.0, self.threshold,
                                   {"dense": 0, "lexical": 0})

        candidates: list[Retrieved] = []
        for idx, fused_score in fused.items():
            meta = self.chunks[idx]
            authority = self.authority_of(meta["source_document"])
            cos = cosine.get(idx)
            if cos is None:
                cos = self._cosine_for(idx, vector)
            candidates.append(
                Retrieved(
                    rank=0,
                    score=fused_score * authority,
                    chunk_id=meta["chunk_id"],
                    source_document=meta["source_document"],
                    slide_or_section=meta["slide_or_section"],
                    text=meta["text"],
                    raw_score=cos,
                    authority=authority,
                    dense_score=cos * authority,
                    dense_rank=dense_rank.get(idx),
                    lexical_rank=lexical_rank.get(idx),
                    fused_score=fused_score,
                )
            )

        candidates.sort(key=lambda p: p.score, reverse=True)

        # --- Cross-encoder re-ranking of the fused shortlist (C2) ---
        #
        # The cross-encoder contributes a third *ranking*, fused with the other
        # two, rather than replacing them. Letting it rank alone was measured
        # and rejected: asked "What is the word limit for the individual
        # project report?", it scored the chunk literally reading "a 4000-word
        # written report" worst of all 150 candidates (-3.82) and promoted two
        # chat messages above it, so the answer became a refusal. These
        # ms-marco models are trained on web prose and are unreliable on terse
        # bureaucratic bullets. Fusing ranks means a passage the bi-encoder is
        # confident about survives the cross-encoder being unsure, and vice
        # versa — which is the entire point of rank fusion.
        if self.rerank and self.reranker is not None:
            shortlist = candidates[:self.rerank_candidates]
            pairs = [(query, p.text) for p in shortlist]
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            for p, s in zip(shortlist, scores):
                p.rerank_score = float(s)

            by_rerank = sorted(shortlist, key=lambda p: p.rerank_score,
                               reverse=True)
            for rank, p in enumerate(by_rerank, start=1):
                p.rerank_rank = rank
                p.fused_score += self.w_rerank / (self.rrf_k + rank)

            for p in shortlist:
                p.score = p.fused_score * p.authority
            shortlist.sort(key=lambda p: p.score, reverse=True)
            candidates = shortlist + candidates[self.rerank_candidates:]

        passages = candidates[:k]
        for rank, p in enumerate(passages, start=1):
            p.rank = rank

        # The gate always reads the dense cosine, so the calibrated threshold
        # means the same thing regardless of which ranking stages are enabled.
        top_score = max((p.dense_score for p in passages), default=0.0)

        return RetrievalResult(
            query=query,
            passages=passages,
            top_score=top_score,
            threshold=self.threshold,
            stages={
                "dense": len(dense),
                "lexical": len(lexical),
                "fused": len(fused),
                "reranked": bool(self.rerank),
            },
        )


if __name__ == "__main__":
    r = Retriever()
    print(f"index: {r.manifest.get('n_chunks')} chunks | k={r.k} "
          f"| threshold={r.threshold} | hybrid={r.hybrid} | rerank={r.rerank}\n")
    for q in [
        "What is the word limit for the individual project report?",
        "Which algorithm does Part C of the assignment ask you to use?",
        "Am I allowed to use AI tools when working on this project?",
        "What is the best pizza topping?",
    ]:
        res = r.retrieve(q)
        flag = "ANSWER" if res.is_confident else "REFUSE"
        print(f"[{flag}] gate={res.top_score:.3f}  {q}")
        for p in res.passages[:3]:
            print(f"     {p.source_document[:44]}/{p.slide_or_section}"
                  f"  cos={p.raw_score:.3f}  via {p.found_by}")
        print()
