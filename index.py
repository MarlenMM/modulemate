"""M2 — Embedding and persistent FAISS index (FR2, FR9).

Vectors are L2-normalised before insertion into a FAISS IndexFlatIP, so the
inner product the index returns *is* cosine similarity in [-1, 1]. That matters
downstream: the refusal gate (M5) thresholds on this score directly, which is
only interpretable because the vectors are unit length.

IndexFlatIP is exact, not approximate. At ~10^3 chunks an exhaustive scan
returns in single-digit milliseconds, so an approximate index (IVF/HNSW) would
add tuning burden and recall loss for no measurable latency gain [15].

Running this file rebuilds the index from the corpus folder in one command
(FR9); it is idempotent and overwrites the previous artefacts.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from ingest import Chunk, build_chunks

INDEX_FILE = "faiss.index"
META_FILE = "chunks.json"
MANIFEST_FILE = "manifest.json"

_model_cache: dict[str, SentenceTransformer] = {}


def get_embedder(model_name: str) -> SentenceTransformer:
    """Load (and cache) the sentence-transformer. Cached because Streamlit
    re-executes the script on every interaction."""
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_texts(
    texts: list[str], cfg=None, show_progress: bool = False
) -> np.ndarray:
    """Embed texts to unit-length float32 vectors."""
    cfg = cfg or config.load()
    model = get_embedder(cfg.require("embedding.model_name"))
    vectors = model.encode(
        texts,
        batch_size=cfg.require("embedding.batch_size"),
        normalize_embeddings=cfg.require("embedding.normalize"),
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype="float32")


def build(cfg=None) -> dict:
    """Full rebuild: ingest -> embed -> index -> persist. Returns a manifest."""
    cfg = cfg or config.load()
    started = time.time()

    chunks: list[Chunk] = build_chunks(cfg)
    if not chunks:
        raise RuntimeError("Ingestion produced no chunks; nothing to index.")

    vectors = embed_texts([c.text for c in chunks], cfg, show_progress=True)
    dim = vectors.shape[1]

    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(vectors)

    out_dir = cfg.path("index.dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(out_dir / INDEX_FILE))
    with open(out_dir / META_FILE, "w", encoding="utf-8") as fh:
        json.dump([asdict(c) for c in chunks], fh, ensure_ascii=False, indent=1)

    manifest = {
        # R1 mitigation: the build timestamp is surfaced in the UI so corpus
        # staleness is visible rather than silent.
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_chunks": len(chunks),
        "n_documents": len({c.source_document for c in chunks}),
        "embedding_model": cfg.require("embedding.model_name"),
        "dim": int(dim),
        "chunk_target_words": cfg.require("index.chunk_target_words"),
        "chunk_overlap_words": cfg.require("index.chunk_overlap_words"),
        "build_seconds": round(time.time() - started, 1),
    }
    with open(out_dir / MANIFEST_FILE, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def load(cfg=None) -> tuple[faiss.Index, list[dict], dict]:
    """Load the persisted index, chunk metadata and manifest."""
    cfg = cfg or config.load()
    out_dir = cfg.path("index.dir")
    index_path = out_dir / INDEX_FILE
    if not index_path.exists():
        raise FileNotFoundError(
            f"No index at {index_path}. Build it with:  python index.py"
        )
    faiss_index = faiss.read_index(str(index_path))
    with open(out_dir / META_FILE, "r", encoding="utf-8") as fh:
        chunks = json.load(fh)
    manifest_path = out_dir / MANIFEST_FILE
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    return faiss_index, chunks, manifest


if __name__ == "__main__":
    m = build()
    print("\nIndex rebuilt")
    for key, value in m.items():
        print(f"  {key:22s} {value}")
