"""Pinned local inference. Only model weights are cached globally, never user text."""

from __future__ import annotations

import importlib.util
import json
import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from modulemate.settings import ROOT

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
READER = "deepset/minilm-uncased-squad2"
READER_REVISION = "934656cdda79824eabf503ed56e15c01ddbdbe3f"
MODEL_KEY = f"{ENCODER}@{ENCODER_REVISION}:mean-v1"
DIMENSIONS = 384
CANDIDATE_ENCODER = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
CANDIDATE_REVISION = "b207367332321f8e44f96e224ef15bc607f4dbf0"


class ModelUnavailable(RuntimeError):
    """Safe local inference failure; raw errors can contain private input."""


def model_dir() -> Path:
    load_dotenv(ROOT / ".env")
    return Path(
        os.getenv("MODULEMATE_MODEL_DIR", str(Path.home() / ".cache/modulemate/models"))
    ).expanduser()


def model_status() -> str:
    load_dotenv(ROOT / ".env")
    if os.getenv("MODULEMATE_LOCAL_ML", "auto").lower() == "off":
        return "disabled"
    try:
        manifest = json.loads((model_dir() / "ready.json").read_text())
        if manifest == {
            "encoder": ENCODER_REVISION,
            "reranker": RERANKER_REVISION,
            "reader": READER_REVISION,
        }:
            if importlib.util.find_spec("sentence_transformers") is None:
                return "dependencies missing"
            if not all(
                (model_dir() / revision / "config.json").is_file()
                for revision in (ENCODER_REVISION, RERANKER_REVISION, READER_REVISION)
            ):
                return "incomplete"
            return "ready"
    except (OSError, ValueError):
        pass
    return "not installed"


def normalized_vectors(values, count, dimensions=None):
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] != count or not np.isfinite(result).all():
        raise ModelUnavailable("Invalid local embedding output.")
    if dimensions is not None and result.shape[1] != dimensions:
        raise ModelUnavailable("Local embedding dimensions changed.")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ModelUnavailable("Empty local embedding output.")
    return result / norms


class LocalModels:
    key = MODEL_KEY
    dimensions = DIMENSIONS

    def __init__(self, directory, *, encoder_name=ENCODER, encoder_revision=ENCODER_REVISION):
        self.key = f"{encoder_name}@{encoder_revision}:mean-v1"
        try:
            import torch
            from sentence_transformers import CrossEncoder, SentenceTransformer

            torch.set_num_threads(2)
            self.encoder = SentenceTransformer(
                str(Path(directory) / encoder_revision),
                device="cpu",
                local_files_only=True,
                trust_remote_code=False,
                model_kwargs={"use_safetensors": True},
            )
            self.reranker = CrossEncoder(
                str(Path(directory) / RERANKER_REVISION),
                device="cpu",
                local_files_only=True,
                trust_remote_code=False,
                model_kwargs={"use_safetensors": True},
                max_length=512,
            )
            # Chunking uses short sentence windows below this limit, including
            # for CJK, where a character can occupy an entire tokenizer token.
            self.encoder.max_seq_length = 256
            self._lock = threading.Lock()
            self._reader_path = str(Path(directory) / READER_REVISION)
            self._reader = None
        except Exception as exc:
            raise ModelUnavailable(
                "Local models could not be loaded. Run the model setup command again."
            ) from exc

    def encode(self, texts):
        try:
            with self._lock:
                values = self.encoder.encode(
                    texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
                )
            return normalized_vectors(values, len(texts), self.dimensions)
        except Exception as exc:
            raise ModelUnavailable("Local embedding inference failed.") from exc

    def rerank(self, query, texts):
        try:
            with self._lock:
                values = np.asarray(
                    self.reranker.predict(
                        [(query, text) for text in texts], batch_size=16, show_progress_bar=False
                    ),
                    dtype=float,
                ).reshape(-1)
            if len(values) != len(texts) or not np.isfinite(values).all():
                raise ValueError("Invalid ranking output")
            return values.tolist()
        except Exception as exc:
            raise ModelUnavailable("Local relevance ranking failed.") from exc

    def extract(self, question, texts):
        try:
            from modulemate.reader import QuestionReader

            with self._lock:
                if self._reader is None:
                    self._reader = QuestionReader(self._reader_path)
                return self._reader.extract(question, texts)
        except Exception as exc:
            raise ModelUnavailable("Local answer extraction failed.") from exc


@lru_cache(maxsize=2)
def _load(directory):
    return LocalModels(directory)


def get_local_models():
    return _load(str(model_dir())) if model_status() == "ready" else None
