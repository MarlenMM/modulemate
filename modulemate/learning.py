"""Small supervised evidence models, with inspectable non-executable weights.

Probabilities are advisory estimates from a small original dataset. They never
authorize sources, establish truth, or veto a generative answer on their own.
No user text is cached and no training or network access occurs during inference.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import snowballstemmer

from modulemate.search import tokens

FEATURE_VERSION = "evidence-features-2"
FEATURES = (
    "query_coverage",
    "source_precision",
    "overlap_dice",
    "bigram_coverage",
    "title_coverage",
    "sentence_coverage",
    "log_words",
    "log_query_words",
    "number_match",
    "number_missing",
    "asks_quantity",
    "quantity_present",
    "asks_time",
    "time_present",
    "asks_person",
    "person_present",
    "asks_location",
    "location_present",
    "asks_reason",
    "reason_present",
    "absence_cue",
    "absence_overlap",
    "condition_present",
    "question_condition",
    "negation_present",
    "definition_present",
    "dense_cosine",
    "reader_span_present",
    "reader_margin",
    "reader_fraction",
)
MODEL_DIRECTORY = Path(__file__).parent / "models"


def _terms(text):
    # A new stemmer has no cross-request mutable buffers containing user text.
    words = tokens(text[:18000])
    return set(snowballstemmer.stemmer("english").stemWords(words))


def features(question, text, title="", *, cosine=0.0, reader=None):
    question, text, title = question[:4000], text[:18000], title[:500]
    q, p = _terms(question), _terms(text)
    overlap = len(q & p)
    qn, pn = max(1, len(q)), max(1, len(p))
    qwords, pwords = tokens(question), tokens(text)
    bigrams = set(zip(qwords, qwords[1:], strict=False))
    source_bigrams = set(zip(pwords, pwords[1:], strict=False))
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    number = r"\b\d+(?:[.:]\d+)?\b"
    qnumbers, pnumbers = set(re.findall(number, question)), set(re.findall(number, text))
    absent = [
        s
        for s in sentences
        if re.search(
            r"\b(?:no (?:stated|fixed|numeric|numerical)|(?:does|do|is|are|was|were) not "
            r"(?:state|give|name|identify|specify|describe|select|include|provide|publish)|"
            r"(?:not|never) (?:given|stated|identified|specified|published|measured|chosen)|"
            r"unnamed|unstated|no .{0,45}(?:estimate|deadline|forecast|price|date|results?))\b",
            s,
            re.I,
        )
    ]

    def has(pattern, value):
        return float(bool(re.search(pattern, value, re.I)))

    values = [
        overlap / qn,
        overlap / pn,
        2 * overlap / (qn + pn),
        len(bigrams & source_bigrams) / max(1, len(bigrams)),
        len(q & _terms(title)) / qn,
        max((len(q & _terms(s)) / qn for s in sentences), default=0),
        math.log1p(len(pwords)),
        math.log1p(len(qwords)),
        len(qnumbers & pnumbers) / max(1, len(qnumbers)),
        len(qnumbers - pnumbers) / max(1, len(qnumbers)),
        has(
            r"\b(?:how (?:many|much|long|wide|far)|number|amount|price|cost|fee|limit|capacity)\b",
            question,
        ),
        has(r"\b\d+\b|\b(?:one|two|three|four|five|six|free|once|twice|without a fare)\b", text),
        has(r"\b(?:when|what time|date|year|deadline)\b", question),
        has(
            r"\b\d{1,2}:\d{2}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december|before|after|overnight|thursday|monday)\b|\b\d{4}\b",
            text,
        ),
        has(r"\b(?:who|whose|name of|which (?:writer|author|person))\b", question),
        has(r"\b(?:named|called|writer|author|librarian|organiser|leader|designer|sender)\b", text),
        has(r"\b(?:where|which (?:room|entrance|place)|location)\b", question),
        has(
            r"\b(?:in|at|beside|inside|under|above|entrance|room|studio|pier|square|door|compartment)\b",
            text,
        ),
        has(r"\b(?:why|reason|purpose)\b", question),
        has(
            r"\b(?:because|so (?:that|it)|in order|to (?:reduce|keep|locate|retain|avoid))\b", text
        ),
        float(bool(absent)),
        max((len(q & _terms(s)) / qn for s in absent), default=0),
        has(r"\b(?:if|unless|only|before|after|must|when|at least)\b", text),
        has(r"\b(?:if|unless|alone|without|guarantee|always|must|required|compulsory)\b", question),
        has(r"\b(?:not|no|never|without)\b", text),
        has(r"\b(?:is|are|means|refers|uses|called|defined|describes)\b", text),
        float(cosine) if math.isfinite(float(cosine)) else 0.0,
        float(bool(reader)),
        min(20.0, max(0.0, reader.get("margin", 0.0))) / 10 if reader else 0.0,
        len(reader.get("text", "").split()) / max(1, len(text.split())) if reader else 0.0,
    ]
    return np.asarray(values, dtype=np.float64)


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -40, 40)))


@dataclass(frozen=True)
class EvidenceScore:
    relevance: float
    answerability: float
    rank: float


class EvidenceModel:
    def __init__(self, artifact):
        if artifact.get("schema") != 1 or artifact.get("features") != list(FEATURES):
            raise ValueError("Evidence model feature schema does not match")
        if artifact.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Evidence model feature version does not match")
        self.identity = artifact["identity"]
        self.encoder = artifact.get("encoder")
        self.reader = artifact.get("reader")
        self.heads = {}
        for name in ("rank", "relevance", "answerability"):
            head = artifact["heads"][name]
            weights = np.asarray(head["weights"], dtype=np.float64)
            bias = float(head["bias"])
            if (
                weights.shape != (len(FEATURES),)
                or not np.isfinite(weights).all()
                or not math.isfinite(bias)
                or np.max(np.abs(weights)) > 1e6
                or abs(bias) > 1e6
            ):
                raise ValueError("Invalid evidence model coefficients")
            self.heads[name] = weights, bias
        self.calibration = {}
        for name in ("relevance", "answerability"):
            values = np.asarray(artifact["calibration"][name], dtype=float)
            if values.shape != (2,) or not np.isfinite(values).all() or values[0] < 0:
                raise ValueError("Invalid evidence calibration")
            self.calibration[name] = values

    def predict(self, question, text, title="", *, cosine=0.0, reader=None):
        row = features(question, text, title, cosine=cosine, reader=reader)
        raw = {name: float(row @ weights + bias) for name, (weights, bias) in self.heads.items()}
        probabilities = {}
        for name, (scale, offset) in self.calibration.items():
            probabilities[name] = float(sigmoid(scale * raw[name] + offset))
        return EvidenceScore(
            probabilities["relevance"], probabilities["answerability"], raw["rank"]
        )


@lru_cache(maxsize=4)
def load_evidence_model(encoder=None, reader=False):
    path = MODEL_DIRECTORY / (
        "evidence-reader.json"
        if reader
        else "evidence-semantic.json"
        if encoder
        else "evidence-lexical.json"
    )
    try:
        if path.stat().st_size > 250000:
            return None
        model = EvidenceModel(json.loads(path.read_text()))
        if reader:
            from modulemate.local_models import READER, READER_REVISION

            if model.reader != f"{READER}@{READER_REVISION}":
                return load_evidence_model(encoder)
        if model.encoder != encoder or bool(model.reader) != reader:
            return (
                load_evidence_model(encoder)
                if reader
                else load_evidence_model()
                if encoder
                else None
            )
        return model
    except (OSError, ValueError, TypeError, KeyError):
        return (
            load_evidence_model(encoder) if reader else load_evidence_model() if encoder else None
        )


def score_passages(question, passages, *, cosines=None, encoder=None, readers=None):
    model = load_evidence_model(encoder, reader=readers is not None)
    if model is None:
        return {}, "unavailable"
    scores = {
        p.id: model.predict(
            question,
            p.text,
            p.name + " " + p.locator,
            cosine=(cosines or {}).get(p.id, 0.0),
            reader=(readers or {}).get(p.id),
        )
        for p in passages
    }
    return scores, model.identity


def reader_signals(engine, question, texts):
    """Validate source identity before using uncalibrated reader span logits."""
    signals = {}
    for span in engine.extract(question, texts):
        if (
            type(span.source) is not int
            or not 1 <= span.source <= len(texts)
            or not math.isfinite(span.margin)
            or not 0 <= span.start < span.end <= len(texts[span.source - 1])
            or texts[span.source - 1][span.start : span.end] != span.text
        ):
            continue
        signals[span.source - 1] = {
            "margin": float(span.margin),
            "text": span.text,
            "start": span.start,
            "end": span.end,
        }
    return signals
