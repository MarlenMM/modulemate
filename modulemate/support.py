"""Independent local natural-language inference; predictions are not proof.

The optional pinned classifier checks source/claim pairs. It never downloads on
a question, executes source instructions, or treats an unverified model verdict
as permission to expose additional documents.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import threading
from functools import lru_cache

import numpy as np

from modulemate.local_models import ModelUnavailable, model_dir

SUPPORT_MODEL = "cross-encoder/nli-deberta-v3-small"
SUPPORT_REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"
LABELS = ("contradiction", "entailment", "neutral")


def support_status():
    if (
        os.getenv("MODULEMATE_LOCAL_ML", "auto").lower() == "off"
        or os.getenv("MODULEMATE_SUPPORT", "off").lower() == "off"
    ):
        return "disabled"
    try:
        directory = model_dir()
        if json.loads((directory / "support-ready.json").read_text()) != {
            "model": SUPPORT_MODEL,
            "revision": SUPPORT_REVISION,
        }:
            return "incomplete"
        if not (directory / SUPPORT_REVISION / "model.safetensors").is_file():
            return "incomplete"
        return "ready" if importlib.util.find_spec("transformers") else "dependencies missing"
    except (OSError, ValueError, TypeError):
        return "not installed"


class SupportModel:
    identity = f"{SUPPORT_MODEL}@{SUPPORT_REVISION}"

    def __init__(self, directory):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            torch.set_num_threads(2)
            path = str(directory / SUPPORT_REVISION)
            self.tokenizer = AutoTokenizer.from_pretrained(
                path, local_files_only=True, trust_remote_code=False
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path, local_files_only=True, trust_remote_code=False, use_safetensors=True
            ).eval()
            mapping = self.model.config.id2label
            if tuple(mapping.get(i, "").lower() for i in range(3)) != LABELS:
                raise ValueError("Unexpected NLI label mapping")
            self._lock = threading.Lock()
        except Exception as exc:
            raise ModelUnavailable("The local support checker could not be loaded.") from exc

    def predict(self, pairs):
        if not pairs:
            return []
        if len(pairs) > 48 or any(
            not isinstance(a, str) or not isinstance(b, str) or len(b) > 1600 for a, b in pairs
        ):
            raise ModelUnavailable("Support check exceeds its input budget.")
        try:
            import torch

            rows = []
            with self._lock, torch.inference_mode():
                for start in range(0, len(pairs), 8):
                    batch = pairs[start : start + 8]
                    encoded = self.tokenizer(
                        [a for a, _ in batch],
                        [b for _, b in batch],
                        padding=True,
                        truncation="only_first",
                        max_length=512,
                        return_tensors="pt",
                    )
                    values = torch.softmax(self.model(**encoded).logits, dim=-1).cpu().numpy()
                    if values.shape != (len(batch), 3) or not np.isfinite(values).all():
                        raise ValueError("Invalid NLI scores")
                    rows.extend(dict(zip(LABELS, map(float, row), strict=True)) for row in values)
            return rows
        except Exception as exc:
            raise ModelUnavailable("The local support check was unavailable.") from exc


@lru_cache(maxsize=1)
def _load(directory):
    from pathlib import Path

    return SupportModel(Path(directory))


def get_support_model():
    if support_status() != "ready":
        return None
    try:
        return _load(str(model_dir()))
    except ModelUnavailable:
        return None


def factual_claims(text, cited_ids):
    """Bounded prose sentences, excluding code, questions and explicit illustrations."""
    clean = re.sub(r"```[\s\S]*?```|`[^`]*`", "", text)
    claims = []
    for paragraph in re.split(r"\n\s*\n|\n(?=\s*[-*]\s)", clean):
        refs = [int(n) for n in re.findall(r"(?<!\w)\[(\d+)\]", paragraph)] or list(cited_ids)
        paragraph = re.sub(r"(?<!\w)\[\d+\]", "", paragraph)
        paragraph = re.sub(r"(?m)^\s*(?:#{1,6}\s+|[-*]\s+|\d+\.\s+)", "", paragraph)
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])|\n", paragraph):
            sentence = sentence.strip().strip("*")
            if not 25 <= len(sentence) <= 1200 or sentence.endswith("?"):
                continue
            if re.match(r"^(?:yes|no)[,;:]\s*(?:it|they|that|this|you)\b", sentence, re.I):
                continue  # An unresolved pronoun is not a standalone NLI claim.
            if re.search(
                r"\b(?:for example|for instance|imagine|illustration|I (?:couldn't|cannot|can't)|not (?:given|stated|specified)|does not (?:state|give|specify))\b",
                sentence,
                re.I,
            ):
                continue
            if refs:
                claims.append({"claim": sentence, "source_ids": list(dict.fromkeys(refs))[:6]})
            if len(claims) >= 8:
                return claims
    return claims


def normalized_claim(text):
    # Judge the proposition, rather than asking a sentence-pair model to infer
    # that 'the guide states X' and X describe the same source attribution.
    return re.sub(
        r"\b(?:according to (?:the |this )?(?:(?:provided|supplied|current|selected) )?"
        r"(?:guide|handout|text|notes|exhibit|notice|materials)[,;:]?\s*|"
        r"(?:the|this|these) (?:(?:provided|supplied|current|selected) )?"
        r"(?:guide|handout|text|notes|exhibit|notice|materials) "
        r"(?:states?|says?|notes?|explains?|explicitly states?)\s+(?:that\s+)?)",
        "",
        text,
        flags=re.I,
    ).strip()


def focused_premise(text, claim, limit=1600):
    """Keep likely support and adjacent qualifiers before the tokenizer limit."""
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+[^\n]+\n?", "", text).strip()
    if len(text) <= limit:
        return text
    from modulemate.learning import features

    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    best = sorted(range(len(sentences)), key=lambda i: -features(claim, sentences[i])[0])[:3]
    selected = set()
    for i in best:
        selected.update(range(max(0, i - 1), min(len(sentences), i + 2)))
    return " ".join(sentences[i] for i in sorted(selected))[:limit]


def premise_variants(text, claim):
    """Whole context plus a focused adjacent-sentence window.

    Natural-language inference models can confuse unrelated dates in a paragraph.
    Keep the whole context and an adjacent window, not just an isolated matching
    word. Neither variant's prediction is a semantic guarantee.
    """
    from modulemate.learning import features

    premise = focused_premise(text, claim)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", premise) if s.strip()]
    if not sentences:
        return []
    best = max(range(len(sentences)), key=lambda i: features(claim, sentences[i])[0])
    window = " ".join(sentences[max(0, best - 1) : best + 2])
    return list(dict.fromkeys([premise, window, sentences[best]]))


def check_claims(text, passages, cited_ids, *, model=None):
    model = model or get_support_model()
    if model is None:
        return {"available": False, "claims": [], "flags": []}
    claims = factual_claims(text, cited_ids)
    pairs, bindings = [], []
    for i, claim in enumerate(claims):
        proposition = normalized_claim(claim["claim"])
        valid = []
        for sid in claim["source_ids"]:
            if sid in cited_ids and 1 <= sid <= len(passages):
                for premise in premise_variants(passages[sid - 1].text, proposition):
                    if len(pairs) >= 47:
                        break
                    pairs.append((premise, proposition))
                    bindings.append((i, (sid,)))
                valid.append(sid)
        if 1 < len(valid) <= 4 and len(pairs) < 48:
            premise = "\n".join(
                focused_premise(passages[sid - 1].text, claim["claim"], limit=500) for sid in valid
            )
            pairs.append((premise, proposition))
            bindings.append((i, tuple(valid)))
    try:
        predictions = model.predict(pairs)
    except ModelUnavailable:
        return {"available": False, "claims": [], "flags": []}
    if len(predictions) != len(pairs):
        return {"available": False, "claims": [], "flags": []}
    try:
        numeric = np.asarray([[row[label] for label in LABELS] for row in predictions], dtype=float)
        if pairs and (
            numeric.shape != (len(pairs), 3)
            or not np.isfinite(numeric).all()
            or np.any(numeric < 0)
            or np.any(numeric > 1)
            or not np.allclose(numeric.sum(axis=1), 1, atol=1e-5)
        ):
            return {"available": False, "claims": [], "flags": []}
    except (KeyError, TypeError, ValueError):
        return {"available": False, "claims": [], "flags": []}
    records = []
    for i, claim in enumerate(claims):
        matches = [
            (sid, score)
            for (index, sid), score in zip(bindings, predictions, strict=True)
            if index == i
        ]
        if not matches:
            continue
        # Another genuinely supporting cited passage outranks a neutral or
        # contradictory snippet. Multiple-source inference may remain unknown.
        supporting_ids, score = max(matches, key=lambda row: row[1]["entailment"])
        label = max(score, key=score.get)
        flagged = score["entailment"] < 0.15 and any(
            candidate["contradiction"] >= 0.90 or candidate["neutral"] >= 0.98
            for _, candidate in matches
        )
        records.append(
            claim
            | {
                "label": label,
                "probabilities": {k: round(v, 4) for k, v in score.items()},
                "best_source_id": supporting_ids[0],
                "checked_source_ids": list(supporting_ids),
                "flagged": flagged,
            }
        )
    return {
        "available": True,
        "model": model.identity,
        "claims": records,
        "flags": [r for r in records if r["flagged"]],
        "limited": len(records) < len(claims) or len(pairs) >= 47,
    }


def remap_support(report, mapping):
    """Persist the compact source numbers that the reader actually sees."""
    records = []
    for record in report.get("claims", []):
        best = record["best_source_id"]
        if best not in mapping:
            continue
        records.append(
            record
            | {
                "best_source_id": mapping[best],
                "source_ids": [mapping[i] for i in record["source_ids"] if i in mapping],
                "checked_source_ids": [
                    mapping[i] for i in record.get("checked_source_ids", [best]) if i in mapping
                ],
            }
        )
    return report | {"claims": records, "flags": [r for r in records if r["flagged"]]}
