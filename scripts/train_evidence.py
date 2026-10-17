"""Train inspectable rank/relevance/answerability heads on original grouped data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulemate.learning import FEATURE_VERSION, FEATURES, EvidenceModel, features, reader_signals  # noqa: E402
from modulemate.local_models import READER, READER_REVISION, get_local_models  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evaluation/evidence_learning.json"


def load_data(path=DATA):
    data = json.loads(path.read_text())
    seen_ids, seen_texts, seen_passages, seen_queries = set(), set(), set(), set()
    if data.get("schema") != 1:
        raise ValueError("Unknown evidence dataset schema")
    for document in data["documents"]:
        if document["split"] not in {"train", "calibration", "test"} or document["id"] in seen_ids:
            raise ValueError("Invalid or repeated document group")
        seen_ids.add(document["id"])
        ids = {p["id"] for p in document["passages"]}
        if len(ids) != len(document["passages"]) or ids & seen_passages:
            raise ValueError("Repeated passage identity")
        seen_passages.update(ids)
        for p in document["passages"]:
            identity = " ".join(p["text"].lower().split())
            if not identity or identity in seen_texts:
                raise ValueError("Empty/repeated source could leak between splits")
            seen_texts.add(identity)
        for q in document["questions"]:
            if (
                q["id"] in seen_queries
                or not set(q["gold"] + q["related"]) <= ids
                or set(q["gold"]) & set(q["related"])
            ):
                raise ValueError("Invalid question identity or evidence labels")
            seen_queries.add(q["id"])
    if any(
        sum(d["split"] == split for d in data["documents"]) < 2
        for split in ("train", "calibration", "test")
    ):
        raise ValueError("Each split needs at least two separate document groups")
    return data


def examples(data, split):
    documents = [d for d in data["documents"] if d["split"] == split]
    rows = []
    for index, document in enumerate(documents):
        # All derived negatives stay in the same partition as the source.
        peer = documents[(index + 1) % len(documents)]
        for q in document["questions"]:
            for p, owner in [(p, document) for p in document["passages"]] + [
                (p, peer) for p in peer["passages"]
            ]:
                grade = (
                    2
                    if len(q["gold"]) == 1 and p["id"] in q["gold"]
                    else int(p["id"] in q["gold"] + q["related"])
                )
                rows.append(
                    {
                        "query_id": q["id"],
                        "document_id": document["id"],
                        "question": q["text"],
                        "text": p["text"],
                        "title": owner["name"],
                        "passage_id": p["id"],
                        "grade": grade,
                        "bundle": False,
                    }
                )
            rows.append(
                {
                    "query_id": q["id"],
                    "document_id": document["id"],
                    "question": q["text"],
                    "text": " ".join(p["text"] for p in document["passages"]),
                    "title": document["name"],
                    "passage_id": document["id"] + "-bundle",
                    "grade": 2 if q["gold"] else int(bool(q["related"])),
                    "bundle": True,
                }
            )
    return rows


def matrix(rows, engine=None, *, reader=False):
    cosines = np.zeros(len(rows))
    if engine:
        texts = list(dict.fromkeys([r["question"] for r in rows] + [r["text"] for r in rows]))
        embeddings = dict(zip(texts, engine.encode(texts), strict=True))
        cosines = [float(embeddings[r["question"]] @ embeddings[r["text"]]) for r in rows]
    signals = {}
    if reader:
        grouped = {}
        for i, row in enumerate(rows):
            grouped.setdefault(row["query_id"], []).append(i)
        for indices in grouped.values():
            local = reader_signals(
                engine, rows[indices[0]]["question"], [rows[i]["text"] for i in indices]
            )
            signals.update({indices[i]: value for i, value in local.items()})
    return np.asarray(
        [
            features(r["question"], r["text"], r["title"], cosine=c, reader=signals.get(i))
            for i, (r, c) in enumerate(zip(rows, cosines, strict=True))
        ]
    )


def fit_head(x, y):
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=0.3, class_weight="balanced", max_iter=1500, random_state=2026)
    model.fit(scaler.transform(x), y)
    weights = model.coef_[0] / scaler.scale_
    bias = float(model.intercept_[0] - weights @ scaler.mean_)
    return {"weights": weights.tolist(), "bias": bias}


def train(data, engine=None, *, reader=False, data_path=DATA):
    if json.loads(data_path.read_text()) != data:
        raise ValueError("Training data differs from the recorded source; provide its data_path")
    train_rows, cal_rows = examples(data, "train"), examples(data, "calibration")
    x, cal_x = matrix(train_rows, engine, reader=reader), matrix(cal_rows, engine, reader=reader)
    heads = {}
    for name, predicate in (
        ("relevance", lambda r: r["grade"] > 0),
        ("answerability", lambda r: r["grade"] == 2),
    ):
        heads[name] = fit_head(x, np.array([predicate(r) for r in train_rows], dtype=int))
    grouped = {}
    for i, row in enumerate(train_rows):
        if not row["bundle"]:
            grouped.setdefault(row["query_id"], []).append(i)
    differences, targets = [], []
    for group in grouped.values():
        for i, j in combinations(group, 2):
            if train_rows[i]["grade"] == train_rows[j]["grade"]:
                continue
            difference = x[i] - x[j]
            target = int(train_rows[i]["grade"] > train_rows[j]["grade"])
            differences.extend([difference, -difference])
            targets.extend([target, 1 - target])
    heads["rank"] = fit_head(np.asarray(differences), np.asarray(targets))
    calibration = {}
    for name, predicate in (
        ("relevance", lambda r: r["grade"] > 0),
        ("answerability", lambda r: r["grade"] == 2),
    ):
        logits = cal_x @ np.asarray(heads[name]["weights"]) + heads[name]["bias"]
        calibrator = LogisticRegression(C=1, random_state=2026).fit(
            logits[:, None], [int(predicate(r)) for r in cal_rows]
        )
        scale, offset = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])
        # A reversed relationship is a failed model, not a useful confidence.
        if scale < 0:
            raise ValueError("Calibration reverses the evidence model; do not publish it")
        calibration[name] = [scale, offset]
    variant = "reader" if reader else "semantic" if engine else "lexical"
    artifact = {
        "schema": 1,
        "identity": f"modulemate-evidence-{variant}-v2",
        "feature_version": FEATURE_VERSION,
        "features": list(FEATURES),
        "encoder": getattr(engine, "key", None),
        "reader": f"{READER}@{READER_REVISION}" if reader else None,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "seed": 2026,
        "algorithm": "standardized logistic heads; pairwise rank differences; disjoint Platt calibration",
        "fit_documents": [d["id"] for d in data["documents"] if d["split"] == "train"],
        "calibration_documents": [
            d["id"] for d in data["documents"] if d["split"] == "calibration"
        ],
        "training_examples": len(train_rows),
        "rank_pairs": len(targets),
        "calibration_examples": len(cal_rows),
        "heads": heads,
        "calibration": calibration,
        "limitations": "Small original synthetic dataset. Advisory probabilities, not factual correctness or general calibration guarantees.",
    }
    EvidenceModel(artifact)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Also train a variant using the pinned local encoder",
    )
    parser.add_argument(
        "--reader", action="store_true", help="Also train with source-bound SQuAD2 span signals"
    )
    args = parser.parse_args()
    data = load_data()
    engines = [(None, False)]
    if args.semantic or args.reader:
        engine = get_local_models()
        if engine is None:
            parser.error("Prepare the local encoder first")
        engines.append((engine, False))
        if args.reader:
            engines.append((engine, True))
    for engine, reader in engines:
        artifact = train(data, engine, reader=reader)
        path = (
            ROOT
            / "modulemate/models"
            / (
                "evidence-reader.json"
                if reader
                else "evidence-semantic.json"
                if engine
                else "evidence-lexical.json"
            )
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "model": artifact["identity"],
                    "training_examples": artifact["training_examples"],
                    "rank_pairs": artifact["rank_pairs"],
                    "calibration_examples": artifact["calibration_examples"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
