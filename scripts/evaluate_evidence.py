"""Grouped evidence diagnostics. Never reads private workspace documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulemate.learning import EvidenceModel, features, reader_signals  # noqa: E402
from modulemate.local_models import get_local_models  # noqa: E402
from scripts.train_evidence import DATA, ROOT, examples, load_data, matrix  # noqa: E402


def probability_metrics(y, probabilities, *, training_prevalence):
    y, p = np.asarray(y, dtype=int), np.asarray(probabilities, dtype=float)
    bins = []
    for low in np.arange(0, 1, 0.1):
        selected = (p >= low) & (p < low + 0.1 + (1e-10 if low > 0.89 else 0))
        if selected.any():
            bins.append(
                {
                    "count": int(selected.sum()),
                    "predicted": round(float(p[selected].mean()), 4),
                    "observed": round(float(y[selected].mean()), 4),
                }
            )
    return {
        "count": len(y),
        "positive_count": int(y.sum()),
        "average_precision": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "balanced_accuracy_at_half": round(float(balanced_accuracy_score(y, p >= 0.5)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "training_prevalence": round(training_prevalence, 6),
        "training_rate_baseline_brier": round(float(np.mean((y - training_prevalence) ** 2)), 4),
        "ece_10_bins": round(
            sum(b["count"] * abs(b["predicted"] - b["observed"]) for b in bins) / len(y), 4
        ),
        "calibration_bins": bins,
        "selective": [
            {
                "threshold": t,
                "accepted": int((p >= t).sum()),
                "coverage": round(float((p >= t).mean()), 4),
                "error": round(float(1 - y[p >= t].mean()), 4) if (p >= t).any() else None,
            }
            for t in (0.3, 0.5, 0.7, 0.85, 0.95)
        ],
    }


def rank_metrics(rows):
    return {
        "count": len(rows),
        "hit_at_1": round(float(np.mean([r["first_rank"] == 1 for r in rows])), 4),
        "hit_at_3": round(float(np.mean([0 < r["first_rank"] <= 3 for r in rows])), 4),
        "all_required_at_3": round(float(np.mean([r["all_rank"] <= 3 for r in rows])), 4),
        "mrr": round(float(np.mean([1 / r["first_rank"] for r in rows])), 4),
        "misses": [r for r in rows if r["first_rank"] != 1 or r["all_rank"] > 3],
    }


def evaluate(data, artifact, split, engine=None, *, reader=False):
    model = EvidenceModel(artifact)
    rows = examples(data, split)
    training_rows = examples(data, "train")
    x = matrix(rows, engine, reader=reader)
    probabilities = {}
    for name in ("relevance", "answerability"):
        weights, bias = model.heads[name]
        scale, offset = model.calibration[name]
        probabilities[name] = 1 / (
            1 + np.exp(-np.clip(scale * (x @ weights + bias) + offset, -40, 40))
        )
    report = {
        "model": model.identity,
        "split": split,
        "documents": [d["id"] for d in data["documents"] if d["split"] == split],
        "relevance": probability_metrics(
            [r["grade"] > 0 for r in rows],
            probabilities["relevance"],
            training_prevalence=float(np.mean([r["grade"] > 0 for r in training_rows])),
        ),
        "answerability": probability_metrics(
            [r["grade"] == 2 for r in rows],
            probabilities["answerability"],
            training_prevalence=float(np.mean([r["grade"] == 2 for r in training_rows])),
        ),
    }
    report["answerability"]["errors_at_half"] = [
        {
            "query": r["query_id"],
            "source": r["passage_id"],
            "expected": r["grade"] == 2,
            "probability": round(float(p), 4),
        }
        for r, p in zip(rows, probabilities["answerability"], strict=True)
        if (p >= 0.5) != (r["grade"] == 2)
    ]
    documents = [d for d in data["documents"] if d["split"] == split]
    candidates = [p | {"title": d["name"]} for d in documents for p in d["passages"]]
    vectors = {}
    if engine:
        texts = list(
            dict.fromkeys(
                [p["text"] for p in candidates]
                + [q["text"] for d in documents for q in d["questions"]]
            )
        )
        vectors = dict(zip(texts, engine.encode(texts), strict=True))
    rankings = {
        key: []
        for key in (
            "lexical",
            "dense",
            "learned",
            "blend_0.0",
            "blend_0.25",
            "blend_0.5",
            "blend_1.0",
        )
    }
    for d in documents:
        for q in d["questions"]:
            if not q["gold"]:
                continue
            cosines = [
                float(vectors[q["text"]] @ vectors[p["text"]]) if engine else 0 for p in candidates
            ]
            spans = (
                reader_signals(engine, q["text"], [p["text"] for p in candidates]) if reader else {}
            )
            scores = [
                model.predict(q["text"], p["text"], p["title"], cosine=c, reader=spans.get(i))
                for i, (p, c) in enumerate(zip(candidates, cosines, strict=True))
            ]
            lexical = sorted(
                range(len(candidates)),
                key=lambda i: -float(features(q["text"], candidates[i]["text"])[0]),
            )
            dense = sorted(range(len(candidates)), key=lambda i: -cosines[i])
            learned = sorted(range(len(candidates)), key=lambda i: -scores[i].rank)
            lex_ranks = {i: r for r, i in enumerate(lexical, 1)}
            dense_ranks = {i: r for r, i in enumerate(dense, 1)}
            learned_ranks = {i: r for r, i in enumerate(learned, 1)}
            orders = {"lexical": lexical, "dense": dense, "learned": learned}
            for weight in (0.0, 0.25, 0.5, 1.0):
                orders[f"blend_{weight}"] = sorted(
                    range(len(candidates)),
                    key=lambda i: (
                        -(
                            1 / (30 + lex_ranks[i])
                            + (1 / (30 + dense_ranks[i]) if engine else 0)
                            + weight / (30 + learned_ranks[i])
                        )
                    ),
                )
            for name, order in orders.items():
                ranks = [r for r, i in enumerate(order, 1) if candidates[i]["id"] in q["gold"]]
                rankings[name].append(
                    {
                        "id": q["id"],
                        "first_rank": min(ranks),
                        "all_rank": max(ranks),
                        "first": candidates[order[0]]["id"],
                    }
                )
    report["ranking"] = {
        name: rank_metrics(rs) for name, rs in rankings.items() if engine or name != "dense"
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--reader", action="store_true")
    parser.add_argument("--split", choices=["calibration", "test"], default="calibration")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    engine = get_local_models() if args.semantic or args.reader else None
    if (args.semantic or args.reader) and engine is None:
        parser.error("Local encoder unavailable")
    artifact = json.loads(
        (
            ROOT
            / "modulemate/models"
            / (
                "evidence-reader.json"
                if args.reader
                else "evidence-semantic.json"
                if engine
                else "evidence-lexical.json"
            )
        ).read_text()
    )
    report = evaluate(load_data(), artifact, args.split, engine, reader=args.reader)
    report["dataset_sha256"] = hashlib.sha256(DATA.read_bytes()).hexdigest()
    report["diagnostic_only"] = True
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(
        json.dumps(
            {
                "model": report["model"],
                "split": report["split"],
                "ranking": {
                    k: {m: v[m] for m in ("hit_at_1", "all_required_at_3", "mrr")}
                    for k, v in report["ranking"].items()
                },
                "answerability": {
                    k: report["answerability"][k]
                    for k in ("average_precision", "brier", "ece_10_bins", "selective")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
