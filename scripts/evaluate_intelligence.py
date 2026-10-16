"""Reproducible, original development benchmark; never reads user data."""

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulemate.db import Database, Store  # noqa: E402
from modulemate.ingest import Chunk, ParsedDocument  # noqa: E402
from modulemate.intent import classify  # noqa: E402
from modulemate.local_models import (  # noqa: E402
    CANDIDATE_ENCODER,
    CANDIDATE_REVISION,
    LocalModels,
    get_local_models,
    model_dir,
)
from modulemate.retrieval import retrieve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_fixture(store, course):
    data = json.loads((ROOT / "evaluation/retrieval.json").read_text())
    grouped = defaultdict(list)
    for p in data["passages"]:
        grouped[p["document"]].append(p)
    for name, passages in grouped.items():
        text = "\n\n".join(p["text"] for p in passages)
        store.add_document(
            course,
            ParsedDocument(
                name,
                hashlib.sha256(text.encode()).hexdigest(),
                text,
                [Chunk(p["locator"], p["text"]) for p in passages],
                [],
            ),
        )
    return data


def evaluate(engine):
    report = {
        "scope": "Original synthetic development benchmark; not a blinded test or general accuracy estimate.",
        "dataset_sha256": hashlib.sha256(
            (ROOT / "evaluation/retrieval.json").read_bytes()
        ).hexdigest(),
        "model": getattr(engine, "key", None),
        "retrieval": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "evaluation.sqlite3")
        store = Store(db)
        course = store.create_course("Original benchmark")
        data = load_fixture(store, course)
        report.update(
            {
                "documents": len(store.documents(course)),
                "passages": len(data["passages"]),
                "queries": len(data["queries"]),
                "unanswerable_queries": len(data["unanswerable"]),
            }
        )
        if engine:
            start = time.perf_counter()
            retrieve(store, course, data["queries"][0]["question"], engine=engine)
            report["cold_index_and_first_search_seconds"] = round(time.perf_counter() - start, 3)
        methods = [("lexical", None, False, "hybrid")]
        if engine:
            methods += [
                ("dense", engine, False, "dense"),
                ("hybrid", engine, False, "hybrid"),
                ("hybrid_reranked", engine, True, "hybrid"),
            ]
        for name, model, rerank, strategy in methods:
            ranks, times, misses, categories = [], [], [], defaultdict(list)
            for row in data["queries"]:
                start = time.perf_counter()
                found = retrieve(
                    store, course, row["question"], engine=model, rerank=rerank, strategy=strategy
                )
                times.append((time.perf_counter() - start) * 1000)
                rank = next(
                    (
                        i
                        for i, p in enumerate(found.passages, 1)
                        if p.name == row["document"] and p.locator == row["locator"]
                    ),
                    0,
                )
                ranks.append(rank)
                categories[row["category"]].append(rank)
                if rank != 1:
                    misses.append(
                        {
                            "id": row["id"],
                            "question": row["question"],
                            "rank": rank,
                            "top": found.passages[0].locator if found.passages else None,
                        }
                    )
            rejected = sum(
                not retrieve(
                    store, course, q, engine=model, rerank=rerank, strategy=strategy
                ).passages
                for q in data["unanswerable"]
            )
            report["retrieval"][name] = {
                "hit_at_1": round(sum(r == 1 for r in ranks) / len(ranks), 4),
                "hit_at_3": round(sum(0 < r <= 3 for r in ranks) / len(ranks), 4),
                "mrr_at_6": round(sum(1 / r if r else 0 for r in ranks) / len(ranks), 4),
                "median_ms": round(statistics.median(times), 2),
                "p95_ms": round(sorted(times)[int(0.95 * len(times))], 2),
                "unanswerable_rejected": rejected,
                "by_category": {
                    c: {"count": len(rs), "hit_at_1": round(sum(r == 1 for r in rs) / len(rs), 4)}
                    for c, rs in categories.items()
                },
                "not_ranked_first": misses,
            }
            print(
                f"{name}: {report['retrieval'][name]['hit_at_1']} hit@1; {rejected}/{len(data['unanswerable'])} rejected",
                flush=True,
            )
    records = json.loads((ROOT / "evaluation/intents.json").read_text())
    predictions = [(r, classify(r["text"])) for r in records]
    report["intent"] = {
        "count": len(records),
        "accuracy": round(sum(r["intent"] == p.label for r, p in predictions) / len(records), 4),
        "misses": [
            {"text": r["text"], "expected": r["intent"], "predicted": p.label}
            for r, p in predictions
            if r["intent"] != p.label
        ],
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-ml",
        action="store_true",
        help="Require installed models and run real neural ablations",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--encoder", choices=["default", "multiqa"], default="default")
    args = parser.parse_args()
    start = time.perf_counter()
    engine = get_local_models() if args.local_ml else None
    if args.local_ml and engine is None:
        parser.error("Local models unavailable; run scripts/prepare_models.py with the ml extra")
    if args.encoder == "multiqa":
        if not args.local_ml:
            parser.error("--encoder multiqa requires --local-ml")
        engine = LocalModels(
            model_dir(), encoder_name=CANDIDATE_ENCODER, encoder_revision=CANDIDATE_REVISION
        )
    report = evaluate(engine)
    report["total_seconds"] = round(time.perf_counter() - start, 3)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized)
    # This is a development regression gate, not a claim of universal quality.
    final = report["retrieval"].get("hybrid", report["retrieval"]["lexical"])
    threshold = 0.85 if args.local_ml else 0.60
    return final["hit_at_3"] >= threshold and report["intent"]["accuracy"] >= 0.75


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
