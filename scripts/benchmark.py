"""Synthetic local retrieval timing at the 100-document course limit."""

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulemate.db import Database, Store  # noqa: E402
from modulemate.ingest import parse_upload  # noqa: E402
from modulemate.local_models import get_local_models  # noqa: E402
from modulemate.retrieval import prepare_index, retrieve  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-ml", action="store_true")
    args = parser.parse_args()
    engine = get_local_models() if args.local_ml else None
    if args.local_ml and engine is None:
        parser.error("Install local models first")
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "benchmark.sqlite3")
        store = Store(db)
        course = store.create_course("Synthetic load")
        start = time.perf_counter()
        for i in range(100):
            text = "\n".join(
                f"Topic{i:03d} section {j}: This original synthetic passage describes a process, its assumptions, observations, and a worked example for retrieval timing."
                for j in range(250)
            )
            store.add_document(
                course, parse_upload(f"topic{i:03d}.txt", text.encode(), isolated=False)
            )
        ingestion = time.perf_counter() - start
        cold = None
        if engine:
            start = time.perf_counter()
            prepare_index(store, course, engine)
            retrieve(store, course, "Topic000 process assumptions", engine=engine)
            cold = round(time.perf_counter() - start, 2)
        durations, hits = [], 0
        for i in range(100):
            start = time.perf_counter()
            found = retrieve(
                store, course, f"Topic{i:03d} process assumptions", engine=engine
            ).passages
            durations.append((time.perf_counter() - start) * 1000)
            hits += bool(found and found[0].name == f"topic{i:03d}.txt")
        print(
            json.dumps(
                {
                    "backend": "local neural hybrid" if engine else "lexical",
                    "cold_index_and_search_seconds": cold,
                    "documents": 100,
                    "passages": sum(d["chunks"] for d in store.documents(course)),
                    "ingest_seconds": round(ingestion, 2),
                    "search_median_ms": round(statistics.median(durations), 2),
                    "search_p95_ms": round(sorted(durations)[94], 2),
                    "exact_topic_hits": hits,
                    "scope": "Synthetic local timing only, excludes parsing subprocess startup and AI latency.",
                },
                indent=2,
            )
        )
        if hits != 100:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
