"""Check complete local answers against original reference facts, with all misses."""

import argparse
import json
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulemate.assistant import answer_question  # noqa: E402
from modulemate.db import Database, Store  # noqa: E402
from modulemate.local_models import get_local_models  # noqa: E402
from scripts.evaluate_intelligence import ROOT, load_fixture  # noqa: E402


def evaluate(engine, *, challenge=False):
    gold = json.loads(
        (
            ROOT
            / ("evaluation/answer_challenge.json" if challenge else "evaluation/answer_checks.json")
        ).read_text()
    )
    expected = (
        {row["id"]: row["required_facts"] for row in gold["queries"]}
        if challenge
        else {row["id"]: row["required_facts"] for row in gold["checks"]}
    )
    rows, timings = [], []
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "answers.sqlite3")
        store = Store(db, db.register("evaluation", "original evaluation passphrase"))
        course = store.create_course("Original answer evaluation")
        data = load_fixture(store, course)
        for row in gold["queries"] if challenge else data["queries"]:
            store.clear_messages(course)
            start = time.perf_counter()
            answer = answer_question(store, course, row["question"], engine=engine)
            timings.append((time.perf_counter() - start) * 1000)
            missed = [
                fact
                for fact in expected[row["id"]]
                if fact.casefold() not in answer.text.casefold()
            ]
            markers = [int(i) for i in re.findall(r"\[(\d+)\]", answer.text)]
            citations_valid = bool(markers) and all(1 <= i <= len(answer.passages) for i in markers)
            rows.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "fact_coverage_pass": not missed,
                    "missing_reference_facts": missed,
                    "citation_markers_resolve": citations_valid,
                    "answer": answer.text,
                    "notice": answer.notice,
                }
            )
    return {
        "scope": gold["scope"],
        "backend": "local neural" if engine else "local lexical",
        "questions": len(rows),
        "reference_fact_passes": sum(r["fact_coverage_pass"] for r in rows),
        "resolvable_citations": sum(r["citation_markers_resolve"] for r in rows),
        "median_answer_ms": round(statistics.median(timings), 2),
        "p95_answer_ms": round(sorted(timings)[int(0.95 * len(timings))], 2),
        "misses": [r["id"] for r in rows if not r["fact_coverage_pass"]],
        "answers": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-ml", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--challenge", action="store_true")
    args = parser.parse_args()
    engine = get_local_models() if args.local_ml else None
    if args.local_ml and engine is None:
        parser.error("Install local models first")
    report = evaluate(engine, challenge=args.challenge)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(json.dumps({k: v for k, v in report.items() if k != "answers"}, indent=2))
    return (
        True
        if args.challenge
        else report["reference_fact_passes"] / report["questions"] >= (0.85 if engine else 0.65)
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
