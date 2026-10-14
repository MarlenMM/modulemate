"""Small original multi-subject retrieval regression; never reads a user's database."""

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulemate.db import Database, Store  # noqa: E402
from modulemate.ingest import parse_upload  # noqa: E402
from modulemate.search import search  # noqa: E402

FIXTURES = {
    "biology.txt": "Photosynthesis converts light energy into chemical energy. Chlorophyll absorbs light. Mitochondria participate in cellular respiration.",
    "statistics.txt": "The median is the middle value of a sorted dataset. The mean is the sum divided by the number of observations. An outlier can shift the mean strongly.",
    "programming.txt": "A binary search repeatedly halves a sorted search interval. A stack removes the most recently added item first. A queue removes the earliest added item first.",
    "schedule.txt": "The fictional portfolio deadline is 20 November 2027. The reflection is worth sixty percent. The presentation lasts five minutes.",
    "korean.txt": "금요일 시험 안내입니다. 복습은 배운 내용을 다시 확인하는 활동입니다.",
    "chinese.txt": "概率是可能性的度量。学习计划帮助分配复习时间。",
}
QUERIES = [
    ("How does photosynthesis use light?", "biology.txt"),
    ("What does chlorophyll absorb?", "biology.txt"),
    ("What are mitochondria involved in?", "biology.txt"),
    ("How is the median found?", "statistics.txt"),
    ("Can an outlier affect the mean?", "statistics.txt"),
    ("How does binary search work?", "programming.txt"),
    ("What order does a stack use?", "programming.txt"),
    ("What order does a queue use?", "programming.txt"),
    ("What is the portfolio deadline?", "schedule.txt"),
    ("How long is the presentation?", "schedule.txt"),
    ("금요일 시험", "korean.txt"),
    ("概率", "chinese.txt"),
]


def run():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "evaluation.sqlite3")
        store = Store(db, db.register("evaluation", "synthetic test passphrase"))
        course = store.create_course("Original evaluation fixtures")
        for name, text in FIXTURES.items():
            store.add_document(course, parse_upload(name, text.encode()))
        latencies, correct = [], 0
        misses = []
        for question, expected in QUERIES:
            start = time.perf_counter()
            found = search(store, course, question)
            latencies.append((time.perf_counter() - start) * 1000)
            passed = bool(found and found[0].name == expected)
            correct += passed
            if not passed:
                misses.append(question)
        no_match = search(store, course, "quasar zeppelin xylophone") == []
        result = {
            "fixture_documents": len(FIXTURES),
            "queries": len(QUERIES),
            "document_hit_at_1": correct / len(QUERIES),
            "median_search_ms": round(statistics.median(latencies), 2),
            "no_match_check": no_match,
            "misses": misses,
            "scope": "Small original lexical regression only; not semantic or live AI accuracy.",
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return correct == len(QUERIES) and no_match


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
