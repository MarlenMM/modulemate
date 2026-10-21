"""Inspect real card output from original material, using isolated storage.

Automatic checks are diagnostics, not semantic accuracy measurements. Read the
questions, answers and evidence in the saved report. No private library is read.
"""

import argparse
import hashlib
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulemate.assistant import generate_cards
from modulemate.db import Database, Store
from modulemate.generation import local_backend
from modulemate.ingest import parse_upload
from modulemate.local_models import get_local_models
from modulemate.retrieval import scoped_passages

ROOT = Path(__file__).resolve().parents[1]


class RecordedBackend:
    def __init__(self, backend):
        self.backend = backend
        self.label = backend.label
        self.calls = []

    def complete(self, system, payload):
        started = time.monotonic()
        raw = self.backend.complete(system, payload)
        self.calls.append(
            {
                "stage": "review"
                if "cards_to_review" in payload
                else "repair"
                if "repair" in payload
                else "generate",
                "output": json.loads(raw),
                "stats": dict(self.backend.last_stats),
                "seconds": round(time.monotonic() - started, 2),
            }
        )
        return raw


def fingerprint():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "modulemate").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach", choices=["legacy", "deck"], default="deck")
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--style", default="Mixed recall")
    parser.add_argument(
        "--fixture", type=Path, nargs="+", default=[ROOT / "evaluation/card_materials.json"]
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Use prepared local semantic selection, QA reading and conservative duplicate checks",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/cards.json")
    args = parser.parse_args()
    backend = local_backend()
    if backend is None:
        parser.error("The explicitly prepared local model is unavailable.")
    cases = [case for path in args.fixture for case in json.loads(path.read_text())["cases"]]
    engine = get_local_models() if args.semantic else None
    report = {
        "at": datetime.now(UTC).isoformat(),
        "fingerprint": fingerprint(),
        "approach": args.approach,
        "style": args.style,
        "model": backend.label,
        "fixtures": [
            str(path.resolve().relative_to(ROOT))
            if path.resolve().is_relative_to(ROOT)
            else path.name
            for path in args.fixture
        ],
        "semantic_duplicates": bool(engine),
        "cases": [],
        "limits": "Authored development fixtures. Checks do not establish semantic correctness. Timings are single-run observations.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="modulemate-cards-") as directory:
            store = Store(Database(Path(directory) / "cards.sqlite3"))
            for case in cases:
                if args.cases and case["id"] not in args.cases:
                    continue
                course = store.create_course(case["id"])
                for document in case["documents"]:
                    store.add_document(
                        course,
                        parse_upload(document["name"], document["text"].encode(), isolated=False),
                    )
                passages = scoped_passages(store, course)
                started = time.monotonic()
                item = {"id": case["id"]}
                recorded = RecordedBackend(backend)
                try:
                    if args.approach == "legacy":
                        cards = generate_cards(passages, recorded)
                    else:
                        from modulemate.cards import draft_deck

                        draft = draft_deck(
                            passages,
                            recorded,
                            count=case["count"],
                            style=args.style,
                            topic=case.get("topic", ""),
                            engine=engine,
                        )
                        cards = draft.cards
                        item["diagnostics"] = draft.diagnostics
                        item["saved"] = store.add_cards(course, cards)
                        sources = {p.id: p for p in passages}

                        def normal(text):
                            return " ".join(text.split())

                        item["exact_source_membership"] = all(
                            normal(s["quote"]) in normal(sources[s["passage_id"]].text)
                            for c in cards
                            for s in c["sources"]
                        )
                        item["exact_answer_membership"] = all(
                            any(normal(c["back"]) in normal(s["quote"]) for s in c["sources"])
                            for c in cards
                        )
                    text = json.dumps(
                        [{k: c[k] for k in ("front", "back")} for c in cards], ensure_ascii=False
                    ).casefold()
                    item.update(
                        cards=cards,
                        concept_mentions={c: c.casefold() in text for c in case["concepts"]},
                        forbidden_mentions=[c for c in case["forbidden"] if c.casefold() in text],
                    )
                except Exception as exc:
                    item["error"] = type(exc).__name__
                item["seconds"] = round(time.monotonic() - started, 2)
                item["calls"] = recorded.calls
                report["cases"].append(item)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                print(
                    f"{item['id']}: {len(item.get('cards', []))} cards, {item['seconds']}s, {item.get('error', 'completed')}",
                    flush=True,
                )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
