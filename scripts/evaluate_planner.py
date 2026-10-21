"""Record actual local-model event outputs from original material, never a private library."""

import argparse
import hashlib
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulemate.calendar_io import events_calendar
from modulemate.db import Database, Store
from modulemate.generation import local_backend
from modulemate.ingest import parse_upload
from modulemate.planner import extract_events
from modulemate.retrieval import scoped_passages
from modulemate.schemas import STRING, _object

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCHEMA = _object(
    {
        "events": {
            "type": "array",
            "maxItems": 24,
            "items": _object(
                {
                    "title": STRING,
                    "start_date": STRING,
                    "end_date": STRING,
                    "start_time": STRING,
                    "end_time": STRING,
                    "timezone": STRING,
                    "description": STRING,
                    "location": STRING,
                    "status": {"enum": ["Confirmed", "Tentative", "Unresolved"]},
                    "recurrence": _object(
                        {
                            "frequency": {"enum": ["none", "weekly"]},
                            "days": {"type": "array", "items": STRING},
                            "until": STRING,
                            "exclude": {"type": "array", "items": STRING},
                        }
                    ),
                }
            ),
        }
    }
)


class RecordedBackend:
    def __init__(self, backend):
        self.backend, self.label, self.calls = backend, backend.label, []

    def complete(self, system, payload):
        started = time.monotonic()
        raw = self.backend.complete(system, payload)
        self.calls.append(
            {
                "stage": "review" if "events_to_review" in payload else "extract",
                "output": json.loads(raw),
                "seconds": round(time.monotonic() - started, 2),
                "stats": dict(self.backend.last_stats),
            }
        )
        return raw


def fingerprint():
    h = hashlib.sha256()
    for p in sorted((ROOT / "modulemate").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def matches(actual, expected):
    return expected["title_contains"].casefold() in actual.get("title", "").casefold() and all(
        actual.get(k) == v for k, v in expected.items() if k != "title_contains"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", nargs="+", type=Path, default=[ROOT / "evaluation/planner_materials.json"]
    )
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--approach", choices=["hybrid", "free_json"], default="hybrid")
    parser.add_argument("--output", type=Path, default=ROOT / "results/planner.json")
    args = parser.parse_args()
    backend = local_backend()
    if backend is None:
        parser.error("The prepared local generator is unavailable.")
    report = {
        "at": datetime.now(UTC).isoformat(),
        "fingerprint": fingerprint(),
        "approach": args.approach,
        "model": backend.label,
        "cases": [],
        "limits": "Authored, reused development cases. Exact-field diagnostics and manual inspection do not establish representative or universal accuracy.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="modulemate-planner-eval-")
    try:
        for case_index, case in enumerate(
            [c for f in args.fixture for c in json.loads(f.read_text())["cases"]], 1
        ):
            if args.cases and case["id"] not in args.cases:
                continue
            store = Store(Database(Path(temporary.name) / f"case-{case_index}.sqlite3"))
            course = store.create_course(case["id"])
            for doc in case["documents"]:
                parsed = parse_upload(doc["name"], doc["text"].encode(), isolated=False)
                store.add_document(course, parsed)
            passages = scoped_passages(store, course)
            recorder = RecordedBackend(backend)
            start = time.monotonic()
            item = {"id": case["id"], "expected": case["expected"]}
            try:
                if args.approach == "free_json":
                    raw = recorder.complete(
                        "Extract only actionable scheduled events and deadlines from the material. Omit history, metadata, hypothetical/worked examples, cancellations and superseded dates. Do not guess missing years, ambiguous numeric dates, times, durations, or zones. Use ISO dates and 24-hour times, empty strings for missing fields. Use Unresolved/Tentative when facts are ambiguous or conditional. JSON only.",
                        {
                            "material": [
                                {
                                    "source_id": i,
                                    "document": p.name,
                                    "location": p.locator,
                                    "text": p.text,
                                }
                                for i, p in enumerate(passages, 1)
                            ],
                            "default_timezone": "Asia/Seoul",
                            "_response_schema": BASELINE_SCHEMA,
                            "_max_tokens": 2800,
                        },
                    )
                    rows = json.loads(raw)["events"]
                    for row in rows:
                        for key in ("start_time", "end_time"):
                            row[key] = row.get(key) or None
                        if row.get("recurrence", {}).get("frequency") == "none":
                            row["recurrence"] = {}
                    ready = [e for e in rows if e.get("status") == "Confirmed"]
                    attention = [e for e in rows if e.get("status") != "Confirmed"]
                    item.update(events=ready, attention=attention)
                else:
                    draft = extract_events(
                        passages,
                        recorder,
                        timezone="Asia/Seoul",
                        year=case.get("year"),
                        date_order=case.get("date_order", "Require unambiguous dates"),
                    )
                    ready, attention = draft.events, draft.attention
                    item.update(
                        events=ready,
                        attention=attention,
                        ignored=draft.ignored,
                        diagnostics=draft.diagnostics,
                    )
                    by_id = {p.id: p for p in passages}
                    item["source_membership"] = all(
                        " ".join(s["quote"].split())
                        in " ".join(by_id[s["passage_id"]].text.split())
                        for e in [*ready, *attention]
                        for s in e["sources"]
                    )
                    item["saved"] = store.add_events(course, ready)
                    from icalendar import Calendar

                    serialized = events_calendar(store.events(course), include_sources=True)
                    item["calendar_event_count"] = len(
                        Calendar.from_ical(serialized).walk("VEVENT")
                    )
                    item["calendar_utf8_folding"] = all(
                        len(line.encode()) <= 75 for line in serialized.split("\r\n")
                    )
                unmatched = list(case["expected"])
                extras = []
                for event in ready:
                    match = next((x for x in unmatched if matches(event, x)), None)
                    if match is None:
                        extras.append(event["title"])
                    else:
                        unmatched.remove(match)
                item.update(
                    exact_matches=len(case["expected"]) - len(unmatched),
                    extra_or_mismatched_ready=extras,
                    missing=unmatched,
                    attention_expected_met=len(attention) >= case.get("attention_at_least", 0),
                )
            except Exception as exc:
                item["error"] = type(exc).__name__ + ": " + str(exc)[:250]
            item.update(calls=recorder.calls, seconds=round(time.monotonic() - start, 2))
            report["cases"].append(item)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(
                case["id"],
                "ready",
                len(item.get("events", [])),
                "attention",
                len(item.get("attention", [])),
                "extra",
                item.get("extra_or_mismatched_ready"),
                "missing",
                len(item.get("missing", [])),
                item.get("error", ""),
                flush=True,
            )
    finally:
        backend.close()
        temporary.cleanup()


if __name__ == "__main__":
    main()
