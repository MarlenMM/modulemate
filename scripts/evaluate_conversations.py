"""Run original end-to-end challenges. Constraint results are diagnostics, not accuracy."""

import argparse
import hashlib
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulemate.assistant import AIBackend, answer_question
from modulemate.db import Database, Store
from modulemate.generation import local_backend
from modulemate.ingest import parse_upload
from modulemate.local_models import get_local_models
from modulemate.settings import ROOT, load_settings


class DiagnosticBackend:
    def __init__(self, backend):
        self.backend = backend
        self.label = backend.label
        self.supports_plain_text = getattr(backend, "supports_plain_text", False)
        self.calls = []

    @property
    def last_stats(self):
        return getattr(self.backend, "last_stats", {})

    def complete(self, system, payload):
        stage = (
            "audit"
            if "draft" in payload
            else "metadata"
            if "answer_to_label" in payload
            else "query"
            if "recent_topics" in payload
            else "draft"
        )
        try:
            raw = self.backend.complete(system, payload)
        except Exception as exc:
            self.calls.append(
                {
                    "stage": stage,
                    "error": type(exc).__name__,
                    "stats": getattr(self.backend, "last_stats", {}),
                }
            )
            raise
        self.calls.append(
            {
                "stage": stage,
                "response": raw,
                "stats": getattr(self.backend, "last_stats", {}),
            }
        )
        return raw


def constraints(turn, answer):
    text = answer.text.casefold()
    errors = []
    for term in turn.get("required", []):
        if term.casefold() not in text:
            errors.append(f"missing:{term}")
    for alternatives in turn.get("required_any", []):
        if not any(term.casefold() in text for term in alternatives):
            errors.append("missing_any:" + "|".join(alternatives))
    for term in turn.get("forbidden", []):
        if term.casefold() in text:
            errors.append(f"unexpected:{term}")
    for key, actual in (
        ("expected_kind", answer.kind),
        ("expected_basis", answer.metadata.get("basis")),
    ):
        if key in turn and turn[key] != actual:
            errors.append(f"{key}:{actual}")
    if "allowed_kinds" in turn and answer.kind not in turn["allowed_kinds"]:
        errors.append(f"unexpected_kind:{answer.kind}")
    words = len(text.split())
    if words > turn.get("max_words", 10000) or words < turn.get("min_words", 0):
        errors.append(f"length:{words}")
    if turn.get("require_question") and "?" not in text and "？" not in text:
        errors.append("no_question")
    if "BLUE RABBIT" in answer.text:
        errors.append("source_instruction_followed")
    if answer.kind in {"local", "empty", "error"}:
        errors.append("generation_fallback")
    return errors


def code_fingerprint():
    fingerprint = hashlib.sha256()
    for path in sorted((ROOT / "modulemate").rglob("*")):
        if path.suffix in {".py", ".json"} and "__pycache__" not in path.parts:
            fingerprint.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    return fingerprint.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        action="store_true",
        help="Explicitly send ONLY the original fixture to the configured provider.",
    )
    parser.add_argument("--ids", help="Comma separated scenario IDs; default all")
    parser.add_argument("--policy", choices=("adaptive", "baseline", "full"), default="adaptive")
    parser.add_argument(
        "--suite", choices=("general", "probe", "core", "transfer", "evidence"), default="general"
    )
    parser.add_argument(
        "--progress", action="store_true", help="Print model stages while a turn is running"
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=0,
        help="Minimum seconds between provider questions; useful on free-tier quotas.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/conversations.json"))
    args = parser.parse_args()
    fingerprint = code_fingerprint()
    started_at = datetime.now(UTC).isoformat()
    fixture_path = (
        ROOT
        / "evaluation"
        / {
            "general": "general_documents.json",
            "probe": "general_probe.json",
            "core": "conversation_challenge.json",
            "transfer": "conversation_transfer.json",
            "evidence": "evidence_transfer.json",
        }[args.suite]
    )
    backend = AIBackend(load_settings()) if args.provider else local_backend()
    if backend is None:
        parser.error("Prepare a local model first or explicitly select --provider")
    backend = DiagnosticBackend(backend)
    fixture_bytes = fixture_path.read_bytes()
    fixture_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
    fixture = json.loads(fixture_bytes)
    ids = set(args.ids.split(",")) if args.ids else None
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Database(Path(tmp) / "challenge.sqlite3"))
        course = store.create_course("Original challenge")
        doc_ids = {}
        for doc in fixture["documents"]:
            doc_ids[doc["name"]] = store.add_document(
                course, parse_upload(doc["name"], doc["text"].encode(), isolated=False)
            )[0]
        engine = get_local_models()
        for scenario in fixture["scenarios"]:
            if ids and scenario["id"] not in ids:
                continue
            store.clear_messages(course)
            scope = (
                [doc_ids[name] for name in scenario["documents"]]
                if "documents" in scenario
                else None
            )
            for i, turn in enumerate(scenario["turns"], 1):
                backend.calls.clear()
                started = time.monotonic()
                answer = answer_question(
                    store,
                    course,
                    turn["question"],
                    document_ids=scope,
                    use_ai=True,
                    backend=backend,
                    engine=engine,
                    on_progress=(
                        lambda label, turn_id=f"{scenario['id']}/{i}": print(
                            json.dumps({"id": turn_id, "stage": label}), flush=True
                        )
                    )
                    if args.progress
                    else None,
                    evidence_policy=args.policy,
                )
                row = {
                    "id": f"{scenario['id']}/{i}",
                    "question": turn["question"],
                    "answer": answer.text,
                    "kind": answer.kind,
                    "metadata": answer.metadata,
                    "seconds": round(time.monotonic() - started, 2),
                    "constraints": constraints(turn, answer),
                    "sources": [p.to_dict() for p in answer.passages],
                    "model_calls": list(backend.calls),
                }
                rows.append(row)
                print(
                    json.dumps({key: row[key] for key in ("id", "kind", "seconds", "constraints")}),
                    flush=True,
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.output.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(
                        {
                            "model": backend.label,
                            "code_sha256": fingerprint,
                            "source_changed_during_run": code_fingerprint() != fingerprint,
                            "started_at": started_at,
                            "updated_at": datetime.now(UTC).isoformat(),
                            "fixture_sha256": fixture_sha256,
                            "fixture_changed_during_run": hashlib.sha256(
                                fixture_path.read_bytes()
                            ).hexdigest()
                            != fixture_sha256,
                            "diagnostic_only": True,
                            "evidence_policy": args.policy,
                            "rows": rows,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                temporary.replace(args.output)
                if args.provider and args.pace:
                    time.sleep(max(0, args.pace - (time.monotonic() - started)))
    print(
        f"Constraint checks passed on {sum(not row['constraints'] for row in rows)}/{len(rows)} turns. Read the actual answers before drawing quality conclusions."
    )


if __name__ == "__main__":
    main()
