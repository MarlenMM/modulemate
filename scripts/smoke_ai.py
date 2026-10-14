"""Opt-in live-provider smoke test using only original fictional example material."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulemate.assistant import AIBackend, ProviderError, answer_question, generate_cards  # noqa: E402
from modulemate.db import Database, Store  # noqa: E402
from modulemate.ingest import parse_upload  # noqa: E402
from modulemate.search import overview  # noqa: E402
from modulemate.settings import ROOT, load_settings  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-provider",
        action="store_true",
        help="Send original demo excerpts to the configured provider; API charges may apply.",
    )
    if not parser.parse_args().allow_provider:
        parser.error("Supply --allow-provider to explicitly enable live requests")
    settings = load_settings()
    if not settings.ai_ready:
        parser.error("Configure the three MODULEMATE_AI_* variables first")
    results = {"provider": settings.provider_host, "model": settings.ai_model, "answers": {}}
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "smoke.sqlite3")
        store = Store(db, db.register("smoke", "temporary smoke passphrase"))
        course = store.create_course("Original Learning Lab")
        store.add_document(
            course,
            parse_upload("Learning Lab.md", (ROOT / "examples/learning-lab.md").read_bytes()),
        )
        backend = AIBackend(settings)
        for mode, question in [
            ("Explain", "What is retrieval practice?"),
            ("Compare", "Compare retrieval practice and rereading."),
            ("Summarize", "Summarize my notes"),
            ("Socratic tutor", "Help me understand spacing with one guiding question."),
        ]:
            answer = answer_question(
                store, course, question, mode=mode, use_ai=True, backend=backend
            )
            results["answers"][mode] = answer.kind
            print(f"{mode}: {answer.kind} ({answer.notice})", flush=True)
        try:
            results["validated_cards"] = len(generate_cards(overview(store, course), backend))
        except (ProviderError, ValueError, TypeError, KeyError) as exc:
            results["validated_cards"] = 0
            results["card_fallback"] = type(exc).__name__
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
