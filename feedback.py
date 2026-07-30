"""C3 — thumbs-up/down feedback logging.

Supports post-hoc error analysis: the gold set measures what its author thought
to ask, and real students ask things its author did not. A rating logged
alongside the retrieval diagnostics for that turn (intent, gate score, which
passages were used) turns "the bot was wrong about X" into a reproducible case
that can be replayed through the pipeline.

Privacy (NFR4): the log records the question the student typed and the answer
they were given, plus retrieval metadata. It stores no identity — no name, no
account, no session cookie, no IP — because ModuleMate has no user accounts to
attach them to. A tutor reviewing the log sees questions, not students. The log
lives locally under results/ and is never transmitted.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import config


def log_path(cfg=None) -> Path:
    cfg = cfg or config.load()
    return cfg.path("evaluation.results_dir") / "feedback.jsonl"


def record(result, rating: str, cfg=None) -> Path:
    """Append one rating. `rating` is "up" or "down".

    Written as JSON Lines so concurrent appends stay well-formed and the file
    can be read incrementally without loading it whole.
    """
    cfg = cfg or config.load()
    path = log_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rating": rating,
        "question": result.question,
        "rewritten_query": result.rewritten_query,
        "intent": result.intent.label,
        "intent_confidence": round(result.intent.confidence, 3),
        "is_refusal": result.is_refusal,
        "refusal_reason": result.refusal_reason,
        "top_score": round(result.top_score, 4),
        "threshold": result.threshold,
        "answer": result.answer,
        "citations": result.citations,
        "retrieved": [
            {
                "rank": p.rank,
                "document": p.source_document,
                "locator": p.slide_or_section,
                "cosine": round(p.raw_score, 4),
                "found_by": p.found_by,
            }
            for p in result.passages
        ],
        "latency_seconds": round(result.latency_seconds, 3),
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def load_all(cfg=None) -> list[dict]:
    """Read every logged rating (empty list if none yet)."""
    path = log_path(cfg)
    if not path.exists():
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def summarise(cfg=None) -> dict:
    """Aggregate ratings for the tutor-facing view and for §4 of the report."""
    entries = load_all(cfg)
    if not entries:
        return {"n": 0}
    up = sum(1 for e in entries if e["rating"] == "up")
    down = sum(1 for e in entries if e["rating"] == "down")
    by_intent: dict[str, dict[str, int]] = {}
    for e in entries:
        bucket = by_intent.setdefault(e["intent"], {"up": 0, "down": 0})
        bucket[e["rating"]] += 1
    return {
        "n": len(entries),
        "up": up,
        "down": down,
        "satisfaction": round(up / len(entries), 3),
        "by_intent": by_intent,
        "negative_on_refusals": sum(
            1 for e in entries if e["rating"] == "down" and e["is_refusal"]
        ),
    }


if __name__ == "__main__":
    s = summarise()
    if not s["n"]:
        print("No feedback logged yet. Rate an answer in the app.")
    else:
        print(json.dumps(s, indent=1))
