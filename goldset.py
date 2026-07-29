"""Shared gold-set loader used by the evaluation harness and the calibrator."""
from __future__ import annotations

import json
from pathlib import Path

OUT_OF_SCOPE = "out-of-scope/social"


def load_gold(path: Path) -> list[dict]:
    """Return the gold-set items, tolerating both the documented
    {"_about": ..., "items": [...]} form and a bare list."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    items = data["items"] if isinstance(data, dict) else data
    if not items:
        raise ValueError(f"Gold set at {path} contains no items.")
    return items


def split_scope(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into (in-scope, out-of-scope) items."""
    in_scope = [i for i in items if i["intent"] != OUT_OF_SCOPE]
    out_scope = [i for i in items if i["intent"] == OUT_OF_SCOPE]
    return in_scope, out_scope
