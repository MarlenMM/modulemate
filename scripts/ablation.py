"""Retrieval ablation study — measures what each stage actually contributes.

Runs the gold set through the retriever under a series of configurations,
reporting exact hit@3, exact hit@5, document-level hit@3 and MRR for each. No
generation calls are made, so a full sweep is free and takes seconds.

This exists because "we added hybrid retrieval and a re-ranker" is an assertion
until someone measures it. Two of the configurations below were expected to
help and did not; that is recorded rather than quietly dropped.

Writes results/ablation.md.

Usage:
    python scripts/ablation.py            # standard sweep
    python scripts/ablation.py --sweep-weights   # + lexical-weight sweep
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import retrieve as retrieve_mod  # noqa: E402
from goldset import load_gold, split_scope  # noqa: E402


def norm(v: str | None) -> str:
    return " ".join((v or "").split()).strip().lower()


class _Overlay(config.Config):
    """A Config with some values overridden, so a Retriever can be built for
    an arbitrary configuration without mutating config.yaml on disk."""

    def __init__(self, base: config.Config, overrides: dict):
        merged = copy.deepcopy(base.as_dict())
        for dotted, value in overrides.items():
            node = merged
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        super().__init__(merged, base.root)


def acceptable(item: dict) -> tuple[set, set]:
    """Every (document, locator) that answers this question, and the set of
    documents among them. Mirrors eval/harness.acceptable_locations so
    ablation numbers are directly comparable with the headline metrics."""
    pairs = {(norm(item["expected_source_doc"]),
              norm(item["expected_slide_or_section"]))}
    for doc, loc in item.get("also_acceptable", []):
        pairs.add((norm(doc), norm(loc)))
    return pairs, {d for d, _ in pairs}


def evaluate(cfg, items: list[dict]) -> dict:
    r = retrieve_mod.Retriever(cfg)
    exact3 = exact5 = doc3 = 0
    reciprocal = 0.0
    for item in items:
        wanted, want_docs = acceptable(item)
        passages = r.retrieve(item["question"], k=5).passages

        rank_of = None
        for i, p in enumerate(passages, start=1):
            if (norm(p.source_document), norm(p.slide_or_section)) in wanted:
                rank_of = i
                break
        if rank_of is not None:
            exact5 += 1
            reciprocal += 1.0 / rank_of
            if rank_of <= 3:
                exact3 += 1
        if any(norm(p.source_document) in want_docs for p in passages[:3]):
            doc3 += 1

    n = len(items)
    return {
        "exact@3": exact3 / n,
        "exact@5": exact5 / n,
        "doc@3": doc3 / n,
        "MRR@5": reciprocal / n,
    }


CONFIGURATIONS: list[tuple[str, dict]] = [
    ("dense only (baseline)", {
        "retrieval.hybrid.enabled": False,
        "retrieval.rerank.enabled": False,
        "retrieval.authority_weights": {},
    }),
    ("dense + authority prior", {
        "retrieval.hybrid.enabled": False,
        "retrieval.rerank.enabled": False,
    }),
    ("dense + authority + BM25 (C1)", {
        "retrieval.hybrid.enabled": True,
        "retrieval.rerank.enabled": False,
    }),
    ("dense + authority + cross-encoder (C2)", {
        "retrieval.hybrid.enabled": False,
        "retrieval.rerank.enabled": True,
    }),
    ("all stages (C1 + C2)", {
        "retrieval.hybrid.enabled": True,
        "retrieval.rerank.enabled": True,
    }),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-weights", action="store_true")
    args = ap.parse_args()

    base = config.load()
    items, _ = split_scope(load_gold(base.path("evaluation.gold_set")))

    rows: list[tuple[str, dict]] = []
    for label, overrides in CONFIGURATIONS:
        cfg = _Overlay(base, overrides)
        metrics = evaluate(cfg, items)
        rows.append((label, metrics))
        print(f"  {label:42s} exact@3={metrics['exact@3']:.3f} "
              f"exact@5={metrics['exact@5']:.3f} doc@3={metrics['doc@3']:.3f} "
              f"MRR={metrics['MRR@5']:.3f}")

    sweep: list[tuple[str, dict]] = []
    if args.sweep_weights:
        print("\n  lexical-weight sweep (hybrid on, re-ranker off):")
        for w in (0.2, 0.4, 0.6, 0.8, 1.0, 1.5):
            cfg = _Overlay(base, {
                "retrieval.hybrid.enabled": True,
                "retrieval.rerank.enabled": False,
                "retrieval.hybrid.weight_lexical": w,
            })
            m = evaluate(cfg, items)
            sweep.append((f"weight_lexical = {w}", m))
            print(f"    w={w:<4} exact@3={m['exact@3']:.3f} "
                  f"exact@5={m['exact@5']:.3f} MRR={m['MRR@5']:.3f}")

    out_dir = base.path("evaluation.results_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "ablation.md"

    def table(title: str, data: list[tuple[str, dict]]) -> list[str]:
        lines = [f"\n## {title}\n",
                 "| Configuration | exact hit@3 | exact hit@5 | document hit@3 | MRR@5 |",
                 "|---|---|---|---|---|"]
        for label, m in data:
            lines.append(
                f"| {label} | {m['exact@3']:.3f} | {m['exact@5']:.3f} "
                f"| {m['doc@3']:.3f} | {m['MRR@5']:.3f} |"
            )
        return lines

    lines = ["# Retrieval ablation\n",
             f"{len(items)} in-scope gold questions, retrieval only (no "
             f"generation). Each row is the full pipeline with one stage "
             f"switched off, so the difference between rows is that stage's "
             f"contribution.\n"]
    lines += table("Stage contributions", rows)
    if sweep:
        lines += table("Lexical weight sweep", sweep)
    lines.append(
        "\n**exact hit@k** requires the precise (document, slide/page) pair "
        "named in the gold set. **document hit@3** requires only the right "
        "document, and is the fairer measure where the corpus answers a "
        "question correctly in more than one place. **MRR@5** is the mean "
        "reciprocal rank of the expected passage within the top 5, so it "
        "rewards moving the right passage up even when it was already "
        "retrieved.\n"
    )
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
