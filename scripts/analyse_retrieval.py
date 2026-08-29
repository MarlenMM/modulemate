"""Retrieval-only diagnostic over the gold set (no generation, no API cost).

The headline hit@k demands the *exact* (document, locator) pair the gold set
names. On a single-module corpus that was a fair test. On the full programme
corpus it is partly unfair, because several questions are answered correctly by
more than one place in the corpus:

  * two programme brochures both state where the summer school is held;
  * the assignment brief and the project brief each carry their own "Penalty"
    section with near-identical wording;
  * regression and classification are taught in both the Week 1 PDFs and the
    Week 2 decks.

Retrieving the other correct source scores zero on exact hit@k while being a
perfectly good answer. So this script reports exact and document-level hit@k
side by side, plus hit@5, so the gap between "wrong" and "differently right"
is visible rather than hidden inside one number.

Runs entirely locally — useful for tuning k, the authority weights or the
chunker without spending API tokens.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from goldset import load_gold, split_scope  # noqa: E402
from retrieve import Retriever  # noqa: E402


def norm(v: str | None) -> str:
    return " ".join((v or "").split()).strip().lower()


def main() -> None:
    cfg = config.load()
    items, _ = split_scope(load_gold(cfg.path("evaluation.gold_set")))
    retriever = Retriever(cfg)

    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rows = []

    for item in items:
        wanted = {(norm(item["expected_source_doc"]),
                   norm(item["expected_slide_or_section"]))}
        for doc, loc in item.get("also_acceptable", []):
            wanted.add((norm(doc), norm(loc)))
        want_docs = {d for d, _ in wanted}
        res = retriever.retrieve(item["question"], k=5)

        def hit_within(n: int) -> bool:
            return any(
                (norm(p.source_document), norm(p.slide_or_section)) in wanted
                for p in res.passages[:n]
            )

        exact3, exact5 = hit_within(3), hit_within(5)
        doc3 = any(norm(p.source_document) in want_docs for p in res.passages[:3])

        intent = item["intent"]
        for key, was_hit in (("exact@3", exact3), ("exact@5", exact5),
                             ("doc@3", doc3)):
            stats[intent][key] += int(was_hit)
            stats["ALL"][key] += int(was_hit)
        stats[intent]["n"] += 1
        stats["ALL"]["n"] += 1

        if not exact3:
            rows.append((item["id"], item["question"], item["expected_source_doc"],
                         item["expected_slide_or_section"], res.passages[:3],
                         exact5, doc3))

    out_dir = cfg.path("evaluation.results_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "retrieval_analysis.md"

    lines = ["# Retrieval analysis (retrieval only, no generation)\n"]
    lines.append(
        f"{stats['ALL']['n']} in-scope gold questions against "
        f"{retriever.manifest.get('n_chunks')} chunks from "
        f"{retriever.manifest.get('n_documents')} documents. "
        f"k={retriever.k}, threshold={retriever.threshold}.\n"
    )
    lines.append("| Intent | n | exact hit@3 | exact hit@5 | document hit@3 |")
    lines.append("|---|---|---|---|---|")
    for intent in [i for i in stats if i != "ALL"] + ["ALL"]:
        s = stats[intent]
        n = s["n"]
        lines.append(
            f"| `{intent}` | {n} | {s['exact@3']/n:.3f} | {s['exact@5']/n:.3f} "
            f"| {s['doc@3']/n:.3f} |"
        )

    lines.append("\n## Questions missing exact hit@3\n")
    lines.append("`@5` and `doc@3` show whether the passage was merely ranked "
                 "lower, or whether a different — often equally correct — "
                 "source was returned instead.\n")
    lines.append("| ID | Expected | In top-5? | Right doc in top-3? | Top-3 returned |")
    lines.append("|---|---|---|---|---|")
    for cid, _q, doc, loc, passages, e5, d3 in rows:
        got = "<br>".join(
            f"`{p.source_document}` / {p.slide_or_section} ({p.score:.3f})"
            for p in passages
        )
        lines.append(
            f"| `{cid}` | `{doc}` / {loc} | {'yes' if e5 else 'no'} | "
            f"{'yes' if d3 else 'no'} | {got} |"
        )

    with open(report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    s = stats["ALL"]
    n = s["n"]
    print(f"in-scope questions: {n}")
    print(f"  exact hit@3     {s['exact@3']/n:.3f}")
    print(f"  exact hit@5     {s['exact@5']/n:.3f}")
    print(f"  document hit@3  {s['doc@3']/n:.3f}")
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
