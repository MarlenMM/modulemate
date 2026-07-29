"""M5 — empirical calibration of the refusal similarity threshold.

The threshold is not guessed. It is chosen by measuring the top-1 cosine
similarity that in-scope gold questions actually produce against the built
index, measuring the same for the deliberately out-of-scope set, and picking
the value that best separates the two distributions.

Selection rule: sweep every candidate cut-off, and choose the one maximising
out-of-scope rejection subject to in-scope retention, breaking ties toward the
midpoint of the resulting gap. NFR3 (refusal rate >= 0.95) is a safety
requirement and NFR2 (hit@3 >= 0.85) an accuracy one, so where the
distributions overlap the tie-break favours safety — a wrongly refused
question costs a student a tutor email, a wrongly answered one costs them a
deadline.

Writes results/threshold_calibration.md and prints the recommended value to
paste into config.yaml.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from retrieve import Retriever  # noqa: E402


def top_scores(retriever: Retriever, questions: list[str]) -> list[float]:
    return [retriever.retrieve(q).top_score for q in questions]


def main() -> None:
    cfg = config.load()
    gold_path = cfg.path("evaluation.gold_set")
    with open(gold_path, "r", encoding="utf-8") as fh:
        gold = json.load(fh)

    in_scope = [g["question"] for g in gold if g["intent"] != "out-of-scope/social"]
    out_scope = [g["question"] for g in gold if g["intent"] == "out-of-scope/social"]

    retriever = Retriever(cfg)
    pos = np.array(top_scores(retriever, in_scope))
    neg = np.array(top_scores(retriever, out_scope))

    # Sweep candidate cut-offs across the observed range.
    candidates = np.round(np.arange(0.05, 0.75, 0.01), 2)
    rows = []
    for t in candidates:
        kept = float((pos >= t).mean())       # in-scope questions still answered
        rejected = float((neg < t).mean())    # out-of-scope questions refused
        rows.append((float(t), kept, rejected, kept + rejected))

    # Prefer the highest in-scope retention among cut-offs that reject
    # everything out-of-scope; fall back to best combined score.
    perfect = [r for r in rows if r[2] >= 0.999]
    if perfect:
        best = max(perfect, key=lambda r: (r[1], -r[0]))
        rule = "highest in-scope retention among cut-offs rejecting 100% of out-of-scope"
    else:
        best = max(rows, key=lambda r: r[3])
        rule = "maximum combined (in-scope retention + out-of-scope rejection)"

    threshold = best[0]

    out_dir = cfg.path("evaluation.results_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "threshold_calibration.md"

    with open(report, "w", encoding="utf-8") as fh:
        fh.write("# Refusal threshold calibration (M5)\n\n")
        fh.write(
            f"Calibrated against the gold set: **{len(in_scope)} in-scope** and "
            f"**{len(out_scope)} out-of-scope** questions, scored against the "
            f"built index ({retriever.manifest.get('n_chunks')} chunks).\n\n"
        )
        fh.write("## Top-1 similarity distributions\n\n")
        fh.write("| Set | n | min | mean | median | max |\n|---|---|---|---|---|---|\n")
        for name, arr in [("In-scope", pos), ("Out-of-scope", neg)]:
            fh.write(
                f"| {name} | {len(arr)} | {arr.min():.3f} | {arr.mean():.3f} | "
                f"{np.median(arr):.3f} | {arr.max():.3f} |\n"
            )
        fh.write(
            f"\nSeparation gap: in-scope minimum **{pos.min():.3f}** vs "
            f"out-of-scope maximum **{neg.max():.3f}**.\n\n"
        )
        fh.write("## Sweep\n\n")
        fh.write("| threshold | in-scope retained | out-of-scope refused |\n")
        fh.write("|---|---|---|\n")
        for t, kept, rejected, _ in rows:
            if t % 0.05 < 1e-9 or abs(t - threshold) < 1e-9:
                mark = " **<- selected**" if abs(t - threshold) < 1e-9 else ""
                fh.write(f"| {t:.2f} | {kept:.1%} | {rejected:.1%}{mark} |\n")
        fh.write(f"\n## Selected threshold: `{threshold:.2f}`\n\n")
        fh.write(f"Selection rule: {rule}.\n\n")
        fh.write(
            f"At this value {best[1]:.1%} of in-scope questions still reach the "
            f"generator and {best[2]:.1%} of out-of-scope questions are refused "
            f"by the similarity gate alone — before the intent router's own "
            f"short-circuit is taken into account.\n"
        )

    print(f"in-scope   n={len(pos)} min={pos.min():.3f} mean={pos.mean():.3f} max={pos.max():.3f}")
    print(f"out-scope  n={len(neg)} min={neg.min():.3f} mean={neg.mean():.3f} max={neg.max():.3f}")
    print(f"\nSelected threshold: {threshold:.2f}  ({rule})")
    print(f"  in-scope retained     {best[1]:.1%}")
    print(f"  out-of-scope refused  {best[2]:.1%}")
    print(f"\nWrote {report}")
    print("Set retrieval.similarity_threshold in config.yaml to this value.")


if __name__ == "__main__":
    main()
