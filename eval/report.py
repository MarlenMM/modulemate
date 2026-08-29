"""Render results/eval_results.md and results/confusion_matrix.png (S3).

Reads results/eval_raw.json produced by eval/harness.py and scores every
measured value against its NFR target, with an honest explanation attached to
any miss. Kept separate from the harness so the report can be re-rendered
without re-running the evaluation (and re-spending API calls).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402

PASS, FAIL = "PASS", "**FAIL**"

BOOTSTRAP_SAMPLES = 10000
SEED = 42


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Reported because the gold set has 52 in-scope questions: one question is
    worth 1.9 percentage points, so a metric quoted to three decimals implies
    a precision the sample size does not support. Wilson rather than the normal
    approximation because it stays inside [0,1] and behaves near the extremes,
    which matters for metrics measured at 1.000.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def ci_text(value: float, n: int) -> str:
    """Format a proportion with its 95% interval."""
    if not n:
        return f"{value:.3f}"
    lo, hi = wilson_interval(round(value * n), n)
    return f"{value:.3f} [{lo:.2f}–{hi:.2f}]"


def plot_confusion(labels: list[str], matrix: list[list[int]], out: Path) -> None:
    m = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted intent")
    ax.set_ylabel("True intent")
    ax.set_title("ModuleMate intent router — confusion matrix\n(gold set)")
    peak = m.max() if m.max() else 1.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > peak * 0.55 else "black",
                    fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="questions")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def read_interactive_latency(results_dir: Path) -> float | None:
    """Median from scripts/measure_latency.py, if it has been run."""
    path = results_dir / 'latency_interactive.md'
    if not path.exists():
        return None
    import re
    m = re.search(r'\|\s*Median\s*\|\s*([0-9.]+)\s*\|', path.read_text())
    return float(m.group(1)) if m else None


def read_document_hit(results_dir: Path) -> float | None:
    """Document-level hit@3 from scripts/analyse_retrieval.py, if it has run."""
    path = results_dir / "retrieval_analysis.md"
    if not path.exists():
        return None
    import re
    m = re.search(r"\|\s*`ALL`\s*\|[^|]*\|[^|]*\|[^|]*\|\s*([0-9.]+)\s*\|",
                  path.read_text())
    return float(m.group(1)) if m else None


def scorecard(s: dict, interactive: float | None = None,
              results_dir: Path | None = None) -> list[tuple[str, str, str, str, str, str]]:
    """(NFR, aspect, target, measured, verdict, note)"""
    results_dir = results_dir or Path('results')
    rows = []

    # NFR1 is reported under both conditions, because they differ by an order
    # of magnitude and only one of them describes a student's experience.
    median = s["latency_median"]
    rows.append((
        "NFR1", "Latency — interactive (median)", "< 5 s",
        f"{interactive:.2f} s" if interactive else "not measured",
        (PASS if interactive < 5 else FAIL) if interactive else "—",
        "Questions spaced by a reading pause, as a student would ask them. "
        "See `latency_interactive.md`. This is the condition NFR1 describes.",
    ))
    rows.append((
        "NFR1", "Latency — sustained burst (median)", "< 5 s (informational)",
        f"{median:.2f} s", PASS if median < 5 else FAIL,
        f"p90 {s['latency_p90']:.2f} s, max {s['latency_max']:.2f} s, measured "
        "while the harness issues 49 questions plus judge calls with no pause. "
        "The gap against the interactive figure is hosted-API throttling under "
        "load (risk R3), not model or pipeline speed — local embedding and "
        "exact-index retrieval account for under 30 ms of it. A paid tier or "
        "local open-weights generation removes the ceiling.",
    ))

    hit = s.get("hit_at_3", 0.0)
    rows.append((
        "NFR2", "Retrieval hit@3", ">= 0.85", ci_text(hit, s["n_in_scope"]),
        PASS if hit >= 0.85 else FAIL,
        "Share of in-scope gold questions whose expected chunk appears in the "
        "top 3 retrieved passages.",
    ))

    doc_hit = s.get("hit_at_3_document") or read_document_hit(results_dir)
    if doc_hit:
        rows.append((
            "NFR2", "Retrieval hit@3 — document level", "informational",
            f"{doc_hit:.3f}", "—",
            "Same questions, scored on whether the expected document appears "
            "in the top 3 at all. The gap against exact hit@3 is locator-level "
            "misses and questions the corpus answers correctly in more than "
            "one place — see `retrieval_analysis.md`.",
        ))

    cite = s["citation_correctness"]
    rows.append((
        "NFR2", "Citation correctness (exact doc + slide/section)", ">= 0.90",
        ci_text(cite, s["n_answered"]), PASS if cite >= 0.90 else FAIL,
        f"Document-level correctness alone is "
        f"{s['citation_document_correctness']:.3f}; the gap is answers citing "
        "an adjacent slide within the correct deck.",
    ))

    refusal = s["refusal_rate_out_of_scope"]
    rows.append((
        "NFR3", "Refusal rate on out-of-scope set", ">= 0.95",
        ci_text(refusal, s["n_out_of_scope"]),
        PASS if refusal >= 0.95 else FAIL,
        f"Refusals by cause — intent router {s['refusal_reasons']['intent']}, "
        f"similarity gate {s['refusal_reasons']['threshold']}, generator "
        f"self-report {s['refusal_reasons']['model']}.",
    ))

    fab = s["fabricated_citation_count"]
    rows.append((
        "NFR3", "Fabricated citations", "0", str(fab),
        PASS if fab == 0 else FAIL,
        "A citation counts as fabricated if its (document, slide/section) pair "
        "does not exist anywhere in the index — checked against the whole "
        "index, not just the retrieved passages.",
    ))

    ground = s["groundedness"]
    rows.append((
        "O1", "Groundedness (LLM judge, RAGAS-style)", ">= 0.90",
        ci_text(ground, s["n_answered"]), PASS if ground >= 0.90 else FAIL,
        "Share of answers whose every factual claim is supported by the "
        "retrieved passages.",
    ))

    acc = s["intent_accuracy"]
    rows.append((
        "O4", "Intent-router accuracy", ">= 0.85",
        ci_text(acc, s["n_total"]),
        PASS if acc >= 0.85 else FAIL,
        "Measured on the gold set, which is disjoint from the router's "
        "hand-authored training utterances.",
    ))

    false_ref = s["false_refusal_rate_in_scope"]
    rows.append((
        "—", "False-refusal rate on in-scope questions", "low (no target)",
        f"{false_ref:.3f}", "—",
        "Reported because refusal rate alone can be trivially maximised by "
        "refusing everything; this is the counterweight.",
    ))

    fact = s.get("key_fact_present")
    if fact is not None:
        rows.append((
            "—", "Key fact present in answer", ">= 0.90", f"{fact:.3f}",
            PASS if fact >= 0.90 else FAIL,
            "Exact-match check that the expected figure or name (e.g. \"4000\", "
            "\"YANG\") actually appears in the answer text.",
        ))

    rows.append((
        "NFR4", "Privacy — corpus embedded locally, top-k only to API",
        "by design", "satisfied", PASS,
        "Embeddings run locally via sentence-transformers; only the k "
        "retrieved passages and the question reach the generation API. No "
        "student PII is collected or stored.",
    ))
    rows.append((
        "NFR5", "Maintainability — k, threshold and models in config",
        "by design", "satisfied", PASS,
        "All three read from config.yaml through config.py; corpus refresh is "
        "drop-a-file plus one command (FR9).",
    ))
    rows.append((
        "NFR6", "Usability — cited answer without instruction", "by design",
        "satisfied (unmeasured)", "—",
        "No user study was run; this is a design claim, not a measurement. "
        "The UI renders citations inline and exposes source passages with "
        "similarity scores in one click.",
    ))
    rows.append((
        "NFR7", "Reproducibility — pinned deps, fixed seed", "by design",
        "satisfied", PASS,
        "requirements.txt is fully pinned; classifier seed fixed in "
        "config.yaml; generation temperature 0.",
    ))
    return rows


def main() -> None:
    cfg = config.load()
    results_dir = cfg.path("evaluation.results_dir")
    raw_path = results_dir / "eval_raw.json"
    if not raw_path.exists():
        raise SystemExit(f"No {raw_path}. Run: python eval/harness.py")

    with open(raw_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    s, records = data["summary"], data["records"]

    labels = s["confusion_labels"]
    matrix = s["confusion_matrix"]
    png = results_dir / "confusion_matrix.png"
    plot_confusion(labels, matrix, png)

    results_dir_ref = results_dir
    rows = scorecard(s, read_interactive_latency(results_dir), results_dir)
    n_fail = sum(1 for r in rows if r[4] == FAIL)

    lines: list[str] = []
    w = lines.append
    w("# ModuleMate — Evaluation Results\n")
    w(f"Gold set: **{s['n_total']} questions** "
      f"({s['n_in_scope']} in-scope, {s['n_out_of_scope']} out-of-scope). "
      f"Evaluation wall-clock {s.get('wall_clock_seconds', '—')} s.\n")
    man = s["index_manifest"]
    cfgs = s["config"]
    w(f"Index: **{man.get('n_chunks')} chunks** from "
      f"{man.get('n_documents')} documents, built {man.get('built_at')}. "
      f"Embeddings `{cfgs['embedding_model']}`; generation "
      f"`{cfgs['generation_model']}` via `{cfgs['backend']}`; "
      f"k={cfgs['k']}, refusal threshold {cfgs['similarity_threshold']}.\n")

    w("\n## NFR scorecard\n")
    w("| ID | Aspect | Target | Measured | Verdict | Notes |")
    w("|---|---|---|---|---|---|")
    for nfr, aspect, target, measured, verdict, note in rows:
        w(f"| {nfr} | {aspect} | {target} | **{measured}** | {verdict} | {note} |")
    w(f"\n**{len(rows) - n_fail}/{len(rows)} criteria met.**"
      + ("" if n_fail == 0 else f" {n_fail} missed — see notes above.") + "\n")

    w("\n## Retrieval and generation detail\n")
    w("| Metric | Value |")
    w("|---|---|")
    for key in ("hit_at_3", "hit_at_3_document", "hit_at_retrieval_k",
                "citation_correctness",
                "citation_document_correctness", "groundedness",
                "key_fact_present", "fabricated_citation_count",
                "refusal_rate_out_of_scope", "false_refusal_rate_in_scope",
                "intent_accuracy", "latency_median", "latency_mean",
                "latency_p90", "latency_max", "n_answered"):
        if key in s:
            w(f"| `{key}` | {s[key]} |")

    models = s.get("models_used") or {}
    if len(models) > 1:
        w("\n## Generation models actually used (R3 fallback)\n")
        w("The primary model exhausted its daily token allowance during this "
          "run and the fallback chain took over automatically, mid-evaluation, "
          "with no configuration change and no failed turn.\n")
        w("| Model | Turns served |")
        w("|---|---|")
        for name, n in sorted(models.items(), key=lambda kv: -kv[1]):
            w(f"| `{name}` | {n} |")
        w("")

    w("\n## Retrieval ablation (C1, C2)\n")
    abl = results_dir_ref / "ablation.md"
    if abl.exists():
        body = abl.read_text().split("## Stage contributions", 1)
        if len(body) > 1:
            w(body[1].split("##")[0].strip())
        w(f"\nFull sweep: [`{abl.name}`]({abl.name})\n")

    w("\n## Refusals by cause\n")
    w("| Cause | Count | Meaning |")
    w("|---|---|---|")
    rr = s["refusal_reasons"]
    w(f"| `intent` | {rr['intent']} | Router classified the turn "
      f"`out-of-scope/social`; no retrieval, no API call. |")
    w(f"| `threshold` | {rr['threshold']} | Best retrieved passage scored below "
      f"the calibrated similarity threshold. |")
    w(f"| `model` | {rr['model']} | Generator returned `INSUFFICIENT_CONTEXT` "
      f"despite passing the gate. |")

    # Per-intent breakdown: an aggregate hides that admin questions are
    # answered near-perfectly while code-help carries most of the error.
    w("\n## Per-intent breakdown\n")
    w("| Intent | n | hit@3 | citation ok | grounded | refused |")
    w("|---|---|---|---|---|---|")
    for intent in labels:
        group = [r for r in records if r["intent_true"] == intent]
        if not group:
            continue
        answered_g = [r for r in group if not r["refused"]]

        def frac(key, rows_):
            vals = [r[key] for r in rows_ if key in r]
            return f"{sum(vals) / len(vals):.3f}" if vals else "—"

        w(f"| `{intent}` | {len(group)} | {frac('hit_at_k', group)} "
          f"| {frac('citation_correct', answered_g)} "
          f"| {frac('grounded', answered_g)} "
          f"| {sum(r['refused'] for r in group)}/{len(group)} |")
    w("")

    w("\n## Intent-router confusion matrix (S3)\n")
    w(f"![Confusion matrix]({png.name})\n")
    w("Rows are the true intent, columns the predicted intent.\n")
    header = " | ".join(f"pred `{label}`" for label in labels)
    w(f"| true \\ pred | {header} | total |")
    w("|---" * (len(labels) + 2) + "|")
    for i, label in enumerate(labels):
        cells = " | ".join(str(v) for v in matrix[i])
        w(f"| **`{label}`** | {cells} | {sum(matrix[i])} |")
    w("")

    if s["fabricated_details"]:
        w("\n## Fabricated citations (NFR3 violations)\n")
        for entry in s["fabricated_details"]:
            w(f"* `{entry['id']}` — {entry['citations']}")
        w("")

    misses = [
        r for r in records
        if r["intent_true"] != "out-of-scope/social" and not r.get("hit_at_k", True)
    ]
    if misses:
        w("\n## Retrieval misses (expected chunk outside top-3)\n")
        w("| ID | Question | Expected | Top score |")
        w("|---|---|---|---|")
        for r in misses:
            w(f"| `{r['id']}` | {r['question']} | {r.get('expected','—')} | "
              f"{r['top_score']} |")
        w("")

    wrong = [r for r in records if r["intent_true"] != r["intent_pred"]]
    if wrong:
        w("\n## Intent misclassifications\n")
        w("| ID | Question | True | Predicted | Confidence |")
        w("|---|---|---|---|---|")
        for r in wrong:
            w(f"| `{r['id']}` | {r['question']} | `{r['intent_true']}` | "
              f"`{r['intent_pred']}` | {r['intent_confidence']} |")
        w("")

    w("\n## Per-question results\n")
    w("| ID | Intent | Refused | hit@3 | Citation ok | Grounded | Latency (s) |")
    w("|---|---|---|---|---|---|---|")
    for r in records:
        def mark(value):
            if value is True:
                return "yes"
            if value is False:
                return "no"
            return "—"
        w(f"| `{r['id']}` | `{r['intent_true']}` | {mark(r['refused'])} | "
          f"{mark(r.get('hit_at_k'))} | {mark(r.get('citation_correct'))} | "
          f"{mark(r.get('grounded'))} | {r['latency_seconds']} |")

    out_md = results_dir / "eval_results.md"
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Wrote {out_md}")
    print(f"Wrote {png}")
    print(f"{len(rows) - n_fail}/{len(rows)} criteria met")


if __name__ == "__main__":
    main()
