"""M8 — Gold-set evaluation harness (FR10, O6).

Drives the same `ModuleMate.ask` path the Streamlit UI uses, so measured
behaviour is the deployed behaviour. Reports, against the NFR targets:

  hit@k                 did retrieval surface the expected chunk in the top k?
  citation correctness  does the parsed citation bracket name the expected
                        document and slide/section? (NFR2 >= 0.90)
  fabricated citations  does any citation name a (document, locator) pair that
                        does not exist in the index? (NFR3 requires zero)
  groundedness          is every claim supported by the retrieved passages?
                        Judged by the LLM, RAGAS-style [16], with a lexical
                        fallback if the judge call fails.
  refusal rate          share of out-of-scope questions refused (NFR3 >= 0.95)
  false refusal rate    share of in-scope questions wrongly refused
  intent accuracy       router accuracy + confusion matrix (S3)
  latency               end-to-end per turn (NFR1 median < 5 s)

Usage:  python eval/harness.py  [--no-judge] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from generate import parse_citations  # noqa: E402
from goldset import OUT_OF_SCOPE, load_gold  # noqa: E402
from pipeline import ModuleMate, TurnResult  # noqa: E402

JUDGE_SYSTEM = """\
You judge whether an ANSWER is fully supported by the supplied CONTEXT passages.

Reply with exactly one word:
GROUNDED   — every factual claim in the ANSWER is stated in or directly follows
             from the CONTEXT.
UNGROUNDED — the ANSWER contains at least one factual claim absent from the
             CONTEXT, or contradicts it.

Ignore style, completeness and helpfulness. Judge support only. Citation
brackets are metadata, not claims — do not judge them.
"""


# --------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------

def normalise(value: str | None) -> str:
    return " ".join((value or "").split()).strip().lower()


def hit_at_k(result: TurnResult, item: dict, k: int) -> bool:
    """True if the expected chunk appears in the top-k retrieved passages."""
    want_doc = normalise(item.get("expected_source_doc"))
    want_loc = normalise(item.get("expected_slide_or_section"))
    for p in result.passages[:k]:
        if normalise(p.source_document) == want_doc and \
           normalise(p.slide_or_section) == want_loc:
            return True
    return False


def citation_correct(result: TurnResult, item: dict) -> bool:
    """True if any emitted citation names the expected document and locator."""
    want_doc = normalise(item.get("expected_source_doc"))
    want_loc = normalise(item.get("expected_slide_or_section"))
    return any(
        normalise(d) == want_doc and normalise(s) == want_loc
        for d, s in result.citations
    )


def citation_doc_correct(result: TurnResult, item: dict) -> bool:
    """Looser check: right document, possibly a neighbouring slide."""
    want_doc = normalise(item.get("expected_source_doc"))
    return any(normalise(d) == want_doc for d, _ in result.citations)


def fabricated_citations(result: TurnResult, known: set[tuple[str, str]]) -> list:
    """Citations naming a (document, locator) pair absent from the index.

    This is the NFR3 'zero fabricated citations' check, and it is deliberately
    checked against the whole index rather than against the retrieved
    passages — a citation to a real slide the model was not shown is still a
    fabrication of provenance.
    """
    return [
        (d, s) for d, s in result.citations
        if (normalise(d), normalise(s)) not in known
    ]


def lexical_groundedness(answer: str, result: TurnResult) -> bool:
    """Fallback judge: are the answer's content words present in the context?"""
    import re
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "and",
        "or", "for", "on", "it", "this", "that", "with", "as", "by", "be",
        "which", "from", "at", "can", "you", "your", "its", "they", "their",
        "has", "have", "not", "but", "also", "these", "those", "than", "then",
        "into", "such", "used", "use", "using", "there", "when", "what",
        "source", "slide", "section",
    }
    context = " ".join(p.text for p in result.passages).lower()
    words = [
        w for w in re.split(r"\W+", answer.lower())
        if len(w) > 3 and w not in stop
    ]
    if not words:
        return True
    covered = sum(1 for w in words if w in context)
    return (covered / len(words)) >= 0.65


def judge_groundedness(bot: ModuleMate, result: TurnResult) -> tuple[bool, str]:
    """LLM-as-judge faithfulness check, with a lexical fallback."""
    context = "\n\n".join(
        f"--- PASSAGE {i} ---\n{p.text}"
        for i, p in enumerate(result.passages, start=1)
    )
    user = f"CONTEXT:\n{context}\n\nANSWER:\n{result.answer}\n\nVerdict:"
    try:
        text, _, _ = bot.generator.backend.complete(
            JUDGE_SYSTEM, user, bot.generator.model, 0.0, 10
        )
        verdict = text.strip().upper()
        if "UNGROUNDED" in verdict:
            return False, "llm"
        if "GROUNDED" in verdict:
            return True, "llm"
    except Exception:
        pass
    return lexical_groundedness(result.answer, result), "lexical"


def confusion_matrix(labels: list[str], truth: list[str],
                     pred: list[str]) -> list[list[int]]:
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for t, p in zip(truth, pred):
        if t in idx and p in idx:
            m[idx[t]][idx[p]] += 1
    return m


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------

def run(limit: int | None = None, use_judge: bool = True) -> dict:
    cfg = config.load()
    k = cfg.require("evaluation.hit_at_k")
    items = load_gold(cfg.path("evaluation.gold_set"))
    if limit:
        items = items[:limit]

    bot = ModuleMate(cfg)
    known = {
        (normalise(c["source_document"]), normalise(c["slide_or_section"]))
        for c in bot.retriever.chunks
    }

    records: list[dict] = []
    for n, item in enumerate(items, start=1):
        started = time.time()
        result = bot.ask(item["question"], history=None, rewrite=False)
        elapsed = time.time() - started

        is_oos = item["intent"] == OUT_OF_SCOPE
        rec: dict = {
            "id": item["id"],
            "question": item["question"],
            "intent_true": item["intent"],
            "intent_pred": result.intent.label,
            "intent_confidence": round(result.intent.confidence, 3),
            "refused": result.is_refusal,
            "refusal_reason": result.refusal_reason,
            "top_score": round(result.top_score, 4),
            "latency_seconds": round(elapsed, 3),
            "answer": result.answer,
            "citations": result.citations,
        }

        if not is_oos:
            rec["hit_at_k"] = hit_at_k(result, item, k)
            rec["expected"] = (
                f"{item['expected_source_doc']} | "
                f"{item['expected_slide_or_section']}"
            )
            if not result.is_refusal:
                rec["citation_correct"] = citation_correct(result, item)
                rec["citation_doc_correct"] = citation_doc_correct(result, item)
                rec["fabricated"] = fabricated_citations(result, known)
                grounded, how = (
                    judge_groundedness(bot, result) if use_judge
                    else (lexical_groundedness(result.answer, result), "lexical")
                )
                rec["grounded"] = grounded
                rec["grounded_by"] = how
                wanted = item.get("expected_answer_contains") or []
                if wanted:
                    rec["fact_present"] = any(
                        w.lower() in result.answer.lower() for w in wanted
                    )
        records.append(rec)
        flag = "REFUSED" if result.is_refusal else "answered"
        print(f"  [{n:2d}/{len(items)}] {item['id']}  {flag:8s} "
              f"{elapsed:5.2f}s  {item['question'][:52]}")

    return summarise(records, cfg, bot, k)


def summarise(records: list[dict], cfg, bot: ModuleMate, k: int) -> dict:
    in_scope = [r for r in records if r["intent_true"] != OUT_OF_SCOPE]
    out_scope = [r for r in records if r["intent_true"] == OUT_OF_SCOPE]
    answered = [r for r in in_scope if not r["refused"]]

    def mean(values) -> float:
        values = list(values)
        return round(sum(values) / len(values), 4) if values else 0.0

    latencies = [r["latency_seconds"] for r in records]
    labels = cfg.require("intents.labels")
    truth = [r["intent_true"] for r in records]
    pred = [r["intent_pred"] for r in records]

    all_fabricated = [
        {"id": r["id"], "citations": r["fabricated"]}
        for r in answered if r.get("fabricated")
    ]

    summary = {
        "n_total": len(records),
        "n_in_scope": len(in_scope),
        "n_out_of_scope": len(out_scope),
        "n_answered": len(answered),
        f"hit_at_{k}": mean(r["hit_at_k"] for r in in_scope),
        "citation_correctness": mean(
            r.get("citation_correct", False) for r in answered
        ),
        "citation_document_correctness": mean(
            r.get("citation_doc_correct", False) for r in answered
        ),
        "fabricated_citation_count": sum(
            len(r["fabricated"]) for r in answered if r.get("fabricated")
        ),
        "fabricated_details": all_fabricated,
        "groundedness": mean(r.get("grounded", False) for r in answered),
        "key_fact_present": mean(
            r["fact_present"] for r in answered if "fact_present" in r
        ),
        "refusal_rate_out_of_scope": mean(r["refused"] for r in out_scope),
        "false_refusal_rate_in_scope": mean(r["refused"] for r in in_scope),
        "intent_accuracy": mean(t == p for t, p in zip(truth, pred)),
        "latency_median": round(statistics.median(latencies), 3),
        "latency_mean": round(statistics.fmean(latencies), 3),
        "latency_p90": round(sorted(latencies)[int(0.9 * len(latencies)) - 1], 3),
        "latency_max": round(max(latencies), 3),
        "confusion_labels": labels,
        "confusion_matrix": confusion_matrix(labels, truth, pred),
        "refusal_reasons": {
            reason: sum(1 for r in records if r["refusal_reason"] == reason)
            for reason in ("intent", "threshold", "model")
        },
        "index_manifest": bot.manifest,
        "config": {
            "k": cfg.require("retrieval.k"),
            "similarity_threshold": cfg.require("retrieval.similarity_threshold"),
            "generation_model": cfg.require("generation.model"),
            "embedding_model": cfg.require("embedding.model_name"),
            "backend": cfg.require("generation.backend"),
        },
    }
    return {"summary": summary, "records": records}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the LLM groundedness judge (lexical only)")
    args = ap.parse_args()

    cfg = config.load()
    print("Running ModuleMate gold-set evaluation…\n")
    started = time.time()
    out = run(limit=args.limit, use_judge=not args.no_judge)
    out["summary"]["wall_clock_seconds"] = round(time.time() - started, 1)

    results_dir = cfg.path("evaluation.results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "eval_raw.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    s = out["summary"]
    print("\n" + "=" * 62)
    for key in (
        "n_total", "n_in_scope", "n_out_of_scope", "n_answered",
        "hit_at_3", "citation_correctness", "citation_document_correctness",
        "fabricated_citation_count", "groundedness", "key_fact_present",
        "refusal_rate_out_of_scope", "false_refusal_rate_in_scope",
        "intent_accuracy", "latency_median", "latency_p90",
    ):
        if key in s:
            print(f"  {key:32s} {s[key]}")
    print("=" * 62)
    print(f"\nWrote {results_dir / 'eval_raw.json'}")
    print("Now run:  python eval/report.py")


if __name__ == "__main__":
    main()
