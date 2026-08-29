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


def acceptable_locations(item: dict) -> set[tuple[str, str]]:
    """Every (document, locator) that correctly answers this question.

    Usually one. Some questions the corpus genuinely answers in more than one
    place — two brochures state the venue, the assignment and project briefs
    carry near-identical Penalty sections — and those items list the
    alternates in `also_acceptable`. Each was verified by reading the chunk;
    the field records a property of the corpus, not a way to absorb misses.
    """
    out = {(normalise(item.get("expected_source_doc")),
            normalise(item.get("expected_slide_or_section")))}
    for doc, loc in item.get("also_acceptable", []):
        out.add((normalise(doc), normalise(loc)))
    return out


def hit_at_k(result: TurnResult, item: dict, k: int) -> bool:
    """True if an acceptable chunk appears in the top-k retrieved passages."""
    wanted = acceptable_locations(item)
    return any(
        (normalise(p.source_document), normalise(p.slide_or_section)) in wanted
        for p in result.passages[:k]
    )


def hit_at_k_document(result: TurnResult, item: dict, k: int) -> bool:
    """True if the expected *document* appears in the top-k, regardless of the
    slide/page within it.

    Reported alongside exact hit@k because the programme-wide corpus answers
    several questions in more than one place — two brochures state the venue,
    the assignment and project briefs each carry a "Penalty" section — so an
    exact-locator miss is often a differently-correct retrieval rather than a
    failure. See results/retrieval_analysis.md.
    """
    wanted = {d for d, _ in acceptable_locations(item)}
    return any(normalise(p.source_document) in wanted
               for p in result.passages[:k])


def citation_correct(result: TurnResult, item: dict) -> bool:
    """True if any emitted citation names an acceptable document and locator."""
    wanted = acceptable_locations(item)
    return any((normalise(d), normalise(s)) in wanted
               for d, s in result.citations)


def citation_doc_correct(result: TurnResult, item: dict) -> bool:
    """Looser check: right document, possibly a neighbouring slide."""
    wanted = {d for d, _ in acceptable_locations(item)}
    return any(normalise(d) in wanted for d, _ in result.citations)


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


def _normalise_for_overlap(text: str) -> str:
    """Flatten typography and markdown before comparing answer to context.

    Strong models write "k‑means" with a non-breaking hyphen, "x₁" with
    subscripts and "**bold**" emphasis; the corpus contains "k-means", "x1"
    and plain text. Comparing raw strings therefore scores a perfectly
    grounded answer as unsupported. Measured: this alone drove the lexical
    fallback's groundedness verdict from 0.59 to a realistic figure.
    """
    import unicodedata
    text = unicodedata.normalize("NFKD", text)
    # Unicode dashes/quotes/spaces -> ASCII equivalents
    for src, dst in (("\u2010", "-"), ("\u2011", "-"), ("\u2012", "-"),
                     ("\u2013", "-"), ("\u2014", "-"), ("\u2018", "'"),
                     ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'),
                     ("\u00a0", " "), ("\u202f", " ")):
        text = text.replace(src, dst)
    # Drop combining marks left by NFKD, and markdown emphasis
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.replace("*", " ").replace("`", " ").replace("_", " ").lower()


def lexical_groundedness(answer: str, result: TurnResult) -> bool:
    """Fallback judge: are the answer's content words present in the context?

    Deliberately crude, and only used when the LLM judge is unavailable. Its
    verdicts are counted separately in the report, because a run dominated by
    this fallback is much weaker evidence than one judged by a model.
    """
    import re
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "and",
        "or", "for", "on", "it", "this", "that", "with", "as", "by", "be",
        "which", "from", "at", "can", "you", "your", "its", "they", "their",
        "has", "have", "not", "but", "also", "these", "those", "than", "then",
        "into", "such", "used", "use", "using", "there", "when", "what",
        "source", "slide", "section",
    }
    context = _normalise_for_overlap(" ".join(p.text for p in result.passages))
    words = [
        w for w in re.split(r"\W+", _normalise_for_overlap(answer))
        if len(w) > 3 and w not in stop
    ]
    if not words:
        return True
    covered = sum(1 for w in words if w in context)
    return (covered / len(words)) >= 0.65


def judge_groundedness(bot: ModuleMate, result: TurnResult,
                       judge_model: str,
                       judge_fallbacks: list[str] | None = None
                       ) -> tuple[bool, str]:
    """LLM-as-judge faithfulness check, with a lexical fallback.

    Judged by a different model from the one that wrote the answer, so the
    generator is not marking its own work.
    """
    judge_fallbacks = judge_fallbacks or []
    context = "\n\n".join(
        f"--- PASSAGE {i} ---\n{p.text}"
        for i, p in enumerate(result.passages, start=1)
    )
    user = f"CONTEXT:\n{context}\n\nANSWER:\n{result.answer}\n\nVerdict:"
    # The judge needs the same fallback chain as generation: it exhausted its
    # own daily allowance mid-run, silently handing 31 of 49 verdicts to the
    # crude lexical fallback and making groundedness look far worse than it
    # was. A metric that degrades silently is worse than one that fails loudly.
    for candidate in [judge_model, *judge_fallbacks]:
        try:
            # Generous token budget: reasoning-style models spend tokens
            # before emitting the verdict, and a tight cap returns empty.
            text, _, _ = bot.generator.backend.complete(
                JUDGE_SYSTEM, user, candidate, 0.0, 600
            )
        except Exception as exc:
            from generate import is_capacity_error
            if is_capacity_error(exc):
                continue
            break
        verdict = text.strip().upper()
        if "UNGROUNDED" in verdict:
            return False, f"llm:{candidate}"
        if "GROUNDED" in verdict:
            return True, f"llm:{candidate}"
    return lexical_groundedness(result.answer, result), "lexical"


def confusion_matrix(labels: list[str], truth: list[str],
                     pred: list[str]) -> list[list[int]]:
    idx = {label: i for i, label in enumerate(labels)}
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
    judge_model = cfg.get("evaluation.judge_model") or cfg.require("generation.model")
    judge_fallbacks = list(cfg.get("evaluation.judge_fallback_models") or [])
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
        # suggest=False: C4 costs an extra call per turn and would
        # inflate measured latency with work no metric reads.
        result = bot.ask(item["question"], history=None,
                         rewrite=False, suggest=False)
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
            "model_used": result.model_used,
        }

        if not is_oos:
            rec["hit_at_k"] = hit_at_k(result, item, k)
            rec["hit_at_k_document"] = hit_at_k_document(result, item, k)
            # The generator is shown all k_retrieval passages, so whether an
            # answer is *possible* depends on hit@k_retrieval, not hit@3.
            # hit@3 remains the NFR2 metric; this is the diagnostic that
            # explains a correct answer whose expected passage ranked 4th.
            rec["hit_at_retrieval_k"] = hit_at_k(
                result, item, cfg.require("retrieval.k")
            )
            rec["expected"] = (
                f"{item['expected_source_doc']} | "
                f"{item['expected_slide_or_section']}"
            )
            if not result.is_refusal:
                rec["citation_correct"] = citation_correct(result, item)
                rec["citation_doc_correct"] = citation_doc_correct(result, item)
                rec["fabricated"] = fabricated_citations(result, known)
                grounded, how = (
                    judge_groundedness(bot, result, judge_model, judge_fallbacks)
                    if use_judge
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
        f"hit_at_{k}_document": mean(
            r.get("hit_at_k_document", False) for r in in_scope
        ),
        "hit_at_retrieval_k": mean(
            r.get("hit_at_retrieval_k", False) for r in in_scope
        ),
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
        "groundedness_llm_judged": mean(
            r.get("grounded", False) for r in answered
            if str(r.get("grounded_by", "")).startswith("llm")
        ),
        "judge_methods": {
            m: sum(1 for r in answered if r.get("grounded_by") == m)
            for m in sorted({r.get("grounded_by") for r in answered
                             if r.get("grounded_by")})
        },
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
        "models_used": {
            m: sum(1 for r in records if r.get("model_used") == m)
            for m in sorted({r.get("model_used") for r in records if r.get("model_used")})
        },
        "refusal_reasons": {
            reason: sum(1 for r in records if r["refusal_reason"] == reason)
            for reason in ("intent", "threshold", "model")
        },
        "index_manifest": bot.manifest,
        "config": {
            "k": cfg.require("retrieval.k"),
            "similarity_threshold": cfg.require("retrieval.similarity_threshold"),
            "generation_model": cfg.require("generation.model"),
            "judge_model": cfg.get("evaluation.judge_model"),
            "rewrite_model": cfg.get("generation.rewrite_model"),
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
