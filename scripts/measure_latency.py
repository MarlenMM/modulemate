"""NFR1 — interactive latency probe.

The gold-set harness fires 49 questions back to back and so measures latency
under sustained burst, where the hosted API throttles and queues. That is the
correct number to report for the harness, but it is not what a student
experiences: real use is one question, then reading time, then another.

This probe measures the interactive case — requests spaced by a pause — so
NFR1 can be reported honestly against both conditions rather than against
whichever is more flattering.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from goldset import load_gold, split_scope  # noqa: E402
from pipeline import ModuleMate  # noqa: E402

GAP_SECONDS = 12   # approximate reading time between student questions
N_QUESTIONS = 10


def main() -> None:
    cfg = config.load()
    items, _ = split_scope(load_gold(cfg.path("evaluation.gold_set")))
    # Spread across intents rather than taking the first N (all admin).
    sample = items[::max(1, len(items) // N_QUESTIONS)][:N_QUESTIONS]

    bot = ModuleMate(cfg)
    bot.ask("warm up the model cache", history=None, rewrite=False)

    latencies: list[float] = []
    for n, item in enumerate(sample, start=1):
        if n > 1:
            time.sleep(GAP_SECONDS)
        started = time.time()
        result = bot.ask(item["question"], history=None, rewrite=False)
        elapsed = time.time() - started
        latencies.append(elapsed)
        print(f"  [{n:2d}/{len(sample)}] {elapsed:5.2f}s  "
              f"{'REFUSED' if result.is_refusal else 'answered'}  "
              f"{item['question'][:50]}")

    median = statistics.median(latencies)
    out_dir = cfg.path("evaluation.results_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "latency_interactive.md"
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("# NFR1 — interactive latency\n\n")
        fh.write(
            f"{len(latencies)} in-scope questions spread across intents, each "
            f"separated by a {GAP_SECONDS}s pause to approximate a student "
            f"reading the previous answer before asking the next. Model "
            f"`{cfg.require('generation.model')}`.\n\n"
        )
        fh.write("| Statistic | Seconds |\n|---|---|\n")
        fh.write(f"| Median | {median:.2f} |\n")
        fh.write(f"| Mean | {statistics.fmean(latencies):.2f} |\n")
        fh.write(f"| Min | {min(latencies):.2f} |\n")
        fh.write(f"| Max | {max(latencies):.2f} |\n\n")
        fh.write("Per-question: " + ", ".join(f"{v:.2f}" for v in latencies) + "\n\n")
        fh.write(
            "Compare with the burst figure in `eval_results.md`, measured "
            "while the harness issues 49 questions plus 38 judge calls with no "
            "pause. The difference between the two is provider throttling "
            "under sustained load, not model or pipeline speed — risk R3 in "
            "the register.\n"
        )

    print(f"\n  median {median:.2f}s | mean {statistics.fmean(latencies):.2f}s "
          f"| min {min(latencies):.2f}s | max {max(latencies):.2f}s")
    print(f"  NFR1 (<5s median): {'PASS' if median < 5 else 'FAIL'}")
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
