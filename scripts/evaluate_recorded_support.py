"""Retrospective NLI flags on public original-fixture answers, not accuracy labels."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulemate.search import Passage  # noqa: E402
from modulemate.support import check_claims, get_support_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/recorded-support.json"))
    args = parser.parse_args()
    os.environ["MODULEMATE_SUPPORT"] = "auto"
    model = get_support_model()
    if model is None:
        parser.error("Prepare the optional support model first")
    path = ROOT / "evaluation/observations/v2.2-conversations.json"
    data = json.loads(path.read_text())
    rows = []
    for run in data["runs"]:
        for row in run["rows"]:
            sources = {p["source_id"]: p for p in row["sources"]}
            passages = [
                Passage(
                    i, str(i), sources[i]["document"], sources[i]["location"], sources[i]["text"]
                )
                if i in sources
                else Passage(i, str(i), "unused", "unused", "")
                for i in range(1, max(sources, default=0) + 1)
            ]
            report = check_claims(row["answer"], passages, list(sources), model=model)
            rows.append({"run": run["name"], "id": row["id"], "report": report})
    report = {
        "diagnostic_only": True,
        "model": model.identity,
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checker_sha256": hashlib.sha256((ROOT / "modulemate/support.py").read_bytes()).hexdigest(),
        "answers": len(rows),
        "answers_flagged": sum(bool(r["report"]["flags"]) for r in rows),
        "qualification": "Flags are model opinions. Actual source reading is required to distinguish false flags and unsupported claims.",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
