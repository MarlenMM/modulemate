"""Evaluate independent NLI on original, document-grouped claim annotations."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulemate.support import LABELS, get_support_model  # noqa: E402
from scripts.train_evidence import DATA, ROOT, load_data  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "calibration", "test"], default="calibration")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ["MODULEMATE_SUPPORT"] = "auto"
    source = {
        p["id"]: (p["text"], d["split"]) for d in load_data()["documents"] for p in d["passages"]
    }
    path = ROOT / "evaluation/claim_support.json"
    rows = [
        r for r in json.loads(path.read_text())["claims"] if source[r["source_id"]][1] == args.split
    ]
    model = get_support_model()
    if model is None:
        parser.error("Prepare the support model first")
    scores = []
    for start in range(0, len(rows), 32):
        scores.extend(
            model.predict(
                [(source[r["source_id"]][0], r["claim"]) for r in rows[start : start + 32]]
            )
        )
    for row, score in zip(rows, scores, strict=True):
        row["source"] = source[row["source_id"]][0]
        row["predicted"] = max(score, key=score.get)
        row["scores"] = score
        row["flagged"] = score["entailment"] < 0.15 and (
            score["contradiction"] >= 0.90 or score["neutral"] >= 0.98
        )
    flags = [r for r in rows if r["flagged"]]
    report = {
        "diagnostic_only": True,
        "split": args.split,
        "model": model.identity,
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "count": len(rows),
        "accuracy": round(float(np.mean([r["label"] == r["predicted"] for r in rows])), 4),
        "macro_f1": round(
            float(
                f1_score(
                    [r["label"] for r in rows], [r["predicted"] for r in rows], average="macro"
                )
            ),
            4,
        ),
        "label_order": list(LABELS),
        "confusion_matrix": confusion_matrix(
            [r["label"] for r in rows], [r["predicted"] for r in rows], labels=list(LABELS)
        ).tolist(),
        "flagged": len(flags),
        "false_flags": sum(r["label"] == "entailment" for r in flags),
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
