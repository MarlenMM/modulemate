"""Explicitly prepare the optional pinned, independent source-support checker."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulemate.local_models import model_dir  # noqa: E402
from modulemate.support import SUPPORT_MODEL, SUPPORT_REVISION, SupportModel  # noqa: E402


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Install the ml extra first.") from None
    directory = model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        SUPPORT_MODEL,
        revision=SUPPORT_REVISION,
        local_dir=directory / SUPPORT_REVISION,
        allow_patterns=["*.json", "model.safetensors", "spm.model"],
        ignore_patterns=["onnx/*", "openvino/*"],
    )
    model = SupportModel(directory)
    rows = model.predict([("A red cup is on the desk.", "There is a cup on the desk.")])
    if rows[0]["entailment"] < 0.5:
        raise SystemExit("Support model installation check failed; readiness was not written.")
    marker = directory / f"support-ready.{os.getpid()}.tmp"
    marker.write_text(json.dumps({"model": SUPPORT_MODEL, "revision": SUPPORT_REVISION}))
    marker.replace(directory / "support-ready.json")
    print("Independent local support checking is ready. No documents were uploaded.")


if __name__ == "__main__":
    main()
