"""Download pinned public weights without sending course text or provider credentials."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulemate.local_models import (  # noqa: E402
    CANDIDATE_ENCODER,
    CANDIDATE_REVISION,
    ENCODER,
    ENCODER_REVISION,
    READER,
    READER_REVISION,
    RERANKER,
    RERANKER_REVISION,
    LocalModels,
    model_dir,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-candidate",
        action="store_true",
        help="Also download the alternative encoder for the documented experiment",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload pinned files to repair a corrupt model cache",
    )
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Install local ML first: uv sync --extra ml") from None
    directory = model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    models_to_prepare = [
        (ENCODER, ENCODER_REVISION),
        (RERANKER, RERANKER_REVISION),
        (READER, READER_REVISION),
    ]
    if args.include_candidate:
        models_to_prepare.append((CANDIDATE_ENCODER, CANDIDATE_REVISION))
    for name, revision in models_to_prepare:
        print(f"Preparing {name} at {revision}", flush=True)
        snapshot_download(
            name,
            revision=revision,
            local_dir=directory / revision,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"],
            ignore_patterns=["onnx/*", "openvino/*"],
            force_download=args.force,
        )
    models = LocalModels(directory)
    models.encode(["Model installation check."])
    models.rerank("installation", ["Model installation check."])
    models.extract("What is being checked?", ["Model installation is being checked."])
    # Only advertise readiness after a successful load and inference. Atomic
    # replace keeps a concurrent application from seeing a partial marker.
    marker = directory / f"ready.{os.getpid()}.tmp"
    marker.write_text(
        json.dumps(
            {"encoder": ENCODER_REVISION, "reranker": RERANKER_REVISION, "reader": READER_REVISION}
        )
    )
    marker.replace(directory / "ready.json")
    print("Local semantic retrieval is ready. Documents stay on this computer.")


if __name__ == "__main__":
    main()
