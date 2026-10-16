"""Download pinned public weights explicitly, then verify offline generation."""

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulemate.generation import LocalBackend, chat_model_dir, chat_spec, chat_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download the pinned weights again to repair a damaged local installation",
    )
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit(
            "This setup uses MLX on Apple Silicon. On other systems, configure an OpenAI-compatible local server; see README.md."
        )
    from huggingface_hub import snapshot_download

    root = chat_model_dir()
    model, revision, _ = chat_spec()
    if chat_status() == "ready" and not args.force:
        print("The pinned local chat model is already prepared. Use --force to download it again.")
        return
    root.mkdir(parents=True, exist_ok=True)
    (root / "ready.json").unlink(missing_ok=True)
    print(
        f"Preparing {model} at revision {revision}. Public weights only; no study text is uploaded.",
        flush=True,
    )
    snapshot_download(
        model,
        revision=revision,
        local_dir=root,
        allow_patterns=["*.json", "*.jinja", "*.safetensors", "README.md", "LICENSE*"],
        force_download=args.force,
    )
    backend = LocalBackend(root)
    try:
        result = backend.complete(
            'Return only JSON: {"ready":true}.', {"request": "Confirm readiness."}
        )
        if json.loads(result).get("ready") is not True:
            raise RuntimeError("Generation smoke test failed; model was not marked ready.")
    finally:
        backend.close()
    weights = {p.name: p.stat().st_size for p in root.glob("*.safetensors")}
    if not weights:
        raise RuntimeError("No model weights found.")
    marker = root / "ready.tmp"
    marker.write_text(
        json.dumps({"model": model, "revision": revision, "weights": weights}, indent=2)
    )
    marker.replace(root / "ready.json")
    print("Local conversational model ready.")


if __name__ == "__main__":
    main()
