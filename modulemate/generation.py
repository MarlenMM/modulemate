"""Explicitly prepared, offline generation. Weights and text never enter Git."""

from __future__ import annotations

import atexit
import importlib.util
import json
import os
import platform
import select
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

CHAT_MODEL = "mlx-community/Qwen3.5-9B-4bit"
CHAT_REVISION = "8b2b98c00a6b4d291155e4890773ca8f769aee53"
CHAT_VARIANTS = {
    "9b": (CHAT_MODEL, CHAT_REVISION, "Qwen3.5 9B"),
    "4b": (
        "mlx-community/Qwen3.5-4B-4bit",
        "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c",
        "Qwen3.5 4B",
    ),
}


def chat_spec():
    return CHAT_VARIANTS.get(
        os.getenv("MODULEMATE_CHAT_VARIANT", "9b").lower(), CHAT_VARIANTS["9b"]
    )


def chat_model_name():
    return chat_spec()[2]


class GenerationError(Exception):
    """A safe generation failure; never contains a prompt or provider response."""


def chat_model_dir():
    return (
        Path(
            os.getenv(
                "MODULEMATE_CHAT_MODEL_DIR",
                "~/.cache/modulemate/chat/" + chat_spec()[0].split("/")[1].lower(),
            )
        )
        .expanduser()
        .resolve()
    )


def chat_status():
    if os.getenv("MODULEMATE_CHAT", "auto").lower() == "off":
        return "disabled"
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return "Apple Silicon runtime unavailable"
    if any(importlib.util.find_spec(name) is None for name in ("mlx_lm", "outlines_core")):
        return "runtime not installed"
    root = chat_model_dir()
    try:
        if any(
            not (root / name).is_file()
            for name in (
                "config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "chat_template.jinja",
            )
        ):
            return "setup required"
        marker = json.loads((root / "ready.json").read_text())
        model, revision, _ = chat_spec()
        if (
            not isinstance(marker, dict)
            or marker.get("model") != model
            or marker.get("revision") != revision
        ):
            return "setup required"
        files = marker.get("weights", {})
        if (
            not isinstance(files, dict)
            or not files
            or any(
                not isinstance(name, str)
                or Path(name).name != name
                or type(size) is not int
                or size <= 0
                or (root / name).stat().st_size != size
                for name, size in files.items()
            )
        ):
            return "setup required"
        return "ready"
    except (OSError, ValueError, TypeError):
        return "setup required"


class LocalBackend:
    """One bounded worker, serialized requests, no shared conversation/KV cache.

    Keeping MLX in a process makes timeouts recoverable, isolates its Metal thread
    state from Streamlit reruns, and releases memory on shutdown or an error.
    """

    label = "Qwen3.5 9B · on this Mac"
    local = True
    supports_plain_text = True

    def __init__(self, root: Path):
        self.root = root
        self.label = chat_model_name() + " · on this Mac"
        self.lock = threading.Lock()
        self.process = None
        self.last_stats = {}
        atexit.register(self.close)

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()

    def complete(self, system: str, payload: dict) -> str:
        if not self.lock.acquire(timeout=2):
            raise GenerationError("The local model is busy. Try again after the current answer.")
        try:
            if not self.process or self.process.poll() is not None:
                self.close()
                env = dict(
                    {
                        k: v
                        for k, v in os.environ.items()
                        if k
                        in {"PATH", "HOME", "LANG", "LC_ALL", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP"}
                    },
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    HF_HUB_DISABLE_TELEMETRY="1",
                    TOKENIZERS_PARALLELISM="false",
                )
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "modulemate.llm_worker", str(self.root)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                    env=env,
                    cwd=Path(__file__).resolve().parents[1],
                )
            request = json.dumps({"system": system, "payload": payload}, ensure_ascii=False)
            if len(request) > 100_000:
                raise GenerationError("The model context is too large. Focus on fewer documents.")
            self.process.stdin.write(request + "\n")
            self.process.stdin.flush()
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    break
                ready, _, _ = select.select(
                    [self.process.stdout], [], [], min(1, deadline - time.monotonic())
                )
                if ready:
                    raw = self.process.stdout.readline()
                    try:
                        response = json.loads(raw)
                    except (ValueError, TypeError):
                        break
                    if response.get("error"):
                        self.last_stats = {"error": response["error"], **response.get("stats", {})}
                        break
                    text = response.get("text")
                    if not isinstance(text, str) or not text.strip() or len(text) > 40_000:
                        break
                    self.last_stats = response.get("stats", {})
                    return text
            self.close()
            raise GenerationError(
                "The local model could not finish an answer. Retry or use a smaller document selection."
            )
        except (OSError, BrokenPipeError) as exc:
            self.close()
            raise GenerationError("The local model stopped. Retry to restart it.") from exc
        finally:
            self.lock.release()


@lru_cache(maxsize=1)
def _backend(root):
    return LocalBackend(root)


def local_backend():
    return _backend(chat_model_dir()) if chat_status() == "ready" else None
