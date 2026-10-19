"""Bounded local OCR. Recognition scores describe transcription, never truth.

Original images and recognition caches live only inside an extraction request.
No URL fetch, hosted service or model download is part of this module.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

from PIL import Image, ImageOps

MAX_PIXELS = 24_000_000
MAX_SIDE = 2600
MAX_IMAGES = 500
OCR_SECONDS = 35


@dataclass(frozen=True)
class Recognition:
    text: str = ""
    uncertain: bool = False


def local_ocr_engine():
    if sys.platform == "darwin" and importlib.util.find_spec("Vision"):
        return "Apple Vision"
    return "Tesseract" if shutil.which("tesseract") else None


def _vision(data):
    import objc
    import Vision

    # Drain native per-image allocations before reading the next page. PyObjC
    # owns the objects: calling dealloc() manually can cause use-after-free.
    with objc.autorelease_pool():
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setRecognitionLanguages_(["en-US"])
        request.setUsesLanguageCorrection_(False)
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
        success, error = handler.performRequests_error_([request], None)
        if not success or error:
            raise ValueError("Local recognition failed")
        lines = []
        uncertain = False
        for observation in request.results() or ():
            candidates = observation.topCandidates_(1)
            if candidates:
                candidate = candidates[0]
                lines.append(str(candidate.string()))
                uncertain |= candidate.confidence() < 0.5
        return Recognition("\n".join(lines), uncertain)


def _tesseract(data):
    result = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", "eng", "tsv"],
        input=data,
        capture_output=True,
        timeout=10,
        check=False,
        env=os.environ | {"OMP_THREAD_LIMIT": "1"},
    )
    if result.returncode or len(result.stdout) > 2_000_000:
        raise ValueError("Local recognition failed")
    lines = {}
    uncertain = False
    for row in csv.DictReader(io.StringIO(result.stdout.decode("utf-8")), delimiter="\t"):
        text = row.get("text", "").strip()
        if text:
            key = tuple(row.get(k) for k in ("page_num", "block_num", "par_num", "line_num"))
            lines.setdefault(key, []).append(text)
            uncertain |= float(row["conf"]) < 50
    return Recognition("\n".join(" ".join(words) for words in lines.values()), uncertain)


def additional_text(recognized, native):
    """Remove duplicate native lines without fuzzy edits to numbers or names."""
    normalized = " ".join(native.split()).casefold()
    lines, seen = [], set()
    for line in recognized.splitlines():
        line = line.replace("\x00", "").strip()
        key = " ".join(line.split()).casefold()
        if (
            key
            and key not in seen
            and not re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", normalized)
        ):
            lines.append(line)
            seen.add(key)
    return "\n".join(lines)


class OCRSession:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.engine = local_ocr_engine() if enabled else None
        self.started = time.monotonic()
        self.cache = {}
        self.attempts = 0
        self.recovered = []
        self.skipped = []
        self.uncertain = []

    @property
    def available(self):
        return (
            bool(self.engine)
            and self.attempts < MAX_IMAGES
            and time.monotonic() - self.started < OCR_SECONDS
        )

    def read(self, data, location, native=""):
        digest = hashlib.sha256(data).hexdigest()
        if digest not in self.cache and not self.available:
            self.skipped.append(location)
            return ""
        try:
            if digest not in self.cache:
                self.attempts += 1
                with Image.open(io.BytesIO(data)) as original:
                    if original.width * original.height > MAX_PIXELS:
                        raise ValueError("Image exceeds pixel limit")
                    image = ImageOps.exif_transpose(original).convert("RGB")
                image.thumbnail((MAX_SIDE, MAX_SIDE))
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                self.cache[digest] = (
                    _vision(buffer.getvalue())
                    if self.engine == "Apple Vision"
                    else _tesseract(buffer.getvalue())
                )
            result = self.cache[digest]
            text = additional_text(result.text, native)
            if text:
                self.recovered.append(location)
                if result.uncertain:
                    self.uncertain.append(location)
            return text
        except (
            OSError,
            ValueError,
            RuntimeError,
            ImportError,
            subprocess.SubprocessError,
            Image.DecompressionBombError,
        ):
            self.skipped.append(location)
            return ""

    def warnings(self):
        messages = []
        if self.recovered:
            messages.append(
                f"Local OCR ({self.engine}) recovered image text at {len(self.recovered)} locations. "
                "OCR locations are labeled; verify names, numbers and formulas against the original. "
                "Text recognition does not interpret diagrams, arrows or chart values."
            )
        if self.uncertain:
            messages.append(
                "Some OCR text has low recognition confidence: "
                + ", ".join(dict.fromkeys(self.uncertain))[:400]
                + ". It is retained for inspection, not treated as less credible source evidence."
            )
        if self.skipped:
            detail = (
                "Local OCR was disabled for this import."
                if not self.enabled
                else "Local OCR is unavailable. On Linux/Windows install Tesseract with English data."
                if not self.engine
                else "Local OCR reached a resource limit or could not read some images."
            )
            messages.append(
                f"{detail} Unread image locations: "
                + ", ".join(dict.fromkeys(self.skipped))[:400]
                + ". Add a searchable copy or paste the missing text."
            )
        return messages
