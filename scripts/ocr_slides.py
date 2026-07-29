"""FR1 stage 0b — OCR recovery of image-only slides.

Why this exists: the module's lecture decks were authored largely as pictures
and embedded OLE objects rather than text boxes. Of 678 slides, only 258
expose any text to python-pptx; the remaining 420 are images, so a text-only
ingestion pipeline would silently omit ~62% of the teaching material and the
assistant would refuse questions the corpus actually answers.

Recovery path: render each deck to PDF via LibreOffice, rasterise each page,
and OCR it with the macOS Vision framework (no extra binary to install, and
markedly better on slide typography than a stock Tesseract install).

Output: corpus/ocr/<deck stem>.json == {"slide 12": "recognised text", ...}
Ingestion merges this with native text; native text always wins where present.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import Quartz
import Vision

ROOT = Path(__file__).resolve().parent.parent
CONVERTED = ROOT / "corpus" / "converted"
PDF_DIR = ROOT / "corpus" / "pdf"
PNG_DIR = ROOT / "corpus" / "png"
OCR_DIR = ROOT / "corpus" / "ocr"

# Vision reports a per-line confidence in [0,1]; below this the "text" is
# usually an artefact of a diagram's axis labels or a decorative glyph.
MIN_CONFIDENCE = 0.30
RENDER_DPI = 130


def ocr_image(png_path: Path) -> str:
    """Run Vision text recognition over one rasterised slide."""
    url = Quartz.CFURLCreateFromFileSystemRepresentation(
        None, str(png_path).encode("utf-8"), len(str(png_path).encode("utf-8")), False
    )
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    if src is None or Quartz.CGImageSourceGetCount(src) == 0:
        return ""
    image = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    if image is None:
        return ""

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
    ok, _err = handler.performRequests_error_([request], None)
    if not ok:
        return ""

    lines: list[str] = []
    for obs in request.results() or []:
        cand = obs.topCandidates_(1)
        if not cand:
            continue
        best = cand[0]
        if best.confidence() < MIN_CONFIDENCE:
            continue
        text = best.string().strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def deck_to_pngs(pptx: Path) -> list[tuple[int, Path]]:
    """Render a deck to one PNG per slide, returning [(slide_no, png)]."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    pdf = PDF_DIR / f"{pptx.stem}.pdf"
    if not pdf.exists():
        subprocess.run(
            ["soffice", "--headless", "--convert-to", "pdf",
             "--outdir", str(PDF_DIR), str(pptx)],
            check=True, capture_output=True,
        )
    if not pdf.exists():
        return []

    out_dir = PNG_DIR / pptx.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    if not any(out_dir.glob("*.png")):
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(RENDER_DPI), str(pdf), str(out_dir / "p")],
            check=True, capture_output=True,
        )

    pages: list[tuple[int, Path]] = []
    for png in out_dir.glob("p-*.png"):
        m = re.search(r"p-(\d+)\.png$", png.name)
        if m:
            pages.append((int(m.group(1)), png))
    return sorted(pages)


def main() -> None:
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    decks = sorted(CONVERTED.glob("*.pptx"))
    if len(sys.argv) > 1:  # optional filter for a quick single-deck run
        decks = [d for d in decks if sys.argv[1].lower() in d.stem.lower()]

    for deck in decks:
        out_path = OCR_DIR / f"{deck.stem}.json"
        if out_path.exists():
            print(f"skip (cached): {deck.stem}")
            continue
        pages = deck_to_pngs(deck)
        result: dict[str, str] = {}
        for slide_no, png in pages:
            text = ocr_image(png)
            if text:
                result[f"slide {slide_no}"] = text
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=1)
        print(f"{deck.stem}: OCR text on {len(result)}/{len(pages)} slides")


if __name__ == "__main__":
    main()
