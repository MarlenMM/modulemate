"""M1 — Ingestion, extraction and metadata-preserving chunking (FR1).

Pipeline stages:
  0. Legacy conversion .ppt/.doc -> .pptx/.docx  (scripts/convert_corpus.sh)
  1. Extraction     — one text unit per slide (PPTX) or per section (DOCX)
  2. Chunking       — dense units split with overlap; sparse units kept whole
  3. Metadata       — {source_document, slide_or_section, chunk_id} on every chunk

Citation granularity differs by source type, because the two formats expose
different natural anchors:
  * PPTX -> slide number, read directly from slide order.
  * DOCX -> section heading. The assessment brief carries no page numbers once
    extracted, and the .doc -> .docx conversion produces no real Word heading
    styles (every paragraph is 'Plain Text'), so headings are recovered by the
    structural heuristic in `_is_heading`.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import docx
from pptx import Presentation

import config

# A separator rule such as "______" or "~~~~~~" carries no meaning.
_SEPARATOR_RE = re.compile(r"^[_~\-=\.\s]+$")


@dataclass
class Chunk:
    """One retrievable passage plus the metadata needed to cite it."""

    chunk_id: str
    source_document: str      # filename stem, as it appears in a citation
    slide_or_section: str     # "slide 12" | "Assessment scheme"
    doc_type: str             # "pptx" | "docx"
    text: str
    n_words: int

    @property
    def citation(self) -> str:
        return f"[Source: {self.source_document}, {self.slide_or_section}]"


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def _shape_text(shape) -> list[str]:
    """Recursively pull text out of a shape, including groups and tables."""
    out: list[str] = []
    if shape.shape_type == 6 and hasattr(shape, "shapes"):  # GROUP
        for sub in shape.shapes:
            out.extend(_shape_text(sub))
        return out
    if getattr(shape, "has_table", False):
        for row in shape.table.rows:
            cells = [c.text.strip() for c in row.cells]
            line = " | ".join(c for c in cells if c)
            if line:
                out.append(line)
        return out
    if shape.has_text_frame:
        for para in shape.text_frame.paragraphs:
            line = "".join(run.text for run in para.runs).strip()
            if line:
                out.append(line)
    return out


def _load_ocr(stem: str, ocr_dir: Path) -> dict[str, str]:
    """OCR text for one deck, keyed by 'slide N' (empty dict if not run)."""
    path = ocr_dir / f"{stem}.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _strip_boilerplate(per_slide: dict[str, str], min_repeats: int) -> dict[str, str]:
    """Remove header/footer lines that recur across most slides in a deck.

    Every slide in these decks is stamped with a course banner
    ("COMP5511 Artificial Intelligence Concepts") and a page marker
    ("Lecture #8 - 5"). Left in, they dominate the embedding of otherwise
    short slides and make unrelated slides look mutually similar.
    """
    counts: dict[str, int] = {}
    for text in per_slide.values():
        for line in set(text.splitlines()):
            key = line.strip().lower()
            if key:
                counts[key] = counts.get(key, 0) + 1

    banned = {k for k, n in counts.items() if n >= min_repeats}
    cleaned: dict[str, str] = {}
    for locator, text in per_slide.items():
        kept = [
            ln for ln in text.splitlines()
            if ln.strip() and ln.strip().lower() not in banned
        ]
        if kept:
            cleaned[locator] = "\n".join(kept)
    return cleaned


def extract_pptx(path: Path, ocr_dir: Path | None = None) -> list[tuple[str, str]]:
    """Return [(slide_or_section, text)] — one entry per slide.

    Native shape text and OCR text are merged per slide. Native text is
    authoritative; OCR supplements it, and is the only source for the
    image-only slides that make up the majority of these decks.
    """
    prs = Presentation(str(path))
    native: dict[str, str] = {}
    for idx, slide in enumerate(prs.slides, start=1):
        lines: list[str] = []
        for shape in slide.shapes:
            lines.extend(_shape_text(shape))
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                lines.append(f"Speaker notes: {notes}")
        text = "\n".join(dict.fromkeys(lines)).strip()  # de-dup, keep order
        if text:
            native[f"slide {idx}"] = text

    ocr = _load_ocr(path.stem, ocr_dir) if ocr_dir else {}
    if ocr:
        # Boilerplate recurs on nearly every slide; 40% is a safe cut-off that
        # catches banners without removing a genuinely repeated definition.
        ocr = _strip_boilerplate(ocr, max(3, int(0.4 * len(ocr))))

    merged: dict[str, str] = {}
    for locator in sorted(
        set(native) | set(ocr), key=lambda s: int(s.split()[1])
    ):
        parts: list[str] = []
        seen: set[str] = set()
        for source in (native.get(locator, ""), ocr.get(locator, "")):
            for line in source.splitlines():
                key = line.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    parts.append(line.strip())
        if parts:
            merged[locator] = "\n".join(parts)

    return list(merged.items())


def extract_pdf(path: Path) -> list[tuple[str, str]]:
    """Return [(page label, text)] — one entry per page.

    Page number is the natural citation anchor for a PDF and is what a student
    sees in a reader, so it is used directly. Pages with no extractable text
    are dropped rather than OCRed: unlike the .ppt decks, every PDF staged by
    the manifest was verified to carry a text layer, and the one scanned
    document in the source tree is excluded there with a reason.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    units: list[tuple[str, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            continue
        if text:
            units.append((f"page {idx}", text))
    return units


def extract_ipynb(path: Path) -> list[tuple[str, str]]:
    """Return [(cell label, text)] — one entry per meaningful notebook cell.

    Markdown cells carry the teaching narrative and code cells the worked
    technique; both are retained, with code fenced so the generator can tell
    prose from an API call it should quote verbatim. Cell outputs are dropped —
    they are mostly dataframes and plot payloads, and a retrieved table of
    floats answers no student question. Citation is by cell index, which is
    stable for a notebook distributed read-only.
    """
    with open(path, "r", encoding="utf-8") as fh:
        nb = json.load(fh)

    units: list[tuple[str, str]] = []
    for idx, cell in enumerate(nb.get("cells", []), start=1):
        source = "".join(cell.get("source", [])).strip()
        if not source:
            continue
        kind = cell.get("cell_type")
        if kind == "code":
            text = f"Code:\n```python\n{source}\n```"
        elif kind == "markdown":
            text = source
        else:
            continue
        units.append((f"cell {idx}", text))
    return units


def extract_txt(path: Path, target_words: int) -> list[tuple[str, str]]:
    """Return [(part label, text)] for plain text with no inherent structure.

    Used for the converted chat exports. A group chat has no headings, pages or
    slides, so there is no natural citation anchor; consecutive lines are
    grouped to the chunk budget and cited as sequential parts. The label is
    honest about granularity rather than inventing a page number.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    pieces = chunk_lines(raw, target_words)
    return [(f"part {i}", piece) for i, piece in enumerate(pieces, start=1)]


def _is_heading(text: str) -> bool:
    """Structural heading test for a converted .doc with no heading styles.

    A heading is short, is not a separator rule or bullet, does not end in
    sentence punctuation, and — if it contains a colon — has nothing after it
    ("Requirements:" is a heading, "Due: Friday 31 July" is a data line).
    """
    t = text.strip()
    if not t or len(t) > 60:
        return False
    if _SEPARATOR_RE.match(t) or t.startswith(("--", "*", "•")):
        return False
    if t.endswith((".", ";", ",")):
        return False
    if ":" in t and t.partition(":")[2].strip():
        return False
    return True


def extract_docx(path: Path) -> list[tuple[str, str]]:
    """Return [(section_title, text)] — one entry per detected section."""
    doc = docx.Document(str(path))
    sections: list[tuple[str, list[str]]] = []
    current_title = "Introduction"
    current_body: list[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text or _SEPARATOR_RE.match(text):
            continue
        if _is_heading(text):
            if current_body:
                sections.append((current_title, current_body))
            current_title = text.rstrip(":").strip()
            current_body = []
        else:
            current_body.append(text)

    if current_body:
        sections.append((current_title, current_body))

    # Tables live outside the paragraph flow; attach them to their own section.
    for t_idx, table in enumerate(doc.tables, start=1):
        rows = []
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            line = " | ".join(c for c in cells if c)
            if line:
                rows.append(line)
        if rows:
            sections.append((f"Table {t_idx}", rows))

    return [(title, "\n".join(body).strip()) for title, body in sections if body]


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def chunk_text(text: str, target_words: int, overlap_words: int) -> list[str]:
    """Split on a word budget with overlap. Short units pass through whole."""
    words = text.split()
    if len(words) <= target_words:
        return [text]
    stride = max(1, target_words - overlap_words)
    pieces: list[str] = []
    for start in range(0, len(words), stride):
        window = words[start:start + target_words]
        if not window:
            break
        pieces.append(" ".join(window))
        if start + target_words >= len(words):
            break
    return pieces


def chunk_lines(text: str, target_words: int) -> list[str]:
    """Line-aware chunking for prose documents, never splitting mid-line.

    Used for the assessment brief. Its sections are lists of semantically
    independent facts -- one bullet gives the word limit, the next names the
    permitted topics, a third covers late penalties. Embedded as one vector
    those facts average into a diffuse point that matches no specific question
    well: measured before this change, "What is the word limit for the report?"
    scored 0.288 and ranked a lecture slide above the brief, while an
    off-topic control question scored 0.349. Grouping whole lines up to a
    tighter budget keeps each fact its own retrievable unit.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for line in lines:
        n = len(line.split())
        if current and count + n > target_words:
            chunks.append("\n".join(current))
            current, count = [], 0
        current.append(line)
        count += n
    if current:
        chunks.append("\n".join(current))
    return chunks or [text]


def latin_ratio(text: str) -> float:
    """Share of letters that are Latin, among all letters present."""
    latin = cjk = 0
    for ch in text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or \
           0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF:
            cjk += 1
        elif ch.isalpha() and code < 0x0250:
            latin += 1
    total = latin + cjk
    return 1.0 if total == 0 else latin / total


def is_english_enough(text: str, min_latin_ratio: float) -> bool:
    """Reject passages that are predominantly CJK.

    The locked scope is English-only and the embedding model
    (all-MiniLM-L6-v2) is English-trained, so a Chinese passage embeds to a
    point that no English question lands near. Indexing it cannot help
    retrieval and can only displace an English chunk from the top-k. The
    group-chat exports are genuinely bilingual, so this is applied per chunk
    rather than per document — the English half of a chat is kept.
    """
    return latin_ratio(text) >= min_latin_ratio


def is_informative(text: str, min_alpha_ratio: float, min_word_tokens: int) -> bool:
    """Reject chunks that carry no answerable language.

    OCR over slide graphics recovers axis tick labels, legend fragments and
    stray glyphs -- e.g. "100 75 50 25÷ 0 10 -10 -20" from a plot. These are
    real strings, so they embed to real vectors and compete for retrieval
    slots, but no student question is ever answered by them. Requiring a
    minimum share of alphabetic characters and a minimum count of word-like
    tokens removes them without touching genuine prose.
    """
    if not text.strip():
        return False
    alpha = sum(ch.isalpha() for ch in text)
    total = sum(not ch.isspace() for ch in text)
    if total == 0 or (alpha / total) < min_alpha_ratio:
        return False
    word_tokens = [t for t in re.split(r"\W+", text) if len(t) > 2 and t.isalpha()]
    return len(word_tokens) >= min_word_tokens


def build_chunks(cfg=None) -> list[Chunk]:
    """Run the whole ingestion pipeline over the converted corpus."""
    cfg = cfg or config.load()
    converted = cfg.path("corpus.converted_dir")
    target = cfg.require("index.chunk_target_words")
    overlap = cfg.require("index.chunk_overlap_words")
    docx_target = cfg.require("index.chunk_target_words_docx")
    min_chars = cfg.require("index.min_chunk_chars")
    min_alpha_ratio = cfg.require("index.min_alpha_ratio")
    min_word_tokens = cfg.require("index.min_word_tokens")
    min_latin_ratio = cfg.require("index.min_latin_ratio")
    min_alpha_ratio_code = cfg.require("index.min_alpha_ratio_code")

    ocr_dir = cfg.path("corpus.ocr_dir")

    files: list[Path] = []
    for ext in ("pptx", "docx", "pdf", "ipynb", "txt"):
        files.extend(sorted(converted.glob(f"*.{ext}")))
    if not files:
        raise FileNotFoundError(
            f"No ingestible files in {converted}. Run:\n"
            f"  python scripts/collect_corpus.py && ./scripts/convert_corpus.sh"
        )

    chunks: list[Chunk] = []
    for path in files:
        doc_type = path.suffix.lstrip(".")
        stem = path.stem  # the name used in citations
        # Beamer-style lecture PDFs emit one page per incremental reveal, so
        # the same paragraph recurs verbatim across up to seven consecutive
        # pages. Measured on this corpus: 133 duplicated chunks. Left in, a
        # single passage can occupy several of the k retrieval slots with
        # identical text, starving the answer of other evidence. The first
        # occurrence wins, so citations point at the page where the content
        # first appears.
        seen_texts: set[str] = set()
        if doc_type == "pptx":
            units = extract_pptx(path, ocr_dir)
        elif doc_type == "docx":
            units = extract_docx(path)
        elif doc_type == "pdf":
            units = extract_pdf(path)
        elif doc_type == "ipynb":
            units = extract_ipynb(path)
        else:  # txt — converted chat exports
            units = extract_txt(path, docx_target)

        for locator, text in units:
            # Slides and PDF pages are visually chunked already, so a word
            # budget with overlap suits them; prose sources chunk by line.
            if doc_type in ("docx", "txt"):
                # Prefix the section heading onto each piece. A lone bullet
                # ("-- You can use AI to the fullest.") embeds ambiguously,
                # and the heading is exactly the disambiguating context a
                # reader would use. It is also what the chunk is cited as, so
                # the retrieved text and its citation stay consistent.
                pieces = (
                    [f"{locator}\n{piece}"
                     for piece in chunk_lines(text, docx_target)]
                    if doc_type == "docx" else chunk_lines(text, docx_target)
                )
            else:
                pieces = chunk_text(text, target, overlap)
            for piece_idx, piece in enumerate(pieces):
                if len(piece) < min_chars:
                    continue
                # Source code is legitimately symbol-dense, so the OCR-noise
                # ratio would discard exactly the notebook cells that answer
                # "which function did the lab use?".
                ratio = (
                    min_alpha_ratio_code
                    if doc_type == "ipynb" and piece.lstrip().startswith("Code:")
                    else min_alpha_ratio
                )
                if not is_informative(piece, ratio, min_word_tokens):
                    continue
                if not is_english_enough(piece, min_latin_ratio):
                    continue
                fingerprint = " ".join(piece.split()).lower()
                if fingerprint in seen_texts:
                    continue
                seen_texts.add(fingerprint)
                slug = re.sub(r"[^a-z0-9]+", "-", locator.lower()).strip("-")
                chunks.append(
                    Chunk(
                        chunk_id=f"{stem}::{slug}::{piece_idx}",
                        source_document=stem,
                        slide_or_section=locator,
                        doc_type=doc_type,
                        text=piece,
                        n_words=len(piece.split()),
                    )
                )
    return chunks


def save_chunks(chunks: list[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump([asdict(c) for c in chunks], fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    cfg = config.load()
    chunks = build_chunks(cfg)
    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c.source_document] = by_doc.get(c.source_document, 0) + 1
    print(f"Extracted {len(chunks)} chunks from {len(by_doc)} documents\n")
    for doc, n in sorted(by_doc.items()):
        print(f"  {n:5d}  {doc}")
    words = [c.n_words for c in chunks]
    print(f"\nwords/chunk: min={min(words)} median={sorted(words)[len(words)//2]} max={max(words)}")
