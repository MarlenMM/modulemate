"""Bounded extraction of user uploads, never imported/executed as code."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

from modulemate.db import UserError

MAX_BYTES = 15 * 1024 * 1024
MAX_CHARACTERS = 2_000_000
MAX_UNITS = 500
MAX_ARCHIVE_BYTES = 60 * 1024 * 1024
SUPPORTED = {".txt", ".md", ".pdf", ".docx", ".pptx", ".ipynb"}


@dataclass(frozen=True)
class Chunk:
    locator: str
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    name: str
    digest: str
    text: str
    chunks: list[Chunk]
    warnings: list[str]


def safe_name(name: str) -> str:
    name = unicodedata.normalize("NFKC", name.replace("\\", "/").split("/")[-1])
    name = "".join(c for c in name if not unicodedata.category(c).startswith("C"))
    return name.strip()[:180] or "Untitled.txt"


def split_text(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Chunk overlap must be smaller than the positive chunk size.")
    text = text.replace("\x00", "").replace("\r\n", "\n").strip()
    result = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(
                text.rfind("\n", start + size // 2, end), text.rfind(" ", start + size // 2, end)
            )
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            result.append(piece)
        if end == len(text):
            break
        candidate = max(start + 1, end - overlap)
        if candidate > 0 and text[candidate - 1].isalnum() and text[candidate].isalnum():
            boundary = max(
                text.rfind(" ", max(start + 1, candidate - 50), candidate + 1),
                text.rfind("\n", max(start + 1, candidate - 50), candidate + 1),
            )
            if boundary > start:
                candidate = boundary + 1
        start = candidate
    return result


def structured_chunks(text: str) -> list[tuple[str, str]]:
    """Keep Markdown sections together and carry their real heading hierarchy.

    Headings inside fenced code are code. A small section is never glued to the
    next unrelated concept merely to fill a character window.
    """
    sections, body, headings = [], [], []
    fence = None

    def flush():
        content = "\n".join(body).strip()
        if not content:
            return
        title = " › ".join(value for _, value in headings)
        prefix = "\n".join("#" * level + " " + value for level, value in headings)
        for piece in split_text(content):
            sections.append((title, (prefix + "\n" + piece).strip()))

    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            kind = marker[1][0]
            fence = None if fence == kind else (kind if fence is None else fence)
        match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line) if fence is None else None
        if match:
            flush()
            body = []
            level = len(match[1])
            headings = [(n, title) for n, title in headings if n < level]
            headings.append((level, match[2]))
        else:
            body.append(line)
    flush()
    # Preserve a heading-only note instead of silently producing an empty file.
    return sections or [("", part) for part in split_text(text)]


def _check_archive(data: bytes, extension: str) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > 2500 or sum(i.file_size for i in entries) > MAX_ARCHIVE_BYTES:
                raise UserError("Office file expands beyond the safe processing limit.")
            required = "word/document.xml" if extension == ".docx" else "ppt/presentation.xml"
            if required not in archive.namelist():
                raise UserError("The file does not match its Office document extension.")
            for item in entries:
                if item.flag_bits & 1 or item.file_size > 12 * 1024 * 1024:
                    raise UserError(
                        "Encrypted or unusually large Office components are not supported."
                    )
                if item.filename.lower().endswith((".xml", ".rels")):
                    xml = archive.read(item).upper()
                    if b"<!DOCTYPE" in xml or b"<!ENTITY" in xml:
                        raise UserError("Office XML containing external entities is not supported.")
    except (zipfile.BadZipFile, RuntimeError, KeyError) as exc:
        raise UserError("This Office file is damaged or unsupported.") from exc


def _extract(data: bytes, extension: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Runs in a resource-limited subprocess in normal application use."""
    warnings = []
    units = []
    if extension in {".txt", ".md"}:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise UserError("Save text files as UTF-8, then upload again.") from exc
        if "\x00" in text:
            raise UserError("This looks like a binary file. Upload UTF-8 text instead.")
        # Splitting before recognizing sections used to orphan a heading from
        # the definition it governs. Chunk only after parsing the structure.
        units = [("part 1", text)]
    elif extension == ".ipynb":
        try:
            notebook = json.loads(data)
            cells = notebook["cells"]
            if not isinstance(cells, list) or len(cells) > MAX_UNITS:
                raise UserError("Notebooks may contain at most 500 cells.")
            for i, cell in enumerate(cells, 1):
                if cell.get("cell_type") not in {"markdown", "code"}:
                    continue
                source = cell.get("source", "")
                source = "".join(source) if isinstance(source, list) else source
                if not isinstance(source, str):
                    raise UserError("Invalid notebook cell text.")
                if source.strip():
                    units.append((f"cell {i}", source))
            warnings.append(
                "Notebook code is read as text. Outputs and attachments are omitted; nothing is executed."
            )
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            if isinstance(exc, UserError):
                raise
            raise UserError("This is not a valid notebook.") from exc
    elif extension == ".pdf":
        from pypdf import PdfReader

        if not data.lstrip().startswith(b"%PDF-"):
            raise UserError("The file does not contain a PDF header.")
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise UserError("Unlock the PDF before uploading it.")
        if len(reader.pages) > MAX_UNITS:
            raise UserError("PDFs may contain at most 500 pages. Split this file first.")
        empty = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                units.append((f"page {i}", text))
            else:
                empty.append(i)
        if empty:
            warnings.append(
                f"{len(empty)} pages have no readable text (first pages: {', '.join(map(str, empty[:10]))}). OCR is not included; upload a searchable PDF for those pages."
            )
    elif extension in {".docx", ".pptx"}:
        _check_archive(data, extension)
        if extension == ".docx":
            from docx import Document
            from docx.table import Table

            doc = Document(io.BytesIO(data))
            lines = []
            # Preserve paragraph/table order rather than moving tables to the end.
            for block in doc.iter_inner_content():
                if isinstance(block, Table):
                    lines.append(
                        "\n".join(" | ".join(cell.text for cell in row.cells) for row in block.rows)
                    )
                else:
                    heading = re.match(r"Heading ([1-6])", block.style.name or "")
                    lines.append(("#" * int(heading[1]) + " " if heading else "") + block.text)
            units = [
                (f"section {i}", text)
                for i, text in enumerate(split_text("\n".join(lines), 5000, 0), 1)
            ]
            warnings.append(
                "DOCX locations are extracted sections, not page numbers. Images, headers, footnotes and text boxes may be omitted."
            )
        else:
            from pptx import Presentation

            presentation = Presentation(io.BytesIO(data))
            if len(presentation.slides) > MAX_UNITS:
                raise UserError("Presentations may contain at most 500 slides.")

            def shape_text(shape):
                if hasattr(shape, "shapes"):
                    return "\n".join(shape_text(child) for child in shape.shapes)
                if shape.has_table:
                    return "\n".join(
                        " | ".join(cell.text for cell in row.cells) for row in shape.table.rows
                    )
                return shape.text if shape.has_text_frame else ""

            missing = 0
            for i, slide in enumerate(presentation.slides, 1):
                text = "\n".join(shape_text(shape) for shape in slide.shapes)
                if slide.has_notes_slide:
                    text += "\n" + slide.notes_slide.notes_text_frame.text
                if text.strip():
                    units.append((f"slide {i}", text))
                else:
                    missing += 1
            warnings.append(
                "Images and diagrams are not read. Export a searchable PDF or add a text description when needed."
            )
            if missing:
                warnings.append(f"{missing} slides have no readable text.")
    if sum(len(text) for _, text in units) > MAX_CHARACTERS:
        raise UserError("Extracted text exceeds 2 million characters. Split this document first.")
    if not any(text.strip() for _, text in units):
        raise UserError(
            "No readable text found. For scans or images, run OCR first or paste your notes."
        )
    return units, warnings


def parse_upload(name: str, data: bytes, *, isolated: bool = True) -> ParsedDocument:
    name = safe_name(name)
    extension = Path(name).suffix.lower()
    if extension not in SUPPORTED:
        raise UserError("Supported files: PDF, DOCX, PPTX, TXT, Markdown and IPYNB.")
    if not data or len(data) > MAX_BYTES:
        raise UserError("Upload a non-empty file no larger than 15 MB.")
    if isolated:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "modulemate.extract_worker", extension],
                input=data,
                env={
                    k: v
                    for k, v in os.environ.items()
                    if k in {"PATH", "LANG", "LC_ALL", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP"}
                },
                capture_output=True,
                timeout=25,
                cwd=Path(__file__).resolve().parent.parent,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise UserError(
                "This file took too long to read. Try a smaller or text-only version."
            ) from exc
        if proc.returncode != 0:
            raise UserError(
                "The document could not be read within safe resource limits. Try a smaller text-only copy."
            )
        try:
            payload = json.loads(proc.stdout)
        except (ValueError, UnicodeDecodeError) as exc:
            raise UserError("The document reader returned an invalid result.") from exc
        if "error" in payload:
            raise UserError(payload["error"])
        units, warnings = payload["units"], payload["warnings"]
    else:
        units, warnings = _extract(data, extension)
    chunks = []
    seen = set()
    for locator, text in units:
        for heading, part in structured_chunks(text):
            location = locator + (" · " + heading[:160] if heading else "")
            # Keep duplicate text on distinct pages: locators remain meaningful.
            key = (location, part)
            if key not in seen:
                chunks.append(Chunk(location, part))
                seen.add(key)
    if len(chunks) > 3000:
        raise UserError("This file produces more than 3,000 passages. Split it first.")
    text = "\n\n".join(f"[{locator}]\n{text.strip()}" for locator, text in units)
    digest = hashlib.sha256(text.encode()).hexdigest()
    return ParsedDocument(name, digest, text, chunks, warnings)
