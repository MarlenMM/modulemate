#!/usr/bin/env bash
# ModuleMate — FR1 stage 0: legacy Office format conversion.
#
# The module corpus ships as legacy binary .ppt / .doc (OLE2 compound files).
# python-pptx and python-docx read only the OOXML formats (.pptx / .docx) and
# raise PackageNotFoundError on the legacy binaries, so conversion is a
# mandatory first step of the ingestion pipeline, not a convenience.
#
# Idempotent: safe to re-run (supports FR9 one-command rebuild).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$HERE/corpus/raw"
OUT="$HERE/corpus/converted"

if ! command -v soffice >/dev/null 2>&1; then
  echo "ERROR: LibreOffice ('soffice') not found on PATH." >&2
  echo "Install with: brew install --cask libreoffice" >&2
  exit 1
fi

mkdir -p "$OUT"

shopt -s nullglob
ppt_files=("$RAW"/*.ppt)
doc_files=("$RAW"/*.doc)

if [ ${#ppt_files[@]} -gt 0 ]; then
  echo "Converting ${#ppt_files[@]} .ppt -> .pptx ..."
  soffice --headless --convert-to pptx --outdir "$OUT" "${ppt_files[@]}" >/dev/null
fi

if [ ${#doc_files[@]} -gt 0 ]; then
  echo "Converting ${#doc_files[@]} .doc -> .docx ..."
  soffice --headless --convert-to docx --outdir "$OUT" "${doc_files[@]}" >/dev/null
fi

# Pass through any files already in a modern format.
for f in "$RAW"/*.pptx "$RAW"/*.docx; do
  cp -n "$f" "$OUT/" 2>/dev/null || true
done

echo "Converted corpus in $OUT:"
ls -1 "$OUT"
