#!/usr/bin/env bash
# ModuleMate — FR1 stage 0: normalise staged sources into ingestible formats.
#
# The corpus arrives in six formats. Three need conversion before any Python
# library will read them:
#   .ppt  -> .pptx   legacy binary OLE2; python-pptx raises PackageNotFoundError
#   .doc  -> .docx   same
#   .rtf  -> .txt    chat exports; no structured reader worth the dependency
# The rest (.pptx, .docx, .pdf, .ipynb) pass through untouched.
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
rtf_files=("$RAW"/*.rtf)

if [ ${#ppt_files[@]} -gt 0 ]; then
  echo "Converting ${#ppt_files[@]} .ppt -> .pptx ..."
  soffice --headless --convert-to pptx --outdir "$OUT" "${ppt_files[@]}" >/dev/null
fi

if [ ${#doc_files[@]} -gt 0 ]; then
  echo "Converting ${#doc_files[@]} .doc -> .docx ..."
  soffice --headless --convert-to docx --outdir "$OUT" "${doc_files[@]}" >/dev/null
fi

if [ ${#rtf_files[@]} -gt 0 ]; then
  echo "Converting ${#rtf_files[@]} .rtf -> .txt ..."
  for f in "${rtf_files[@]}"; do
    base="$(basename "${f%.rtf}")"
    if command -v textutil >/dev/null 2>&1; then
      # macOS native: handles the RTF variants these exports use, and keeps
      # the UTF-8 of mixed Chinese/English chat text intact.
      textutil -convert txt -encoding UTF-8 -output "$OUT/$base.txt" "$f"
    else
      soffice --headless --convert-to txt:Text --outdir "$OUT" "$f" >/dev/null
    fi
  done
fi

# Formats that need no conversion.
for f in "$RAW"/*.pptx "$RAW"/*.docx "$RAW"/*.pdf "$RAW"/*.ipynb; do
  cp -n "$f" "$OUT/" 2>/dev/null || true
done

echo
echo "Converted corpus in $OUT:"
for ext in pptx docx pdf ipynb txt; do
  n=$(find "$OUT" -maxdepth 1 -name "*.$ext" | wc -l | tr -d ' ')
  [ "$n" -gt 0 ] && printf "  %-6s %s\n" ".$ext" "$n"
done
