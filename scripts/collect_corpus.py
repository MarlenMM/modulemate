"""FR1 stage -1: collect eligible source files into corpus/raw/.

Applies the include/exclude rules declared in corpus_sources.yaml and stages
matching files into corpus/raw/, from where the existing conversion and
ingestion stages take over. Keeping the rules in data rather than in code means
the corpus can be re-scoped — a module added, the chat sources dropped —
without touching the pipeline (NFR5, and the "drop a file in a folder" workflow
the Dr Novák persona needs).

Two behaviours worth knowing:
  * Files are de-duplicated by content hash. Several lecture PDFs appear in
    more than one folder of the source tree; indexing both copies would let one
    document occupy two of the k retrieval slots with identical text.
  * Names are flattened with their parent folder as a prefix, so
    "clustering/Lecture_5.pdf" becomes "clustering__Lecture_5.pdf" and stays
    distinguishable from a same-named file elsewhere. That prefixed stem is
    what appears in citations.

Prints a full audit of what was included, skipped and de-duplicated.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "corpus_sources.yaml"
RAW_DIR = ROOT / "corpus" / "raw"

CHAT_PREFIX = "chats/"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def opening_text(path: Path, max_chars: int = 4000) -> str:
    """Cheap peek at a file's opening text, for the authorship guard.

    Only the first pages/cells are read: a cover page carries the author, and
    scanning whole 100-page decks at collection time would cost more than the
    guard is worth. Returns "" for formats not worth parsing here.
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            import warnings

            from pypdf import PdfReader
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reader = PdfReader(str(path))
                out = []
                for page in reader.pages[:3]:
                    out.append(page.extract_text() or "")
                    if sum(map(len, out)) > max_chars:
                        break
                return "\n".join(out)[:max_chars]
        if suffix == ".docx":
            import docx
            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs[:60])[:max_chars]
        if suffix == ".ipynb":
            import json as _json
            with open(path, "r", encoding="utf-8") as fh:
                nb = _json.load(fh)
            cells = nb.get("cells", [])[:6]
            return "\n".join("".join(c.get("source", [])) for c in cells)[:max_chars]
        if suffix in (".rtf", ".txt"):
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read(max_chars)
    except Exception:
        return ""
    return ""


def authored_by_student(path: Path, identifiers: list[str]) -> str | None:
    """Return the matching identifier if this looks like the student's work."""
    if not identifiers:
        return None
    text = opening_text(path)
    if not text:
        return None
    lowered = text.lower()
    for ident in identifiers:
        if ident.lower() in lowered:
            return ident
    return None


def flat_name(rel: Path) -> str:
    """Flatten a relative path to a unique, readable filename stem."""
    parts = list(rel.parts)
    stem = Path(parts[-1]).stem
    suffix = Path(parts[-1]).suffix
    parent = parts[-2] if len(parts) > 1 else ""
    # Only prefix when the parent adds information the filename lacks.
    if parent and parent.lower() not in stem.lower():
        stem = f"{parent}__{stem}"
    return f"{stem}{suffix}"


def main() -> None:
    with open(MANIFEST, "r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)

    source_root = Path(spec["source_root"]).expanduser()
    if not source_root.is_dir():
        raise SystemExit(f"source_root does not exist: {source_root}")

    allowed = {e.lower() for e in spec["allowed_extensions"]}
    includes = spec.get("include_globs", [])
    excludes = spec.get("exclude_globs", [])
    include_chats = spec.get("include_chats", True)
    identifiers = spec.get("student_identifiers", [])
    dedupe = spec.get("deduplicate_by_hash", True)
    renames = spec.get("rename", {}) or {}

    # --- Gather candidates ---
    candidates: list[Path] = []
    for pattern in includes:
        for path in sorted(source_root.glob(pattern)):
            if path.is_file():
                candidates.append(path)

    # --- Filter ---
    staged: dict[str, Path] = {}      # hash -> source path
    kept: list[tuple[Path, str]] = []
    skipped: list[tuple[Path, str]] = []

    for path in candidates:
        rel = path.relative_to(source_root)
        rel_posix = rel.as_posix()

        if path.suffix.lower() not in allowed:
            skipped.append((rel, "extension not allowed"))
            continue
        if any(rel.full_match(pat) if hasattr(rel, "full_match")
               else rel.match(pat) for pat in excludes) or \
           any(_glob_match(rel_posix, pat) for pat in excludes):
            skipped.append((rel, "excluded by manifest"))
            continue
        if not include_chats and rel_posix.startswith(CHAT_PREFIX):
            skipped.append((rel, "chats disabled"))
            continue

        owner = authored_by_student(path, identifiers)
        if owner:
            skipped.append((rel, f"AUTHORSHIP GUARD: contains '{owner}'"))
            continue

        digest = file_hash(path)
        if dedupe and digest in staged:
            skipped.append((rel, f"duplicate of {staged[digest].name}"))
            continue
        staged[digest] = rel
        kept.append((rel, renames.get(rel_posix) or flat_name(rel)))

    # --- Stage ---
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in RAW_DIR.iterdir() if p.is_file()}
    wanted = {name for _, name in kept}

    for stale in sorted(existing - wanted):
        (RAW_DIR / stale).unlink()
        print(f"  removed (no longer in manifest): {stale}")

    copied = 0
    for rel, name in kept:
        target = RAW_DIR / name
        src = source_root / rel
        if target.exists() and target.stat().st_size == src.stat().st_size:
            continue
        shutil.copy2(src, target)
        copied += 1

    # --- Audit ---
    by_ext: dict[str, int] = {}
    for _, name in kept:
        ext = Path(name).suffix.lower()
        by_ext[ext] = by_ext.get(ext, 0) + 1

    print(f"\nStaged {len(kept)} files into {RAW_DIR} ({copied} newly copied)")
    for ext, n in sorted(by_ext.items()):
        print(f"  {ext:8s} {n}")

    guarded = [r for r in skipped if r[1].startswith("AUTHORSHIP GUARD")]
    if guarded:
        print(f"\n!! Authorship guard rejected {len(guarded)} file(s) "
              f"as the student's own work:")
        for rel, reason in guarded:
            print(f"     {rel}  ({reason.split(': ', 1)[1]})")

    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for rel, reason in skipped:
            print(f"  [{reason}] {rel}")

    print("\nNext:  ./scripts/convert_corpus.sh "
          "&& python scripts/ocr_slides.py && python index.py")


def _glob_match(rel_posix: str, pattern: str) -> bool:
    """`**`-aware match against a POSIX relative path."""
    from fnmatch import fnmatch
    if fnmatch(rel_posix, pattern):
        return True
    # fnmatch treats '*' as crossing '/', which is close enough for '**'
    # patterns, but "a/**" must also match "a/b/c".
    if pattern.endswith("/**"):
        return rel_posix.startswith(pattern[:-3] + "/")
    return False


if __name__ == "__main__":
    main()
