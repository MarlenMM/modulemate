"""Fail CI if runtime data, legacy corpus artifacts, or obvious API keys are tracked."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS = {"data", "corpus", "index_store", "results", "backups", ".venv"}
FORBIDDEN_FILES = {
    "corpus_sources.yaml",
    "IMPLEMENTATION_NOTES.md",
    "IMPLEMENTATION_SUMMARY_FOR_REPORT.md",
    "eval/gold_set.json",
    ".streamlit/secrets.toml",
}
SECRET = re.compile(
    r"(?:gsk_[A-Za-z0-9]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{30,}|ghp_[A-Za-z0-9]{30,})"
)


def main():
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    failures = []
    checked = 0
    for name in filter(None, names):
        path = ROOT / name
        if (
            name.split("/")[0] in FORBIDDEN_DIRS
            or name in FORBIDDEN_FILES
            or (Path(name).name.startswith(".env") and name != ".env.example")
            or path.suffix in {".sqlite3", ".sqlite", ".db"}
        ):
            failures.append(f"Private or legacy artifact tracked: {name}")
            continue
        if not path.exists():
            failures.append(f"Tracked file missing from checkout: {name}")
            continue
        if path.is_symlink():
            failures.append(f"Review symlink before publishing: {name}")
            continue
        checked += 1
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if SECRET.search(text):
            failures.append(f"Possible credential in: {name}")
    for failure in failures:
        print(failure)
    if failures:
        raise SystemExit(1)
    print(
        f"Release-tree checks passed for {checked} tracked files. Git history is not scanned or erased by this check."
    )


if __name__ == "__main__":
    main()
