"""Subprocess boundary for parsers. No files are persisted or code executed."""

import json
import sys

from modulemate.db import UserError
from modulemate.ingest import MAX_BYTES, _extract


def main():
    try:
        import resource

        cpu_limit = 15 if "--no-ocr" in sys.argv or sys.argv[1] in {".txt", ".md"} else 55
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        if sys.platform.startswith("linux"):
            resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024,) * 2)
    except (ImportError, ValueError, OSError):
        pass  # Windows uses the parent timeout; document this deployment boundary.
    try:
        data = sys.stdin.buffer.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise UserError("Upload limit exceeded.")
        units, warnings = _extract(data, sys.argv[1], use_ocr="--no-ocr" not in sys.argv[2:])
        output = {"units": units, "warnings": warnings}
    except UserError as exc:
        output = {"error": str(exc)}
    except Exception:
        output = {"error": "This document is damaged or unsupported. Try exporting a fresh copy."}
    sys.stdout.write(json.dumps(output))


if __name__ == "__main__":
    main()
