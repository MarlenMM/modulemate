"""Operator-only consistent backups and explicit session revocation."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulemate.db import Database  # noqa: E402
from modulemate.settings import load_settings  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", type=Path)
    parser.add_argument(
        "--revoke-sessions",
        action="store_true",
        help="Revoke all sessions in the configured database (e.g. after a restore).",
    )
    args = parser.parse_args()
    if not args.destination and not args.revoke_sessions:
        parser.error("Supply a new backup destination or --revoke-sessions")
    path = load_settings().data_dir / "modulemate.sqlite3"
    if not path.exists():
        parser.error("No configured database exists yet")
    db = Database(path)
    if args.revoke_sessions:
        with db.connect() as con:
            con.execute("DELETE FROM sessions")
        print("All sessions revoked in the configured database.")
    if args.destination:
        db.backup(args.destination.expanduser().resolve())
        print("Consistent private backup written.")


if __name__ == "__main__":
    main()
