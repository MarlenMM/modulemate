import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pytest

from modulemate.db import SCHEMA_VERSION, Database, Store, UserError
from modulemate.ingest import parse_upload
from modulemate.search import search


def test_schema_initialization_is_concurrent_and_idempotent(tmp_path):
    path = tmp_path / "race.sqlite3"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: Database(path), range(4)))
    with sqlite3.connect(path) as con:
        assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_course_filters_do_not_cross_even_for_same_user(store, populated):
    course, document = populated
    second = store.create_course("Math")
    with pytest.raises(UserError):
        search(store, second, "retrieval", [document])
    assert search(store, second, "retrieval") == []
    assert search(store, course, "retrieval", []) == []


def test_document_dedupe_is_course_local(store, course):
    parsed = parse_upload("note.txt", b"A very short note.")
    doc, added = store.add_document(course, parsed)
    assert added
    renamed = parse_upload("renamed.txt", b"A very short note.")
    assert store.add_document(course, renamed) == (doc, False)
    second = store.create_course("Second")
    assert store.add_document(second, parsed)[1]


def test_deletion_cascades_and_fts_is_fresh(db, store, populated):
    course, document = populated
    passage = search(store, course, "retrieval")[0]
    store.add_card(course, "Question", "Answer", document, passage.locator)
    card = store.cards(course)[0]
    store.review_card(card["id"], "Good", today=date.fromisoformat(card["due"]))
    store.add_message(course, "q", "contains private information", [passage.to_dict()], "evidence")
    store.delete_document(document)
    assert search(store, course, "retrieval") == []
    assert store.cards(course) == []
    assert store.messages(course) == []
    with db.connect() as con:
        assert con.execute("SELECT count(*) FROM reviews").fetchone()[0] == 0
        con.execute("INSERT INTO chunk_search(chunk_search,rank) VALUES('integrity-check',1)")


def test_online_backup_roundtrip(db, store, populated, tmp_path):
    path = tmp_path / "backup.sqlite3"
    db.backup(path)
    restored = Store(Database(path))
    assert restored.export()["courses"] == store.export()["courses"]
    with pytest.raises(UserError):
        db.backup(path)


def test_failed_ingestion_is_atomic(db, store, course):
    parsed = parse_upload("ok.txt", b"Some sample text.")
    with db.connect() as con:
        con.execute(
            "CREATE TRIGGER fail_chunk BEFORE INSERT ON chunks BEGIN SELECT RAISE(ABORT,'test failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        store.add_document(course, parsed)
    assert store.documents(course) == []


def test_task_edit_preserves_id_and_enforces_ownership(db, store, course):
    store.add_task(course, "Read", "2026-10-20", 20)
    task_id = store.tasks(course)[0]["id"]
    store.update_task(task_id, "Read more", "2026-10-21", 40)
    updated = store.tasks(course)[0]
    assert updated["id"] == task_id and updated["minutes"] == 40
    with pytest.raises(UserError):
        store.update_task("missing-task", "Missing", "2026-10-21", 40)
    with pytest.raises(UserError):
        store.update_task(task_id, "Invalid", "not-a-date", 40)


def test_local_workspace_has_no_credentials_and_is_shared(db, store, populated):
    assert Store(db).courses() == store.courses()
    with db.connect() as con:
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert not names & {"users", "sessions", "login_attempts"}
        assert {r[1] for r in con.execute("PRAGMA table_info(workspace)")} == {
            "id",
            "name",
            "created_at",
        }
    raw = json.dumps(store.export())
    assert '"password"' not in raw and '"token"' not in raw
    store.reset_workspace()
    assert store.courses() == []
    assert store.create_course("Start again")


def test_v2_migration_preserves_all_accounts_and_dependent_data(tmp_path):
    from pathlib import Path

    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as con:
        con.executescript((Path(__file__).parent / "fixtures/legacy_v2.sql").read_text())
        for i in (1, 2):
            con.execute(
                "INSERT INTO users VALUES (?,?,?,?,?)",
                (str(i), f"student{i}", "legacy-hash", "legacy-salt", "2026-01-01"),
            )
            con.execute("INSERT INTO sessions VALUES (?,?,?)", (f"token{i}", str(i), 9999999999))
            con.execute(
                "INSERT INTO courses VALUES (?,?,?,?,?)",
                (f"course{i}", str(i), "Same name", "", "2026-01-01"),
            )
            con.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?)",
                (
                    f"doc{i}",
                    f"course{i}",
                    "Notes",
                    f"digest{i}",
                    f"Private fact {i}",
                    "[]",
                    "2026-01-01",
                ),
            )
            con.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                (i, f"doc{i}", "part 1", f"Private fact {i}", f"private fact {i}", 0),
            )
            con.execute(
                "INSERT INTO embeddings VALUES (?,?,?,?,?)", (i, "test", "digest", 1, b"1234")
            )
            con.execute(
                "INSERT INTO cards VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    f"card{i}",
                    f"course{i}",
                    f"doc{i}",
                    "Question",
                    "Answer",
                    "part 1",
                    "2026-01-01",
                    0,
                    0,
                    "2026-01-01",
                ),
            )
            con.execute(
                "INSERT INTO reviews VALUES (?,?,?,?)",
                (f"review{i}", f"card{i}", "Good", "2026-01-01"),
            )
            con.execute(
                "INSERT INTO tasks VALUES (?,?,?,?,?,?,?)",
                (f"task{i}", f"course{i}", "Read", "2026-01-01", 30, 0, "2026-01-01"),
            )
            con.execute(
                "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)",
                (
                    f"message{i}",
                    f"course{i}",
                    "Question",
                    "Answer",
                    "[]",
                    "Explain",
                    "2026-01-01",
                    '{"topic":"preserved"}',
                ),
            )
    store = Store(Database(path))
    assert [c["id"] for c in store.courses()] == ["course1", "course2"]
    assert store.document("doc2")["text"] == "Private fact 2"
    assert store.messages("course1")[0]["metadata"]["topic"] == "preserved"
    with store.db.connect() as con:
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not con.execute("PRAGMA foreign_key_check").fetchall()
        assert con.execute("SELECT id FROM workspace").fetchone()[0] == "local"
        for table in (
            "courses",
            "documents",
            "chunks",
            "embeddings",
            "cards",
            "reviews",
            "tasks",
            "messages",
        ):
            assert con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 2
    backups = list(tmp_path.glob("legacy.pre-v3-*.sqlite3"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        assert backup.execute("SELECT count(*) FROM users").fetchone()[0] == 2
    Database(path)
    assert len(list(tmp_path.glob("legacy.pre-v3-*.sqlite3"))) == 1


@pytest.mark.parametrize("legacy_version", [1, 2])
def test_empty_legacy_workspace_starts_without_an_account(tmp_path, legacy_version):
    from pathlib import Path

    path = tmp_path / "empty-legacy.sqlite3"
    with sqlite3.connect(path) as con:
        con.executescript((Path(__file__).parent / "fixtures/legacy_v2.sql").read_text())
        if legacy_version == 1:
            con.execute("DROP TABLE embeddings")
            con.execute("ALTER TABLE messages DROP COLUMN metadata")
            con.execute("PRAGMA user_version=1")
    migrated = Store(Database(path))
    assert migrated.courses() == []
    course = migrated.create_course("First local course")
    assert migrated.courses()[0]["id"] == course
    with migrated.db.connect() as con:
        assert not con.execute("PRAGMA foreign_key_check").fetchall()
    backups = list(tmp_path.glob("empty-legacy.pre-v3-*.sqlite3"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        assert backup.execute("PRAGMA user_version").fetchone()[0] == legacy_version
