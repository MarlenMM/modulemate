import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pytest

from modulemate.db import AuthenticationError, Database, Store, UserError, token_hash
from modulemate.ingest import parse_upload
from modulemate.search import overview, search


def test_schema_initialization_is_concurrent_and_idempotent(tmp_path):
    path = tmp_path / "race.sqlite3"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: Database(path), range(4)))
    with sqlite3.connect(path) as con:
        assert con.execute("PRAGMA user_version").fetchone()[0] == 1
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_password_and_session_are_hashed(db, store):
    with db.connect() as con:
        user = con.execute("SELECT * FROM users").fetchone()
        session = con.execute("SELECT * FROM sessions").fetchone()
        assert user["password"] != "a strong password"
        assert session["token"] != store.token
        assert session["token"] == token_hash(store.token)
    other = Store(db, db.login("STUDENT", "a strong password"))
    assert other.profile()["username"] == "student"
    db.logout(other.token)
    with pytest.raises(AuthenticationError):
        other.profile()


def test_rate_limit_failures_survive_transaction(db, store):
    for _ in range(5):
        with pytest.raises(AuthenticationError, match="Incorrect"):
            db.login("student", "wrong")
    with pytest.raises(AuthenticationError, match="Too many"):
        db.login("student", "a strong password")


def test_expired_session_cannot_read(db, store):
    with db.connect() as con:
        con.execute("UPDATE sessions SET expires_at=0")
    with pytest.raises(AuthenticationError):
        store.courses()


def test_registration_validation(db):
    for name, password in [("bad space", "a valid password"), ("valid", "short")]:
        with pytest.raises(UserError):
            db.register(name, password)
    with pytest.raises(UserError, match="disabled"):
        db.register("valid", "a valid password", allowed=False)


def test_cross_account_operations_are_denied(db, store, populated):
    course, document = populated
    store.add_card(course, "What?", "This.", document, "part 1")
    card = store.cards(course)[0]["id"]
    store.add_task(course, "Read", "2026-10-20", 30)
    task = store.tasks(course)[0]["id"]
    other = Store(db, db.register("second", "another password"))
    own_course = other.create_course("Other")
    parsed = parse_upload("secret.txt", b"This is confidential source material.")
    operations = [
        lambda: other.documents(course),
        lambda: other.document(document),
        lambda: other.document_passages(document),
        lambda: other.delete_document(document),
        lambda: other.delete_course(course),
        lambda: other.update_course(course, "stolen", ""),
        lambda: other.add_document(course, parsed),
        lambda: search(other, course, "retrieval"),
        lambda: overview(other, course),
        lambda: other.messages(course),
        lambda: other.add_message(course, "question", "answer", [], "mode"),
        lambda: other.clear_messages(course),
        lambda: other.cards(course),
        lambda: other.add_card(course, "q", "a"),
        lambda: other.review_counts(course),
        lambda: other.update_card(card, "changed", "answer"),
        lambda: other.review_card(card, "Good"),
        lambda: other.delete_card(card),
        lambda: other.tasks(course),
        lambda: other.modify_task(task, done=True),
        lambda: other.modify_task(task, delete=True),
        lambda: other.add_task(course, "x", "2026-10-20", 20),
        lambda: other.add_card(own_course, "q", "a", document, "part 1"),
        lambda: search(other, own_course, "retrieval", [document]),
        lambda: other.add_message(own_course, "q", "a", [{"document_id": document}], "x"),
    ]
    for operation in operations:
        with pytest.raises(UserError):
            operation()
    assert search(other, own_course, "retrieval") == []
    assert "retrieval practice" not in json.dumps(other.export()).lower()
    assert store.document(document)["name"] == "Learning Lab.md"


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


def test_password_change_invalidates_other_sessions(db, store):
    other = Store(db, db.login("student", "a strong password"))
    store.change_password("a strong password", "replacement password")
    assert store.profile()["username"] == "student"
    with pytest.raises(AuthenticationError):
        other.profile()
    with pytest.raises(AuthenticationError):
        db.login("student", "a strong password")
    assert db.login("student", "replacement password")


def test_export_and_delete_account(db, store, populated):
    course, _ = populated
    store.add_card(course, "q", "a")
    store.add_task(course, "read", "2026-10-20", 30)
    exported = store.export()
    assert exported["courses"][0]["documents"]
    raw = json.dumps(exported)
    assert store.token not in raw and "password" not in raw
    with pytest.raises(AuthenticationError):
        store.delete_account("wrong password")
    store.delete_account("a strong password")
    with db.connect() as con:
        for table in ("users", "sessions", "courses", "documents", "chunks", "cards", "tasks"):
            assert con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_online_backup_roundtrip(db, store, populated, tmp_path):
    path = tmp_path / "backup.sqlite3"
    db.backup(path)
    restored = Store(Database(path), store.token)
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


def test_password_hash_parameters_and_legacy_upgrade(db, store):
    import hashlib

    with db.connect() as con:
        user = con.execute("SELECT * FROM users").fetchone()
        assert user["password"].startswith("scrypt$32768$8$3$")
        legacy = hashlib.scrypt(
            b"a strong password", salt=bytes.fromhex(user["salt"]), n=16384, r=8, p=1
        ).hex()
        con.execute("UPDATE users SET password=?", (legacy,))
    assert db.login("student", "a strong password")
    with db.connect() as con:
        assert con.execute("SELECT password FROM users").fetchone()[0].startswith("scrypt$")


def test_task_edit_preserves_id_and_enforces_ownership(db, store, course):
    store.add_task(course, "Read", "2026-10-20", 20)
    task_id = store.tasks(course)[0]["id"]
    store.update_task(task_id, "Read more", "2026-10-21", 40)
    updated = store.tasks(course)[0]
    assert updated["id"] == task_id and updated["minutes"] == 40
    other = Store(db, db.register("second", "another password"))
    with pytest.raises(UserError):
        other.update_task(task_id, "Stolen", "2026-10-21", 40)
    with pytest.raises(UserError):
        store.update_task(task_id, "Invalid", "not-a-date", 40)
