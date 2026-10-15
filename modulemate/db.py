"""Transactional SQLite storage. Only authenticated Stores expose user content."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

SCHEMA_VERSION = 2
SESSION_SECONDS = 12 * 60 * 60
MAX_DOCUMENTS = 100
MAX_CHUNKS = 12000


class UserError(ValueError):
    """A safe, actionable error that can be shown in the interface."""


class AuthenticationError(UserError):
    pass


def uid() -> str:
    return uuid.uuid4().hex


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def checked_text(value: str, label: str, limit: int) -> str:
    value = value.strip()
    if not value or len(value) > limit or "\x00" in value:
        raise UserError(f"{label} must contain 1–{limit:,} characters and no null bytes.")
    return value


def password_hash(password: str, salt: str) -> str:
    derived = hashlib.scrypt(
        password.encode(), salt=bytes.fromhex(salt), n=32768, r=8, p=3, maxmem=128 * 1024 * 1024
    ).hex()
    return "scrypt$32768$8$3$" + derived


def password_matches(password: str, user) -> bool:
    stored = user["password"]
    if stored.startswith("scrypt$32768$8$3$"):
        derived = password_hash(password, user["salt"])
    elif re.fullmatch(r"[0-9a-f]{128}", stored):
        # Compatibility with early local development accounts; upgrade on login.
        derived = hashlib.scrypt(
            password.encode(), salt=bytes.fromhex(user["salt"]), n=16384, r=8, p=1
        ).hex()
    else:
        return False
    return hmac.compare_digest(derived, stored)


def validate_password(password: str) -> None:
    if not 12 <= len(password) <= 256:
        raise UserError("Use a password with 12–256 characters, ideally a unique passphrase.")


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


SCHEMA = """
CREATE TABLE users (
 id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE, password TEXT NOT NULL,
 salt TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE sessions (
 token TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 expires_at REAL NOT NULL
);
CREATE TABLE login_attempts (username TEXT PRIMARY KEY, failures INTEGER NOT NULL, started REAL NOT NULL);
CREATE TABLE courses (
 id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 name TEXT NOT NULL, description TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX courses_owner ON courses(user_id);
CREATE TABLE documents (
 id TEXT PRIMARY KEY, course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
 name TEXT NOT NULL, digest TEXT NOT NULL, text TEXT NOT NULL, warnings TEXT NOT NULL,
 created_at TEXT NOT NULL, UNIQUE(course_id, digest)
);
CREATE INDEX documents_course ON documents(course_id);
CREATE TABLE chunks (
 id INTEGER PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
 locator TEXT NOT NULL, text TEXT NOT NULL, search_text TEXT NOT NULL, position INTEGER NOT NULL
);
CREATE INDEX chunks_document ON chunks(document_id);
CREATE VIRTUAL TABLE chunk_search USING fts5(search_text, content='chunks', content_rowid='id', tokenize='unicode61 remove_diacritics 2');
CREATE TRIGGER chunk_insert AFTER INSERT ON chunks BEGIN
 INSERT INTO chunk_search(rowid, search_text) VALUES (new.id, new.search_text);
END;
CREATE TRIGGER chunk_delete AFTER DELETE ON chunks BEGIN
 INSERT INTO chunk_search(chunk_search, rowid, search_text) VALUES ('delete', old.id, old.search_text);
END;
CREATE TABLE messages (
 id TEXT PRIMARY KEY, course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
 question TEXT NOT NULL, answer TEXT NOT NULL, sources TEXT NOT NULL, mode TEXT NOT NULL,
 created_at TEXT NOT NULL
);
CREATE INDEX messages_course ON messages(course_id, created_at);
CREATE TABLE cards (
 id TEXT PRIMARY KEY, course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
 document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
 front TEXT NOT NULL, back TEXT NOT NULL, locator TEXT NOT NULL,
 due TEXT NOT NULL, interval INTEGER NOT NULL DEFAULT 0, repetitions INTEGER NOT NULL DEFAULT 0,
 created_at TEXT NOT NULL, UNIQUE(course_id, front, back)
);
CREATE INDEX cards_course_due ON cards(course_id, due);
CREATE TABLE reviews (
 id TEXT PRIMARY KEY, card_id TEXT NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
 rating TEXT NOT NULL, reviewed_at TEXT NOT NULL
);
CREATE TABLE tasks (
 id TEXT PRIMARY KEY, course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
 title TEXT NOT NULL, due TEXT NOT NULL, minutes INTEGER NOT NULL,
 done INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
);
CREATE INDEX tasks_course_due ON tasks(course_id, due);
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as con:
            # Serialize first-run migration across sessions/processes.
            con.execute("BEGIN IMMEDIATE")
            version = con.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError("This database needs a newer ModuleMate version.")
            if version == 0:
                statement = ""
                for line in SCHEMA.splitlines(keepends=True):
                    statement += line
                    if sqlite3.complete_statement(statement):
                        con.execute(statement)
                        statement = ""
                version = 1
            if version < 2:
                con.execute("""CREATE TABLE embeddings (
                    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    model TEXT NOT NULL, digest TEXT NOT NULL, dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL, PRIMARY KEY(chunk_id, model)
                )""")
                con.execute("ALTER TABLE messages ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")
            con.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        self.path.chmod(0o600)

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=15)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA foreign_keys=ON")
            # Switching a brand-new database to WAL can return SQLITE_BUSY
            # immediately, even with busy_timeout, during concurrent first boot.
            for attempt in range(6):
                try:
                    con.execute("PRAGMA journal_mode=WAL")
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc) or attempt == 5:
                        raise
                    time.sleep(0.02 * (2**attempt))
            con.execute("PRAGMA secure_delete=ON")
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    def register(self, username: str, password: str, *, allowed: bool = True) -> str:
        if not allowed:
            raise UserError("Registration is disabled by the operator.")
        username = username.strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{2,39}", username):
            raise UserError(
                "Username: 3–40 lowercase letters, numbers, dots, dashes or underscores."
            )
        validate_password(password)
        salt = secrets.token_hex(16)
        hashed = password_hash(password, salt)
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            if con.execute("SELECT count(*) FROM users").fetchone()[0] >= 100:
                raise UserError("This installation has reached its 100-account limit.")
            user_id = uid()
            try:
                con.execute(
                    "INSERT INTO users VALUES (?,?,?,?,?)",
                    (user_id, username, hashed, salt, now_iso()),
                )
            except sqlite3.IntegrityError as exc:
                raise UserError("That username is unavailable.") from exc
            return self._new_session(con, user_id)

    def _new_session(self, con, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        con.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
        con.execute(
            "INSERT INTO sessions VALUES (?,?,?)",
            (token_hash(token), user_id, time.time() + SESSION_SECONDS),
        )
        return token

    def login(self, username: str, password: str) -> str:
        username = username.strip().lower()[:100]
        if len(password) > 256:
            raise AuthenticationError("Incorrect username or password.")
        error = None
        token = None
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            current = time.time()
            con.execute("DELETE FROM login_attempts WHERE started < ?", (current - 900,))
            attempt = con.execute(
                "SELECT * FROM login_attempts WHERE username=?", (username,)
            ).fetchone()
            if attempt and attempt["failures"] >= 5:
                raise AuthenticationError("Too many attempts. Try again in 15 minutes.")
            user = con.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
            candidate = user or {"salt": "00" * 16, "password": "scrypt$32768$8$3$" + "00" * 64}
            valid = password_matches(password, candidate)
            if not user or not valid:
                con.execute(
                    "INSERT INTO login_attempts VALUES (?,1,?) ON CONFLICT(username) DO UPDATE SET failures=failures+1",
                    (username, current),
                )
                error = "Incorrect username or password."
            else:
                if not user["password"].startswith("scrypt$"):
                    con.execute(
                        "UPDATE users SET password=? WHERE id=?",
                        (password_hash(password, user["salt"]), user["id"]),
                    )
                con.execute("DELETE FROM login_attempts WHERE username=?", (username,))
                token = self._new_session(con, user["id"])
        if error:
            raise AuthenticationError(error)
        return token

    def logout(self, token: str) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM sessions WHERE token=?", (token_hash(token),))

    def backup(self, destination: Path) -> None:
        """SQLite online backup; operator only, never exposed through the UI."""
        if destination.exists():
            raise UserError("Choose a new backup path; existing files are not overwritten.")
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as source, sqlite3.connect(destination) as target:
            source.backup(target)
        destination.chmod(0o600)


class Store:
    def __init__(self, db: Database, token: str):
        self.db = db
        self.token = token

    @contextmanager
    def transaction(self):
        with self.db.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            user = con.execute(
                "SELECT u.* FROM users u JOIN sessions s ON s.user_id=u.id WHERE s.token=? AND s.expires_at>?",
                (token_hash(self.token), time.time()),
            ).fetchone()
            if not user:
                raise AuthenticationError("Your session expired. Sign in again.")
            yield con, user

    def _course(self, con, user, course_id):
        row = con.execute(
            "SELECT * FROM courses WHERE id=? AND user_id=?", (course_id, user["id"])
        ).fetchone()
        if not row:
            raise UserError("Course not found in your account.")
        return row

    def _document(self, con, user, document_id):
        row = con.execute(
            "SELECT d.* FROM documents d JOIN courses c ON c.id=d.course_id WHERE d.id=? AND c.user_id=?",
            (document_id, user["id"]),
        ).fetchone()
        if not row:
            raise UserError("Document not found in your account.")
        return row

    def profile(self):
        with self.transaction() as (_, user):
            return {"username": user["username"], "created_at": user["created_at"]}

    def courses(self):
        with self.transaction() as (con, user):
            return [
                dict(r)
                for r in con.execute(
                    "SELECT c.*, (SELECT count(*) FROM documents d WHERE d.course_id=c.id) AS documents FROM courses c WHERE user_id=? ORDER BY created_at,id",
                    (user["id"],),
                )
            ]

    def create_course(self, name, description=""):
        name = checked_text(name, "Course name", 100)
        description = description.strip()[:1000]
        with self.transaction() as (con, user):
            if (
                con.execute(
                    "SELECT count(*) FROM courses WHERE user_id=?", (user["id"],)
                ).fetchone()[0]
                >= 30
            ):
                raise UserError("Limit: 30 courses per account. Remove an unused course first.")
            course_id = uid()
            con.execute(
                "INSERT INTO courses VALUES (?,?,?,?,?)",
                (course_id, user["id"], name, description, now_iso()),
            )
            return course_id

    def update_course(self, course_id, name, description):
        name = checked_text(name, "Course name", 100)
        description = description.strip()[:1000]
        with self.transaction() as (con, user):
            self._course(con, user, course_id)
            con.execute(
                "UPDATE courses SET name=?,description=? WHERE id=?", (name, description, course_id)
            )

    def delete_course(self, course_id):
        with self.transaction() as (con, user):
            self._course(con, user, course_id)
            con.execute("DELETE FROM courses WHERE id=?", (course_id,))

    def documents(self, course_id):
        with self.transaction() as (con, user):
            self._course(con, user, course_id)
            return [
                dict(r)
                for r in con.execute(
                    "SELECT d.id,d.name,d.created_at,d.warnings,length(d.text) AS characters,count(c.id) AS chunks FROM documents d LEFT JOIN chunks c ON c.document_id=d.id WHERE d.course_id=? GROUP BY d.id ORDER BY d.created_at,d.id",
                    (course_id,),
                )
            ]

    def document(self, document_id):
        with self.transaction() as (con, user):
            return dict(self._document(con, user, document_id))

    def document_passages(self, document_id):
        with self.transaction() as (con, user):
            self._document(con, user, document_id)
            return [
                dict(row)
                for row in con.execute(
                    "SELECT locator,text FROM chunks WHERE document_id=? ORDER BY position",
                    (document_id,),
                )
            ]

    def add_document(self, course_id, parsed):
        from modulemate.search import index_text

        with self.transaction() as (con, user):
            self._course(con, user, course_id)
            existing = con.execute(
                "SELECT id FROM documents WHERE course_id=? AND digest=?",
                (course_id, parsed.digest),
            ).fetchone()
            if existing:
                return existing[0], False
            total = con.execute(
                "SELECT count(*) FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.course_id=?",
                (course_id,),
            ).fetchone()[0]
            count = con.execute(
                "SELECT count(*) FROM documents WHERE course_id=?", (course_id,)
            ).fetchone()[0]
            if count >= MAX_DOCUMENTS or total + len(parsed.chunks) > MAX_CHUNKS:
                raise UserError(
                    "Course limit reached (100 documents / 12,000 passages). Remove old material first."
                )
            document_id = uid()
            con.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?)",
                (
                    document_id,
                    course_id,
                    parsed.name,
                    parsed.digest,
                    parsed.text,
                    json.dumps(parsed.warnings),
                    now_iso(),
                ),
            )
            con.executemany(
                "INSERT INTO chunks(document_id,locator,text,search_text,position) VALUES (?,?,?,?,?)",
                [
                    (document_id, c.locator, c.text, index_text(c.text), i)
                    for i, c in enumerate(parsed.chunks)
                ],
            )
            return document_id, True

    def delete_document(self, document_id):
        with self.transaction() as (con, user):
            doc = self._document(con, user, document_id)
            # Chat text can paraphrase a removed source: clear the whole course history.
            con.execute("DELETE FROM messages WHERE course_id=?", (doc["course_id"],))
            con.execute("DELETE FROM documents WHERE id=?", (document_id,))

    def messages(self, course_id, limit=40):
        with self.transaction() as (con, user):
            self._course(con, user, course_id)
            rows = con.execute(
                "SELECT * FROM messages WHERE course_id=? ORDER BY created_at DESC,id DESC LIMIT ?",
                (course_id, min(max(limit, 1), 200)),
            ).fetchall()
            return [
                dict(r)
                | {"sources": json.loads(r["sources"]), "metadata": json.loads(r["metadata"])}
                for r in reversed(rows)
            ]

    def add_message(self, course_id, question, answer, sources, mode, metadata=None):
        question = checked_text(question, "Question", 4000)
        answer = checked_text(answer, "Answer", 30000)
        with self.transaction() as (con, user):
            self._course(con, user, course_id)
            for source in sources:
                doc = self._document(con, user, source["document_id"])
                if doc["course_id"] != course_id:
                    raise UserError("Source is not in the selected course.")
            con.execute(
                "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)",
                (
                    uid(),
                    course_id,
                    question,
                    answer,
                    json.dumps(sources),
                    mode,
                    now_iso(),
                    json.dumps(metadata or {}),
                ),
            )
            con.execute(
                "DELETE FROM messages WHERE course_id=? AND id NOT IN (SELECT id FROM messages WHERE course_id=? ORDER BY created_at DESC,id DESC LIMIT 200)",
                (course_id, course_id),
            )

    def clear_messages(self, course_id):
        with self.transaction() as (con, user):
            self._course(con, user, course_id)
            con.execute("DELETE FROM messages WHERE course_id=?", (course_id,))

    def add_card(self, course_id, front, back, document_id=None, locator="", *, today=None):
        front = checked_text(front, "Card question", 1000)
        back = checked_text(back, "Card answer", 4000)
        with self.transaction() as (con, user):
            self._course(con, user, course_id)
            if document_id:
                doc = self._document(con, user, document_id)
                if doc["course_id"] != course_id:
                    raise UserError("Card source is not in the selected course.")
                if not con.execute(
                    "SELECT 1 FROM chunks WHERE document_id=? AND locator=?", (document_id, locator)
                ).fetchone():
                    raise UserError("Card source location is no longer available.")
            if (
                con.execute(
                    "SELECT count(*) FROM cards WHERE course_id=?", (course_id,)
                ).fetchone()[0]
                >= 2000
            ):
                raise UserError("Limit: 2,000 cards per course.")
            card_id = uid()
            try:
                con.execute(
                    "INSERT INTO cards(id,course_id,document_id,front,back,locator,due,created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        card_id,
                        course_id,
                        document_id,
                        front,
                        back,
                        locator,
                        (today or date.today()).isoformat(),
                        now_iso(),
                    ),
                )
            except sqlite3.IntegrityError:
                return False
            return True

    def cards(self, course_id):
        with self.transaction() as (con, user):
            self._course(con, user, course_id)
            return [
                dict(r)
                for r in con.execute(
                    "SELECT c.*,d.name AS document_name FROM cards c LEFT JOIN documents d ON d.id=c.document_id WHERE c.course_id=? ORDER BY c.due,c.created_at,c.id",
                    (course_id,),
                )
            ]

    def _card(self, con, user, card_id):
        row = con.execute(
            "SELECT k.* FROM cards k JOIN courses c ON c.id=k.course_id WHERE k.id=? AND c.user_id=?",
            (card_id, user["id"]),
        ).fetchone()
        if not row:
            raise UserError("Card not found in your account.")
        return row

    def update_card(self, card_id, front, back):
        front = checked_text(front, "Card question", 1000)
        back = checked_text(back, "Card answer", 4000)
        with self.transaction() as (con, user):
            self._card(con, user, card_id)
            try:
                con.execute("UPDATE cards SET front=?,back=? WHERE id=?", (front, back, card_id))
            except sqlite3.IntegrityError as exc:
                raise UserError("An identical card already exists.") from exc

    def delete_card(self, card_id):
        with self.transaction() as (con, user):
            self._card(con, user, card_id)
            con.execute("DELETE FROM cards WHERE id=?", (card_id,))

    def review_card(self, card_id, rating, *, today=None):
        from modulemate.study import schedule_review

        today = today or date.today()
        with self.transaction() as (con, user):
            card = self._card(con, user, card_id)
            if date.fromisoformat(card["due"]) > today:
                raise UserError("This card is not due yet. Use Practice for extra revision.")
            interval, repetitions = schedule_review(card["interval"], card["repetitions"], rating)
            # Again returns tomorrow; an in-session practice quiz is available for repeats.
            due = (today + timedelta(days=interval)).isoformat()
            con.execute(
                "UPDATE cards SET interval=?,repetitions=?,due=? WHERE id=?",
                (interval, repetitions, due, card_id),
            )
            con.execute("INSERT INTO reviews VALUES (?,?,?,?)", (uid(), card_id, rating, now_iso()))

    def review_counts(self, course_id):
        with self.transaction() as (con, user):
            self._course(con, user, course_id)
            return dict(
                con.execute(
                    "SELECT r.rating,count(*) FROM reviews r JOIN cards c ON c.id=r.card_id WHERE c.course_id=? GROUP BY r.rating",
                    (course_id,),
                ).fetchall()
            )

    def add_task(self, course_id, title, due, minutes):
        title = checked_text(title, "Task title", 200)
        try:
            date.fromisoformat(due)
        except (TypeError, ValueError) as exc:
            raise UserError("Choose a valid due date.") from exc
        if isinstance(minutes, bool) or not isinstance(minutes, int) or not 5 <= minutes <= 2400:
            raise UserError("Estimate 5–2,400 minutes.")
        with self.transaction() as (con, user):
            self._course(con, user, course_id)
            if (
                con.execute(
                    "SELECT count(*) FROM tasks WHERE course_id=?", (course_id,)
                ).fetchone()[0]
                >= 1000
            ):
                raise UserError("Limit: 1,000 tasks per course.")
            con.execute(
                "INSERT INTO tasks VALUES (?,?,?,?,?,0,?)",
                (uid(), course_id, title, due, minutes, now_iso()),
            )

    def update_task(self, task_id, title, due, minutes):
        title = checked_text(title, "Task title", 200)
        try:
            date.fromisoformat(due)
        except (TypeError, ValueError) as exc:
            raise UserError("Choose a valid due date.") from exc
        if isinstance(minutes, bool) or not isinstance(minutes, int) or not 5 <= minutes <= 2400:
            raise UserError("Estimate 5–2,400 minutes.")
        with self.transaction() as (con, user):
            row = con.execute(
                "SELECT t.id FROM tasks t JOIN courses c ON c.id=t.course_id WHERE t.id=? AND c.user_id=?",
                (task_id, user["id"]),
            ).fetchone()
            if not row:
                raise UserError("Task not found in your account.")
            con.execute(
                "UPDATE tasks SET title=?,due=?,minutes=? WHERE id=?",
                (title, due, minutes, task_id),
            )

    def tasks(self, course_id):
        with self.transaction() as (con, user):
            self._course(con, user, course_id)
            return [
                dict(r)
                for r in con.execute(
                    "SELECT * FROM tasks WHERE course_id=? ORDER BY done,due,created_at",
                    (course_id,),
                )
            ]

    def modify_task(self, task_id, *, done=None, delete=False):
        with self.transaction() as (con, user):
            row = con.execute(
                "SELECT t.id FROM tasks t JOIN courses c ON c.id=t.course_id WHERE t.id=? AND c.user_id=?",
                (task_id, user["id"]),
            ).fetchone()
            if not row:
                raise UserError("Task not found in your account.")
            if delete:
                con.execute("DELETE FROM tasks WHERE id=?", (task_id,))
            elif isinstance(done, bool):
                con.execute("UPDATE tasks SET done=? WHERE id=?", (int(done), task_id))

    def change_password(self, old, new):
        validate_password(new)
        if len(old) > 256:
            raise AuthenticationError("Incorrect password.")
        with self.transaction() as (con, user):
            if not password_matches(old, user):
                raise AuthenticationError("Incorrect password.")
            salt = secrets.token_hex(16)
            con.execute(
                "UPDATE users SET salt=?,password=? WHERE id=?",
                (salt, password_hash(new, salt), user["id"]),
            )
            con.execute(
                "DELETE FROM sessions WHERE user_id=? AND token!=?",
                (user["id"], token_hash(self.token)),
            )

    def delete_account(self, password):
        if len(password) > 256:
            raise AuthenticationError("Incorrect password.")
        with self.transaction() as (con, user):
            if not password_matches(password, user):
                raise AuthenticationError("Incorrect password.")
            con.execute("DELETE FROM users WHERE id=?", (user["id"],))
            con.execute("DELETE FROM login_attempts WHERE username=?", (user["username"],))

    def export(self):
        with self.transaction() as (con, user):
            output = {
                "format": "modulemate-export-v1",
                "exported_at": now_iso(),
                "username": user["username"],
                "courses": [],
            }
            for row in con.execute(
                "SELECT * FROM courses WHERE user_id=? ORDER BY created_at", (user["id"],)
            ):
                course = dict(row)
                course.pop("user_id")
                for table in ("documents", "cards", "tasks", "messages"):
                    course[table] = [
                        dict(r)
                        for r in con.execute(
                            f"SELECT * FROM {table} WHERE course_id=?", (row["id"],)
                        )
                    ]
                course["reviews"] = [
                    dict(r)
                    for r in con.execute(
                        "SELECT r.* FROM reviews r JOIN cards c ON c.id=r.card_id WHERE c.course_id=?",
                        (row["id"],),
                    )
                ]
                output["courses"].append(course)
            return output
