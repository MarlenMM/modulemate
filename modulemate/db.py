"""Transactional single-user storage, with lossless migration from account databases."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

SCHEMA_VERSION = 4
MAX_DOCUMENTS = 100
MAX_CHUNKS = 12000


class UserError(ValueError):
    """A safe, actionable error that can be shown in the interface."""


def uid() -> str:
    return uuid.uuid4().hex


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def checked_text(value: str, label: str, limit: int) -> str:
    value = value.strip()
    if not value or len(value) > limit or "\x00" in value:
        raise UserError(f"{label} must contain 1–{limit:,} characters and no null bytes.")
    return value


CARD_PROVENANCE_SCHEMA = """
CREATE TABLE card_sources (
 card_id TEXT NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
 document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
 passage_id INTEGER NOT NULL, locator TEXT NOT NULL, quote TEXT NOT NULL,
 position INTEGER NOT NULL, PRIMARY KEY(card_id, position)
);
CREATE INDEX card_sources_document ON card_sources(document_id);
CREATE TRIGGER document_cards BEFORE DELETE ON documents BEGIN
 DELETE FROM cards WHERE id IN (SELECT card_id FROM card_sources WHERE document_id=old.id);
END;
"""


def _execute_schema(con, script):
    # executescript would implicitly commit the surrounding migration.
    statement = ""
    for line in script.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            con.execute(statement)
            statement = ""


SCHEMA = (
    """
CREATE TABLE workspace (
 id TEXT PRIMARY KEY CHECK(id='local'), name TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE courses (
 id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
 name TEXT NOT NULL, description TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX courses_owner ON courses(workspace_id);
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
 created_at TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX messages_course ON messages(course_id, created_at);
CREATE TABLE cards (
 id TEXT PRIMARY KEY, course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
 document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
 front TEXT NOT NULL, back TEXT NOT NULL, locator TEXT NOT NULL,
 due TEXT NOT NULL, interval INTEGER NOT NULL DEFAULT 0, repetitions INTEGER NOT NULL DEFAULT 0,
 created_at TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}', UNIQUE(course_id, front, back)
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
CREATE TABLE embeddings (
 chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
 model TEXT NOT NULL, digest TEXT NOT NULL, dimensions INTEGER NOT NULL,
 vector BLOB NOT NULL, PRIMARY KEY(chunk_id, model)
);
"""
    + CARD_PROVENANCE_SCHEMA
)


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Back up the complete legacy database before changing any account tables.
        # The online backup includes committed WAL content; copies are private and ignored.
        if self.path.exists():
            with sqlite3.connect(self.path) as source:
                source_version = source.execute("PRAGMA user_version").fetchone()[0]
                if 1 <= source_version < SCHEMA_VERSION:
                    target_version = 3 if source_version < 3 else SCHEMA_VERSION
                    backup = self.path.with_name(
                        f"{self.path.stem}.pre-v{target_version}-{uid()[:8]}.sqlite3"
                    )
                    with sqlite3.connect(backup) as target:
                        source.backup(target)
                    backup.chmod(0o600)
        with self.connect() as con:
            # Serialize first-run migration across sessions/processes.
            con.execute("BEGIN IMMEDIATE")
            version = con.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError("This database needs a newer ModuleMate version.")
            if version == 0:
                _execute_schema(con, SCHEMA)
                con.execute("INSERT INTO workspace VALUES ('local','My workspace',?)", (now_iso(),))
                version = SCHEMA_VERSION
            if version < 2:
                con.execute("""CREATE TABLE embeddings (
                    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    model TEXT NOT NULL, digest TEXT NOT NULL, dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL, PRIMARY KEY(chunk_id, model)
                )""")
                con.execute("ALTER TABLE messages ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")
            if version < 3:
                # All previous local accounts now share this one workspace. Keep course
                # IDs, names and every dependent record, including duplicate course names.
                # SQLite renames update dependent foreign keys. Merge into an existing
                # owner first, then remove credential columns without dropping a parent.
                owner = con.execute(
                    "SELECT id FROM users ORDER BY created_at,id LIMIT 1"
                ).fetchone()
                if owner:
                    owner_id = owner[0]
                    con.execute("UPDATE courses SET user_id=?", (owner_id,))
                    con.execute("DELETE FROM sessions")
                    con.execute("DELETE FROM users WHERE id!=?", (owner_id,))
                else:
                    owner_id = "local"
                    con.execute("INSERT INTO users VALUES ('local','local','','',?)", (now_iso(),))
                con.execute("DROP TABLE sessions")
                con.execute("DROP TABLE login_attempts")
                con.execute("ALTER TABLE users DROP COLUMN password")
                con.execute("ALTER TABLE users DROP COLUMN salt")
                con.execute("ALTER TABLE users RENAME COLUMN username TO name")
                con.execute("ALTER TABLE users RENAME TO workspace")
                con.execute("ALTER TABLE courses RENAME COLUMN user_id TO workspace_id")
                con.execute("UPDATE workspace SET name='My workspace'")
                # Legacy IDs can remain internal; the Store never needs a credential.
            if version < 4:
                con.execute("ALTER TABLE cards ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")
                _execute_schema(con, CARD_PROVENANCE_SCHEMA)
            owner = con.execute("SELECT id FROM workspace LIMIT 1").fetchone()
            if owner and owner[0] != "local":
                # Canonicalize the internal owner without deleting/reinserting
                # a parent (which would cascade through the user's library).
                con.execute("PRAGMA defer_foreign_keys=ON")
                con.execute("UPDATE workspace SET id='local' WHERE id=?", (owner[0],))
                con.execute(
                    "UPDATE courses SET workspace_id='local' WHERE workspace_id=?", (owner[0],)
                )
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

    def backup(self, destination: Path) -> None:
        """SQLite online backup; operator only, never exposed through the UI."""
        if destination.exists():
            raise UserError("Choose a new backup path; existing files are not overwritten.")
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as source, sqlite3.connect(destination) as target:
            source.backup(target)
        destination.chmod(0o600)


class Store:
    def __init__(self, db: Database):
        self.db = db

    @contextmanager
    def transaction(self):
        with self.db.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            workspace = con.execute("SELECT * FROM workspace LIMIT 1").fetchone()
            if not workspace:
                raise UserError("The workspace is missing. Restore a database backup.")
            yield con, workspace

    def _course(self, con, workspace, course_id):
        row = con.execute(
            "SELECT * FROM courses WHERE id=? AND workspace_id=?", (course_id, workspace["id"])
        ).fetchone()
        if not row:
            raise UserError("Course not found in this workspace.")
        return row

    def _document(self, con, workspace, document_id):
        row = con.execute(
            "SELECT d.* FROM documents d JOIN courses c ON c.id=d.course_id WHERE d.id=? AND c.workspace_id=?",
            (document_id, workspace["id"]),
        ).fetchone()
        if not row:
            raise UserError("Document not found in this workspace.")
        return row

    def profile(self):
        with self.transaction() as (_, workspace):
            return {"workspace": workspace["name"], "created_at": workspace["created_at"]}

    def courses(self):
        with self.transaction() as (con, workspace):
            return [
                dict(r)
                for r in con.execute(
                    "SELECT c.*, (SELECT count(*) FROM documents d WHERE d.course_id=c.id) AS documents FROM courses c WHERE workspace_id=? ORDER BY created_at,id",
                    (workspace["id"],),
                )
            ]

    def create_course(self, name, description=""):
        name = checked_text(name, "Course name", 100)
        description = description.strip()[:1000]
        with self.transaction() as (con, workspace):
            if (
                con.execute(
                    "SELECT count(*) FROM courses WHERE workspace_id=?", (workspace["id"],)
                ).fetchone()[0]
                >= 30
            ):
                raise UserError("Limit: 30 courses per workspace. Remove an unused course first.")
            course_id = uid()
            con.execute(
                "INSERT INTO courses VALUES (?,?,?,?,?)",
                (course_id, workspace["id"], name, description, now_iso()),
            )
            return course_id

    def update_course(self, course_id, name, description):
        name = checked_text(name, "Course name", 100)
        description = description.strip()[:1000]
        with self.transaction() as (con, workspace):
            self._course(con, workspace, course_id)
            con.execute(
                "UPDATE courses SET name=?,description=? WHERE id=?", (name, description, course_id)
            )

    def delete_course(self, course_id):
        with self.transaction() as (con, workspace):
            self._course(con, workspace, course_id)
            con.execute("DELETE FROM courses WHERE id=?", (course_id,))

    def documents(self, course_id):
        with self.transaction() as (con, workspace):
            self._course(con, workspace, course_id)
            return [
                dict(r)
                for r in con.execute(
                    "SELECT d.id,d.name,d.created_at,d.warnings,length(d.text) AS characters,count(c.id) AS chunks FROM documents d LEFT JOIN chunks c ON c.document_id=d.id WHERE d.course_id=? GROUP BY d.id ORDER BY d.created_at,d.id",
                    (course_id,),
                )
            ]

    def document(self, document_id):
        with self.transaction() as (con, workspace):
            return dict(self._document(con, workspace, document_id))

    def document_passages(self, document_id):
        with self.transaction() as (con, workspace):
            self._document(con, workspace, document_id)
            return [
                dict(row)
                for row in con.execute(
                    "SELECT locator,text FROM chunks WHERE document_id=? ORDER BY position",
                    (document_id,),
                )
            ]

    def add_document(self, course_id, parsed):
        from modulemate.search import index_text

        with self.transaction() as (con, workspace):
            self._course(con, workspace, course_id)
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
                    (
                        document_id,
                        c.locator,
                        c.text,
                        index_text(f"{parsed.name}\n{c.locator}\n{c.text}"),
                        i,
                    )
                    for i, c in enumerate(parsed.chunks)
                ],
            )
            return document_id, True

    def delete_document(self, document_id):
        with self.transaction() as (con, workspace):
            doc = self._document(con, workspace, document_id)
            # Chat text can paraphrase a removed source: clear the whole course history.
            con.execute("DELETE FROM messages WHERE course_id=?", (doc["course_id"],))
            con.execute("DELETE FROM documents WHERE id=?", (document_id,))

    def messages(self, course_id, limit=40):
        with self.transaction() as (con, workspace):
            self._course(con, workspace, course_id)
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
        with self.transaction() as (con, workspace):
            self._course(con, workspace, course_id)
            for source in sources:
                doc = self._document(con, workspace, source["document_id"])
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
        with self.transaction() as (con, workspace):
            self._course(con, workspace, course_id)
            con.execute("DELETE FROM messages WHERE course_id=?", (course_id,))

    def save_answer_feedback(self, course_id, message_id, rating, comment=""):
        """Local review labels; they are never automatically treated as training truth."""
        if rating not in {"helpful", "incorrect", "unsupported", "unclear"}:
            raise UserError("Choose a supported answer rating.")
        if not isinstance(comment, str) or len(comment) > 2000 or "\x00" in comment:
            raise UserError("Keep the feedback note within 2,000 characters.")
        with self.transaction() as (con, workspace):
            self._course(con, workspace, course_id)
            row = con.execute(
                "SELECT metadata FROM messages WHERE id=? AND course_id=?", (message_id, course_id)
            ).fetchone()
            if row is None:
                raise UserError("Answer not found in this course.")
            metadata = json.loads(row["metadata"])
            metadata["feedback"] = {
                "rating": rating,
                "comment": comment.strip(),
                "updated_at": now_iso(),
            }
            con.execute(
                "UPDATE messages SET metadata=? WHERE id=? AND course_id=?",
                (json.dumps(metadata), message_id, course_id),
            )

    def add_card(
        self,
        course_id,
        front,
        back,
        document_id=None,
        locator="",
        *,
        today=None,
        sources=(),
        metadata=None,
    ):
        return bool(
            self.add_cards(
                course_id,
                [
                    {
                        "front": front,
                        "back": back,
                        "document_id": document_id,
                        "locator": locator,
                        "sources": sources,
                        "metadata": metadata or {},
                    }
                ],
                today=today,
            )
        )

    def add_cards(self, course_id, drafts, *, today=None):
        """Validate and save a whole selection atomically, including provenance.

        A deleted/reimported source or invalid edit cannot leave a half-saved deck.
        Recheck source identity inside the write transaction after model inference.
        """
        if not isinstance(drafts, list) or len(drafts) > 24:
            raise UserError("Save at most 24 card drafts at a time.")
        with self.transaction() as (con, workspace):
            self._course(con, workspace, course_id)
            existing = {
                (" ".join(r[0].split()).casefold(), " ".join(r[1].split()).casefold())
                for r in con.execute("SELECT front,back FROM cards WHERE course_id=?", (course_id,))
            }
            prepared = []
            for draft in drafts:
                front = checked_text(draft.get("front", ""), "Card question", 1000)
                back = checked_text(draft.get("back", ""), "Card answer", 4000)
                identity = (" ".join(front.split()).casefold(), " ".join(back.split()).casefold())
                if identity in existing:
                    continue
                document_id, locator = draft.get("document_id"), draft.get("locator", "")
                if not isinstance(locator, str) or len(locator) > 1000:
                    raise UserError("Invalid card source location.")
                if document_id:
                    doc = self._document(con, workspace, document_id)
                    if doc["course_id"] != course_id:
                        raise UserError("Card source is not in the selected course.")
                    if not con.execute(
                        "SELECT 1 FROM chunks WHERE document_id=? AND locator=?",
                        (document_id, locator),
                    ).fetchone():
                        raise UserError("Card source location is no longer available.")
                sources = draft.get("sources", [])
                if not isinstance(sources, (list, tuple)) or len(sources) > 3:
                    raise UserError("A card can have at most three supporting excerpts.")
                for source in sources:
                    if not isinstance(source, dict) or type(source.get("passage_id")) is not int:
                        raise UserError("Invalid card evidence.")
                    row = con.execute(
                        "SELECT c.*,d.course_id FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id=? AND d.course_id=?",
                        (source["passage_id"], course_id),
                    ).fetchone()
                    quote = source.get("quote")
                    if (
                        not row
                        or row["document_id"] != source.get("document_id")
                        or row["locator"] != source.get("locator")
                        or not isinstance(quote, str)
                        or not quote.strip()
                        or len(quote) > 1500
                        or " ".join(quote.split()) not in " ".join(row["text"].split())
                    ):
                        raise UserError(
                            "Card evidence changed or is no longer available. Generate new drafts from the current material."
                        )
                if sources and (
                    document_id != sources[0]["document_id"] or locator != sources[0]["locator"]
                ):
                    raise UserError("Card source does not match its supporting evidence.")
                metadata = draft.get("metadata", {})
                if not isinstance(metadata, dict) or len(json.dumps(metadata)) > 4000:
                    raise UserError("Invalid card metadata.")
                prepared.append((front, back, document_id, locator, sources, metadata))
                existing.add(identity)
            current = con.execute(
                "SELECT count(*) FROM cards WHERE course_id=?", (course_id,)
            ).fetchone()[0]
            if current + len(prepared) > 2000:
                raise UserError("Limit: 2,000 cards per course.")
            for front, back, document_id, locator, sources, metadata in prepared:
                card_id = uid()
                con.execute(
                    "INSERT INTO cards(id,course_id,document_id,front,back,locator,due,created_at,metadata) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        card_id,
                        course_id,
                        document_id,
                        front,
                        back,
                        locator,
                        (today or date.today()).isoformat(),
                        now_iso(),
                        json.dumps(metadata),
                    ),
                )
                con.executemany(
                    "INSERT INTO card_sources VALUES (?,?,?,?,?,?)",
                    [
                        (card_id, s["document_id"], s["passage_id"], s["locator"], s["quote"], i)
                        for i, s in enumerate(sources)
                    ],
                )
            return len(prepared)

    def cards(self, course_id):
        with self.transaction() as (con, workspace):
            self._course(con, workspace, course_id)
            cards = [
                dict(r)
                for r in con.execute(
                    "SELECT c.*,d.name AS document_name FROM cards c LEFT JOIN documents d ON d.id=c.document_id WHERE c.course_id=? ORDER BY c.due,c.created_at,c.id",
                    (course_id,),
                )
            ]
            evidence = defaultdict(list)
            for row in con.execute(
                "SELECT s.*,d.name AS document_name FROM card_sources s JOIN cards c ON c.id=s.card_id JOIN documents d ON d.id=s.document_id WHERE c.course_id=? ORDER BY s.position",
                (course_id,),
            ):
                item = dict(row)
                evidence[item.pop("card_id")].append(item)
            for card in cards:
                card["metadata"] = json.loads(card["metadata"])
                card["sources"] = evidence[card["id"]]
            return cards

    def _card(self, con, workspace, card_id):
        row = con.execute(
            "SELECT k.* FROM cards k JOIN courses c ON c.id=k.course_id WHERE k.id=? AND c.workspace_id=?",
            (card_id, workspace["id"]),
        ).fetchone()
        if not row:
            raise UserError("Card not found in this workspace.")
        return row

    def update_card(self, card_id, front, back):
        front = checked_text(front, "Card question", 1000)
        back = checked_text(back, "Card answer", 4000)
        with self.transaction() as (con, workspace):
            card = self._card(con, workspace, card_id)
            try:
                metadata = json.loads(card["metadata"])
                if front != card["front"] or back != card["back"]:
                    metadata["review"] = "user_edited"
                con.execute(
                    "UPDATE cards SET front=?,back=?,metadata=? WHERE id=?",
                    (front, back, json.dumps(metadata), card_id),
                )
            except sqlite3.IntegrityError as exc:
                raise UserError("An identical card already exists.") from exc

    def delete_card(self, card_id):
        with self.transaction() as (con, workspace):
            self._card(con, workspace, card_id)
            con.execute("DELETE FROM cards WHERE id=?", (card_id,))

    def review_card(self, card_id, rating, *, today=None):
        from modulemate.study import schedule_review

        today = today or date.today()
        with self.transaction() as (con, workspace):
            card = self._card(con, workspace, card_id)
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
        with self.transaction() as (con, workspace):
            self._course(con, workspace, course_id)
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
        with self.transaction() as (con, workspace):
            self._course(con, workspace, course_id)
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
        with self.transaction() as (con, workspace):
            row = con.execute(
                "SELECT t.id FROM tasks t JOIN courses c ON c.id=t.course_id WHERE t.id=? AND c.workspace_id=?",
                (task_id, workspace["id"]),
            ).fetchone()
            if not row:
                raise UserError("Task not found in this workspace.")
            con.execute(
                "UPDATE tasks SET title=?,due=?,minutes=? WHERE id=?",
                (title, due, minutes, task_id),
            )

    def tasks(self, course_id):
        with self.transaction() as (con, workspace):
            self._course(con, workspace, course_id)
            return [
                dict(r)
                for r in con.execute(
                    "SELECT * FROM tasks WHERE course_id=? ORDER BY done,due,created_at",
                    (course_id,),
                )
            ]

    def modify_task(self, task_id, *, done=None, delete=False):
        with self.transaction() as (con, workspace):
            row = con.execute(
                "SELECT t.id FROM tasks t JOIN courses c ON c.id=t.course_id WHERE t.id=? AND c.workspace_id=?",
                (task_id, workspace["id"]),
            ).fetchone()
            if not row:
                raise UserError("Task not found in this workspace.")
            if delete:
                con.execute("DELETE FROM tasks WHERE id=?", (task_id,))
            elif isinstance(done, bool):
                con.execute("UPDATE tasks SET done=? WHERE id=?", (int(done), task_id))

    def reset_workspace(self):
        """Clear study content; the local workspace remains usable immediately."""
        with self.transaction() as (con, workspace):
            con.execute("DELETE FROM courses WHERE workspace_id=?", (workspace["id"],))

    def export(self):
        with self.transaction() as (con, workspace):
            output = {
                "format": "modulemate-export-v2",
                "exported_at": now_iso(),
                "workspace": workspace["name"],
                "courses": [],
            }
            for row in con.execute(
                "SELECT * FROM courses WHERE workspace_id=? ORDER BY created_at", (workspace["id"],)
            ):
                course = dict(row)
                course.pop("workspace_id")
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
                course["card_sources"] = [
                    dict(r)
                    for r in con.execute(
                        "SELECT s.* FROM card_sources s JOIN cards c ON c.id=s.card_id WHERE c.course_id=? ORDER BY s.card_id,s.position",
                        (row["id"],),
                    )
                ]
                output["courses"].append(course)
            return output
