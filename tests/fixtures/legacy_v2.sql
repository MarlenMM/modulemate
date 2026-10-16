-- Original 2.1 database schema, retained only as a migration fixture.

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

CREATE TABLE embeddings (chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE, model TEXT NOT NULL, digest TEXT NOT NULL, dimensions INTEGER NOT NULL, vector BLOB NOT NULL, PRIMARY KEY(chunk_id,model));
ALTER TABLE messages ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}';
PRAGMA user_version=2;
