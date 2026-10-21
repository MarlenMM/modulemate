# Upgrading to ModuleMate 2.6

Version 2.6 adds planner events and supporting excerpts. Opening schema 4 creates
a private online backup named `modulemate.pre-v5-<id>.sqlite3`, then atomically
adds schema 5 event tables, indexes and a supporting-document deletion trigger.
Existing courses, documents, passages, conversations, cards, review schedules,
study tasks and embeddings are retained. Existing tasks remain study tasks;
the upgrade does not invent calendar events from them.

Selected-event saves recheck all source identities and excerpts before committing.
Deleting any supporting document removes its derived events. JSON workspace
exports include `planner_events` and `event_sources`; JSON import is still not
implemented. The Planner's separate .ics export is intended for calendar import.
Stop the app before restoring or downgrading; an older version must not open a
schema 5 database.

## Earlier card evidence migration

Version 2.5 adds durable supporting excerpts and generation metadata for cards.
Opening a schema 3 database creates `modulemate.pre-v4-<id>.sqlite3` beside it,
then atomically upgrades through schema 4 to schema 5. Existing document, card and review IDs,
answers, due dates and review history are retained. Existing cards do not acquire
invented evidence or an AI-review label. Subsequent opens do not migrate again.

New cards can retain several source excerpts. Deleting any supporting document
removes the derived card and its reviews, including when that document was a
secondary source. Saving selected drafts rechecks every excerpt and commits the
whole selection together. JSON exports now include `card_sources` alongside the
existing course data. JSON import is still not implemented.

Stop the app before restoring a backup or switching versions. An older app must
not open a schema 4 database; restore the appropriate pre-upgrade backup first.

## Earlier account migration

Stop the old app, update the source, and run the setup commands in README.md.
Start the app normally: the library opens immediately with no login.

On the first schema 1/2 database open, ModuleMate writes a private online backup
named `modulemate.pre-v3-<id>.sqlite3` beside the database. It then migrates
through the schema 3 account migration and then to schema 5. Every previous account's courses are retained in the one
local workspace, including duplicate course names. Document/chunk/message/card/
review/task IDs and existing vector data are preserved. Password, salt, session
and login-attempt storage is removed from the active schema. A second open does
not migrate again. Do not run an old account-based app against the upgraded file.

The default database remains `data/modulemate.sqlite3`; `MODULEMATE_DATA_DIR`
continues to override it. `MODULEMATE_ALLOW_REGISTRATION` no longer has any effect
and can be removed. Provider keys/settings remain outside the database.

Backups are retained for recovery and are not part of a public release. To restore,
stop the app, preserve the current file and its WAL companions, then restore the
chosen backup using SQLite's backup API. Opening it with the current app migrates it again.
JSON exports are portable readouts; JSON re-import is not implemented.

Prepared retrieval weights remain compatible. Conversational weights are a separate
explicit setup, outside the repository. Use the same `MODULEMATE_CHAT_VARIANT`
when preparing and running a local model. Unsupported hardware can use a compatible
local server, an explicitly enabled provider, or the limited excerpt fallback.

Historical 2.1 benchmark files describe that version's extractive behavior. The
new conversation challenge records its own model identity, actual answers and
latencies; do not quote old scores as a measure of the new generator.
