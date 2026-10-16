# Upgrading to ModuleMate 2.2

Stop the old app, update the source, and run the setup commands in README.md.
Start the app normally: the library opens immediately with no login.

On the first schema 1/2 database open, ModuleMate writes a private online backup
named `modulemate.pre-v3-<id>.sqlite3` beside the database. It then migrates
atomically to schema 3. Every previous account's courses are retained in the one
local workspace, including duplicate course names. Document/chunk/message/card/
review/task IDs and existing vector data are preserved. Password, salt, session
and login-attempt storage is removed from the active schema. A second open does
not migrate again. Do not run an old account-based app against the upgraded file.

The default database remains `data/modulemate.sqlite3`; `MODULEMATE_DATA_DIR`
continues to override it. `MODULEMATE_ALLOW_REGISTRATION` no longer has any effect
and can be removed. Provider keys/settings remain outside the database.

Backups are retained for recovery and are not part of a public release. To restore,
stop the app, preserve the current file and its WAL companions, then restore the
chosen backup using SQLite's backup API. Opening it with 2.2 migrates it again.
JSON exports are portable readouts; JSON re-import is not implemented.

Prepared retrieval weights remain compatible. Conversational weights are a separate
explicit setup, outside the repository. Use the same `MODULEMATE_CHAT_VARIANT`
when preparing and running a local model. Unsupported hardware can use a compatible
local server, an explicitly enabled provider, or the limited excerpt fallback.

Historical 2.1 benchmark files describe that version's extractive behavior. The
new conversation challenge records its own model identity, actual answers and
latencies; do not quote old scores as a measure of the new generator.
