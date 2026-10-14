# Privacy and deployment

## Intended deployment

Run locally or on a small trusted self-hosted installation. The checked-in server
configuration binds to loopback and enables XSRF/CORS protection. Do not expose
the development server directly to the internet. For shared access use HTTPS,
a reverse proxy with request/authentication rate limits, private persistent disk,
process/container resource limits, backups and an operator who accepts access
to stored materials. Close registration after provisioning accounts if appropriate.
An external identity provider, MFA, password recovery, malware scanning and a
formal security audit are outside this release.

## Stored and transmitted data

- Usernames, salted scrypt password hashes, expiring hashed session tokens and
  short-lived login failure counters.
- Courses, extracted document text, local FTS index, saved chats, cards, review
  events and tasks. No uploaded originals, provider keys or raw passwords in DB.
- Local mode sends no material to an AI service. Streamlit usage stats are off.
- Optional AI shares selected excerpts and recent conversation with the displayed
  provider only after the user opts in. A second request sends the draft and excerpts to the same provider to check support. Consent is session-local and clears on
  sign-out or browser reload. The operator controls the endpoint/model.
- The server operator can read the database. This is not end-to-end encryption.
  Use encrypted storage and OS account controls where needed.
- Account exports contain private content; store them safely. Source deletion
  clears linked cards/reviews and course chats. Operator backups and previously
  downloaded or transmitted copies are not automatically removed.

## Configuration

| Variable | Default / purpose |
|---|---|
| `MODULEMATE_DATA_DIR` | `./data`; use a persistent private directory outside web roots |
| `MODULEMATE_AI_BASE_URL` | `https://api.openai.com/v1`; HTTPS required except loopback |
| `MODULEMATE_AI_MODEL` | Empty: no default model commitment |
| `MODULEMATE_AI_API_KEY` | Empty: local tools still work |
| `MODULEMATE_ALLOW_REGISTRATION` | `true`; set `false` to close new account creation |

There are no per-user provider overrides. Only trusted operators should change
environment configuration. Restart the app after changing provider settings.
Password hashing uses scrypt N=32768, r=8, p=3 (32 MiB working memory), one of the configurations in the [OWASP password storage guidance](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html). Passwords require 12–256 characters; five failed logins per username in 15
minutes cause a temporary lockout. Sessions expire after 12 hours; password
change revokes other sessions. There is no recovery email or admin backdoor.

## Resource limits

- 100 accounts per installation, 30 courses per account.
- 100 documents / 12,000 passages per course.
- 15 MB per file; five files / 30 MB per batch.
- 2 million extracted characters / 3,000 passages per document.
- 500 PDF pages, slides or notebook cells; office archives at most 2,500 entries,
  60 MB expanded total and 12 MB per member. External-entity XML is rejected.
- 25-second parser wall timeout, 15 CPU seconds on POSIX; 768 MB address space
  ceiling on Linux. macOS uses CPU/wall limits; Windows has wall timeout only.
- 2,000 cards and 1,000 tasks per course; the most recent 200 chat exchanges are
  retained. UI displays the latest 40; full account export includes all retained.

The parser process is a fault/resource boundary, **not an OS security sandbox**.
For untrusted public uploads, container isolation, malware scanning and additional
CPU/disk/network quotas are required. Parser libraries remain part of the trusted
computing base. No remote URL fetching, notebook execution or macros are enabled.

## Backup and restore

Create a consistent backup using SQLite's backup API, not a live copy of only the
main DB file while WAL writes are in flight:

```bash
uv run python scripts/backup.py /private/backups/modulemate-2026-10-14.sqlite3
```

The destination must not exist; backups receive owner-only file permissions.
They include all accounts, password hashes and sessions and must stay private.
To restore: stop the server, preserve the existing data directory as a separate
backup, put the backup at `MODULEMATE_DATA_DIR/modulemate.sqlite3` in a fresh
private directory, and restart. Do not mix an old main DB with another database's
`-wal` or `-shm` files. Invalidate restored sessions using `scripts/backup.py
--revoke-sessions` before serving. Test restoration before relying on backups.

The account JSON export is a portable human/machine-readable copy; automatic
JSON account re-import is not supported in this release.

## Technical references

Implementation choices were checked against the official
[Streamlit upload documentation](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader),
[server configuration](https://docs.streamlit.io/develop/api-reference/configuration/config.toml),
and [Chat Completions response formats](https://platform.openai.com/docs/api-reference/chat).
Upload extension filters alone are not validation; the application also validates
content and parser limits. JSON mode alone is not grounding; application-side
source validation still runs.
