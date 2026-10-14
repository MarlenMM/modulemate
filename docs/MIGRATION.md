# Migration from the fixed-corpus prototype

Version 2 replaces the former shared university index and course-specific intent
classifier with per-account, user-supplied libraries. The old corpus, OCR caches,
course manifests, evaluation answers and reports are removed from the current
tree. They are not imported into the new database. The old accuracy figures do
not describe this implementation and must not be reused in new claims.

The implementation migration preserved a private copy of the original code and
local course artifacts outside the repository. This archive is a local recovery
copy, not part of the software distribution. Do not upload it or copy the old
`corpus`, `index_store`, `results`, `.env`, runtime database or personal exports
into a public release. The legacy `.env` is kept untracked; configuring the new
provider variables is a separate local step.

## Git history and public copies

Deleting a tracked file affects the new tree, not earlier commits, tags, forks,
clones, cached pages or release assets. The existing repository was public when
this migration was performed. A normal update cannot guarantee that its previous
course material is no longer accessible.

For a distribution with no inherited history, export only the verified current
commit:

```bash
git archive --format=zip --prefix=modulemate/ HEAD -o /private/path/modulemate-source.zip
```

Review that archive before publishing. It contains tracked files only and no Git
history. A separate history-removal operation on the existing remote would need
a complete path/content audit, backup, rewriting every affected branch/tag,
coordinated force updates, and review of hosting caches/releases/forks. Do not
claim that operation has happened merely because the latest branch is clean.
GitHub support may be needed for retained/cached objects; third-party clones are
outside this project's control.

## New installation

1. Install with `uv sync --locked` (Python 3.12/3.13).
2. Start the app, create an account and a course.
3. Add only material you are permitted to use. There is no implicit filesystem
   crawl, university download, legacy index loading or automatic corpus rebuild.
4. Use local tools immediately. Configure optional AI and explicitly opt in from
   the sidebar if desired.
5. Keep the data directory private and back it up with the documented backup API.
