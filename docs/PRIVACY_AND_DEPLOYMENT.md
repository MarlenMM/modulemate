# Local data and provider boundaries

ModuleMate 2.2 is a single-user local application for a GitHub portfolio/demo.
It has no account, password, session token, sign-in or registration flow. Anyone
who can access this local app shares the same workspace. Its checked-in server
configuration binds to `127.0.0.1`; public/shared hosting is outside its scope.

The SQLite database stores courses, extracted text, passages, local vectors,
conversation, cards, reviews and tasks. Uploaded original files are discarded
after extraction. Data files have private filesystem permissions and are ignored
by Git. They are not encrypted by ModuleMate. JSON exports contain study content;
keep them private. Deleting a document removes its passages/vectors/linked cards
and clears that course's chat, which may contain paraphrases of the source.

OCR runs locally through Apple Vision or an installed Tesseract engine. PDF
rendering, image normalization and recognition use bounded request-local buffers;
original images are discarded after import. Only the extracted text and reading
notices are saved. No cloud vision endpoint receives uploaded pictures. A hosted
conversational provider, if explicitly enabled, can receive selected OCR text
under the same provider boundary as native document text.

Local model setup explicitly downloads public weights. Inference uses local files,
disables Hub networking/telemetry in the generative worker, and loads no custom
remote model code. The worker is bounded, serializes requests and discards its
prompt cache after each call. Text is not retained in a global conversation cache.
Only the database retains saved conversation. No Kaggle/Colab training is required.

A configured hosted provider is separate: the sidebar requires explicit activation.
It receives questions, bounded selected passages, recent turns, drafts and review
requests. Keys remain in `.env` or the process environment, never in SQLite or
exports. Errors shown in the app contain no raw provider responses. There is no
silent network fallback when the local generator fails.

Legacy account databases are backed up before the schema 3 upgrade. All previous
accounts' courses become accessible in this one workspace, retaining their IDs
and dependent content. Password/session tables are removed from the active schema.
Backups preserve the old database and therefore can contain old credential hashes;
keep those local. Clearing the active workspace does not erase backups or exports.

Document and course selection still constrain retrieval and model context. Changing
filters excludes earlier turns that relied on unselected documents. The project
release check excludes runtime data, private corpora, model weights and credentials.
Source text is untrusted input: it cannot grant the assistant permissions or make
it execute notebook code. Citation checks and model review reduce some errors but
are not guarantees of factual correctness or resistance to every adversarial input.
