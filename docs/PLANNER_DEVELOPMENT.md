# Material-based planner events

Work starts from `56784df` on 21 October 2026. The original local suite passes
406 tests with real weights, OCR and OCR stress enabled. Existing manual study
tasks, Cards and the local-first, user-owned-material boundary remain in scope.

## Design and evaluation decisions

The user wants selected-material extraction, descriptions and necessary event
details, editable/selectable drafts, durable events and portable calendar export.
Precision takes priority over an apparently complete but invented schedule.

Compare a freely generated date/JSON baseline with a hybrid: deterministic,
literal temporal candidates; AI semantic event selection; app-owned date/time
normalization; individual factual review. Do not ask a language model to invent
an ISO timestamp and then treat valid JSON as factual verification.

Use RFC 5545 iCalendar for a portable batch file. Apple Calendar supports `.ics`
import; Google Calendar's desktop Import & Export accepts `.ics`. An import is a
snapshot, not ongoing synchronization. A direct link can assist with a single
event, but is not a substitute for a batch file or authenticated calendar sync.

Primary references:

- [RFC 5545](https://datatracker.ietf.org/doc/html/rfc5545)
- [icalendar parser/generator](https://icalendar.readthedocs.io/)
- [Google import instructions](https://support.google.com/calendar/answer/37118?hl=en)
- [Apple Calendar import instructions](https://support.apple.com/guide/calendar/import-or-export-calendars-icl1023/mac)

Evaluate positive and negative cases across prose, tables, imported formats,
long documents, named and numeric dates, missing years, locale ambiguity,
time zones/DST, deadlines without end times, multiday events, bounded weekly
recurrence/exceptions, relative dates, conflicting/rescheduled/cancelled events,
historical examples, metadata and model-directed source instructions. Inspect
actual events and quotations, not only substring checks. Hold unresolved facts
for user correction; never silently roll a missing year forward from today's date.

Migration must back up first, preserve existing rows/schedules and reauthorize
every supporting excerpt inside a transactional selected-event save. Calendar
exports need escaping, UTF-8 line folding, stable UIDs, exclusive all-day ends,
unambiguous instants and bounded recurrence. Existing calendar events will not be
modified through an external account during development tests.

## Cycles and observations

The retained reports include actual model output, source excerpts, timings and
implementation fingerprints. Counts below describe the authored development
fixtures, not a representative population.

| Checkpoint | Cases | Ready events matching specified fields | Extra or mismatched ready events | Observed issue |
|---|---:|---:|---:|---|
| Free JSON baseline | 11 | 6 | 2 | Invented or mismatched temporal fields; seven expected events missing |
| Hybrid cycle 1 | 11 | 9 | 0 | Useful dates dropped; ambiguous entries absent |
| Cycle 2 | 11 | 13 | 0 | Two uncertainty queues still incomplete |
| Cycle 3 | 11 | 13 | 0 | Original cases complete; transfer cases added next |
| Cycle 4 transfer | 12 | 4 | 0 | Lists, numeric appointment and submission window lost; unresolved bounds omitted |
| Cycle 5 combined | 23 | 20 | 0 | Reviewer rejected a real poster session over its organizational category; writer excluded ambiguous dates |
| Cycle 6 combined | 23 | 21 | 0 | All specified resolved fields and uncertainty counts matched |
| Cycle 7 acceptance | 31 | 27 | 0 | Eight newly authored acceptance cases also matched |
| Cycle 8 dense table | 1 | 0 | — | A generation-capacity failure returned no approved drafts |
| Cycle 9 row batches | 32 | 51 | 0 | Two final table rows rejected over irrelevant recurrence IDs |
| Cycle 10 structured columns and final matrix | 32 | 53 | 0 | All specified resolved fields and uncertainty counts matched |

The free-JSON baseline was a real alternative: the same prepared local generator
was asked to identify actual events and output ISO values without guessing.
Structured JSON alone did not make those values source-supported. The hybrid
keeps semantic identification in the model but makes date/time/duration choices
refer to application-owned literal candidates. Its additional review costs
inference time; the baseline's 11 cases took 89.61 seconds, versus 162.99 seconds
for the 11 cycle-3 cases. These single runs include changed prompts/logic and do
not isolate a causal model-stage effect.

Failures changed the design:

- A rescheduling row included an irrelevant excluded-date ID. Rejecting the entire
  event lost a correctly selected replacement. Nonrecurring exception IDs are now
  irrelevant rather than a reason to erase a valid one-off appointment.
- A critic reading a whole batch borrowed objections from another event. Each
  proposed event now carries its own evidence and explicit year basis.
- A critic dropped dates merely because they were unresolved. Unknown years,
  numeric order, weekday mismatches, bounds, relative dates and unclear clocks
  remain unchecked correction candidates. Non-event narratives and cancelled or
  sample calendars still stay excluded.
- Source headings matter: “Sample calendar data, not actual appointments” is
  evidence, not just a filename or locator. Actual heading excerpts are retained
  and available to the factual reviewer and to the learner.
- “October 21 and 28” is a list, not a week-long event. The app expands independent
  occurrences and reviews every one, even when expansion exceeds a model batch.
- The label “Workshop” versus “Other” is organizational, not a factual premise.
  Literal title words determine the category, and category disagreement cannot
  make the factual critic erase a correctly sourced event.
- A short UTC-offset pattern once swallowed the end of an ordinary 14:00–15:30
  range. Native-format ingestion regressions exposed it. Offset recognition now
  requires the actual UTC/GMT or ISO-offset context.
- An explicitly excluded DST-gap occurrence must serialize as an EXDATE, even
  though that clock cannot be used as a real event instant. Recurrence bounds use
  the last actual occurrence rather than an arbitrary end-of-day wall clock.
- A first-request section bound must not become an unreachable tail. The UI
  preserves a continuation position after save/discard, and invalidates it when
  material/date settings change. No unsaved preview is replaced by continuation.
- The browser caught stale export tags after renaming a saved event. The app now
  refreshes labels while preserving the existing ID selection; the real download
  was rechecked after two saved edits.
- A fresh 26-row calendar table overflowed one writer request (88.78 seconds,
  GenerationError, no approved drafts). Independent rows and date-aware batches
  recovered 24 rows, but cost 196.13 seconds and still lost two rows to irrelevant
  zero-valued recurrence IDs. The app now reads unambiguous calendar columns
  directly, keeps the AI relevance/source review, and ignores recurrence IDs for
  nonrecurring events. This is a separate structured-reading path, rather than
  repeatedly tuning the same writer prompt.
  The final dense-table run returned all 26 rows in 73.62 seconds with five review
  calls, compared with 24 in 196.13 seconds with the previous writer/reviewer path.
- Overlapping ingestion can cut a row mid-cell. The planner prefers its complete
  row in the adjacent same-document, same-heading passage, without merging other
  documents or sample/actual heading contexts. Prefix lookup uses sorted groups
  rather than a quadratic comparison across all table rows.

The later acceptance material adds unrelated nearby clocks, multiple deadlines
in one paragraph, cancellation beside a distinct replacement activity,
conditional proposals, date-shaped identifiers, evening weekly office hours,
two-digit years and explicitly archived schedules. It was authored after the
earlier cycles; once inspected, it too is development material.

## Calendar and storage choices

Use the maintained iCalendar library rather than a custom line encoder. Exported
events have stable UIDs; saved edits and explicit replacements increment SEQUENCE.
All-day date ranges include the UI's final day. A timed deadline without a supplied
end remains a point event, rather than an invented hour-long meeting. Weekly
series retain local clock time across DST, their final bound and exception dates.

A Google template link is useful for a one-off event with a known end. Weekly
series and timed points use the portable .ics file. The app does not claim that
opening a file always imports without a client prompt, or that a repeated file
import synchronizes edits. External calendar accounts were not modified.

Selected saves reauthorize every supporting excerpt and commit atomically.
Re-extraction skips duplicates and preserves saved edits. An explicit replacement
choice updates the selected saved event while preserving its calendar identity.
Deleting a supporting document removes locally derived events; it cannot revoke
copies previously imported into another calendar.

## Deliberate limits

English text/date expressions are the supported extraction target. Relative dates
without an unambiguous absolute anchor, ambiguous day boundaries/time zones,
unsupported recurrence and conflicting notices require user correction. A separate
date scan is available when no generator is selected; its entries are unreviewed
mentions and never preselected.

The service scans all selected passages and reviews up to 180 date-bearing sections
per request, with continuation for the rest. The source parser still has its
document/OCR bounds and can miss or misread content. Arbitrary table geometry,
diagrams and implicit multi-document schedule logic are not guaranteed. Model
selection and review remain fallible; zero observed extra ready events in these
cases cannot establish zero false positives on arbitrary material.
