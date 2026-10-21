"""Source-backed planner previews and editable events, separate from study capacity."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from uuid import uuid4

import streamlit as st

from modulemate.calendar_io import KINDS, STATUSES, events_calendar, google_calendar_link
from modulemate.db import UserError
from modulemate.generation import GenerationError
from modulemate.planner import extract_events
from modulemate.planner_dates import DATE_ORDER, WEEKDAYS
from modulemate.retrieval import scoped_passages


def plain(value):
    return re.sub(r"([\\\x60*_{}\[\]()<>#+.!|~-])", r"\\\1", str(value))


def _date(value):
    return date.fromisoformat(value) if value else None


def quantity(count, noun):
    return f"{count} {noun}{'' if count == 1 else 's'}"


def event_editor(event, key):
    title = st.text_input("Event title", event.get("title", ""), max_chars=200, key=key + "_title")
    a, b = st.columns(2)
    start = a.date_input(
        "Start date",
        value=_date(event.get("start_date")),
        key=key + "_start",
        min_value=date(1000, 1, 1),
        max_value=date(2199, 12, 31),
    )
    end = b.date_input(
        "Last day / end date",
        value=_date(event.get("end_date")),
        key=key + "_end",
        min_value=date(1000, 1, 1),
        max_value=date(2199, 12, 31),
    )
    st.caption(
        "For all-day events, the last day is included. An empty last day uses the start date."
    )
    all_day = st.checkbox(
        "All-day / date only",
        value=not event.get("start_time") and not event.get("metadata", {}).get("requires_time"),
        key=key + "_all_day",
    )
    a, b = st.columns(2)
    start_time = a.text_input(
        "Start time (24-hour)",
        event.get("start_time") or "",
        placeholder="HH:MM",
        key=key + "_start_time",
    )
    end_time = b.text_input(
        "End time (optional)",
        event.get("end_time") or "",
        placeholder="HH:MM",
        key=key + "_end_time",
    )
    st.caption(
        "Times apply when All-day is unchecked. A blank end time keeps a point-in-time event; no duration is assumed."
    )
    zone = st.text_input(
        "Event time zone",
        event.get("timezone", "UTC"),
        key=key + "_zone",
        max_chars=100,
        help="IANA zone, such as Asia/Seoul or America/New_York, or a fixed offset such as UTC-04:00.",
    )
    a, b = st.columns(2)
    kind = a.selectbox(
        "Event type", KINDS, index=KINDS.index(event.get("kind", "Other")), key=key + "_kind"
    )
    status = b.selectbox(
        "Status",
        STATUSES,
        index=STATUSES.index(event.get("status", "Confirmed")),
        key=key + "_status",
    )
    location = st.text_input(
        "Location (optional)", event.get("location", ""), max_chars=300, key=key + "_location"
    )
    description = st.text_area(
        "Description", event.get("description", ""), max_chars=4000, key=key + "_description"
    )
    repeat = event.get("recurrence") or {}
    with st.expander("Weekly recurrence and exceptions", expanded=bool(repeat)):
        weekly = st.checkbox("Repeat weekly", bool(repeat), key=key + "_weekly")
        days = st.multiselect(
            "Repeat on", WEEKDAYS, default=repeat.get("days", []), key=key + "_days"
        )
        until = st.date_input(
            "Repeat through",
            value=_date(repeat.get("until")),
            key=key + "_until",
            min_value=date(1000, 1, 1),
            max_value=date(2199, 12, 31),
        )
        exceptions = st.text_input(
            "Skip dates (comma-separated YYYY-MM-DD)",
            ", ".join(repeat.get("exclude", [])),
            key=key + "_exclude",
            max_chars=1500,
        )
        st.caption(
            "Weekly series need a final date. Changing the start preserves the selected weekdays."
        )
    return {
        "title": title,
        "description": description,
        "location": location,
        "kind": kind,
        "status": status,
        "start_date": start.isoformat() if start else None,
        "end_date": end.isoformat() if end else None,
        "start_time": None if all_day else (start_time.strip() or "Time required"),
        "end_time": None if all_day else end_time.strip(),
        "timezone": zone.strip(),
        "recurrence": {
            "frequency": "weekly",
            "days": days,
            "until": until.isoformat() if until else "",
            "exclude": [x.strip() for x in exceptions.split(",") if x.strip()],
        }
        if weekly
        else {},
    }


def event_sources(event, names):
    if not event.get("sources"):
        st.caption("Manual event · no supporting material linked.")
        return
    with st.expander("Supporting material"):
        review = event.get("metadata", {}).get("review")
        if review in {"user_edited", "user_confirmed"}:
            st.caption(
                "You reviewed or edited this event. The excerpts are retained for comparison."
            )
        if event.get("metadata", {}).get("year_basis"):
            st.text(event["metadata"]["year_basis"])
        for source in event["sources"]:
            st.text(
                f"{names.get(source['document_id'], source.get('document_name', 'Material'))} · {source['locator']}"
            )
            st.text(source["quote"])


def extract_panel(store, course_id, backend, today, timezone, select_documents, document_labels):
    st.subheader("Find dates in your materials")
    st.write("Extract scheduled events and deadlines, then check and edit what you want to keep.")
    docs = store.documents(course_id)
    if not docs:
        st.info("Add material in Library first, or add an event manually in Events and dates.")
        return
    ids = select_documents(docs, f"planner_docs_{course_id}")
    selected = [d for d in docs if ids is None or d["id"] in ids]
    order = st.selectbox(
        "Numeric date format",
        DATE_ORDER,
        help="A date such as 03/04 stays unresolved unless the order is clear or you select a convention.",
    )
    supplied = st.checkbox(
        "Supply a year for dates without one",
        help="Explicit source years and unambiguous term headings take precedence. Today's year is never silently substituted.",
    )
    year = (
        int(
            st.number_input(
                "Year for incomplete dates", min_value=1000, max_value=2199, value=today.year
            )
        )
        if supplied
        else None
    )
    st.caption(
        f"Default time zone: {timezone}. An explicit source zone takes precedence. Change Your timezone in the sidebar or edit an individual event."
    )
    if not backend:
        st.info(
            "No AI generator is selected. A source date scan can list literal mentions for manual review; it does not decide which mentions are real events."
        )
    for doc in selected:
        notices = json.loads(doc["warnings"])
        if notices:
            with st.expander("Reading notices · " + plain(doc["name"])):
                for notice in notices:
                    st.text(notice)
    signature = (
        [(d["id"], d["created_at"], d["chunks"]) for d in selected],
        order,
        year,
        timezone,
        bool(backend),
    )
    key = f"planner_preview_{course_id}"
    position_key = f"planner_position_{course_id}"
    previous = st.session_state.get(key)
    position = st.session_state.get(position_key)
    if (previous and previous["signature"] != signature) or (
        position and position["signature"] != signature
    ):
        st.session_state.pop(key, None)
        st.session_state.pop(position_key, None)
        st.info("Your source selection or date settings changed. Extract a new preview.")
    restart = st.button("Extract dates" if backend else "Scan date mentions", type="primary")
    position = st.session_state.get(position_key)
    more = False
    if position and position["remaining"]:
        st.caption(f"{position['remaining']} date-bearing sections remain in this selection.")
        more = st.button(
            "Continue extracting dates",
            disabled=key in st.session_state,
            help="Save or discard the current preview before reviewing the next sections.",
        )
    if restart or more:
        progress = st.progress(0, text="Scanning the selected material…")
        try:
            result = extract_events(
                scoped_passages(store, course_id, ids),
                backend,
                timezone=timezone,
                year=year,
                date_order=order,
                progress=lambda fraction, text: progress.progress(fraction, text=text),
                start_at=position["next"] if more else 0,
            )
            if result.events or result.attention:
                st.session_state[key] = {
                    "id": uuid4().hex,
                    "signature": signature,
                    "events": result.events,
                    "attention": result.attention,
                    "ignored": result.ignored,
                    "diagnostics": result.diagnostics,
                }
            if result.events or result.attention or key not in st.session_state:
                st.session_state[position_key] = {
                    "signature": signature,
                    "next": result.diagnostics["review_end"],
                    "remaining": result.diagnostics["remaining"],
                }
            else:
                st.info(
                    "No suitable events were identified. Inspect the material or try another selection. Any earlier preview is retained."
                )
            for warning in result.diagnostics["warnings"]:
                st.warning(warning)
            st.caption(
                f"Scanned {quantity(result.diagnostics['scanned_passages'], 'passage')}; found {quantity(result.diagnostics['candidate_blocks'], 'date-bearing section')}."
            )
            if result.diagnostics["remaining"] and key not in st.session_state:
                st.session_state.flash = "No events to keep in the reviewed sections. Continue to review the next sections."
                st.rerun()
        except (GenerationError, ValueError, TypeError, KeyError):
            st.warning(
                "Date extraction could not finish. Try a smaller selection or retry. Any earlier preview is retained."
            )
        finally:
            progress.empty()
    preview = st.session_state.get(key)
    if not preview:
        return
    st.subheader(
        f"Event preview · {len(preview['events'])} ready · {len(preview['attention'])} {'needs' if len(preview['attention']) == 1 else 'need'} review"
    )
    info = preview["diagnostics"]
    st.caption(
        f"{info['engine']} · Scanned {quantity(info['scanned_passages'], 'passage')} across {quantity(info['documents'], 'document')}. Reviewed sections {info['review_start'] + 1}–{info['review_end']} of {info['candidate_blocks']}."
    )
    st.caption(
        "AI checks can miss mistakes. Verify dates, time zones, conditions and source reading. Unresolved drafts are unchecked until you choose to keep them."
    )
    names = document_labels(docs)
    saved_events = {e["id"]: e for e in store.events(course_id)}
    with st.form("planner_save_" + preview["id"]):
        selection = []
        for i, event in enumerate([*preview["events"], *preview["attention"]], 1):
            editor_key = f"planner_draft_{preview['id']}_{i}"
            with st.expander(
                f"{i}. {plain(event['title'])} · {event['start_date'] or 'Date unresolved'}",
                expanded=i == 1 or bool(event["issues"]),
            ):
                conflicts = [
                    e
                    for e in saved_events.values()
                    if e["title"].casefold() == event["title"].casefold()
                    and e["kind"] == event["kind"]
                    and e["kind"] in {"Exam", "Deadline"}
                    and any(
                        e.get(k) != event.get(k)
                        for k in ("start_date", "end_date", "start_time", "end_time", "location")
                    )
                ]
                keep = st.checkbox(
                    f"Keep event {i}",
                    value=not event["issues"] and not conflicts,
                    key=editor_key + "_keep",
                )
                if conflicts:
                    st.warning(
                        "A saved event with this name has different details. Choose whether to replace it or keep a separate event."
                    )
                replacement = (
                    st.selectbox(
                        f"Existing event to replace · draft {i}",
                        [None, *saved_events],
                        format_func=lambda identity: (
                            "Add as a separate event"
                            if identity is None
                            else f"Replace {saved_events[identity]['title']} · {saved_events[identity]['start_date']} · {saved_events[identity]['start_time'] or 'date only'}"
                        ),
                        key=editor_key + "_replace",
                    )
                    if saved_events
                    else None
                )
                for issue in event["issues"]:
                    st.warning(plain(issue))
                values = event_editor(event, editor_key)
                event_sources(event, names)
                edited = any(values.get(k) != event.get(k) for k in values)
                selection.append(
                    event
                    | values
                    | {
                        "replace_id": replacement,
                        "metadata": event["metadata"]
                        | {"review": "user_edited" if edited else "user_confirmed"},
                    }
                    if keep
                    else None
                )
        if st.form_submit_button("Save selected events", type="primary"):
            chosen = [e for e in selection if e is not None]
            if not chosen:
                st.info("Choose at least one event to keep, or discard the preview.")
            else:
                try:
                    count = store.add_events(course_id, chosen)
                    st.session_state.pop(key, None)
                    st.session_state.flash = f"Saved {quantity(count, 'event')}. Repeated events are skipped; replacements keep their calendar IDs."
                    st.rerun()
                except UserError as exc:
                    st.error(str(exc))
    if preview["ignored"]:
        with st.expander(f"Excluded drafts · {len(preview['ignored'])}"):
            for ignored in preview["ignored"]:
                st.text(f"{ignored['title']}: {ignored['reason']}")
                event_sources(ignored, names)
    if st.button("Discard event preview"):
        st.session_state.pop(key, None)
        st.rerun()


def events_panel(store, course_id, today, timezone, document_labels):
    events = store.events(course_id)
    st.subheader("Your events and dates")
    names = document_labels(store.documents(course_id))
    for event in events:
        for source in event["sources"]:
            source["document_name"] = names.get(source["document_id"], source["document_name"])
    with st.expander("Add an event manually", expanded=not events):
        with st.form(f"manual_event_{course_id}"):
            values = event_editor(
                {
                    "start_date": today.isoformat(),
                    "end_date": today.isoformat(),
                    "timezone": timezone,
                },
                f"manual_event_{course_id}",
            )
            if st.form_submit_button("Save manual event", type="primary"):
                try:
                    count = store.add_events(
                        course_id,
                        [values | {"metadata": {"origin": "manual", "review": "user_edited"}}],
                    )
                    st.session_state.flash = (
                        "Manual event saved." if count else "This event already exists."
                    )
                    st.rerun()
                except UserError as exc:
                    st.error(str(exc))
    if not events:
        st.info("No saved events yet. Extract dates from your material or add one above.")
        return
    st.caption(
        f"{len(events)} saved {'event' if len(events) == 1 else 'events'}. Changes here affect future exports. Earlier calendar imports do not stay in sync."
    )
    a, b = st.columns(2)
    first = a.date_input("Show from (optional)", value=None, key=f"event_filter_first_{course_id}")
    last = b.date_input("Show through (optional)", value=None, key=f"event_filter_last_{course_id}")
    shown = [
        e
        for e in events
        if (not first or (e["recurrence"].get("until") or e["end_date"]) >= first.isoformat())
        and (not last or e["start_date"] <= last.isoformat())
    ]
    by_id = {e["id"]: e for e in shown}
    labels = {
        e["id"]: f"{e['start_date']} · {e['title']}"
        + (f" · {e['start_time']} {e['timezone']}" if e["start_time"] else "")
        for e in shown
    }
    export_signature = hashlib.sha256(
        json.dumps(labels, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:12]
    export_key = f"export_events_{course_id}_{export_signature}"
    selection_key = f"export_selection_{course_id}"
    previous = st.session_state.get(selection_key)
    defaults = list(by_id)
    if previous and set(previous["ids"]) == set(by_id):
        defaults = [i for i in st.session_state.get(previous["key"], defaults) if i in by_id]
    selected = st.multiselect(
        "Events to export",
        list(by_id),
        default=defaults,
        format_func=labels.get,
        key=export_key,
    )
    st.session_state[selection_key] = {"ids": list(by_id), "key": export_key}
    include_sources = st.checkbox("Include supporting material in the calendar file", value=False)
    export = [by_id[i] for i in selected]
    st.download_button(
        "Export selected events (.ics)",
        events_calendar(
            export,
            include_sources=include_sources,
            name="ModuleMate · " + next(c["name"] for c in store.courses() if c["id"] == course_id),
        ),
        "modulemate-events.ics",
        "text/calendar",
        disabled=not export,
    )
    with st.expander("Import into your calendar"):
        st.write(
            "Apple Calendar: open or import the .ics file and choose a calendar. Google Calendar on a computer: Settings → Import & Export → select this .ics file → choose a calendar → Import."
        )
        st.caption(
            "The file is a snapshot. Reimport behavior differs between calendars; it is not a subscription. Event IDs stay stable when you edit them here."
        )
        st.link_button(
            "Google Calendar import help", "https://support.google.com/calendar/answer/37118?hl=en"
        )
        st.link_button(
            "Apple Calendar import help",
            "https://support.apple.com/guide/calendar/import-or-export-calendars-icl1023/mac",
        )
    pages = max(1, (len(shown) + 19) // 20)
    page = (
        int(
            st.number_input(
                "Event page",
                min_value=1,
                max_value=pages,
                value=1,
                key=f"event_page_{course_id}_{pages}",
            )
        )
        if pages > 1
        else 1
    )
    if pages > 1:
        st.caption(
            f"Showing up to 20 of {len(shown)} filtered events. Export selection covers all filtered events."
        )
    for event in shown[(page - 1) * 20 : page * 20]:
        with st.expander(plain(labels[event["id"]])):
            st.text(event["description"])
            if event["location"]:
                st.text("Location: " + event["location"])
            if event["recurrence"]:
                st.text(
                    "Weekly: "
                    + ", ".join(event["recurrence"]["days"])
                    + " through "
                    + event["recurrence"]["until"]
                )
            event_sources(event, names)
            link = google_calendar_link(event)
            if link:
                st.link_button("Add this event to Google Calendar", link)
                st.caption(
                    "Opens Google with this event's details. Review and save it there yourself."
                )
            with st.form(f"edit_event_{event['id']}_{event['sequence']}"):
                values = event_editor(event, f"edit_event_{event['id']}_{event['sequence']}")
                if st.form_submit_button("Save event changes"):
                    try:
                        store.update_event(event["id"], values)
                        st.session_state.flash = (
                            "Event updated. Export it again to use these changes elsewhere."
                        )
                        st.rerun()
                    except UserError as exc:
                        st.error(str(exc))
            if st.button("Remove event", key="delete_event_" + event["id"]):
                store.delete_event(event["id"])
                st.session_state.flash = "Event removed from this planner."
                st.rerun()
