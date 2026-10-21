import json
import sqlite3

import pytest
from icalendar import Calendar

from modulemate.calendar_io import events_calendar
from modulemate.db import Database, Store, UserError
from modulemate.ingest import parse_upload
from modulemate.retrieval import scoped_passages


def draft(store, course, text="Final exam on November 2, 2026."):
    document, _ = store.add_document(
        course, parse_upload("Schedule.md", text.encode(), isolated=False)
    )
    passage = scoped_passages(store, course, [document])[0]
    return {
        "title": "Final exam",
        "description": text,
        "kind": "Exam",
        "status": "Confirmed",
        "location": "",
        "start_date": "2026-11-02",
        "end_date": "2026-11-02",
        "start_time": None,
        "end_time": None,
        "timezone": "Asia/Seoul",
        "recurrence": {},
        "sources": [
            {
                "document_id": document,
                "passage_id": passage.id,
                "locator": passage.locator,
                "quote": text,
            }
        ],
        "metadata": {"origin": "material_ai", "review": "model_checked", "origin_key": "a" * 64},
    }


def test_event_save_edit_export_stable_uid_and_source_retention(store, course):
    value = draft(store, course)
    assert store.add_events(course, [value]) == 1
    original = store.events(course)[0]
    uid = Calendar.from_ical(events_calendar([original])).walk("VEVENT")[0]["UID"]
    store.update_event(
        original["id"],
        {"title": "My exam wording", "start_date": "2026-11-03", "end_date": "2026-11-03"},
    )
    updated = store.events(course)[0]
    exported = Calendar.from_ical(events_calendar([updated], include_sources=True)).walk("VEVENT")[
        0
    ]
    assert updated["id"] == original["id"] and exported["UID"] == uid
    assert updated["sequence"] == 1 and exported["SEQUENCE"] == 1
    assert updated["metadata"]["review"] == "user_edited"
    assert updated["sources"] == original["sources"]
    assert value["description"] in str(exported["DESCRIPTION"])
    # Re-extraction cannot overwrite an edited source event or duplicate it.
    assert store.add_events(course, [value]) == 0
    assert store.events(course)[0]["title"] == "My exam wording"
    export = store.export()["courses"][0]
    assert len(export["planner_events"]) == 1 and len(export["event_sources"]) == 1
    store.delete_event(original["id"])
    assert store.events(course) == []


def test_selected_event_save_is_atomic_and_reauthorizes_even_duplicates(store, course):
    value = draft(store, course)
    bad = value | {"title": "Other event", "metadata": {}}
    bad["sources"] = [value["sources"][0] | {"quote": "Invented material"}]
    with pytest.raises(UserError, match="evidence"):
        store.add_events(course, [value, bad])
    assert store.events(course) == []
    store.add_events(course, [value])
    with pytest.raises(UserError, match="evidence"):
        store.add_events(course, [value | {"sources": bad["sources"]}])
    with pytest.raises(UserError):
        store.add_events(course, [value | {"start_date": None}])


def test_event_evidence_course_and_unknown_identifiers_are_rejected(store, course):
    value = draft(store, course)
    other = store.create_course("Second")
    with pytest.raises(UserError):
        store.add_events(other, [value])
    store.add_events(course, [value])
    for call in [
        lambda: store.update_event("unknown", {"title": "Intrusion"}),
        lambda: store.delete_event("unknown"),
        lambda: store.events("unknown"),
    ]:
        with pytest.raises(UserError):
            call()


def test_secondary_source_deletion_removes_derived_events(store, course):
    value = draft(store, course)
    second, _ = store.add_document(
        course, parse_upload("Term.txt", b"Spring 2026.", isolated=False)
    )
    p = scoped_passages(store, course, [second])[0]
    value["sources"].append(
        {"document_id": second, "passage_id": p.id, "locator": p.locator, "quote": "Spring 2026."}
    )
    store.add_events(course, [value])
    store.delete_document(second)
    assert store.events(course) == []


def test_unresolved_draft_needs_explicit_user_review_and_valid_values(store, course):
    value = draft(store, course)
    value["issues"] = ["Weekday disagreement"]
    with pytest.raises(UserError, match="Review"):
        store.add_events(course, [value])
    value["metadata"]["review"] = "user_confirmed"
    assert store.add_events(course, [value]) == 1


def test_v4_migration_backs_up_and_preserves_every_original_row(tmp_path):
    file = tmp_path / "workspace.sqlite3"
    store = Store(Database(file))
    course = store.create_course("Original course")
    store.add_card(course, "Original question", "Original answer")
    store.add_task(course, "Original task", "2026-11-02", 30)
    snapshot = store.export()["courses"][0]
    with sqlite3.connect(file) as con:
        con.execute("DROP TRIGGER document_events")
        con.execute("DROP TABLE event_sources")
        con.execute("DROP TABLE planner_events")
        con.execute("PRAGMA user_version=4")
    migrated = Store(Database(file))
    actual = migrated.export()["courses"][0]
    assert actual == snapshot
    backups = list(tmp_path.glob("workspace.pre-v5-*.sqlite3"))
    assert len(backups) == 1 and backups[0].stat().st_mode & 0o777 == 0o600
    with sqlite3.connect(backups[0]) as con:
        assert con.execute("PRAGMA user_version").fetchone()[0] == 4
        assert con.execute("SELECT front FROM cards").fetchone()[0] == "Original question"
    assert json.loads(actual["cards"][0]["metadata"]) == {}


def test_revised_material_can_replace_a_reviewed_event_without_changing_uid(store, course):
    old = draft(store, course)
    store.add_events(course, [old])
    original = store.events(course)[0]
    changed = draft(store, course, text="Final exam moved to November 4, 2026.")
    changed.update(start_date="2026-11-04", end_date="2026-11-04", replace_id=original["id"])
    changed["metadata"]["origin_key"] = "b" * 64
    assert store.add_events(course, [changed]) == 1
    actual = store.events(course)[0]
    assert actual["id"] == original["id"] and actual["sequence"] == 1
    assert actual["start_date"] == "2026-11-04"
    assert actual["created_at"] == original["created_at"]
    assert actual["sources"][0]["document_id"] != original["sources"][0]["document_id"]
    store.delete_document(original["sources"][0]["document_id"])
    assert store.events(course)[0]["id"] == original["id"]


def test_replacements_reject_cross_course_and_double_targets_atomically(store, course):
    value = draft(store, course)
    store.add_events(course, [value])
    saved = store.events(course)[0]
    changed = value | {
        "start_date": "2026-11-04",
        "end_date": "2026-11-04",
        "replace_id": saved["id"],
    }
    with pytest.raises(UserError, match="at most once"):
        store.add_events(course, [changed, changed])
    assert store.events(course)[0]["start_date"] == "2026-11-02"
    other = store.create_course("Other")
    with pytest.raises(UserError):
        store.add_events(other, [changed | {"sources": []}])
