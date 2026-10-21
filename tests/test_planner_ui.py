import json
from datetime import date
from pathlib import Path

from streamlit.testing.v1 import AppTest

from modulemate.db import Database, Store
from modulemate.ingest import parse_upload


def widget(items, label):
    return next(x for x in items if x.label == label)


def click(app, label):
    widget(app.button, label).click().run()
    assert not app.exception


class PlannerBackend:
    label = "Local planner test backend"

    def __init__(self):
        self.calls = []

    def complete(self, system, payload):
        self.calls.append(payload)
        if "events_to_review" in payload:
            return json.dumps(
                {
                    "reviews": [
                        {"event_id": i, "verdict": "keep", "reason": "", "source_quote": ""}
                        for i in range(1, len(payload["events_to_review"]) + 1)
                    ]
                }
            )
        b = payload["candidates"][0]
        title = "Final exam" if "Final exam" in b["text"] else "Lab meeting"
        return json.dumps(
            {
                "events": [
                    {
                        "candidate_id": b["id"],
                        "title": title,
                        "detail": b["text"],
                        "location": "",
                        "kind": "Exam" if title == "Final exam" else "Meeting",
                        "disposition": "scheduled",
                        "reason": "",
                        "start_date_id": 1,
                        "end_date_id": 0,
                        "start_time_id": 0,
                        "end_time_id": 0,
                        "duration_id": 0,
                        "weekly_days": [],
                        "exclude_date_ids": [],
                    }
                ]
            }
        )


def planner_app(tmp_path, monkeypatch, *, missing_year=False):
    import modulemate.generation

    monkeypatch.setenv("MODULEMATE_DATA_DIR", str(tmp_path))
    backend = PlannerBackend()
    monkeypatch.setattr(modulemate.generation, "chat_status", lambda: "ready")
    monkeypatch.setattr(modulemate.generation, "local_backend", lambda: backend)
    store = Store(Database(tmp_path / "modulemate.sqlite3"))
    course = store.create_course("Planner lab")
    docs = []
    for name, text in [
        ("Cedar", "Final exam on November 2" + ("." if missing_year else ", 2026.")),
        ("Birch", "Lab meeting on November 3, 2026."),
    ]:
        doc, _ = store.add_document(
            course, parse_upload(name + ".md", text.encode(), isolated=False)
        )
        docs.append(doc)
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    widget(app.radio, "Workspace").set_value("Planner").run()
    assert not app.exception
    widget(app.multiselect, "Focus on documents (leave empty for all)").set_value([docs[0]]).run()
    return app, store, course, docs, backend


def test_planner_extract_edit_select_save_export_and_remove(tmp_path, monkeypatch):
    app, store, course, docs, backend = planner_app(tmp_path, monkeypatch)
    click(app, "Extract dates")
    assert widget(app.checkbox, "Keep event 1").value
    assert {b["document"] for b in backend.calls[0]["candidates"]} == {"Cedar.md"}
    widget(app.text_input, "Event title").set_value("My final exam")
    click(app, "Save selected events")
    assert store.events(course)[0]["title"] == "My final exam"
    assert store.events(course)[0]["metadata"]["review"] == "user_edited"
    widget(app.radio, "Planner tools").set_value("Events and dates").run()
    assert not app.exception
    assert any(x.label == "Export selected events (.ics)" for x in app.get("download_button"))
    assert len(widget(app.multiselect, "Events to export").value) == 1
    assert any(t.value == "Final exam on November 2, 2026." for t in app.text)
    click(app, "Remove event")
    assert store.events(course) == []


def test_unresolved_dates_are_unchecked_and_can_be_corrected(tmp_path, monkeypatch):
    app, store, course, docs, backend = planner_app(tmp_path, monkeypatch, missing_year=True)
    click(app, "Extract dates")
    assert not widget(app.checkbox, "Keep event 1").value
    assert widget(app.date_input, "Start date").value is None
    click(app, "Save selected events")
    assert store.events(course) == []
    widget(app.checkbox, "Keep event 1").check()
    widget(app.date_input, "Start date").set_value(date(2027, 11, 2))
    click(app, "Save selected events")
    assert store.events(course)[0]["start_date"] == "2027-11-02"


def test_saved_edit_refreshes_export_labels_without_reselecting_an_excluded_event(
    tmp_path, monkeypatch
):
    app, store, course, docs, backend = planner_app(tmp_path, monkeypatch)
    click(app, "Extract dates")
    click(app, "Save selected events")
    saved = store.events(course)[0]
    widget(app.radio, "Planner tools").set_value("Events and dates").run()
    exports = widget(app.multiselect, "Events to export")
    original_key = exports.key
    exports.set_value([]).run()
    title = next(
        x
        for x in app.text_input
        if x.label == "Event title" and x.key.startswith("edit_event_" + saved["id"])
    )
    title.set_value("Renamed final exam")
    click(app, "Save event changes")
    exports = widget(app.multiselect, "Events to export")
    assert exports.key != original_key
    assert exports.value == []
    assert any("Renamed final exam" in option for option in exports.options)
    actual = store.events(course)[0]
    assert actual["id"] == saved["id"] and actual["sequence"] == saved["sequence"] + 1


def test_large_selection_continuation_retains_preview_and_resets_with_focus(tmp_path, monkeypatch):
    import modulemate.planner

    monkeypatch.setattr(modulemate.planner, "REVIEW_BLOCKS_PER_REQUEST", 1)
    app, store, course, docs, backend = planner_app(tmp_path, monkeypatch)
    widget(app.multiselect, "Focus on documents (leave empty for all)").set_value([]).run()
    click(app, "Extract dates")
    app.run()
    assert widget(app.button, "Continue extracting dates").disabled
    assert widget(app.text_input, "Event title").value == "Final exam"
    click(app, "Save selected events")
    widget(app.radio, "Planner tools").set_value("Extract from materials").run()
    assert not widget(app.button, "Continue extracting dates").disabled
    click(app, "Continue extracting dates")
    assert widget(app.text_input, "Event title").value == "Lab meeting"
    click(app, "Save selected events")
    assert len(store.events(course)) == 2
    widget(app.radio, "Planner tools").set_value("Extract from materials").run()
    assert not any(b.label == "Continue extracting dates" for b in app.button)
    widget(app.multiselect, "Focus on documents (leave empty for all)").set_value([docs[0]]).run()
    click(app, "Extract dates")
    assert widget(app.text_input, "Event title").value == "Final exam"


def test_planner_preview_focus_context_and_fresh_editor_identity(tmp_path, monkeypatch):
    app, store, course, docs, backend = planner_app(tmp_path, monkeypatch)
    click(app, "Extract dates")
    widget(app.text_input, "Event title").set_value("Old unsaved wording")
    click(app, "Extract dates")
    assert widget(app.text_input, "Event title").value == "Final exam"
    widget(app.selectbox, "Numeric date format").set_value("Day / month").run()
    assert not any(b.label == "Save selected events" for b in app.button)
    click(app, "Extract dates")
    widget(app.multiselect, "Focus on documents (leave empty for all)").set_value([docs[1]]).run()
    assert not any(b.label == "Save selected events" for b in app.button)
    click(app, "Extract dates")
    assert widget(app.text_input, "Event title").value == "Lab meeting"


def test_revised_notice_requires_deliberate_choice_and_can_keep_calendar_id(tmp_path, monkeypatch):
    app, store, course, docs, backend = planner_app(tmp_path, monkeypatch)
    click(app, "Extract dates")
    click(app, "Save selected events")
    original = store.events(course)[0]
    revised, _ = store.add_document(
        course, parse_upload("Revised.md", b"Final exam on November 4, 2026.", isolated=False)
    )
    widget(app.radio, "Planner tools").set_value("Extract from materials").run()
    widget(app.multiselect, "Focus on documents (leave empty for all)").set_value([revised]).run()
    click(app, "Extract dates")
    assert not widget(app.checkbox, "Keep event 1").value
    widget(app.checkbox, "Keep event 1").check()
    widget(app.selectbox, "Existing event to replace · draft 1").set_value(original["id"])
    click(app, "Save selected events")
    actual = store.events(course)[0]
    assert len(store.events(course)) == 1
    assert actual["id"] == original["id"] and actual["sequence"] == 1
    assert actual["start_date"] == "2026-11-04"


def test_unselected_source_time_is_not_silently_saved_as_an_all_day_event(tmp_path, monkeypatch):
    app, store, course, docs, backend = planner_app(tmp_path, monkeypatch)
    uncertain, _ = store.add_document(
        course,
        parse_upload("Clock.md", b"Final exam on November 2, 2026 at 8 CST.", isolated=False),
    )
    app.run()
    widget(app.multiselect, "Focus on documents (leave empty for all)").set_value([uncertain]).run()
    click(app, "Extract dates")
    assert not widget(app.checkbox, "All-day / date only").value
    widget(app.checkbox, "Keep event 1").check()
    click(app, "Save selected events")
    assert store.events(course) == [] and app.error
    widget(app.text_input, "Start time (24-hour)").set_value("20:00")
    widget(app.text_input, "Event time zone").set_value("Asia/Shanghai")
    click(app, "Save selected events")
    actual = store.events(course)[0]
    assert actual["start_time"] == "20:00" and actual["timezone"] == "Asia/Shanghai"
