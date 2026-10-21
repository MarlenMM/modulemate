"""Real Streamlit widget flows against a fresh private database, without API calls."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def widget(elements, label):
    return next(e for e in elements if e.label == label)


def click(app, label):
    widget(app.button, label).click().run()
    assert not app.exception


def page(app, label):
    widget(app.radio, "Workspace").set_value(label).run()
    assert not app.exception


def test_complete_local_study_journey(tmp_path, monkeypatch):
    monkeypatch.setenv("MODULEMATE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MODULEMATE_AI_API_KEY", "")
    monkeypatch.setenv("MODULEMATE_AI_MODEL", "")
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    assert not app.exception
    widget(app.text_input, "Course name").set_value("Learning Lab")
    click(app, "Create course")
    click(app, "Add original Learning Lab example")
    assert any(m.value == "1" for m in app.metric)
    page(app, "Assistant")
    app.chat_input[0].set_value("What is retrieval practice?").run()
    assert not app.exception
    assert any("Retrieval practice means" in t.value for t in app.markdown)
    widget(app.selectbox, "Your assessment").set_value("unsupported")
    widget(app.text_area, "Correction or note (optional)").set_value("Please check the qualifier.")
    click(app, "Save answer feedback")
    assert any(d.label == "Export answer review cases (JSON)" for d in app.get("download_button"))
    click(app, "Quiz me on that")
    assert any("in your own words" in t.value for t in app.markdown)
    page(app, "Cards")
    widget(app.radio, "Study tools").set_value("Create cards").run()
    click(app, "Generate learning cards")
    click(app, "Save selected cards")
    widget(app.radio, "Study tools").set_value("Review").run()
    assert not any(b.label == "Good" for b in app.button)
    click(app, "Reveal answer")
    click(app, "Good")
    widget(app.radio, "Study tools").set_value("Practice quiz").run()
    click(app, "Start a fresh 5-card quiz")
    click(app, "Check answer")
    click(app, "I recalled it")
    page(app, "Planner")
    widget(app.text_input, "Task or assignment").set_value("Review chapter one")
    click(app, "Add task")
    assert app.dataframe
    click(app, "Mark done")
    assert any("All tasks are complete" in s.value for s in app.success)
    page(app, "Settings")
    assert not app.exception
    assert any(d.label == "Export all my data (JSON)" for d in app.get("download_button"))
    page(app, "Library")
    widget(
        app.checkbox, "Delete this document, its linked cards and this course's chat history"
    ).check().run()
    click(app, "Delete document")
    assert any("Your library is empty" in s.value for s in app.info)
    page(app, "Cards")
    assert any(m.label == "Saved cards" and m.value == "0" for m in app.metric)

    assert not any(e.label in {"Username", "Password", "Repeat password"} for e in app.text_input)
    assert not any(e.label == "Sign out" for e in app.button)


def test_new_browser_opens_same_local_workspace_without_login(tmp_path, monkeypatch):
    monkeypatch.setenv("MODULEMATE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MODULEMATE_CHAT", "off")
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    widget(app.text_input, "Course name").set_value("Local learning")
    click(app, "Create course")
    second = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    assert not second.exception
    assert widget(second.selectbox, "Current course").value
    assert any(r.label == "Workspace" for r in second.radio)
    assert not any(r.label == "Account" for r in second.radio)


def test_document_focus_survives_navigation_without_crossing_course_or_tool(tmp_path, monkeypatch):
    from modulemate.db import Database, Store
    from modulemate.ingest import parse_upload

    monkeypatch.setenv("MODULEMATE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MODULEMATE_CHAT", "off")
    store = Store(Database(tmp_path / "modulemate.sqlite3"))
    first, second = store.create_course("First lab"), store.create_course("Second lab")
    documents = []
    for course_id, name in ((first, "Cedar"), (first, "Birch"), (second, "Maple")):
        document, _ = store.add_document(
            course_id,
            parse_upload(f"{name}.txt", f"The {name} label reads pH = 7.".encode(), isolated=False),
        )
        documents.append(document)
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    widget(app.selectbox, "Current course").set_value(first).run()
    page(app, "Assistant")
    widget(app.multiselect, "Focus on documents (leave empty for all)").set_value(
        [documents[0]]
    ).run()
    page(app, "Cards")
    widget(app.radio, "Study tools").set_value("Create cards").run()
    focus = widget(app.multiselect, "Focus on documents (leave empty for all)")
    assert focus.value == []
    focus.set_value([documents[1]]).run()
    page(app, "Assistant")
    assert widget(app.multiselect, "Focus on documents (leave empty for all)").value == [
        documents[0]
    ]
    widget(app.selectbox, "Current course").set_value(second).run()
    assert widget(app.multiselect, "Focus on documents (leave empty for all)").value == []
    widget(app.multiselect, "Focus on documents (leave empty for all)").set_value(
        [documents[2]]
    ).run()
    widget(app.selectbox, "Current course").set_value(first).run()
    assert widget(app.multiselect, "Focus on documents (leave empty for all)").value == [
        documents[0]
    ]
    page(app, "Cards")
    widget(app.radio, "Study tools").set_value("Create cards").run()
    assert widget(app.multiselect, "Focus on documents (leave empty for all)").value == [
        documents[1]
    ]


class CardBackend:
    label = "Card test model"

    def __init__(self):
        self.calls = []

    def complete(self, system, payload):
        import json

        self.calls.append(payload)
        if "cards_to_review" in payload:
            return json.dumps(
                {
                    "reviews": [
                        {"card_id": c["card_id"], "verdict": "keep", "issue": ""}
                        for c in payload["cards_to_review"]
                    ]
                }
            )
        evidence = next(
            e for s in payload["source_evidence"] for e in s["excerpts"] if "violet" in e["quote"]
        )
        return json.dumps(
            {
                "cards": [
                    {
                        "kind": "qa",
                        "front": "What colour is the Cedar sample?",
                        "back": "violet",
                        "evidence_ids": [evidence["evidence_id"]],
                    }
                ]
            }
        )


def card_app(tmp_path, monkeypatch):
    import modulemate.generation
    from modulemate.db import Database, Store
    from modulemate.ingest import parse_upload

    monkeypatch.setenv("MODULEMATE_DATA_DIR", str(tmp_path))
    store = Store(Database(tmp_path / "modulemate.sqlite3"))
    course = store.create_course("Card lab")
    documents = []
    for name, colour in (("Cedar", "violet"), ("Birch", "amber")):
        document, _ = store.add_document(
            course,
            parse_upload(f"{name}.txt", f"The {name} sample is {colour}.".encode(), isolated=False),
        )
        documents.append(document)
    backend = CardBackend()
    monkeypatch.setattr(modulemate.generation, "chat_status", lambda: "ready")
    monkeypatch.setattr(modulemate.generation, "local_backend", lambda: backend)
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    page(app, "Cards")
    widget(app.radio, "Study tools").set_value("Create cards").run()
    widget(app.multiselect, "Focus on documents (leave empty for all)").set_value(
        [documents[0]]
    ).run()
    return app, store, course, documents, backend


def test_ai_card_preview_edit_save_and_reveal_sources(tmp_path, monkeypatch):
    app, store, course, documents, backend = card_app(tmp_path, monkeypatch)
    click(app, "Generate learning cards")
    assert widget(app.text_area, "Answer · draft 1").value == "violet"
    generation = next(c for c in backend.calls if "source_evidence" in c)
    assert {s["document"] for s in generation["source_evidence"]} == {"Cedar.txt"}
    widget(app.text_area, "Answer · draft 1").set_value("Violet, according to my edited wording.")
    click(app, "Save selected cards")
    saved = store.cards(course)[0]
    assert saved["document_id"] == documents[0]
    assert saved["metadata"]["review"] == "user_edited"
    assert saved["sources"][0]["quote"] == "The Cedar sample is violet."
    widget(app.radio, "Study tools").set_value("Review").run()
    assert not any(e.label == "Supporting material" for e in app.expander)
    click(app, "Reveal answer")
    assert any(e.label == "Supporting material" for e in app.expander)
    assert any(t.value == "The Cedar sample is violet." for t in app.text)


def test_preview_survives_navigation_and_is_invalidated_by_source_change(tmp_path, monkeypatch):
    app, store, course, documents, backend = card_app(tmp_path, monkeypatch)
    click(app, "Generate learning cards")
    page(app, "Library")
    page(app, "Cards")
    widget(app.radio, "Study tools").set_value("Create cards").run()
    assert widget(app.text_area, "Answer · draft 1").value == "violet"
    widget(app.multiselect, "Focus on documents (leave empty for all)").set_value(
        [documents[1]]
    ).run()
    assert not any(b.label == "Save selected cards" for b in app.button)
    assert store.cards(course) == []


def test_regenerated_preview_does_not_reuse_old_editor_values(tmp_path, monkeypatch):
    app, store, course, documents, backend = card_app(tmp_path, monkeypatch)
    click(app, "Generate learning cards")
    widget(app.text_area, "Answer · draft 1").set_value("A stale editor value")
    click(app, "Generate learning cards")
    assert widget(app.text_area, "Answer · draft 1").value == "violet"
    widget(app.checkbox, "Save draft 1").uncheck()
    click(app, "Save selected cards")
    assert store.cards(course) == []
    click(app, "Discard preview")
    assert not any(b.label == "Save selected cards" for b in app.button)


def test_saved_recall_cards_distinguish_repeated_filenames(tmp_path, monkeypatch):
    from modulemate.db import Database, Store
    from modulemate.ingest import parse_upload

    monkeypatch.setenv("MODULEMATE_DATA_DIR", str(tmp_path))
    store = Store(Database(tmp_path / "modulemate.sqlite3"))
    course = store.create_course("Two imports")
    for value in (7, 8):
        document, _ = store.add_document(
            course, parse_upload("Signal.txt", f"pH = {value}".encode(), isolated=False)
        )
        store.add_card(course, "What pH does this import state?", str(value), document, "part 1")
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    page(app, "Cards")
    assert any("Signal.txt · import 1" in c.value.replace("\\", "") for c in app.caption)
    click(app, "Reveal answer")
    click(app, "Good")
    assert any("Signal.txt · import 2" in c.value.replace("\\", "") for c in app.caption)
