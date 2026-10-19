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
    click(app, "Draft cards from material")
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
