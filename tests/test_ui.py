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
    widget(app.radio, "Account").set_value("Create account").run()
    widget(app.text_input, "Username").set_value("ui_student")
    widget(app.text_input, "Password").set_value("a unique passphrase")
    widget(app.text_input, "Repeat password").set_value("a unique passphrase")
    click(app, "Create account")
    widget(app.text_input, "Course name").set_value("Learning Lab")
    click(app, "Create course")
    click(app, "Add original Learning Lab example")
    assert any(m.value == "1" for m in app.metric)
    page(app, "Assistant")
    app.chat_input[0].set_value("What is retrieval practice?").run()
    assert not app.exception
    assert any("Matching excerpts" in t.value for t in app.text)
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
    page(app, "Account")
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
    click(app, "Sign out")
    assert any(t.value == "Your materials. Your study space." for t in app.title)


def test_session_does_not_leak_to_another_browser(tmp_path, monkeypatch):
    monkeypatch.setenv("MODULEMATE_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    widget(app.radio, "Account").set_value("Create account").run()
    widget(app.text_input, "Username").set_value("private_user")
    widget(app.text_input, "Password").set_value("a unique passphrase")
    widget(app.text_input, "Repeat password").set_value("a unique passphrase")
    click(app, "Create account")
    second = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    assert any(t.value == "Your materials. Your study space." for t in second.title)
    assert not any(r.label == "Workspace" for r in second.radio)
