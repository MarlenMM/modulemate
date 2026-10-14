from datetime import date

import pytest

from modulemate.db import UserError
from modulemate.search import overview
from modulemate.study import calendar_export, cloze_cards, plan_tasks, schedule_review


def test_review_progression_and_lapse():
    assert schedule_review(0, 0, "Good") == (1, 1)
    assert schedule_review(1, 1, "Good") == (3, 2)
    assert schedule_review(3, 2, "Good") == (6, 3)
    assert schedule_review(50, 8, "Again") == (1, 0)
    assert schedule_review(300, 8, "Easy") == (365, 9)
    with pytest.raises(UserError):
        schedule_review(1, 1, "invalid")


def test_review_double_submission_rejected(store, course):
    today = date(2026, 10, 14)
    store.add_card(course, "Question", "Answer", today=today)
    card = store.cards(course)[0]
    store.review_card(card["id"], "Good", today=today)
    assert store.cards(course)[0]["due"] == "2026-10-15"
    with pytest.raises(UserError, match="not due"):
        store.review_card(card["id"], "Good", today=today)
    assert store.review_counts(course) == {"Good": 1}


def test_cloze_cards_grounded_and_deduplicated(store, populated):
    course, _ = populated
    passages = overview(store, course)
    drafts = cloze_cards(passages)
    assert drafts and "_____" in drafts[0]["front"]
    assert store.add_card(course, **drafts[0])
    assert not store.add_card(course, **drafts[0])


def task(name, due, minutes=30, done=0):
    return {
        "id": name,
        "title": name,
        "due": due,
        "minutes": minutes,
        "done": done,
        "created_at": name,
    }


def test_plan_enforces_capacity_and_warns_unallocated():
    tasks = [
        task("urgent", "2026-10-14", 90),
        task("later", "2026-10-16", 90),
        task("done", "2026-10-14", 90, 1),
    ]
    sessions, warnings = plan_tasks(tasks, daily_minutes=60, horizon=3, today=date(2026, 10, 14))
    assert sessions[0]["title"] == "urgent"
    assert sum(s["minutes"] for s in sessions if s["date"] == "2026-10-14") == 60
    assert all(s["minutes"] <= 50 for s in sessions)
    assert sum(s["minutes"] for s in sessions) == 150
    assert any("30 minutes" in w for w in warnings)
    assert all(s["title"] != "done" for s in sessions)


def test_plan_overdue_is_visible():
    sessions, warnings = plan_tasks([task("late", "2026-10-13")], today=date(2026, 10, 14))
    assert sessions[0]["date"] == "2026-10-14"
    assert any("Overdue" in w for w in warnings)


def test_calendar_escaping_and_unicode_line_folding():
    title = "공부 " * 50 + ", deadline; test\nEND:VEVENT"
    text = calendar_export([task(title, "2026-10-20")])
    assert "DTEND;VALUE=DATE:20261021" in text
    assert text.count("\r\nEND:VEVENT\r\n") == 1
    assert "\\, deadline\\; test\\nEND:VEVENT" in text.replace("\r\n ", "")
    assert all(len(line.encode()) <= 75 for line in text.split("\r\n"))


def test_cloze_prefers_learning_content_over_example_boilerplate(store, populated):
    course, _ = populated
    drafts = cloze_cards(overview(store, course))
    assert "university" not in drafts[0]["back"].lower()
    assert "retrieval practice" in drafts[0]["back"].lower()
