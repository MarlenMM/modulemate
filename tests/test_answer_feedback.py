"""Local human review is scoped, durable and distinct from training labels."""

import math

import pytest

from modulemate.db import UserError
from modulemate.feedback import review_queue


def add_answer(store, course):
    store.add_message(
        course,
        "What does this mean?",
        "A supported explanation.",
        [],
        "Explain",
        {
            "model": "test",
            "notice": "Original notice",
            "evidence_assessment": {"parts": [{"estimate": 0.5}]},
        },
    )
    return store.messages(course)[-1]


def test_feedback_updates_only_its_message_and_preserves_provenance(store, course):
    message = add_answer(store, course)
    store.save_answer_feedback(course, message["id"], "unsupported", "The condition was omitted.")
    saved = store.messages(course)[-1]
    assert saved["question"] == message["question"] and saved["answer"] == message["answer"]
    assert saved["metadata"]["notice"] == "Original notice"
    assert saved["metadata"]["feedback"]["rating"] == "unsupported"
    assert saved["metadata"]["feedback"]["comment"] == "The condition was omitted."
    store.save_answer_feedback(course, message["id"], "helpful")
    assert store.messages(course)[-1]["metadata"]["feedback"]["rating"] == "helpful"


def test_feedback_cannot_cross_course_boundary(store, course):
    message = add_answer(store, course)
    other = store.create_course("Other")
    with pytest.raises(UserError):
        store.save_answer_feedback(other, message["id"], "incorrect")
    assert "feedback" not in store.messages(course)[-1]["metadata"]
    with pytest.raises(UserError):
        store.save_answer_feedback(course, "missing", "helpful")


@pytest.mark.parametrize(
    "rating,note", [("automatic truth", ""), ("helpful", "x" * 2001), ("helpful", None)]
)
def test_feedback_validates_labels_and_note_budget(store, course, rating, note):
    message = add_answer(store, course)
    with pytest.raises(UserError):
        store.save_answer_feedback(course, message["id"], rating, note)


def test_clear_conversation_also_removes_feedback(store, course):
    message = add_answer(store, course)
    store.save_answer_feedback(course, message["id"], "incorrect")
    store.clear_messages(course)
    assert not store.messages(course)


def test_uncertainty_queue_prioritizes_reported_errors_without_declaring_truth():
    messages = [
        {
            "id": str(i),
            "question": "q",
            "answer": "a",
            "metadata": {
                "evidence_assessment": {"parts": [{"estimate": p}]},
                "feedback": {"rating": rating},
            },
        }
        for i, p, rating in [(1, 0.5, None), (2, 0.99, "incorrect"), (3, 0.5, "helpful")]
    ]
    queue = review_queue(messages)
    assert [row["message_id"] for row in queue] == ["2", "1", "3"]
    assert queue[0]["label_status"].startswith("Human review required")
    assert len(review_queue(messages, limit=1)) == 1
    assert review_queue(messages, limit=-1) == []


def test_bad_confidence_values_cannot_poison_review_priority():
    queue = review_queue(
        [
            {
                "id": "m",
                "question": "q",
                "answer": "a",
                "metadata": {
                    "evidence_assessment": {
                        "parts": [{"estimate": p} for p in [float("nan"), float("inf"), -1, 2, "x"]]
                    }
                },
            }
        ]
    )
    assert math.isfinite(queue[0]["priority"]) and queue[0]["uncertainty"] == 0
