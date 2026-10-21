"""Behavior regressions with original content; neural weights tested separately."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from modulemate.assistant import answer_question
from modulemate.db import Database, UserError
from modulemate.dialogue import plan_turn, sentences
from modulemate.ingest import parse_upload
from modulemate.local_models import ModelUnavailable
from modulemate.retrieval import retrieve


class TinyEngine:
    key = "test-vectors-v1"
    dimensions = 3

    def __init__(self):
        self.encoded = []

    def encode(self, texts):
        self.encoded.extend(texts)
        return np.array([[1, len(text) % 7 + 1, 1] for text in texts], dtype=np.float32)

    def rerank(self, query, texts):
        return [3.0] * len(texts)


def add(store, course, name, text):
    return store.add_document(course, parse_upload(name, text.encode(), isolated=False))[0]


def test_vectors_persist_and_deleted_documents_cascade(store, populated):
    course, doc = populated
    engine = TinyEngine()
    found = retrieve(store, course, "retrieval practice", engine=engine)
    assert found.indexed > 0
    engine.encoded.clear()
    second = retrieve(store, course, "retrieval practice", engine=engine)
    assert second.indexed == 0
    assert engine.encoded == ["retrieval practice"]
    with store.db.connect() as con:
        assert con.execute("SELECT count(*) FROM embeddings").fetchone()[0] > 0
    store.delete_document(doc)
    assert retrieve(store, course, "retrieval practice", engine=engine).passages == []
    with store.db.connect() as con:
        assert con.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0


def test_neural_scope_is_checked_before_inference(db, store, populated):
    course, doc = populated
    engine = TinyEngine()
    with pytest.raises(UserError):
        retrieve(store, "missing-course", "retrieval", engine=engine)
    other = store.create_course("Other")
    with pytest.raises(UserError):
        retrieve(store, other, "retrieval", [doc], engine=engine)
    assert engine.encoded == []
    assert retrieve(store, course, "retrieval", [], engine=engine).passages == []


def test_filters_prevent_unselected_text_from_reaching_model(store, course):
    selected = add(store, course, "one.txt", "A stack removes the most recent item first.")
    add(store, course, "two.txt", "Secret unselected topic: probabilities represent uncertainty.")
    engine = TinyEngine()
    found = retrieve(store, course, "stack", [selected], engine=engine)
    assert found.passages
    assert not any("Secret unselected" in text for text in engine.encoded)


def test_corrupt_cache_rebuilt_and_model_change_invalidates(store, populated):
    course, _ = populated
    engine = TinyEngine()
    first = retrieve(store, course, "retrieval", engine=engine)
    with store.db.connect() as con:
        con.execute("UPDATE embeddings SET vector=?", (b"broken",))
    assert retrieve(store, course, "retrieval", engine=engine).indexed == first.indexed
    engine.key = "test-vectors-v2"
    assert retrieve(store, course, "retrieval", engine=engine).indexed == first.indexed


def test_deletion_during_embedding_cannot_resurrect_source(store, populated):
    course, doc = populated

    class DeletingEngine(TinyEngine):
        deleted = False

        def encode(self, texts):
            if not self.deleted:
                self.deleted = True
                store.delete_document(doc)
            return super().encode(texts)

    result = retrieve(store, course, "retrieval practice", engine=DeletingEngine())
    assert result.passages == []
    with store.db.connect() as con:
        assert con.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0


def test_model_failure_and_nan_return_explicit_fallback(store, populated):
    course, _ = populated

    class Broken(TinyEngine):
        def encode(self, texts):
            return np.full((len(texts), 3), np.nan)

    result = retrieve(store, course, "retrieval", engine=Broken())
    assert result.passages and result.method == "lexical fallback" and result.notice

    class Unavailable(TinyEngine):
        def encode(self, texts):
            raise ModelUnavailable("Sensitive internal failure")

    result = retrieve(store, course, "retrieval", engine=Unavailable())
    assert "Sensitive" not in result.notice


def test_version_one_migration_preserves_existing_data(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as con:
        legacy = (
            (Path(__file__).parent / "fixtures/legacy_v2.sql")
            .read_text()
            .split("CREATE TABLE embeddings")[0]
        )
        con.executescript(legacy)
        con.execute("PRAGMA user_version=1")
        con.execute("INSERT INTO users VALUES ('u','student','hash','salt','today')")
        con.execute("INSERT INTO courses VALUES ('c','u','Old course','','today')")
        con.execute(
            "INSERT INTO messages VALUES ('m','c','question','answer','[]','Explain','today')"
        )
    database = Database(path)
    with database.connect() as con:
        assert con.execute("PRAGMA user_version").fetchone()[0] == 4
        assert con.execute("SELECT metadata FROM messages WHERE id='m'").fetchone()[0] == "{}"
        assert con.execute("SELECT name FROM courses WHERE id='c'").fetchone()[0] == "Old course"


def test_greetings_do_not_search_or_call_provider(store, course):
    result = answer_question(store, course, "Hello!")
    assert result.kind == "conversation" and not result.passages
    assert "Hi!" in result.text
    assert answer_question(store, course, "What does it mean?").kind == "clarification"


def test_comparison_second_reference_and_topic_switch(store, course):
    add(
        store,
        course,
        "structures.md",
        "# Stack\nA stack removes the newest item first.\n\n# Queue\nA queue removes the oldest item first.",
    )
    add(store, course, "plants.txt", "Photosynthesis converts light energy into chemical energy.")
    result = answer_question(store, course, "Compare stack and queue")
    assert "**stack**" in result.text and "**queue**" in result.text
    assert "newest" in result.text and "oldest" in result.text
    ambiguous = answer_question(store, course, "Explain it")
    assert ambiguous.kind == "clarification"
    second = answer_question(store, course, "Explain the second one")
    assert second.metadata["query"] == "queue" and "oldest" in second.text
    followup = answer_question(store, course, "Quiz me on that")
    assert followup.metadata["topic"] == "queue"
    assert "oldest" not in followup.text  # Recall before revealing.
    new = answer_question(store, course, "What is photosynthesis?")
    assert not new.metadata["used_context"]
    assert "queue" not in new.metadata["query"]


def test_filter_change_drops_prior_topic_and_provider_history(store, course):
    stack = add(store, course, "stack.txt", "A stack removes the newest item first.")
    queue = add(store, course, "queue.txt", "A queue removes the oldest item first.")
    answer_question(store, course, "What is a stack?", document_ids=[stack])
    result = answer_question(store, course, "Explain it again", document_ids=[queue])
    assert result.kind == "clarification" and not result.passages


def test_missing_comparison_side_is_explicit(store, course):
    add(store, course, "stack.txt", "A stack removes the newest item first.")
    result = answer_question(store, course, "Compare stack and quantum teleportation")
    assert "couldn't find enough evidence for this side" in result.text


def test_sentences_remove_headings_and_keep_source_membership(store, populated):
    from modulemate.search import overview

    course, _ = populated
    passages = overview(store, course)
    extracted = sentences(passages)
    assert all(
        " ".join(s.text.split()) in " ".join(passages[s.source - 1].text.split()) for s in extracted
    )
    assert not any("Retrieval practice Retrieval practice" in s.text for s in extracted)
    assert not any(s.text == "five-minute presentation." for s in extracted)


def test_summary_has_multiple_topics_and_persisted_limit(store, populated):
    course, _ = populated
    result = answer_question(store, course, "Summarize my notes")
    assert "Retrieval practice" in result.text
    assert "Spacing" in result.text
    assert "Interleaving" in result.text
    persisted = store.messages(course)[-1]["metadata"]
    assert persisted["notice"] == result.notice and "not every page" in result.notice


def test_unsupported_detail_not_claimed_as_answer(store, populated):
    course, _ = populated
    result = answer_question(store, course, "Who invented retrieval practice?")
    assert "don't establish" in result.text
    assert "Partial evidence" in result.notice


def test_intent_training_is_separate_from_evaluation():
    train = json.loads(Path("modulemate/intent_training.json").read_text())
    evaluation = json.loads(Path("evaluation/intents.json").read_text())
    assert not {r["text"].casefold() for r in train} & {r["text"].casefold() for r in evaluation}
    assert len({r["intent"] for r in train}) == 4


def test_self_contained_question_does_not_inherit_context():
    history = [
        {
            "question": "What is a stack?",
            "sources": [{"document_id": "d"}],
            "metadata": {"topic": "stack"},
        }
    ]
    plan = plan_turn("How does this neural network learn its parameters?", "Auto", history)
    assert not plan.used_context and "stack" not in plan.query


def test_recall_attempt_continues_topic_without_claiming_a_grade(store, populated):
    course, _ = populated
    answer_question(store, course, "What is retrieval practice?")
    answer_question(store, course, "Quiz me on that")
    review = answer_question(store, course, "I think it means rereading the chapter")
    assert review.metadata["recall_review"]
    assert review.metadata["query"] == "retrieval practice"
    assert "Check your explanation" in review.text
    assert "Retrieval practice means" in review.text
    assert "not been automatically graded" in review.notice
    switched = answer_question(store, course, "What is interleaving?")
    assert not switched.metadata["used_context"]


def test_reader_hints_must_be_real_spans_in_the_selected_context():
    from modulemate.dialogue import Plan, compose_local
    from modulemate.reader import Span
    from modulemate.search import Passage

    passages = [
        Passage(
            1,
            "d",
            "notes.txt",
            "part 1",
            "Photosynthesis converts light energy into chemical energy. Chlorophyll absorbs light.",
        )
    ]

    class ForgedReader(TinyEngine):
        def extract(self, question, contexts):
            return [Span(1, "A fabricated sentence", 0, 21, 1000)]

    answer, _, _ = compose_local(
        Plan("What is photosynthesis?", "photosynthesis", "Explain"), passages, ForgedReader()
    )
    assert "fabricated" not in answer
    assert "Photosynthesis converts" in answer


def test_lightweight_answers_distinguish_word_count_and_duration(store, course):
    add(
        store,
        course,
        "brief.txt",
        "The report contains 1800 words. The presentation lasts six minutes. The presentation contributes 25 percent of the mark.",
    )
    assert "1800 words" in answer_question(store, course, "How long should the report be?").text
    assert "six minutes" in answer_question(store, course, "How long is the presentation?").text
    assert (
        "25 percent" in answer_question(store, course, "How much is the presentation worth?").text
    )


def test_cjk_sentences_are_split_and_ranked_by_the_actual_question(store, course):
    add(
        store,
        course,
        "korean.txt",
        "금요일 시험 안내입니다. 시험 시간은 오전 열 시입니다. 복습은 배운 내용을 다시 확인하는 활동입니다.",
    )
    answer = answer_question(store, course, "금요일 시험 시간")
    assert "오전 열 시" in answer.text and "복습은" not in answer.text


def test_exact_identifiers_constrain_neural_and_lexical_candidates(store, course):
    selected = add(
        store, course, "one.txt", "CS142 studies memory allocation and linked structures."
    )
    add(store, course, "two.txt", "CS143 studies memory allocation and linked structures.")
    for engine in (None, TinyEngine()):
        found = retrieve(store, course, "CS142 memory allocation", engine=engine)
        assert found.passages and {p.document_id for p in found.passages} == {selected}
        assert not retrieve(store, course, "CS999 memory allocation", engine=engine).passages


def test_document_identifiers_do_not_exclude_later_sections(store, course):
    doc = add(
        store,
        course,
        "lec1.md",
        "# CS142\nCourse introduction.\n\n## Median\nThe median is the middle value after sorting observations.",
    )
    add(store, course, "lec10.md", "The median appears in a different lecture.")
    for engine in (None, TinyEngine()):
        for question in ("CS142 median", "lec1 median"):
            found = retrieve(store, course, question, engine=engine)
            assert found.passages and all(p.document_id == doc for p in found.passages)
            assert any("middle value" in p.text for p in found.passages)
