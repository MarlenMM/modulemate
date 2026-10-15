"""Explicit real-weight integration checks, separate from network-free unit tests."""

import os

import pytest

from modulemate.assistant import answer_question
from modulemate.local_models import get_local_models
from modulemate.retrieval import retrieve
from scripts.evaluate_intelligence import load_fixture

pytestmark = pytest.mark.skipif(
    os.getenv("MODULEMATE_TEST_LOCAL_ML") != "1",
    reason="Set MODULEMATE_TEST_LOCAL_ML=1 after model setup",
)


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setenv("MODULEMATE_LOCAL_ML", "auto")
    models = get_local_models()
    assert models is not None, "Run scripts/prepare_models.py first"
    return models


def test_real_paraphrase_and_out_of_scope(store, course, engine):
    load_fixture(store, course)
    question = "How can revision be spread out over time?"
    lexical = retrieve(store, course, question)
    neural = retrieve(store, course, question, engine=engine)
    assert not lexical.passages or lexical.passages[0].locator != "spacing"
    assert neural.passages[0].locator == "spacing"
    answer = answer_question(store, course, question, engine=engine)
    assert "Spacing distributes review" in answer.text
    assert not retrieve(store, course, "How do tectonic plates move?", engine=engine).passages


def test_real_comparison_continuity_and_answer_length(store, populated, engine):
    course, _ = populated
    comparison = answer_question(store, course, "Compare spacing and interleaving", engine=engine)
    assert "Spacing means" in comparison.text and "Interleaving alternates" in comparison.text
    followup = answer_question(store, course, "Explain the second one", engine=engine)
    assert "Interleaving alternates" in followup.text
    assert followup.metadata["topic"] == "interleaving"
    fact = answer_question(store, course, "How long do the study periods last?", engine=engine)
    assert "25 minutes" in fact.text and len(fact.text) < 500
    assert all("[" in answer.text for answer in [comparison, followup, fact])


def test_real_summary_then_recall_prompt(store, populated, engine):
    course, _ = populated
    summary = answer_question(store, course, "Summarize my notes", engine=engine)
    assert all(topic in summary.text for topic in ["Retrieval practice", "Spacing", "Interleaving"])
    prompt = answer_question(store, course, "Quiz me on that", engine=engine)
    assert prompt.kind == "local" and "in your own words" in prompt.text


def test_real_reader_span_membership_and_empty_context(engine):
    context = "The demonstration report must contain 1800 words."
    spans = engine.extract("How many words must the report contain?", [context])
    assert spans
    assert all(span.source == 1 and context[span.start : span.end] == span.text for span in spans)
    assert "1800" in spans[0].text
    assert engine.extract("What is missing?", []) == []


def test_models_load_and_infer_with_network_connections_denied(engine, monkeypatch):
    import socket

    from modulemate.local_models import LocalModels, model_dir

    def deny(*args, **kwargs):
        raise AssertionError("Unexpected network connection during local inference")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    models = LocalModels(model_dir())
    assert models.encode(["An offline inference check."]).shape == (1, 384)
    assert len(models.rerank("offline check", ["An offline inference check."])) == 1
    assert models.extract("What is being checked?", ["Offline inference is being checked."])
