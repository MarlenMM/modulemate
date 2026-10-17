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


def test_learned_context_assessment_uses_real_reader_and_bound_spans(engine):
    from modulemate.adaptive import assess_evidence
    from modulemate.search import Passage

    passages = [
        Passage(
            1,
            "a",
            "Orientation.md",
            "part 1 › Session",
            "The Kestrel session follows the Foot protection rules.",
        ),
        Passage(
            2,
            "b",
            "Equipment.md",
            "part 1 › Foot protection",
            "Closed shoes are compulsory. Sandals are prohibited.",
        ),
    ]
    report = assess_evidence(
        "What footwear do I need for the Kestrel session?", passages, engine=engine
    )
    assert report["model"] == "modulemate-evidence-reader-v2"
    assert report["advisory"] and report["parts"]
    for part in report["parts"]:
        assert 0 <= part["estimate"] <= 1
        for candidate in part["candidate_spans"]:
            p = next(p for p in passages if p.id == candidate["passage_id"])
            assert p.text[candidate["start"] : candidate["end"]] == candidate["text"]


def test_batched_request_index_matches_individual_real_retrieval(store, course, engine):
    from modulemate.retrieval import RetrievalIndex

    load_fixture(store, course)
    queries = [
        "How can revision be spread out over time?",
        "Which container gives back the newest item first?",
    ]
    index = RetrievalIndex(store, course, None, engine, queries)
    for query in queries:
        normal = retrieve(store, course, query, engine=engine)
        batched = retrieve(store, course, query, engine=engine, _index=index)
        assert [p.id for p in normal.passages[:3]] == [p.id for p in batched.passages[:3]]


def test_optional_support_model_is_offline_and_label_order_is_correct(engine, monkeypatch):
    import socket

    from modulemate.local_models import model_dir
    from modulemate.support import SupportModel, support_status

    monkeypatch.setenv("MODULEMATE_SUPPORT", "auto")
    if support_status() != "ready":
        pytest.skip("The optional support model has not been prepared")

    def deny(*args, **kwargs):
        raise AssertionError("Unexpected network connection")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    model = SupportModel(model_dir())
    rows = model.predict(
        [
            ("A red cup is on the desk.", "There is a cup on the desk."),
            ("The workshop has ten places.", "The workshop has twenty places."),
        ]
    )
    assert max(rows[0], key=rows[0].get) == "entailment"
    assert max(rows[1], key=rows[1].get) == "contradiction"
