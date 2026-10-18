"""Evidence learning: provenance, scope, recovery and fallible model boundaries."""

import copy
import hashlib
import json

import numpy as np
import pytest

from modulemate.adaptive import adaptive_context, assess_evidence, question_parts
from modulemate.context import Context, build_context
from modulemate.db import UserError
from modulemate.dialogue import plan_turn
from modulemate.ingest import Chunk, ParsedDocument
from modulemate.learning import FEATURES, EvidenceModel, features, load_evidence_model
from modulemate.retrieval import RetrievalIndex, learned_order, retrieve
from modulemate.search import Passage
from modulemate.support import check_claims, factual_claims, focused_premise
from scripts.train_evidence import DATA, ROOT, examples, load_data, train


def artifact():
    return json.loads((ROOT / "modulemate/models/evidence-lexical.json").read_text())


def test_training_sources_and_artifact_are_reproducible():
    data = load_data()
    original = artifact()
    assert original["dataset_sha256"] == hashlib.sha256(DATA.read_bytes()).hexdigest()
    trained = train(data)
    assert np.allclose(trained["heads"]["rank"]["weights"], original["heads"]["rank"]["weights"])
    for name in trained["calibration"]:
        assert np.allclose(trained["calibration"][name], original["calibration"][name], atol=1e-7)
    fit, calibration, test = (
        {d["id"] for d in data["documents"] if d["split"] == s}
        for s in ("train", "calibration", "test")
    )
    assert not (fit & calibration or fit & test or calibration & test)
    assert set(original["fit_documents"]) == fit
    assert set(original["calibration_documents"]) == calibration
    for split, expected in (("train", fit), ("calibration", calibration), ("test", test)):
        assert all(row["document_id"] in expected for row in examples(data, split))


@pytest.mark.parametrize("mutation", ["nan", "shape", "feature", "negative_calibration"])
def test_corrupt_numeric_models_are_rejected(mutation):
    item = artifact()
    if mutation == "nan":
        item["heads"]["rank"]["weights"][0] = float("nan")
    elif mutation == "shape":
        item["heads"]["answerability"]["weights"].pop()
    elif mutation == "feature":
        item["features"][0] = "unrecognised"
    else:
        item["calibration"]["relevance"][0] = -1
    with pytest.raises(ValueError):
        EvidenceModel(item)


def test_finite_features_and_deterministic_predictions():
    model = load_evidence_model()
    for question, source in [
        ("", ""),
        ("???", "..."),
        ("Where is it?", "A cup is in the room."),
        ("How many?", "1 " * 20000),
    ]:
        row = features(question, source, cosine=float("nan"))
        assert row.shape == (len(FEATURES),) and np.isfinite(row).all()
        a, b = model.predict(question, source), model.predict(question, source)
        assert a == b
        assert 0 <= a.relevance <= 1 and 0 <= a.answerability <= 1


def test_unsupported_encoder_falls_back_to_lexical_artifact():
    assert load_evidence_model("not-the-trained-encoder").encoder is None


def test_rank_vote_does_not_remove_candidates():
    passages = [
        Passage(1, "x", "Workshop", "place", "The workshop takes place in Blue Room."),
        Passage(2, "x", "Workshop", "cost", "A seat costs 9 tokens."),
    ]
    ranked, scores, model = learned_order("Where is the workshop?", passages, {1: 0.03, 2: 0.03})
    assert {p.id for p in ranked} == {1, 2}
    assert model and set(scores) == {1, 2}


def test_learned_retrieval_preserves_explicit_source_scope(store, populated):
    course, doc = populated
    other = store.create_course("Other")
    with pytest.raises(UserError):
        retrieve(store, other, "spacing", [doc], strategy="learned")
    result = retrieve(store, course, "spacing", [doc], strategy="learned")
    assert result.passages and all(p.document_id == doc for p in result.passages)
    store.delete_document(doc)
    assert not retrieve(store, course, "spacing", strategy="learned").passages


def test_parts_preserve_original_and_do_not_split_noun_phrases():
    question = "Where is the meeting and who organised it?"
    parts = question_parts(question)
    assert parts[0] == question and "who organised it" in parts
    assert question_parts("Compare salt and pepper") == ["Compare salt and pepper"]
    code = 'Rewrite "if a and b" in plain English'
    assert question_parts(code) == [code]


def test_whole_short_document_uses_source_order(store, populated):
    course, doc = populated
    plan = plan_turn("What is spacing?", "Explain", [])
    ctx = build_context(store, course, plan.question, plan, [], [doc], policy="adaptive")
    assert ctx.complete
    assert ctx.method == "full short selection"
    assert [p.id for p in ctx.passages] == sorted(p.id for p in ctx.passages)


def test_soft_document_focus_does_not_claim_library_complete(store, course):
    for name, text in [
        ("Fruit one.md", "Apple tokens are red."),
        ("Fruit two.md", "Orange tokens are blue."),
    ]:
        store.add_document(
            course,
            ParsedDocument(
                name, hashlib.sha256(text.encode()).hexdigest(), text, [Chunk("note", text)], []
            ),
        )
    question = "What colour are Apple tokens?"
    ctx = build_context(store, course, question, plan_turn(question, "Explain", []), [])
    assert ctx.total_passages == 2 and len(ctx.passages) == 1
    assert not ctx.complete


def test_full_document_baseline_keeps_complete_selection(store, populated):
    course, doc = populated
    question = "What is spacing?"
    ctx = build_context(
        store, course, question, plan_turn(question, "Explain", []), [], [doc], policy="full"
    )
    assert ctx.complete and ctx.method == "full selected documents"


def test_empty_assessment_never_claims_support():
    result = assess_evidence("Who organised this?", [])
    assert result["advisory"] and result["parts"][0]["estimate"] == 0


def test_recovery_is_bounded_and_does_not_widen_explicit_scope(monkeypatch):
    from modulemate import adaptive

    calls = []
    passage = Passage(1, "allowed", "Allowed", "section", "A relevant statement.")

    def build(*args, **kwargs):
        calls.append((args, kwargs))
        return Context(passages=[passage], total_passages=4)

    monkeypatch.setattr(adaptive, "build_context", build)
    monkeypatch.setattr(
        adaptive,
        "assess_evidence",
        lambda *args, **kwargs: {"model": "test", "parts": [{"question": "q", "estimate": 0.1}]},
    )
    plan = plan_turn("Find the answer", "Explain", [])
    result = adaptive_context(None, "course", plan.question, plan, [], ["allowed"])
    assert len(calls) == 2 and calls[1][1]["recover"]
    assert all(call[0][5] == ["allowed"] for call in calls)
    assert result.evidence["recovery_attempted"] and not result.evidence["recovery_applied"]


def test_recovery_cannot_sacrifice_another_question_part(monkeypatch):
    from modulemate import adaptive

    contexts = [
        Context(passages=[Passage(i, "d", "d", "s", str(i))], total_passages=5) for i in (1, 2)
    ]
    reports = [
        {
            "model": "test",
            "parts": [{"question": "a", "estimate": a}, {"question": "b", "estimate": b}],
        }
        for a, b in ((0.9, 0.1), (0.2, 0.8))
    ]
    monkeypatch.setattr(adaptive, "build_context", lambda *a, **k: contexts.pop(0))
    monkeypatch.setattr(adaptive, "assess_evidence", lambda *a, **k: reports.pop(0))
    plan = plan_turn("Where and who?", "Explain", [])
    result = adaptive_context(None, "course", plan.question, plan, [])
    assert result.passages[0].id == 1
    assert not result.evidence["recovery_applied"]


class StubSupport:
    identity = "test-support"

    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, pairs):
        assert len(pairs) == len(self.predictions)
        return copy.deepcopy(self.predictions)


def test_claim_segmentation_excludes_code_questions_and_illustrations():
    text = "The workshop starts in North Hall. [1]\n\n```python\nprint('The workshop costs 10.')\n```\n\nFor example, imagine a second event.\n\nWhat would you bring?"
    claims = factual_claims(text, [1])
    assert len(claims) == 1 and claims[0]["source_ids"] == [1]


def test_supported_source_can_outweigh_unrelated_citation():
    passages = [Passage(i, "d", "d", "s", "The meeting starts at 10:00.") for i in (1, 2)]
    model = StubSupport(
        [
            {"entailment": 0.005, "contradiction": 0.005, "neutral": 0.99},
            {"entailment": 0.99, "contradiction": 0.005, "neutral": 0.005},
            {"entailment": 0.99, "contradiction": 0.005, "neutral": 0.005},
        ]
    )
    result = check_claims(
        "The meeting starts at ten in the morning. [1] [2]", passages, [1, 2], model=model
    )
    assert result["available"] and not result["flags"]
    assert result["claims"][0]["best_source_id"] == 2


def test_contradiction_flag_is_a_prediction_not_a_changed_answer():
    model = StubSupport([{"entailment": 0.005, "contradiction": 0.99, "neutral": 0.005}])
    passage = Passage(1, "d", "d", "s", "The meeting starts at 10:00.")
    result = check_claims(
        "The meeting begins at eleven in the morning. [1]", [passage], [1], model=model
    )
    assert result["flags"][0]["label"] == "contradiction"
    assert result["claims"][0]["claim"].startswith("The meeting begins")


def test_unavailable_support_checker_does_not_invent_a_verdict():
    assert check_claims("A sufficiently long claim for checking.", [], []) == {
        "available": False,
        "claims": [],
        "flags": [],
    }


def test_long_premise_keeps_relevant_context_and_budget():
    source = (
        "Unrelated material about flowers. " * 200
        + "The telescope uses a red filter. It is supplied by staff. "
    )
    focused = focused_premise(source, "Staff provide the telescope filter.")
    assert "supplied by staff" in focused and len(focused) <= 1600


def test_single_question_is_not_scored_twice_for_trailing_punctuation():
    assert question_parts("Where does the meeting take place?") == [
        "Where does the meeting take place?"
    ]


def test_short_library_keeps_semantically_linked_document(store, course):
    for name, text in [
        ("Orientation.md", "The Stride session uses the rules in Foot protection."),
        ("Equipment.md", "Foot protection: Closed shoes are compulsory. Sandals are prohibited."),
    ]:
        store.add_document(
            course,
            ParsedDocument(
                name, hashlib.sha256(text.encode()).hexdigest(), text, [Chunk("section", text)], []
            ),
        )
    question = "What footwear do I need for the Stride session?"
    plan = plan_turn(question, "Explain", [])
    old = build_context(store, course, question, plan, [])
    new = build_context(store, course, question, plan, [], policy="adaptive")
    assert len({p.document_id for p in old.passages}) == 1
    assert len({p.document_id for p in new.passages}) == 2 and new.complete
    assert any("Closed shoes" in p.text for p in new.passages)


def test_named_topic_does_not_pull_in_an_unrelated_reference_graph(store, course):
    from modulemate.ingest import parse_upload

    fixture = json.loads((ROOT / "evaluation/evidence_transfer.json").read_text())
    for doc in fixture["documents"]:
        store.add_document(course, parse_upload(doc["name"], doc["text"].encode(), isolated=False))
    question = "I need the sculpture's weight, but I have no measurements. What can these directions help me do?"
    context = build_context(
        store, course, question, plan_turn(question, "Explain", []), [], policy="adaptive"
    )
    assert {p.name for p in context.passages} == {"Paper sculpture.md"}
    assert not context.links and not context.complete
    assert any("clips" in p.text for p in context.passages)
    assert any("no dimensions" in p.text for p in context.passages)
    comparison = "Compare the sculpture and the Stride session."
    both = build_context(
        store, course, comparison, plan_turn(comparison, "Auto", []), [], policy="adaptive"
    )
    assert {"Paper sculpture.md", "Orientation.md"} <= {p.name for p in both.passages}


def test_title_focus_still_follows_an_allowed_reference(store, course):
    from modulemate.ingest import parse_upload

    for name, text in (
        (
            "Timber sculpture.md",
            "# Assembly\nSee Measurements.md, Height, for the sculpture dimensions.",
        ),
        ("Measurements.md", "# Height\nThe assembled timber sculpture is 30 centimetres tall."),
        ("Other notes.md", "# Measurements\nThe pool's depth is two metres."),
    ):
        store.add_document(course, parse_upload(name, text.encode(), isolated=False))
    question = "How tall is the sculpture?"
    context = build_context(
        store, course, question, plan_turn(question, "Explain", []), [], policy="adaptive"
    )
    assert {p.name for p in context.passages} == {"Timber sculpture.md", "Measurements.md"}
    assert [link["name"] for link in context.links] == ["Measurements.md"]
    compound = "How tall is the sculpture?\nWhat is the pool depth?"
    both = build_context(
        store, course, compound, plan_turn(compound, "Explain", []), [], policy="adaptive"
    )
    assert {"Timber sculpture.md", "Other notes.md"} <= {p.name for p in both.passages}


class CountingEncoder:
    key = "counting-encoder"
    dimensions = 3

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array([[1, len(t) % 7 + 1, 1] for t in texts], dtype=np.float32)


def test_request_index_batches_queries_and_reuses_vectors(store, populated, monkeypatch):
    from modulemate import retrieval

    course, doc = populated
    engine = CountingEncoder()
    calls = []
    original = retrieval._vectors

    def vectors(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(retrieval, "_vectors", vectors)
    queries = ["spacing", "retrieval practice", "interleaving"]
    index = RetrievalIndex(store, course, [doc], engine, queries)
    first = retrieve(store, course, queries[0], [doc], engine=engine, _index=index)
    assert first.passages
    for query in queries[1:]:
        assert retrieve(store, course, query, [doc], engine=engine, _index=index).passages
    assert len(calls) == 1
    assert any(set(batch) == set(queries) for batch in engine.calls)
    assert len(engine.calls) == 2  # one source batch, one query batch


def test_request_index_refuses_reuse_in_a_different_scope(store, populated):
    course, doc = populated
    engine = CountingEncoder()
    index = RetrievalIndex(store, course, [doc], engine)
    with pytest.raises(ValueError, match="scopes"):
        retrieve(store, course, "spacing", engine=engine, _index=index)
    assert not engine.calls
    engine.key = "a-different-model"
    with pytest.raises(ValueError, match="scopes"):
        retrieve(store, course, "spacing", [doc], engine=engine, _index=index)


def test_request_index_never_returns_a_deleted_snapshot(store, populated):
    course, doc = populated
    engine = CountingEncoder()
    index = RetrievalIndex(store, course, None, engine)
    assert retrieve(store, course, "spacing", engine=engine, _index=index).passages
    store.delete_document(doc)
    assert not retrieve(store, course, "spacing", engine=engine, _index=index).passages


def test_reader_signals_reject_foreign_spans_and_nonfinite_logits():
    from modulemate.learning import reader_signals
    from modulemate.reader import Span

    class Reader:
        def extract(self, question, texts):
            return [
                Span(1, "real", 0, 4, 1.0),
                Span(1, "fabricated", 0, 4, 2.0),
                Span(1, "real", 0, 4, float("nan")),
                Span(2, "real", 0, 4, 10.0),
            ]

    assert reader_signals(Reader(), "What?", ["real source"]) == {
        0: {"margin": 1.0, "text": "real", "start": 0, "end": 4}
    }


def test_support_source_numbers_follow_compact_answer_citations():
    from modulemate.support import remap_support

    report = {
        "available": True,
        "claims": [
            {
                "claim": "A statement",
                "source_ids": [3, 7],
                "best_source_id": 7,
                "checked_source_ids": [3, 7],
                "flagged": True,
            }
        ],
    }
    mapped = remap_support(report, {3: 1, 7: 2})
    assert mapped["flags"][0]["source_ids"] == [1, 2]
    assert mapped["claims"][0]["best_source_id"] == 2


def test_explicit_reference_follows_uploaded_sections_within_allowed_scope(store, course):
    from modulemate.context import referenced_sections

    origin = Passage(
        1,
        "intro",
        "Orientation.md",
        "overview",
        "Read the Foot protection section of Equipment.md before the trip.",
    )
    equipment = Passage(
        2, "kit", "Equipment.md", "part 1 › Foot protection", "Closed shoes are compulsory."
    )
    groups = {"intro": [origin], "kit": [equipment]}
    found, links = referenced_sections([origin], groups, store, course, "What must I wear?")
    assert found == [equipment] and links[0]["to_document"] == "kit"
    restricted, links = referenced_sections(
        [origin], {"intro": [origin]}, store, course, "What must I wear?"
    )
    assert restricted == [] and links == []


def test_document_reference_cannot_open_a_path_or_an_ambiguous_filename(store, course):
    from modulemate.context import referenced_sections

    origin = Passage(
        1, "intro", "Orientation.md", "overview", "Read /private/Secret.md and Equipment.md."
    )
    one = Passage(2, "one", "Equipment.md", "section", "First version.")
    two = Passage(3, "two", "Equipment.md", "section", "Second version.")
    found, links = referenced_sections(
        [origin], {"intro": [origin], "one": [one], "two": [two]}, store, course, "What applies?"
    )
    assert not found and not links


def test_document_references_have_one_hop_and_two_target_budget(store, course):
    from modulemate.context import referenced_sections

    origin = Passage(1, "a", "A.md", "a", "See B.md, C.md and D.md. B section C section D section")
    b = Passage(2, "b", "B.md", "part 1 › B section", "Read C.md. C section")
    c = Passage(3, "c", "C.md", "part 1 › C section", "Closed shoes are compulsory.")
    d = Passage(4, "d", "D.md", "part 1 › D section", "A third reference.")
    found, links = referenced_sections(
        [origin], {"a": [origin], "b": [b], "c": [c], "d": [d]}, store, course, "What applies?"
    )
    assert {p.document_id for p in found} == {"b", "c"} and len(links) == 2
    found, links = referenced_sections(
        [Passage(1, "a", "A.md", "a", "See B.md. B section")],
        {"a": [origin], "b": [b], "c": [c]},
        store,
        course,
        "What applies?",
    )
    assert found == [b] and len(links) == 1


def test_training_rejects_cross_partition_copies_and_false_provenance(tmp_path):
    data = load_data()
    modified = copy.deepcopy(data)
    modified["documents"][-1]["passages"][0]["text"] = modified["documents"][0]["passages"][0][
        "text"
    ]
    path = tmp_path / "bad-data.json"
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="leak"):
        load_data(path)
    with pytest.raises(ValueError, match="recorded source"):
        train(modified)


def test_malformed_secondary_support_scores_do_not_erase_a_primary_answer():
    model = StubSupport([{"entailment": float("nan"), "contradiction": 0.5, "neutral": 0.5}])
    p = Passage(1, "d", "d", "s", "The meeting starts at 10:00.")
    report = check_claims("The meeting starts at ten in the morning. [1]", [p], [1], model=model)
    assert not report["available"] and not report["flags"]


def test_writer_view_omits_only_duplicated_leading_headings():
    from modulemate.evidence import source_excerpts, writer_text

    original = "# Loop sculpture\n## Limits\nThe note gives no material masses."
    p = Passage(1, "d", "Sculpture.md", "part 1 · Loop sculpture › Limits", original)
    assert writer_text(p) == "The note gives no material masses."
    assert p.text == original
    _, bindings = source_excerpts([p])
    assert bindings[1]["quote"] in original
    code = Passage(
        2, "d", "Code.md", "part 1 › Code", "```python\n# Keep this comment\nvalues = [1, 2]\n```"
    )
    assert writer_text(code) == code.text
    unrelated = Passage(3, "d", "Note.txt", "page 1", original)
    assert writer_text(unrelated) == original


def test_citation_hints_find_omitted_details_without_inventing_sources():
    from modulemate.evidence import citation_coverage_hints, source_excerpts

    ps = [
        Passage(
            1,
            "d",
            "Sculpture.md",
            "Assembly",
            "Fold paper strips into loops and join the loops using clips.",
        ),
        Passage(
            2,
            "d",
            "Sculpture.md",
            "Limits",
            "The directions give no material masses or measurements.",
        ),
    ]
    _, bindings = source_excerpts(ps)
    text = "The directions describe paper loops joined using clips, but give no material masses."
    hints = citation_coverage_hints(text, ps, [2], bindings)
    assert hints and hints[0]["source_id"] == 1
    assert {"loop", "clip"} <= set(hints[0]["details"])
    assert citation_coverage_hints(text, ps, [1, 2], bindings) == []
    assert citation_coverage_hints("The author is unknown.", ps, [], bindings) == []
    assert citation_coverage_hints("```text\npaper loops using clips\n```", ps, [2], bindings) == []
