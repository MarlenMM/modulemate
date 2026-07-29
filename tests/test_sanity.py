"""Sanity tests for the deterministic parts of the pipeline.

Deliberately narrow. The gold-set harness is the real testing deliverable;
these cover the pure functions where a silent regression would corrupt every
downstream metric — chunking, citation parsing, the refusal decision and the
metric helpers themselves. Nothing here calls the generation API.

Run:  .venv/bin/python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from generate import build_context_block, parse_citations  # noqa: E402
from ingest import chunk_lines, chunk_text, is_informative  # noqa: E402
from retrieve import RetrievalResult, Retrieved  # noqa: E402


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def test_short_text_is_not_split():
    text = "A decision tree is a supervised learning algorithm."
    assert chunk_text(text, target_words=230, overlap_words=45) == [text]


def test_long_text_splits_with_overlap():
    words = [f"w{i}" for i in range(500)]
    pieces = chunk_text(" ".join(words), target_words=100, overlap_words=20)
    assert len(pieces) > 1
    assert all(len(p.split()) <= 100 for p in pieces)
    # Consecutive pieces must share the overlap window, or context is lost
    # across a split and a fact straddling the boundary becomes unretrievable.
    first_tail = pieces[0].split()[-20:]
    second_head = pieces[1].split()[:20]
    assert first_tail == second_head


def test_chunk_text_covers_every_word():
    words = [f"w{i}" for i in range(313)]
    pieces = chunk_text(" ".join(words), target_words=50, overlap_words=10)
    seen = {w for p in pieces for w in p.split()}
    assert seen == set(words)


def test_chunk_lines_never_splits_mid_line():
    text = "-- first bullet here\n-- second bullet here\n-- third bullet here"
    pieces = chunk_lines(text, target_words=5)
    for piece in pieces:
        for line in piece.splitlines():
            assert line in text.splitlines()


def test_chunk_lines_groups_up_to_budget():
    text = "\n".join(["one two three"] * 6)  # 3 words per line
    pieces = chunk_lines(text, target_words=7)
    assert len(pieces) > 1
    assert all(len(p.split()) <= 7 for p in pieces)


# --------------------------------------------------------------------------
# OCR noise filter
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "100\n75\n50\n25\n0\n10\n-10\n-20",          # chart axis labels
    "16@151*1 | 16@149*1 | 64@47*1",             # layer-shape fragments
    "3 > 4 5 = 7",                               # stray glyphs
])
def test_noise_chunks_rejected(text):
    assert not is_informative(text, min_alpha_ratio=0.55, min_word_tokens=6)


@pytest.mark.parametrize("text", [
    "A convolutional neural network is commonly applied to analyse imagery.",
    "Decision tree induction partitions the training set into disjoint subsets.",
])
def test_real_prose_kept(text):
    assert is_informative(text, min_alpha_ratio=0.55, min_word_tokens=6)


# --------------------------------------------------------------------------
# Citation parsing — the harness scores against this exact bracket
# --------------------------------------------------------------------------

def test_parse_single_citation():
    answer = "The limit is 4000 words.\n[Source: HIT summer course_AI and ML_project_July2026, Requirements]"
    assert parse_citations(answer) == [
        ("HIT summer course_AI and ML_project_July2026", "Requirements")
    ]


def test_parse_two_citations_on_one_line():
    answer = "Text. [Source: AI notes8_deep learning, slide 8] [Source: AI notes8_deep learning, slide 11]"
    assert parse_citations(answer) == [
        ("AI notes8_deep learning", "slide 8"),
        ("AI notes8_deep learning", "slide 11"),
    ]


def test_document_names_containing_spaces_and_underscores_survive():
    answer = "[Source: AI notes5_machine learning_part 1, slide 25]"
    assert parse_citations(answer) == [
        ("AI notes5_machine learning_part 1", "slide 25")
    ]


def test_no_citation_returns_empty():
    assert parse_citations("I could not find that in the materials.") == []


# --------------------------------------------------------------------------
# Refusal gate
# --------------------------------------------------------------------------

def _passage(score: float) -> Retrieved:
    return Retrieved(1, score, "c1", "doc", "slide 1", "text")


def test_gate_refuses_below_threshold():
    r = RetrievalResult("q", [_passage(0.30)], top_score=0.30, threshold=0.35)
    assert not r.is_confident


def test_gate_accepts_at_threshold():
    r = RetrievalResult("q", [_passage(0.35)], top_score=0.35, threshold=0.35)
    assert r.is_confident


def test_gate_refuses_when_nothing_retrieved():
    r = RetrievalResult("q", [], top_score=0.0, threshold=0.35)
    assert not r.is_confident


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

def test_context_block_states_the_exact_citation_to_copy():
    """Each passage carries the literal bracket the model must emit; without
    it, citation correctness cannot be scored."""
    block = build_context_block([
        Retrieved(1, 0.8, "c1", "AI notes8_deep learning", "slide 8", "CNN text")
    ])
    assert "Cite this passage as: [Source: AI notes8_deep learning, slide 8]" in block
    assert "CNN text" in block


# --------------------------------------------------------------------------
# Config contract (NFR5)
# --------------------------------------------------------------------------

def test_tunables_are_present_and_typed():
    cfg = config.load(reload=True)
    assert isinstance(cfg.require("retrieval.k"), int)
    assert 0.0 < cfg.require("retrieval.similarity_threshold") < 1.0
    assert cfg.require("generation.model")
    assert cfg.require("embedding.model_name")


def test_four_intent_labels_configured():
    cfg = config.load(reload=True)
    labels = cfg.require("intents.labels")
    assert len(labels) == 4
    assert "out-of-scope/social" in labels


# --------------------------------------------------------------------------
# Harness metric helpers
# --------------------------------------------------------------------------

def test_hit_at_k_and_fabrication_detection():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
    from harness import fabricated_citations, hit_at_k  # noqa: E402

    class Fake:
        passages = [
            Retrieved(1, 0.8, "c1", "AI notes8_deep learning", "slide 8", "t"),
            Retrieved(2, 0.7, "c2", "AI notes7_neural networks", "slide 9", "t"),
        ]
        citations = [("AI notes8_deep learning", "slide 8"),
                     ("AI notes8_deep learning", "slide 999")]

    item = {"expected_source_doc": "AI notes8_deep learning",
            "expected_slide_or_section": "slide 8"}
    assert hit_at_k(Fake(), item, k=3)

    item_miss = {"expected_source_doc": "AI notes8_deep learning",
                 "expected_slide_or_section": "slide 42"}
    assert not hit_at_k(Fake(), item_miss, k=3)

    known = {("ai notes8_deep learning", "slide 8"),
             ("ai notes7_neural networks", "slide 9")}
    # slide 999 exists in no document, so it is a fabricated provenance claim.
    assert fabricated_citations(Fake(), known) == [
        ("AI notes8_deep learning", "slide 999")
    ]
