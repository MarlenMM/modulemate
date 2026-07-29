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
from ingest import (  # noqa: E402
    chunk_lines, chunk_text, is_english_enough, is_informative, latin_ratio,
)
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
# Language filter — the embedding model is English-only
# --------------------------------------------------------------------------

def test_english_text_passes_language_filter():
    assert is_english_enough("The deadline for the report is Friday.", 0.60)


def test_cjk_dominant_text_rejected():
    assert not is_english_enough("请进群的留学生同学请将昵称改成姓名", 0.60)


def test_bilingual_chunk_keeps_its_english_half():
    """Chat exports mix languages line by line; a mostly-English message with
    a short Chinese fragment must survive."""
    text = "Dear students, the deadline is Friday 31 July. Please submit on time. 谢谢"
    assert is_english_enough(text, 0.60)


def test_latin_ratio_of_pure_scripts():
    assert latin_ratio("hello world") == 1.0
    assert latin_ratio("你好世界") == 0.0
    assert latin_ratio("12345 -+/") == 1.0  # no letters at all -> not penalised


def test_real_code_cells_survive_the_relaxed_threshold():
    """Notebook code cells in the built index must pass the code threshold.

    Checked against the real index rather than a synthetic string: the point of
    the separate `min_alpha_ratio_code` is that actual lab notebooks survive
    ingestion, and those cells are what answer "which function did the lab
    use?". A regression here would silently empty the code-help evidence.
    """
    from ingest import is_informative as _inf

    root = Path(__file__).resolve().parent.parent
    chunks_path = root / "index_store" / "chunks.json"
    if not chunks_path.exists():
        pytest.skip("index not built yet")

    import json
    with open(chunks_path, encoding="utf-8") as fh:
        chunks = json.load(fh)
    code_cells = [
        c["text"] for c in chunks
        if c["doc_type"] == "ipynb" and c["text"].lstrip().startswith("Code:")
    ]
    if not code_cells:
        pytest.skip("no notebook code cells in the index")

    cfg = config.load(reload=True)
    code_ratio = cfg.require("index.min_alpha_ratio_code")
    min_tokens = cfg.require("index.min_word_tokens")
    survivors = sum(1 for t in code_cells if _inf(t, code_ratio, min_tokens))
    assert survivors == len(code_cells), (
        f"{len(code_cells) - survivors} indexed code cells would now be "
        f"filtered out by min_alpha_ratio_code={code_ratio}"
    )


def test_prose_threshold_is_stricter_than_code_threshold():
    """The two thresholds must not collapse into one, or the distinction the
    ingestion pipeline draws between prose and code is meaningless."""
    cfg = config.load(reload=True)
    assert cfg.require("index.min_alpha_ratio_code") < \
        cfg.require("index.min_alpha_ratio")


# --------------------------------------------------------------------------
# Source authority prior
# --------------------------------------------------------------------------

class _FakeRetriever:
    """Exercises Retriever.authority_of without loading the FAISS index."""

    from retrieve import Retriever
    authority_of = Retriever.authority_of

    def __init__(self, weights, default=1.0):
        self.authority_weights = weights
        self.default_authority = default


def test_chat_sources_are_demoted():
    r = _FakeRetriever({"chats__": 0.85})
    assert r.authority_of("chats__HITSummerSchool2026") == 0.85
    assert r.authority_of("Lectures Week 2__AI notes8_deep learning") == 1.0


def test_authority_reorders_a_near_tie():
    """The measured case: a chat chunk at 0.567 outranked the assessment
    brief at 0.552 for 'what is the word limit?'. Weighting must flip it."""
    r = _FakeRetriever({"chats__": 0.85})
    chat = 0.567 * r.authority_of("chats__Code Review")
    brief = 0.552 * r.authority_of("Lectures Week 2__HIT summer course")
    assert brief > chat


def test_authority_does_not_demote_unmatched_prefixes():
    r = _FakeRetriever({"chats__": 0.85}, default=1.0)
    assert r.authority_of("Code Review Project__lab_guide") == 1.0


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


# --------------------------------------------------------------------------
# Corpus manifest — the include/exclude contract the user relies on
# --------------------------------------------------------------------------

def test_manifest_excludes_the_students_own_work():
    """The corpus must contain course-issued material only. These paths are
    the student's own project, plan, report template and answered assignment."""
    import yaml
    with open(Path(__file__).resolve().parent.parent / "corpus_sources.yaml",
              encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    excludes = " ".join(spec["exclude_globs"])
    for owned in ("smart-chatbot", "code-review-ai", "Code_Review_Project_Plan",
                  "Experiment_Report__Eng_", "ANSWERED",
                  "lab1.pdf", "lab2.pdf", "lab3.pdf"):
        assert owned in excludes, f"{owned} is not excluded by the manifest"


def test_authorship_guard_identifiers_are_configured():
    import yaml
    with open(Path(__file__).resolve().parent.parent / "corpus_sources.yaml",
              encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    assert spec.get("student_identifiers"), \
        "authorship guard needs identifiers or it silently passes everything"


def test_no_staged_file_is_the_students_own_work():
    """End-to-end check against what is actually staged in corpus/raw."""
    import yaml
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from collect_corpus import authored_by_student

    root = Path(__file__).resolve().parent.parent
    with open(root / "corpus_sources.yaml", encoding="utf-8") as fh:
        identifiers = yaml.safe_load(fh).get("student_identifiers", [])

    raw = root / "corpus" / "raw"
    if not raw.is_dir():
        pytest.skip("corpus not collected yet")
    offenders = [
        p.name for p in sorted(raw.iterdir())
        if p.is_file() and authored_by_student(p, identifiers)
    ]
    assert not offenders, f"student-authored files staged: {offenders}"
