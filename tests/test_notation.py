"""Original native-PDF fraction layout and non-vetoing notation notices."""

import base64
import json
from pathlib import Path

from modulemate.assistant import answer_question
from modulemate.evidence import source_notation_limits
from modulemate.ingest import parse_upload
from modulemate.search import Passage


def test_stacked_pdf_notation_reaches_the_writer_without_hiding_source_text(store, course):
    fixture = json.loads(
        (Path(__file__).resolve().parents[1] / "evaluation/notation_transfer.json").read_text()
    )
    source = fixture["documents"][0]
    parsed = parse_upload(
        source["name"], base64.b64decode(source["base64"], validate=True), isolated=False
    )
    assert "r + s\n3" in parsed.text
    document, _ = store.add_document(course, parsed)

    class NotationAware:
        label = "scripted"
        supports_plain_text = True

        def complete(self, system, payload):
            if "answer_to_label" in payload:
                return '{"basis":"materials","citations":[{"evidence_id":1}]}'
            assert payload["context_complete"]
            assert payload["source_notation_limits"][0]["source_id"] == 1
            assert any(r.get("notation") for r in payload["source_reading_limits"])
            assert "r + s\n3" in payload["passages"][0]["text"]
            return "[ANSWER] The score combines two readings into one scaled index. Its fraction grouping should be checked against the original layout."

    answer = answer_question(
        store,
        course,
        "Explain the main concept of this presentation.",
        document_ids=[document],
        use_ai=True,
        backend=NotationAware(),
    )
    assert answer.kind == "generated" and answer.metadata["context_complete"]
    assert answer.passages and "r + s\n3" in answer.passages[0].text
    assert any(r.get("notation") for r in answer.metadata["source_reading_limits"])


def test_ordinary_prose_code_and_inline_equations_do_not_receive_a_stack_notice():
    for name, text in (
        ("Code.py", "r + s\n3"),
        ("Normal.pdf", "S = (r + s)/3"),
        ("Normal.pdf", "Day-night\n3"),
        ("Normal.pdf", "A source title\n3"),
    ):
        assert source_notation_limits([Passage(1, "one", name, "part 1", text)]) == []
    footer = Passage(1, "one", "Normal.pdf", "page 3", "S = 4 + 2\n3")
    assert source_notation_limits([footer]) == []
    fraction = Passage(2, "one", "Normal.pdf", "page 4", "r + s\n3")
    assert len(source_notation_limits([fraction, fraction])) == 1
