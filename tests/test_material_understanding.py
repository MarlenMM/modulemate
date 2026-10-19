"""Evidence recall and synthesis regressions, independent of model phrasing."""

import json

import pytest

from modulemate.assistant import answer_question
from modulemate.context import broad_request, build_context
from modulemate.conversation import parse_response
from modulemate.dialogue import plan_turn
from modulemate.evidence import source_excerpts
from modulemate.ingest import parse_upload
from modulemate.search import Passage


def test_supported_conditional_facts_do_not_gain_an_implicit_fifty_word_limit(store, course):
    source = (
        "Day teams bring a paper map and a blue pencil, record the gate number before entry, "
        "and meet the guide at the east gate. Night teams bring a torch and a red pencil, "
        "record the gate number and arrival time before entry, and meet the guide at the west gate. "
        "Both teams wait outside until the guide arrives."
    )
    store.add_document(course, parse_upload("Visit.txt", source.encode(), isolated=False))

    class ConditionalFacts:
        label = "scripted"
        supports_plain_text = True
        calls = 0

        def complete(self, system, payload):
            self.calls += 1
            if "answer_to_label" in payload:
                return '{"basis":"mixed","citations":[{"evidence_id":1}]}'
            return "[ANSWER] " + source

    backend = ConditionalFacts()
    answer = answer_question(
        store,
        course,
        "Are the conditions the same for day and night teams, and what must each team bring, record and do before entry?",
        use_ai=True,
        backend=backend,
    )
    assert len(source.split()) > 50
    assert answer.kind == "generated" and source in answer.text
    assert answer.passages and backend.calls == 2


def test_source_topic_words_are_not_a_verdict_of_outside_information():
    from modulemate.conversation import check_fact_scope

    source = "The Example Calculation workshop starts at 09:10 in the north room."
    passage = Passage(1, "one", "Workshop.txt", "part 1", source)
    check_fact_scope(source, "Is the Example Calculation workshop in the north room?")
    parsed = parse_response(
        json.dumps(
            {
                "status": "answered",
                "basis": "materials",
                "answer": source,
                "citations": [{"source_id": 1, "quote": source}],
            }
        ),
        [passage],
    )
    assert parsed.basis == "materials" and parsed.cited_ids == [1]


def test_attributed_quotations_cannot_copy_an_assistant_paraphrase_as_source_words():
    from modulemate.conversation import unsupported_attributed_quotes

    source = Passage(1, "one", "Overview.txt", "part 1", "No actual design outcome is reported.")
    assert unsupported_attributed_quotes(
        'The available text states that "this check guarantees success".', [source]
    ) == ["this check guarantees success"]
    assert unsupported_attributed_quotes(
        'According to the closing principle, it "guarantees success".', [source]
    ) == ["guarantees success"]
    for legitimate in (
        "The text states: “No actual design outcome is reported.”",
        'The text states "No actual design outcome is reported," so no measurement is given.',
        'The text does not say "this check guarantees success".',
        'Interpretation: the process is a "second look" before deciding.',
        'Example: a document says "try a second route" before the imaginary traveler leaves.',
        '```text\nThe source states "this check guarantees success".\n```',
        'The code literal is `The source says "this check guarantees success"`.',
    ):
        assert unsupported_attributed_quotes(legitimate, [source]) == []


def test_a_false_verbatim_quote_repairs_the_prose_without_refusing_its_supported_idea(
    store, course
):
    text = "The method connects observing existing features with reviewing proposed changes."
    document, _ = store.add_document(
        course, parse_upload("Overview.txt", text.encode(), isolated=False)
    )

    class RepairQuote:
        label = "scripted"
        supports_plain_text = True
        drafts = 0

        def complete(self, system, payload):
            if "answer_to_label" in payload:
                return '{"basis":"materials","citations":[{"evidence_id":1}]}'
            self.drafts += 1
            if self.drafts == 1:
                return '[ANSWER] The text states that "observing guarantees a successful change".'
            assert payload["unmatched_source_quotations"] == [
                "observing guarantees a successful change"
            ]
            assert payload["recent_conversation"] == []
            assert "observing guarantees" in payload["quoted_answer_to_correct"]
            return "[ANSWER] The method connects observations of existing features with reviewing proposed changes."

    backend = RepairQuote()
    answer = answer_question(
        store,
        course,
        "Explain the main concept.",
        document_ids=[document],
        use_ai=True,
        backend=backend,
    )
    assert answer.kind == "generated" and backend.drafts == 2
    assert "reviewing proposed changes" in answer.text and "guarantees" not in answer.text
    assert answer.passages and answer.metadata["repaired"]

    class RepeatsQuote(RepairQuote):
        def complete(self, system, payload):
            if "answer_to_label" in payload:
                return super().complete(system, payload)
            return '[ANSWER] The method connects observation with review. The text states "this check guarantees success".'

    advisory = answer_question(
        store,
        course,
        "Explain the main concept.",
        document_ids=[document],
        use_ai=True,
        backend=RepeatsQuote(),
    )
    assert advisory.kind == "generated" and advisory.passages
    assert "Quoted wording needs review" in advisory.notice
    assert any(
        item.get("type") == "unmatched_attributed_quote"
        for item in advisory.metadata["citation_review"]
    )


def test_quote_correction_keeps_the_missing_result_when_the_writer_changes_its_marker(
    store, course
):
    store.add_document(
        course,
        parse_upload(
            "Method.txt", b"The process supplies checks. No result is reported.", isolated=False
        ),
    )

    class GapCorrection:
        label = "scripted"
        supports_plain_text = True
        drafts = 0

        def complete(self, system, payload):
            if "answer_to_label" in payload:
                return '{"basis":"materials","citations":[{"evidence_id":1}]}'
            self.drafts += 1
            if self.drafts == 1:
                return '[MISSING] The text states "the checks do not guarantee a result". No measured result is given.'
            return '[ANSWER] No measured result is given. The text states "No result is reported."'

    result = answer_question(
        store,
        course,
        "What improvement was measured?",
        use_ai=True,
        backend=GapCorrection(),
    )
    assert result.kind == "insufficient" and "No measured result is given" in result.text
    assert result.passages and result.metadata["repaired"]


def test_a_fresh_material_question_does_not_reuse_an_old_claim_but_its_followup_keeps_context(
    store, course
):
    from modulemate.retrieval import scoped_passages

    store.add_document(
        course,
        parse_upload(
            "Workflow.txt", b"Observe the site, then review the proposed change.", isolated=False
        ),
    )
    source = scoped_passages(store, course)[0].to_dict()
    store.add_message(
        course,
        "Explain the earlier idea.",
        "OLD_UNSUPPORTED_GUARANTEE",
        [source],
        "Explain",
        {"kind": "generated", "used_context": False, "topic": "Old interpretation"},
    )

    class ObserveHistory:
        label = "scripted"
        supports_plain_text = True
        histories = []

        def complete(self, system, payload):
            if "answer_to_label" in payload:
                return '{"basis":"materials","citations":[{"evidence_id":1}],"topic":"Observation and review"}'
            self.histories.append(payload["recent_conversation"])
            return "[ANSWER] The method connects observations of the site with reviewing a proposed change."

    backend = ObserveHistory()
    first = answer_question(
        store, course, "Explain the main concept.", use_ai=True, backend=backend
    )
    assert first.kind == "generated" and backend.histories[-1] == []
    second = answer_question(store, course, "Why does that matter?", use_ai=True, backend=backend)
    assert second.kind == "generated" and backend.histories[-1]
    assert len(backend.histories[-1]) == 1
    assert backend.histories[-1][0]["question"] == "Explain the main concept."
    assert "OLD_UNSUPPORTED_GUARANTEE" not in json.dumps(backend.histories[-1])


def test_explicit_scope_or_sentence_count_does_not_invent_a_word_limit():
    from modulemate.conversation import check_requested_format, narrow_fact_request

    assert not narrow_fact_request("I just want to understand the time complexity of this code.")
    assert narrow_fact_request("For B2, give only the current word count and what it excludes.")
    long_fact = (
        "The required format includes "
        + ", ".join(f"the entry for item {i}" for i in range(14))
        + "."
    )
    check_requested_format(long_fact, "Give only the current format.")
    check_requested_format(long_fact, "Describe the format in one sentence.")
    with pytest.raises(ValueError, match="requested_format"):
        check_requested_format("Use the form. Add the date.", "Describe it in one sentence.")


def test_empty_trailing_writer_marker_does_not_override_the_answer():
    from modulemate.conversation import parse_draft

    assert parse_draft("[ANSWER]\nInspect all zones before concluding.\n\n[MISSING]") == (
        "Inspect all zones before concluding.",
        "answered",
    )
    literal = 'The literal tag is "[MISSING]".'
    assert parse_draft("[ANSWER] " + literal)[0] == literal
    code = "```text\n[MISSING]\n```"
    assert parse_draft("[ANSWER]\n" + code)[0] == code
    supplied = "first line\n[MISSING]"
    assert parse_draft("[ANSWER]\n" + supplied, preserve_literal=True)[0] == supplied


def test_flashcard_back_keeps_a_complete_short_fact_but_rejects_partial_or_changed_facts():
    from modulemate.assistant import generate_cards

    passage = Passage(1, "one", "Signal card.txt", "part 1 · Reading", "# Reading\npH = 7")

    class ShortCards:
        def __init__(self, back=None):
            self.back = back

        def complete(self, system, payload):
            if "answer_to_check" in payload:
                return '{"supported":true}'
            card = {"front": "What pH is on the card?", "evidence_id": 1}
            if self.back is not None:
                card = {"front": card["front"], "source_id": 1, "back": self.back}
            return json.dumps({"cards": [card]})

    cards = generate_cards([passage], ShortCards())
    assert len(cards) == 1 and cards[0]["back"] == "pH = 7"
    assert cards[0]["document_id"] == "one"
    for invalid in ("7", "pH = 8", ""):
        with pytest.raises(ValueError, match="No card answers matched"):
            generate_cards([passage], ShortCards(invalid))


@pytest.mark.parametrize(
    "literal", ["samples[1] = 7", "https://example.org/field-notes", "<label>42</label>"]
)
def test_card_text_preserves_literal_source_data(literal):
    from modulemate.assistant import generate_cards

    class LiteralCard:
        def complete(self, system, payload):
            if "answer_to_check" in payload:
                return '{"supported":true}'
            return (
                '{"cards":[{"front":"What literal value appears in the source?","evidence_id":1}]}'
            )

    passage = Passage(1, "one", "Literal.txt", "part 1", literal)
    cards = generate_cards([passage], LiteralCard())
    assert len(cards) == 1 and cards[0]["back"] == literal


def test_code_literals_are_data_but_active_remote_markup_still_fails():
    from modulemate.conversation import check_visible_text

    check_visible_text("The location is `https://example.org/field-notes`.")
    check_visible_text("The location is https://example.org/field-notes.")
    check_visible_text('```html\n<img src="https://example.org/picture.png">\n```')
    check_visible_text("The protocol token is `</think>`.")
    for active in (
        "![picture](https://example.org/picture.png)",
        '<img src="https://example.org/picture.png">',
        "<think>trace</think>",
    ):
        with pytest.raises(ValueError, match="unsafe_markup"):
            check_visible_text(active)


def test_a_citable_source_url_is_not_lost_as_unsafe_markup(store, course):
    text = "Reference location: https://example.org/field-notes"
    store.add_document(course, parse_upload("Reference.txt", text.encode(), isolated=False))

    class SourceURL:
        label = "scripted"

        def complete(self, system, payload):
            return json.dumps(
                {
                    "answer": "https://example.org/field-notes [1]",
                    "status": "answered",
                    "basis": "materials",
                    "citations": [{"source_id": 1, "quote": text}],
                }
            )

    answer = answer_question(
        store, course, "What is the reference location?", use_ai=True, backend=SourceURL()
    )
    assert answer.kind == "generated" and "https://example.org/field-notes" in answer.text
    assert answer.passages


def test_preparation_before_having_inputs_keeps_all_available_actions(store, course):
    text = (
        "# Preparing a sketch\nFold one sheet in half. Title one side with the site name. "
        "Draw a blank grid. Reserve a box for observations.\n"
        "## Open information\nNo site measurements or visit date are supplied in these directions."
    )
    store.add_document(course, parse_upload("Sketch.md", text.encode(), isolated=False))

    class Prepare:
        label = "scripted"
        supports_plain_text = True

        def complete(self, system, payload):
            if "answer_to_label" in payload:
                return '{"basis":"materials","citations":[{"evidence_id":1}]}'
            assert payload["source_only_facts"] is False
            steps = " ".join(s["step"] for s in payload["provided_preparation"])
            assert all(term in steps for term in ("Fold", "Title", "Draw", "Reserve"))
            assert all("No site measurements" not in p["text"] for p in payload["passages"])
            return "[ANSWER] Fold the sheet, title one side, draw a grid and reserve an observation box."

    answer = answer_question(
        store,
        course,
        "Can I prepare a useful sketch before I have the site measurements?",
        use_ai=True,
        backend=Prepare(),
    )
    assert answer.kind == "generated" and "No site measurements" in answer.text


def test_information_gap_hints_do_not_turn_physical_negatives_into_unknowns():
    from modulemate.evidence import (
        preparation_omissions,
        source_information_limits,
        source_negative_statements,
    )

    passages = [
        Passage(1, "one", "Notice", "part 1", "The notice names no sponsor."),
        Passage(2, "two", "Open day", "part 1", "The open day has no sponsor."),
        Passage(3, "one", "Notice", "part 2", "Thin shoes provide no protection."),
        Passage(4, "one", "Notice", "part 3", "No measurements are supplied in these directions."),
    ]
    hints = source_information_limits(passages)
    assert [h["source_id"] for h in hints] == [1, 4]
    assert source_negative_statements(passages) == [{"source_id": 2, "quote": passages[1].text}]
    assert all(h["quote"] == passages[h["source_id"] - 1].text for h in hints)
    gaps = preparation_omissions(passages, [{"source_id": 1, "step": "Prepare the page."}])
    assert [h["source_id"] for h in gaps] == [1, 4]


def test_source_information_limits_do_not_remove_the_original_evidence(store, course):
    text = "The session note does not name a coordinator."
    store.add_document(course, parse_upload("Session.txt", text.encode(), isolated=False))

    class Scope:
        label = "scripted"

        def complete(self, system, payload):
            assert payload["source_information_limits"] == [{"source_id": 1, "quote": text}]
            assert payload["passages"][0]["text"] == text
            return json.dumps(
                {
                    "answer": "The coordinator is not named in the note. [1]",
                    "status": "insufficient",
                    "basis": "materials",
                    "citations": [{"source_id": 1, "quote": text}],
                }
            )

    answer = answer_question(store, course, "Explain this note.", use_ai=True, backend=Scope())
    assert answer.kind == "insufficient" and answer.passages


@pytest.mark.parametrize(
    "question",
    [
        "Explain the main concept of the presentation.",
        "What is this slide deck getting at?",
        "What is the big picture in these materials?",
        "What are the central ideas?",
        "Explain the key message of this reading.",
        "Explain this presentation to me.",
        "What is this about?",
        "What should I take away from these slides?",
        "Can you expalin the main concept from here?",
        "Could you explain the main idea?",
    ],
)
def test_abstract_document_requests_read_broadly(question):
    assert broad_request(question, "Explain")


@pytest.mark.parametrize(
    "question",
    [
        "What is the deadline in these materials?",
        "Explain the valve shown on slide 6.",
        "Give an analogy for linked lists.",
        "Summarize the evidence on lunar dust.",
        "Can you explain the main concept of entropy?",
    ],
)
def test_specific_questions_do_not_become_whole_document_overviews(question):
    assert not broad_request(question, "Explain")


def test_main_idea_context_covers_beginning_middle_and_end(store, course):
    text = "\n".join(
        f"## Section {i}\n" + (f"Distinct observation {i} from the fictional inspection. " * 25)
        for i in range(72)
    )
    doc, _ = store.add_document(course, parse_upload("Deck.md", text.encode(), isolated=False))
    question = "Explain the main concept of this presentation."
    context = build_context(
        store, course, question, plan_turn(question, "Auto", []), [], [doc], policy="adaptive"
    )
    locations = " ".join(p.locator for p in context.passages)
    assert "Section 0" in locations and "Section 71" in locations
    assert any(f"Section {i}" in locations for i in range(30, 42))
    assert not context.complete and context.characters <= 30000


def test_many_short_slides_keep_the_end_when_passage_count_is_the_limit(store, course):
    text = "\n".join(
        f"## Slide {i}\nOriginal observation {i} describes another part of the fictional inspection."
        for i in range(72)
    )
    doc, _ = store.add_document(course, parse_upload("Deck.md", text.encode(), isolated=False))
    question = "Explain the main concept of the presentation."
    context = build_context(
        store, course, question, plan_turn(question, "Auto", []), [], [doc], policy="adaptive"
    )
    assert any("Slide 0" in p.locator for p in context.passages)
    assert any("Slide 71" in p.locator for p in context.passages)
    assert not context.complete and len(context.passages) <= 48


def test_current_material_overview_excludes_explicitly_retired_versions(store, course):
    current = store.add_document(
        course, parse_upload("Presentation.md", b"Inspect all three zones.", isolated=False)
    )[0]
    store.add_document(
        course,
        parse_upload("Retired presentation.md", b"Inspect only the front zone.", isolated=False),
    )
    question = "Explain the main idea of the current presentation."
    context = build_context(
        store, course, question, plan_turn(question, "Auto", []), [], policy="adaptive"
    )
    assert not context.clarification
    assert {p.document_id for p in context.passages} == {current}
    assert context.complete and context.total_passages == len(context.passages) == 1


def test_ambiguous_main_idea_does_not_merge_unrelated_files(store, course):
    for name, text in (
        ("Tea.md", "Tea invitations are repeated in this fictional story."),
        ("Kiln.md", "A kiln inspection samples all three zones in this fictional example."),
    ):
        store.add_document(course, parse_upload(name, text.encode(), isolated=False))
    question = "What is the main concept?"
    context = build_context(
        store, course, question, plan_turn(question, "Auto", []), [], policy="adaptive"
    )
    assert context.clarification and not context.passages


def test_title_preference_does_not_hide_same_subject_facts_in_a_differently_named_file(
    store, course
):
    for name, text in (
        ("Timber sculpture.md", "# Assembly\nAssemble the sculpture from three wooden rings."),
        ("Workshop ledger.md", "# Height\nThe timber sculpture is 30 centimetres tall."),
        ("Field trip.md", "# Height\nThe viewing tower is 11 metres tall. Wear closed shoes."),
    ):
        store.add_document(course, parse_upload(name, text.encode(), isolated=False))
    question = "How tall is the sculpture?"
    context = build_context(
        store, course, question, plan_turn(question, "Auto", []), [], policy="adaptive"
    )
    assert {p.name for p in context.passages} == {"Timber sculpture.md", "Workshop ledger.md"}
    assert any("30 centimetres" in p.text for p in context.passages)
    only = store.documents(course)[0]["id"]
    restricted = build_context(
        store, course, question, plan_turn(question, "Auto", []), [], [only], policy="adaptive"
    )
    assert {p.document_id for p in restricted.passages} <= {only}


def test_one_identifiable_presentation_can_be_explained_among_other_files(store, course):
    doc, _ = store.add_document(
        course,
        parse_upload("Kiln presentation.md", b"Inspect two tiles from each zone.", isolated=False),
    )
    store.add_document(course, parse_upload("Travel.md", b"Bring closed shoes.", isolated=False))
    question = "What is the main concept of the presentation?"
    context = build_context(
        store, course, question, plan_turn(question, "Auto", []), [], policy="adaptive"
    )
    assert not context.clarification and {p.document_id for p in context.passages} == {doc}


def test_short_complete_evidence_survives_both_labeling_and_source_binding(store, course):
    store.add_document(course, parse_upload("Card.txt", b"pH = 7", isolated=False))

    class ShortFact:
        label = "scripted"
        supports_plain_text = True

        def complete(self, system, payload):
            if "answer_to_label" in payload:
                excerpt = payload["source_evidence"][0]["excerpts"][0]
                assert excerpt["quote"] == "pH = 7"
                return json.dumps(
                    {"basis": "materials", "citations": [{"evidence_id": excerpt["evidence_id"]}]}
                )
            return "[ANSWER] The pH is 7."

    result = answer_question(
        store, course, "What pH is on the card?", use_ai=True, backend=ShortFact()
    )
    assert result.kind == "generated" and result.passages[0].text == "pH = 7"


def test_short_quote_must_be_the_whole_short_source():
    passage = Passage(1, "one", "Card", "image 1 · OCR", "pH = 7")
    catalog, _ = source_excerpts([passage])
    assert catalog[0]["excerpts"][0]["quote"] == "pH = 7"
    for quote in ("7", "pH = 8", ""):
        raw = json.dumps(
            {
                "status": "answered",
                "basis": "materials",
                "answer": "The pH is 7. [1]",
                "citations": [{"source_id": 1, "quote": quote}],
            }
        )
        with pytest.raises(ValueError, match="quote_mismatch"):
            parse_response(raw, [passage])


def test_evidence_retry_preserves_a_valid_explanation_instead_of_rewriting_as_missing(
    store, course
):
    store.add_document(
        course,
        parse_upload("Notes.txt", b"Inspect two tiles from each of three zones.", isolated=False),
    )

    class EvidenceRetry:
        label = "scripted"
        supports_plain_text = True

        def __init__(self):
            self.writes = 0
            self.labels = 0

        def complete(self, system, payload):
            if "answer_to_label" not in payload:
                self.writes += 1
                return "[ANSWER] The idea is to make every zone visible before judging the batch."
            self.labels += 1
            return json.dumps(
                {
                    "basis": "materials",
                    "citations": [] if self.labels == 1 else [{"evidence_id": 1}],
                }
            )

    backend = EvidenceRetry()
    result = answer_question(
        store, course, "Explain the main concept.", use_ai=True, backend=backend
    )
    assert result.kind == "generated" and "make every zone visible" in result.text
    assert backend.writes == 1 and backend.labels == 2


@pytest.mark.parametrize(
    "question",
    [
        "Can I think of the sample as a window?",
        "Can I connect these two ideas?",
        "Could I interpret the repeated cup as a symbol of trust?",
        "Can I use a window analogy to explain this?",
        "Can we regard this as checking the message rather than the measuring instrument?",
        "May we see the sample as a window onto the whole collection?",
        "Can I prepare a useful sketch before I have the site measurements?",
    ],
)
def test_conceptual_modal_questions_do_not_become_permission_checks(question):
    from modulemate.conversation import institutional_permission, source_fact_request

    assert not institutional_permission(question)
    assert not source_fact_request(question)


def test_reading_limits_reach_writer_without_becoming_a_refusal(store, course):
    from modulemate.ingest import Chunk, ParsedDocument

    text = "Inspect two tiles from each zone before deciding."
    store.add_document(
        course,
        ParsedDocument(
            "Deck.pdf",
            "original",
            text,
            [Chunk("page 1", text)],
            ["Some image locations were unread; page 2 needs supplied text."],
        ),
    )

    class Explain:
        label = "scripted"

        def complete(self, system, payload):
            assert payload["source_reading_limits"][0]["document"] == "Deck.pdf"
            return json.dumps(
                {
                    "status": "answered",
                    "basis": "materials",
                    "answer": "Inspect all zones before drawing a conclusion. [1]",
                    "citations": [{"source_id": 1, "quote": text}],
                }
            )

    result = answer_question(
        store, course, "Explain the main idea of the presentation.", use_ai=True, backend=Explain()
    )
    assert result.kind == "generated"
    assert result.metadata["source_reading_limits"]


def test_main_idea_reads_complete_selection_without_qa_scoring_or_search(
    store, course, monkeypatch
):
    from modulemate import adaptive, context

    store.add_document(
        course,
        parse_upload("Notes.txt", b"A useful plan inspects all three zones.", isolated=False),
    )

    def not_needed(*args, **kwargs):
        raise AssertionError("A whole short selection needs no factual QA gate or search")

    monkeypatch.setattr(context, "retrieve", not_needed)
    monkeypatch.setattr(adaptive, "assess_evidence", not_needed)
    question = "Explain the main concept of these materials."
    assembled = adaptive.adaptive_context(
        store, course, question, plan_turn(question, "Auto", []), []
    )
    assert assembled.complete and len(assembled.passages) == 1


def test_repeated_label_failure_is_not_reported_as_missing_material(store, course):
    store.add_document(
        course, parse_upload("Card.txt", b"The pH is seven in this example.", isolated=False)
    )

    class NoEvidence:
        label = "scripted"
        supports_plain_text = True

        def complete(self, system, payload):
            return (
                '{"basis":"materials","citations":[]}'
                if "answer_to_label" in payload
                else "[ANSWER] The pH is seven."
            )

    result = answer_question(
        store, course, "What pH is on the card?", use_ai=True, backend=NoEvidence()
    )
    assert result.kind == "error" and result.metadata["model_failure_stage"] == "evidence"
    assert "couldn't attach source evidence" in result.text
    assert "not in" not in result.text


def test_optional_source_footers_do_not_compete_with_actual_evidence():
    from modulemate.conversation import without_source_footer

    for footer in (
        '**Source Support:**\nPassage 1 says "a rule".',
        '<evidence>\nsource_id 1: "a rule"\n</evidence>',
        "### Supporting evidence\nA quotation.",
    ):
        text = "Inspect all three zones.\n\n" + footer
        assert (
            without_source_footer(text, "Explain the main idea of these sources.")
            == "Inspect all three zones."
        )
        assert without_source_footer(text, "Include supporting evidence quotes.") == text


def test_source_binding_json_is_omitted_without_damaging_requested_code():
    from modulemate.conversation import without_source_footer

    packet = '```json\n[{"source_id": 1, "text": "An exact source sentence."}]\n```'
    text = "An analogy about a window.\n\n" + packet + "\n\nA closing explanation."
    cleaned = without_source_footer(text, "Give me an analogy.")
    assert "source_id" not in cleaned and "A closing explanation." in cleaned
    assert without_source_footer(text, "Show the source JSON schema.") == text
    code = "An example.\n```python\nvalues = [1, 2, 3]\n```"
    assert without_source_footer(code, "Give me an analogy.") == code


def test_confident_estimate_cannot_prevent_a_second_look_after_a_missing_answer(
    store, course, monkeypatch
):
    from modulemate import adaptive
    from modulemate.context import Context
    from modulemate.retrieval import scoped_passages

    for name, text in (
        ("Overview.txt", "The lunar tiles were checked before packing."),
        ("Ledger.txt", "Petra checked the lunar tiles before packing."),
    ):
        store.add_document(course, parse_upload(name, text.encode(), isolated=False))
    passages = scoped_passages(store, course)
    first = next(p for p in passages if p.name == "Overview.txt")
    second = next(p for p in passages if p.name == "Ledger.txt")
    scope = [p.document_id for p in passages]
    searches = []

    def build(*args, **kwargs):
        assert args[5] == scope
        searches.append(kwargs.get("recover", False))
        return Context(
            passages=[second if kwargs.get("recover") else first],
            total_passages=2,
            policy="adaptive",
        )

    monkeypatch.setattr(adaptive, "build_context", build)
    monkeypatch.setattr(
        adaptive,
        "assess_evidence",
        lambda *args, **kwargs: {
            "model": "confident but mistaken head",
            "parts": [{"question": "Who checked the lunar tiles?", "estimate": 0.99}],
        },
    )

    class Answer:
        label = "scripted"

        def complete(self, system, payload):
            if "draft" in payload:
                return '{"approved":true,"issues":[]}'
            found = "Petra" in payload["passages"][0]["text"]
            return json.dumps(
                {
                    "status": "answered" if found else "insufficient",
                    "basis": "materials" if found else "none",
                    "answer": "Petra checked them. [1]"
                    if found
                    else "The available passage does not name the checker.",
                    "citations": [{"source_id": 1, "quote": second.text}] if found else [],
                }
            )

    result = answer_question(
        store,
        course,
        "Who checked the lunar tiles?",
        use_ai=True,
        backend=Answer(),
        document_ids=scope,
    )
    assert result.kind == "generated" and "Petra" in result.text
    assert searches == [False, True]
    assert result.metadata["evidence_assessment"]["response_recovery_applied"]
    assert result.passages == [second]


def test_already_retrieved_extra_context_is_reused_without_another_search(monkeypatch):
    from modulemate import adaptive
    from modulemate.context import Context

    one = Passage(1, "one", "one.txt", "part 1", "The tiles were checked.")
    two = Passage(2, "two", "two.txt", "part 1", "Petra checked the tiles.")
    candidate = Context(passages=[one, two], policy="adaptive")
    context = Context(
        passages=[one],
        policy="adaptive",
        evidence={"recovery_attempted": True},
        recovery_candidate=candidate,
    )

    def no_search(*args, **kwargs):
        raise AssertionError("A previous broader search must be reused")

    monkeypatch.setattr(adaptive, "build_context", no_search)
    assert adaptive.recover_missing_context(context, None, "course", "Who?", None, []) is candidate
    context.complete = True
    assert adaptive.recover_missing_context(context, None, "course", "Who?", None, []) is None


def test_preparation_task_separates_missing_results_from_available_setup(store, course):
    source = "Use the same paper and launch point. Record each flight and repeat each design three times."
    store.add_document(course, parse_upload("Trial.txt", source.encode(), isolated=False))

    class Preparation:
        label = "scripted"

        def complete(self, system, payload):
            if "draft" in payload:
                return '{"approved":true,"issues":[]}'
            assert "BEFORE the unavailable inputs arrive" in payload["turn_task"]
            assert "no flight times" in payload["request_background"]
            return json.dumps(
                {
                    "status": "answered",
                    "basis": "materials",
                    "answer": "Prepare the same paper and launch point, then plan three launches per design. [1]",
                    "citations": [{"source_id": 1, "quote": source}],
                }
            )

    result = answer_question(
        store,
        course,
        "I have no flight times yet. How can I prepare the comparison?",
        use_ai=True,
        backend=Preparation(),
    )
    assert result.kind == "generated"


@pytest.mark.parametrize(
    "question",
    [
        "The budget is missing, so I cannot finish the funding request. What can I prepare using the brief?",
        "How can I prepare these records?",
        "Can I think of this as a fair comparison?",
    ],
)
def test_preparation_and_conceptual_modals_do_not_enter_policy_critic(question):
    from modulemate.conversation import needs_review

    assert not needs_review(question, "Explain")


def test_actual_policy_modal_still_receives_targeted_review():
    from modulemate.conversation import needs_review

    assert needs_review("Can I wear sandals instead of closed shoes?", "Explain")


def test_preparation_inventory_and_literal_omissions_preserve_all_source_information(store, course):
    source = (
        "# Brief\n## Draft\nStart with a title. Write an issue paragraph. "
        "Finish with a proposed action and a list of unresolved questions.\n"
        "## Unspecified\nThe brief does not specify a budget, funding source or deadline."
    )
    store.add_document(course, parse_upload("Brief.md", source.encode(), isolated=False))

    class Prepare:
        label = "scripted"
        supports_plain_text = True

        def complete(self, system, payload):
            if "answer_to_label" in payload:
                return '{"basis":"materials","citations":[{"evidence_id":1}]}'
            assert len(payload["passages"]) == 1
            assert "proposed action" in payload["provided_preparation"][-1]["step"]
            assert "does not specify" not in payload["passages"][0]["text"]
            return "[ANSWER] Prepare the title, issue paragraph, proposed action and unresolved questions."

    result = answer_question(
        store,
        course,
        "The budget is missing. What can I prepare from this brief?",
        use_ai=True,
        backend=Prepare(),
    )
    assert result.kind == "generated" and len(result.passages) == 2
    assert "Open inputs" in result.text and "does not specify a budget" in result.text
    assert "[1]" in result.text and "[2]" in result.text


def test_omission_display_never_hides_mixed_steps_or_policy_conditions():
    from modulemate.evidence import preparation_omissions, preparation_steps

    steps = Passage(
        1, "d", "Brief", "Draft", "Write an issue paragraph and list unresolved questions."
    )
    for text in (
        "Write a proposal. The brief does not specify a budget.",
        "No exceptions are provided before approval.",
        "The brief does not provide a URL: https://example.invalid.",
    ):
        passage = Passage(2, "d", "Brief", "Other", text)
        assert not preparation_omissions([steps, passage], preparation_steps([steps, passage]))


def test_input_rewrite_does_not_enter_the_course_policy_critic(store, course):
    class Rewrite:
        label = "scripted"

        def complete(self, system, payload):
            assert "draft" not in payload
            return json.dumps(
                {
                    "answer": "Please remember the submission deadline.",
                    "status": "answered",
                    "basis": "input",
                    "citations": [],
                }
            )

    result = answer_question(
        store,
        course,
        'Rewrite politely: "Remember the submission deadline."',
        use_ai=True,
        backend=Rewrite(),
    )
    assert result.kind == "generated"
