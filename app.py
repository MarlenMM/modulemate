"""Single-user local study workspace."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from datetime import date, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

import streamlit as st

from modulemate.assistant import MODES, AIBackend, answer_question
from modulemate.cards import STYLES, draft_deck
from modulemate.db import Database, Store, UserError
from modulemate.feedback import review_queue
from modulemate.generation import GenerationError, chat_model_name, chat_status, local_backend
from modulemate.ingest import SUPPORTED, parse_upload
from modulemate.local_models import ModelUnavailable, get_local_models, model_status
from modulemate.retrieval import prepare_index, scoped_passages
from modulemate.search import search
from modulemate.settings import ROOT, load_settings
from modulemate.study import calendar_export, plan_tasks

st.set_page_config(page_title="ModuleMate · Your study workspace", page_icon="📖", layout="wide")
st.markdown(
    """<style>
.block-container {max-width:1180px;padding-top:2rem;padding-bottom:3rem}
[data-testid="stSidebar"] {border-right:1px solid #dddcd5}
h1,h2,h3 {letter-spacing:-.025em}
[data-testid="stMetric"] {border:1px solid #deded6;border-radius:12px;padding:16px}
[data-testid="stChatMessage"] {border:1px solid #e5e4dc;border-radius:12px}
.stButton button {border-radius:8px}
pre {white-space:pre-wrap!important;overflow-wrap:anywhere!important}
</style>""",
    unsafe_allow_html=True,
)


def plain(value):
    """Escape Markdown in user-derived widget labels/notices (including images)."""
    return re.sub(r"([\\`*_{}\[\]()<>#+.!|~-])", r"\\\1", str(value))


def reset_session():
    st.session_state.clear()


def flash(message):
    st.session_state.flash = message
    st.rerun()


def source_panel(sources, key, *, related=False, metadata=None):
    if not sources:
        return
    with st.expander(
        f"{'Related passages' if related else 'Check sources'} · {len(sources)} {'passage' if len(sources) == 1 else 'passages'}"
    ):
        metadata = metadata or {}
        read, total = metadata.get("context_passages"), metadata.get("context_total_passages")
        if (
            metadata.get("overview_sampled")
            and type(read) is int
            and type(total) is int
            and 0 < read < total
        ):
            st.caption(
                f"This overview read {read} of {total} available passages. Long materials use a sample; focus on a section or use Teach for guided coverage."
            )
        for reading in metadata.get("source_reading_limits", []):
            st.caption(plain(reading["document"]) + ": " + plain(" ".join(reading["notices"])))
        for review in metadata.get("citation_review", []):
            if review.get("type") == "unmatched_attributed_quote":
                st.caption(
                    "This attributed phrase could not be matched verbatim. Check the source wording: "
                    + plain(review["quote"])
                )
        for i, passage in enumerate(sources, 1):
            st.text(f"[{i}] {passage['name']} · {passage['locator']}")
            st.text(passage["text"])
            st.divider()


def answer_feedback(store, course_id, message):
    labels = {
        "helpful": "Helpful",
        "incorrect": "Incorrect answer",
        "unsupported": "Not supported by the sources",
        "unclear": "Unclear or incomplete",
    }
    previous = message.get("metadata", {}).get("feedback", {})
    with st.expander("Rate this answer"), st.form(f"feedback_{message['id']}"):
        rating = st.selectbox(
            "Your assessment",
            list(labels),
            index=list(labels).index(previous["rating"]) if previous.get("rating") in labels else 0,
            format_func=labels.get,
        )
        note = st.text_area(
            "Correction or note (optional)", value=previous.get("comment", ""), max_chars=2000
        )
        st.caption("Saved locally for review. Feedback does not automatically retrain the model.")
        if st.form_submit_button("Save answer feedback"):
            store.save_answer_feedback(course_id, message["id"], rating, note)
            flash("Answer feedback saved.")


def passage_count(count):
    return f"{count} passage{'s' if count != 1 else ''}"


def document_labels(documents):
    counts = Counter(d["name"] for d in documents)
    seen = Counter()
    labels = {}
    for document in documents:
        name = document["name"]
        seen[name] += 1
        labels[document["id"]] = (
            f"{name} · import {seen[name]} · {passage_count(document['chunks'])}"
            if counts[name] > 1
            else name
        )
    return labels


def select_documents(documents, key):
    names = document_labels(documents)
    # Streamlit drops a widget's state when its page is hidden. Keep the chosen
    # document scope separately so navigation cannot silently widen it to all.
    saved_key = f"saved_{key}"
    previous = st.session_state.get(key, st.session_state.get(saved_key, []))
    available = [document for document in previous if document in names]
    if key not in st.session_state or previous != available:
        st.session_state[key] = available
    selected = st.multiselect(
        "Focus on documents (leave empty for all)", list(names), format_func=names.get, key=key
    )
    st.session_state[saved_key] = list(selected)
    return selected or None


def library(store, course_id):
    st.header("Course library")
    st.write(
        "Add material you have permission to use. Search and study tools use only this course's library."
    )
    st.caption(
        "Extracted text is saved; originals are not retained. Local OCR reads text in scans and embedded images. Diagrams and handwriting may still need a description."
    )
    with (
        st.expander("Upload documents", expanded=True),
        st.form(f"upload_{course_id}", clear_on_submit=True),
    ):
        uploads = st.file_uploader(
            "PDF, DOCX, PPTX, TXT, Markdown, notebook or image",
            type=[s[1:] for s in sorted(SUPPORTED)],
            accept_multiple_files=True,
        )
        use_ocr = st.checkbox("Read image text with local OCR", value=True)
        st.caption(
            "OCR stays on this computer. Reimport an original to recover images omitted by an older import. Changed text creates a separate copy; select the updated import or remove the old copy after checking its text preview."
        )
        rights = st.checkbox(
            "I have permission to use these materials here.", key=f"upload_rights_{course_id}"
        )
        submitted = st.form_submit_button("Add to library", type="primary")
    if submitted:
        if not rights:
            st.error("Confirm you have permission to use the materials.")
        elif not uploads:
            st.info("Choose at least one document.")
        elif len(uploads) > 5 or sum(u.size for u in uploads) > 30 * 1024 * 1024:
            st.error("Add at most 5 files / 30 MB per batch, with at most 15 MB per file.")
        else:
            for upload in uploads:
                try:
                    with st.spinner(f"Reading {plain(upload.name)}…"):
                        parsed = parse_upload(upload.name, upload.getvalue(), use_ocr=use_ocr)
                        _, created = store.add_document(course_id, parsed)
                    st.success(
                        f"{plain(parsed.name)}: {len(parsed.chunks)} passages added."
                        if created
                        else f"{plain(parsed.name)}: already in this course; skipped."
                    )
                    for warning in parsed.warnings:
                        st.warning(plain(warning))
                except UserError as exc:
                    st.error(f"{plain(upload.name)}: {plain(exc)}")
    with st.expander("Paste notes"), st.form(f"notes_{course_id}", clear_on_submit=True):
        title = st.text_input("Note title", max_chars=150)
        content = st.text_area("Your notes", height=200, max_chars=100000)
        note_rights = st.checkbox("These are my notes, or I have permission to use them.")
        if st.form_submit_button("Save notes"):
            if not note_rights:
                st.error("Confirm you have permission to use these notes.")
            elif not title.strip():
                st.error("Give the note a title.")
            else:
                parsed = parse_upload(title + ".txt", content.encode())
                _, created = store.add_document(course_id, parsed)
                st.success(
                    "Notes saved." if created else "These notes are already in your library."
                )
    docs = store.documents(course_id)
    if not docs:
        st.info(
            "Your library is empty. Upload a document, paste notes, or try the original example below."
        )
        if st.button("Add original Learning Lab example"):
            parsed = parse_upload(
                "Learning Lab.md", (ROOT / "examples/learning-lab.md").read_bytes()
            )
            store.add_document(course_id, parsed)
            flash("Original example added. Try asking: What is retrieval practice?")
        return
    a, b, c = st.columns(3)
    a.metric("Documents", len(docs))
    b.metric("Searchable passages", sum(d["chunks"] for d in docs))
    c.metric("Text characters", f"{sum(d['characters'] for d in docs):,}")
    if docs and model_status() == "ready":
        with st.expander("Prepare local semantic search"):
            st.caption(
                "New passages are indexed on your first question. Prepare them now to avoid that first-question delay; existing vectors are reused."
            )
            if st.button("Prepare this course"):
                try:
                    with st.spinner("Preparing local semantic index…"):
                        engine = get_local_models()
                        if engine is None:
                            raise ModelUnavailable("Local models unavailable")
                        count = prepare_index(store, course_id, engine)
                    st.success(f"Ready. Indexed {count} new passages locally.")
                except ModelUnavailable:
                    st.error(
                        "Local models could not be used. Rerun the model setup command with --force; word search is still available."
                    )
    with st.form(f"search_{course_id}"):
        query = st.text_input(
            "Find something in your materials",
            max_chars=4000,
            placeholder="A concept, deadline, formula or distinctive phrase",
        )
        go = st.form_submit_button("Search library")
    if go:
        hits = search(store, course_id, query)
        if not hits:
            st.info("No matching passages. Try another term or add the missing material.")
        for hit in hits:
            st.text(f"{hit.name} · {hit.locator}")
            st.text(hit.text)
            st.divider()
    labels = document_labels(docs)
    for doc in docs:
        title = labels[doc["id"]]
        if title == doc["name"]:
            title += f" · {passage_count(doc['chunks'])}"
        with st.expander(plain(title)):
            for warning in json.loads(doc["warnings"]):
                st.warning(plain(warning))
            full = store.document(doc["id"])
            st.text(full["text"][:16000])
            if len(full["text"]) > 16000:
                st.caption(
                    "Preview shortened to 16,000 characters; download contains all extracted text."
                )
            st.download_button(
                "Download extracted text",
                full["text"],
                file_name="extracted-notes.txt",
                mime="text/plain",
                key=f"download_{doc['id']}",
            )
            confirm = st.checkbox(
                "Delete this document, its linked cards and this course's chat history",
                key=f"confirm_{doc['id']}",
            )
            if st.button("Delete document", key=f"delete_{doc['id']}", disabled=not confirm):
                store.delete_document(doc["id"])
                for key in list(st.session_state):
                    if key.startswith(("draft_", "quiz_", "reveal_")):
                        del st.session_state[key]
                flash("Document and derived study content removed.")


def render_answer(text):
    # Preserve code, citations and LaTeX. Escaping every backslash used to break
    # equations. Neutralize remote media/links and HTML without changing math.
    segments = re.split(r"(```[\s\S]*?```|`[^`]*`)", text)
    for i in range(0, len(segments), 2):
        segments[i] = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", segments[i]).replace("![", "\\![")
    text = "".join(segments)
    st.markdown(text, unsafe_allow_html=False)


def assistant(store, course_id, settings, backend):
    use_ai = backend is not None
    st.header("Ask, understand, connect")
    docs = store.documents(course_id)
    if not docs:
        st.info(
            "Add materials in Library, or paste text into your question for help working with it."
        )
    mode = st.selectbox("How would you like to study?", list(MODES), key=f"mode_{course_id}")
    doc_ids = select_documents(docs, f"assistant_docs_{course_id}")
    if mode == "Summarize":
        st.caption(
            "Summaries cover selected excerpts. Focus on one document or topic for more useful coverage."
        )
    if backend:
        st.caption(f"Conversational tutor · {backend.label}")
        st.caption(
            "English answers from your materials or text you provide. Missing information is stated clearly; open Check sources to inspect the evidence."
        )
    else:
        st.warning(
            "Excerpt mode is active. Enable a conversational model in the sidebar for explanations, examples and tutoring."
        )
    status = model_status()
    if status != "ready":
        with st.expander("Set up semantic retrieval"):
            st.code(
                "uv sync --extra ml --extra local\nuv run --extra ml --extra local python scripts/prepare_models.py"
            )
            st.caption(f"Retrieval model status: {status}. Word matching is available.")
    messages = store.messages(course_id)
    if messages:
        with st.expander("Conversation options"):
            transcript = "\n\n".join(
                f"You: {m['question']}\n\nModuleMate: {m['answer']}" for m in messages
            )
            st.download_button(
                "Export displayed conversation", transcript, "conversation.txt", "text/plain"
            )
            st.download_button(
                "Export answer review cases (JSON)",
                json.dumps(
                    {
                        "schema": 1,
                        "private_user_content": True,
                        "cases": review_queue(store.messages(course_id, 200)),
                    },
                    indent=2,
                ),
                "answer-review.json",
                "application/json",
            )
            st.caption(
                "Review cases include your questions, answers and source excerpts, ordered by feedback and estimated uncertainty."
            )
            if st.button("Clear this course's conversation"):
                store.clear_messages(course_id)
                flash("Conversation cleared.")
    else:
        st.caption(
            "Try a question about a specific concept, ask for a comparison, or choose Summarize and ask for a study outline."
        )
    for message in messages:
        with st.chat_message("user"):
            st.text(message["question"])
        with st.chat_message("assistant"):
            render_answer(message["answer"])
            st.caption(message.get("metadata", {}).get("notice") or message["mode"])
            source_panel(
                message["sources"],
                message["id"],
                related=message.get("metadata", {}).get("kind") == "error",
                metadata=message.get("metadata"),
            )
            answer_feedback(store, course_id, message)
    if messages:
        if messages[-1].get("metadata", {}).get("kind") == "error" and st.button(
            "Retry this question"
        ):
            st.session_state[f"pending_question_{course_id}"] = messages[-1]["question"]
        suggestions = messages[-1].get("metadata", {}).get("suggestions", [])
        for col, suggestion in zip(st.columns(max(1, len(suggestions))), suggestions, strict=False):
            if col.button(suggestion, key=f"followup_{messages[-1]['id']}_{suggestion}"):
                st.session_state[f"pending_question_{course_id}"] = suggestion
    typed = st.chat_input("Ask about your materials…", max_chars=4000)
    question = typed or st.session_state.pop(f"pending_question_{course_id}", None)
    if question:
        with st.chat_message("user"):
            st.text(question)
        with st.chat_message("assistant"):
            with st.status("Reading your question…", expanded=False) as progress:
                try:
                    engine = get_local_models()
                except ModelUnavailable:
                    engine = None
                    st.warning(
                        "Local models could not be loaded. Using word matching; rerun the model setup command with --force to repair them."
                    )
                result = answer_question(
                    store,
                    course_id,
                    question,
                    mode=mode,
                    document_ids=doc_ids,
                    use_ai=use_ai,
                    backend=backend,
                    engine=engine,
                    on_progress=lambda label: progress.update(label=label),
                )
                progress.update(label="Answer ready", state="complete")
            render_answer(result.text)
            if result.notice:
                st.caption(result.notice)
            source_panel(
                [p.to_dict() for p in result.passages],
                "latest",
                related=result.kind == "error",
                metadata=result.metadata,
            )
        st.rerun()


def card_sources(card):
    sources = card.get("sources", [])
    if sources:
        with st.expander("Supporting material"):
            if card.get("metadata", {}).get("review") == "user_edited":
                st.caption("You edited this card. The excerpts are retained for your comparison.")
            for source in sources:
                st.text(f"{source.get('document_name', '')} · {source['locator']}".strip(" ·"))
                st.text(source["quote"])
    elif card.get("document_name"):
        st.caption(f"{plain(card['document_name'])} · {plain(card['locator'])}")


def create_cards_page(store, course_id, backend, today, saved_cards):
    docs = store.documents(course_id)
    generated, manual = st.tabs(["Generate from materials", "Write your own"])
    with manual:
        source_names = {None: "My own card (no source link)", **document_labels(docs)}
        source_id = st.selectbox(
            "Optional card source", list(source_names), format_func=source_names.get
        )
        locator = ""
        if source_id:
            source_passages = store.document_passages(source_id)
            locator = st.selectbox(
                "Source location", list(dict.fromkeys(p["locator"] for p in source_passages))
            )
            st.text(
                "\n\n".join(p["text"] for p in source_passages if p["locator"] == locator)[:6000]
            )
        with st.form(f"manual_card_{course_id}", clear_on_submit=True):
            front = st.text_area("Question", max_chars=1000)
            back = st.text_area("Answer", max_chars=4000)
            if st.form_submit_button("Save card"):
                created = store.add_card(course_id, front, back, source_id, locator, today=today)
                flash("Card saved." if created else "That card already exists.")
    with generated:
        st.subheader("Turn your materials into learning cards")
        st.write(
            "Choose your material. Get focused questions, concise answers and the excerpts behind them."
        )
        if not docs:
            st.info(
                "Add material in Library to generate cards. You can always write your own card."
            )
            return
        doc_ids = select_documents(docs, f"card_docs_{course_id}")
        selected_docs = [d for d in docs if doc_ids is None or d["id"] in doc_ids]
        scope = [(d["id"], d["created_at"], d["chunks"]) for d in selected_docs]
        draft_key = f"draft_{course_id}"
        previous = st.session_state.get(draft_key)
        if previous and previous.get("scope") != scope:
            st.session_state.pop(draft_key, None)
            st.info("Your material selection changed. Generate a new preview for this selection.")
        topic = st.text_input(
            "Optional topic for new cards",
            max_chars=500,
            placeholder="Leave empty for the main ideas across your selection",
        )
        if backend:
            method = st.radio("Create with", ["AI learning cards", "Source cloze"], horizontal=True)
            st.caption(
                "AI chooses learning targets and writes questions. Answers preserve the source wording. Inspect, edit or discard any draft before saving."
            )
        else:
            method = "Source cloze"
            st.info(
                "Source cloze mode makes literal fill-in-the-blank cards. Select a prepared AI model in the sidebar for AI-written questions and concise answers."
            )
        left, right = st.columns(2)
        count = left.number_input("Number of cards", min_value=1, max_value=24, value=8, step=1)
        style = right.selectbox(
            "Card style", list(STYLES) if method == "AI learning cards" else ["Fill in the blank"]
        )
        for document in selected_docs:
            notices = json.loads(document["warnings"])
            if notices:
                with st.expander("Reading notices · " + plain(document["name"])):
                    for notice in notices:
                        st.text(notice)
        if st.button("Generate learning cards", type="primary"):
            progress = st.progress(0, text="Reading the selected material…")
            try:
                passages = scoped_passages(store, course_id, doc_ids)
                try:
                    engine = get_local_models() if method == "AI learning cards" else None
                except ModelUnavailable:
                    engine = None
                deck = draft_deck(
                    passages,
                    backend if method == "AI learning cards" else None,
                    count=int(count),
                    topic=topic,
                    style=style,
                    existing=saved_cards,
                    engine=engine,
                    progress=lambda fraction, text: progress.progress(fraction, text=text),
                )
                if deck.cards:
                    st.session_state[draft_key] = {
                        "id": uuid4().hex,
                        "cards": deck.cards,
                        "diagnostics": deck.diagnostics,
                        "scope": scope,
                        "topic": topic.strip(),
                        "method": "Source cloze"
                        if all(c["metadata"]["origin"] == "source_cloze" for c in deck.cards)
                        else method,
                    }
                else:
                    st.info(
                        "No new suitable cards were found. Try a different topic or document, or choose Source cloze. Any earlier preview is retained."
                    )
                for warning in deck.diagnostics.get("warnings", []):
                    st.warning(warning)
            except (GenerationError, ModelUnavailable, ValueError, TypeError):
                st.warning(
                    "Card generation could not finish. Try fewer cards or a smaller selection. Any earlier preview is retained."
                )
            finally:
                progress.empty()
        preview = st.session_state.get(draft_key)
        if not preview:
            return
        drafts, info = preview["cards"], preview["diagnostics"]
        st.subheader(f"Your preview · {len(drafts)} cards")
        st.caption(
            f"{preview['method']} · Read {info['read_passages']} of {info['available_passages']} passages "
            f"across {info['read_documents']} of {info['available_documents']} documents. "
            + (f"Topic: {plain(preview['topic'])}. " if preview["topic"] else "")
            + f"Target: {info['requested']} cards. Weak or repeated targets are omitted."
        )
        if info["read_passages"] < info["available_passages"]:
            st.info(
                "This preview samples the selected material. Focus on a smaller section or topic for more targeted coverage."
            )
        st.caption(
            "AI checks can miss mistakes. Compare important details with the supporting material, especially scanned text and formulas."
        )
        names = document_labels(docs)
        with st.form(f"save_drafts_{preview['id']}"):
            selection = []
            for i, draft in enumerate(drafts):
                with st.container(border=True):
                    st.caption(
                        plain(names.get(draft["document_id"], "Source"))
                        + " · "
                        + plain(draft["locator"])
                    )
                    selected = st.checkbox(
                        f"Save draft {i + 1}", value=True, key=f"select_{preview['id']}_{i}"
                    )
                    front = st.text_area(
                        f"Question · draft {i + 1}",
                        value=draft["front"],
                        max_chars=1000,
                        key=f"front_{preview['id']}_{i}",
                    )
                    back = st.text_area(
                        f"Answer · draft {i + 1}",
                        value=draft["back"],
                        max_chars=4000,
                        key=f"back_{preview['id']}_{i}",
                    )
                    display = draft | {
                        "sources": [
                            s | {"document_name": names.get(s["document_id"], "Source")}
                            for s in draft["sources"]
                        ]
                    }
                    card_sources(display)
                    edited = front.strip() != draft["front"] or back.strip() != draft["back"]
                    selection.append(
                        (
                            selected,
                            draft
                            | {
                                "front": front,
                                "back": back,
                                "metadata": draft["metadata"]
                                | ({"review": "user_edited"} if edited else {}),
                            },
                        )
                    )
            if st.form_submit_button("Save selected cards", type="primary"):
                to_save = [draft for selected, draft in selection if selected]
                if not to_save:
                    st.info("Select at least one card to save, or discard the preview.")
                else:
                    saved = store.add_cards(course_id, to_save, today=today)
                    st.session_state.pop(draft_key, None)
                    flash(f"Saved {saved} new cards. Identical cards are skipped.")
        if st.button("Discard preview"):
            st.session_state.pop(draft_key, None)
            st.rerun()


def cards_page(store, course_id, backend, today):
    st.header("Recall beats recognition")
    st.write("Create focused cards, try the answer from memory, then check your source.")
    cards = store.cards(course_id)
    # Saved recall/quiz views need the same import distinction as the picker.
    names = document_labels(store.documents(course_id))
    for card in cards:
        if card["document_id"]:
            card["document_name"] = names.get(card["document_id"], card["document_name"])
        for source in card["sources"]:
            source["document_name"] = names.get(source["document_id"], source["document_name"])
    due = [c for c in cards if c["due"] <= today.isoformat()]
    counts = store.review_counts(course_id)
    a, b, c = st.columns(3)
    a.metric("Saved cards", len(cards))
    b.metric("Due today", len(due))
    c.metric("Reviews completed", sum(counts.values()))
    tool = st.radio(
        "Study tools",
        ["Review", "Create cards", "Practice quiz", "Manage cards"],
        index=0 if cards else 1,
        horizontal=True,
    )
    if tool == "Review":
        if not due:
            st.success(
                "You're up to date. Create some cards or use Practice quiz for another round."
            )
            if cards:
                st.caption(f"Next scheduled review: {min(c['due'] for c in cards)}")
            return
        card = due[0]
        st.caption(f"{len(due)} cards due · recall first")
        if card["document_name"]:
            st.caption(f"{plain(card['document_name'])} · {plain(card['locator'])}")
        st.text(card["front"])
        reveal_key = f"reveal_{card['id']}"
        if st.button("Reveal answer", key=f"show_{card['id']}"):
            st.session_state[reveal_key] = True
        if st.session_state.get(reveal_key):
            st.text(card["back"])
            card_sources(card)
            st.caption(
                "Again: tomorrow · Hard: a short interval · Good/Easy: increasing intervals. These are self-ratings, not an exam score."
            )
            for col, rating in zip(st.columns(4), ["Again", "Hard", "Good", "Easy"], strict=True):
                if col.button(rating, key=f"{rating}_{card['id']}", width="stretch"):
                    store.review_card(card["id"], rating, today=today)
                    st.session_state.pop(reveal_key, None)
                    st.rerun()
    elif tool == "Create cards":
        create_cards_page(store, course_id, backend, today, cards)
    elif tool == "Practice quiz":
        if not cards:
            st.info("Create some cards to start a quiz.")
            return
        quiz_key = f"quiz_{course_id}"
        if st.button("Start a fresh 5-card quiz"):
            st.session_state[quiz_key] = {
                "ids": [c["id"] for c in random.sample(cards, min(5, len(cards)))],
                "position": 0,
                "correct": 0,
                "revealed": False,
            }
        quiz = st.session_state.get(quiz_key)
        if not quiz:
            st.caption(
                "Try answering each question before revealing it. Practice does not change your review schedule."
            )
            return
        if quiz["position"] >= len(quiz["ids"]):
            st.success(
                f"Finished: you marked {quiz['correct']} of {len(quiz['ids'])} answers as recalled."
            )
            return
        by_id = {c["id"]: c for c in cards}
        card = by_id.get(quiz["ids"][quiz["position"]])
        if not card:
            st.session_state.pop(quiz_key, None)
            st.rerun()
        st.progress(quiz["position"] / len(quiz["ids"]))
        if card["document_name"]:
            st.caption(f"{plain(card['document_name'])} · {plain(card['locator'])}")
        st.text(card["front"])
        st.text_area(
            "Your recall attempt (not saved)", key=f"quiz_attempt_{course_id}_{quiz['position']}"
        )
        if st.button("Check answer"):
            quiz["revealed"] = True
        if quiz["revealed"]:
            st.text(card["back"])
            card_sources(card)
            for col, label, score in zip(
                st.columns(2), ["I recalled it", "Needs more practice"], [1, 0], strict=True
            ):
                if col.button(label):
                    quiz["correct"] += score
                    quiz["position"] += 1
                    quiz["revealed"] = False
                    st.rerun()
    else:
        if not cards:
            st.info("No saved cards yet.")
        for card in cards:
            with st.expander(plain(card["front"][:90])):
                card_sources(card)
                with st.form(f"edit_{card['id']}"):
                    front = st.text_area("Question", value=card["front"], max_chars=1000)
                    back = st.text_area("Answer", value=card["back"], max_chars=4000)
                    st.caption(
                        f"Due {card['due']} · {card['repetitions']} successful reviews since last Again"
                    )
                    if st.form_submit_button("Save changes"):
                        store.update_card(card["id"], front, back)
                        flash("Card updated.")
                if st.button("Delete card", key=f"delete_card_{card['id']}"):
                    store.delete_card(card["id"])
                    flash("Card deleted.")


def planner(store, course_id, today):
    st.header("Make room for learning")
    st.write(
        "Set your own deadlines and estimates. The plan fits work into your available time, earliest deadlines first."
    )
    with st.form(f"new_task_{course_id}", clear_on_submit=True):
        title = st.text_input("Task or assignment", max_chars=200)
        a, b = st.columns(2)
        due = a.date_input("Due date", value=today)
        minutes = b.number_input("Estimated minutes", min_value=5, max_value=2400, value=30, step=5)
        if st.form_submit_button("Add task", type="primary"):
            store.add_task(course_id, title, due.isoformat(), int(minutes))
            flash("Task added.")
    tasks = store.tasks(course_id)
    if not tasks:
        st.info("No tasks yet. Add a reading session, revision topic or assignment deadline.")
        return
    for task in tasks:
        a, b, c = st.columns([6, 2, 1])
        a.text(f"{task['title']} · {task['minutes']} min · due {task['due']}")
        if b.button("Reopen" if task["done"] else "Mark done", key=f"done_{task['id']}"):
            store.modify_task(task["id"], done=not task["done"])
            st.rerun()
        if c.button("Remove", key=f"remove_{task['id']}"):
            store.modify_task(task["id"], delete=True)
            st.rerun()
    with st.expander("Edit a task"):
        by_id = {t["id"]: t for t in tasks}
        task_id = st.selectbox(
            "Task to edit", list(by_id), format_func=lambda key: by_id[key]["title"]
        )
        selected = by_id[task_id]
        with st.form(f"edit_task_{task_id}"):
            new_title = st.text_input("Updated title", selected["title"], max_chars=200)
            new_due = st.date_input("Updated due date", date.fromisoformat(selected["due"]))
            new_minutes = st.number_input(
                "Updated estimated minutes",
                min_value=5,
                max_value=2400,
                value=selected["minutes"],
                step=5,
            )
            if st.form_submit_button("Save task changes"):
                store.update_task(task_id, new_title, new_due.isoformat(), int(new_minutes))
                flash("Task updated.")
    st.subheader("Your study plan")
    a, b = st.columns(2)
    capacity = a.slider("Minutes available each day", 5, 240, 60, step=5)
    horizon = b.slider("Days to plan", 1, 30, 7)
    sessions, warnings = plan_tasks(tasks, capacity, horizon, today=today)
    for warning in warnings:
        st.warning(plain(warning))
    if sessions:
        st.dataframe(
            [{k: s[k] for k in ("date", "title", "minutes")} for s in sessions],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Blocks are at most 50 minutes. Breaks are extra. Mark tasks done as you finish; use Edit a task when your deadline or estimate changes."
        )
    elif not warnings:
        st.success("All tasks are complete.")
    st.download_button(
        "Export deadlines to calendar (.ics)",
        calendar_export(tasks),
        "study-deadlines.ics",
        "text/calendar",
    )


def workspace_settings(store, course_id):
    st.header("Workspace settings")
    st.write("One local workspace. No account or password is needed.")
    st.caption(
        "Your notes, chats, cards and tasks stay in the local database. Uploaded originals are discarded after extraction."
    )
    st.download_button(
        "Export all my data (JSON)",
        json.dumps(store.export(), ensure_ascii=False, indent=2),
        "modulemate-export.json",
        "application/json",
    )
    st.caption("Keep exports private. JSON re-import is not currently supported.")
    if course_id:
        course = next(c for c in store.courses() if c["id"] == course_id)
        with st.expander("Course settings"):
            with st.form("course_settings"):
                name = st.text_input("Course name", course["name"], max_chars=100)
                description = st.text_area("Description", course["description"], max_chars=1000)
                if st.form_submit_button("Save course settings"):
                    store.update_course(course_id, name, description)
                    flash("Course settings saved.")
            confirm = st.checkbox(
                "Delete this course and all its documents, cards, tasks and chats"
            )
            if st.button("Delete course permanently", disabled=not confirm):
                store.delete_course(course_id)
                reset_session()
                flash("Course deleted.")
    with st.expander("Clear workspace"):
        st.write(
            "Removes all courses and study content from the active database. Backups and exports are separate."
        )
        confirmed = st.checkbox("Permanently delete all courses and study content")
        if st.button("Clear all study content", disabled=not confirmed):
            store.reset_workspace()
            reset_session()
            flash("Workspace cleared. Create a course to start again.")


def main():
    settings = load_settings()
    db = Database(settings.data_dir / "modulemate.sqlite3")
    store = Store(db)
    with st.sidebar:
        st.title("ModuleMate")
        st.caption("A little structure. A lot of progress.")
        st.caption("Local workspace · No sign-in")
        courses = store.courses()
        duplicate_names = {
            c["name"] for c in courses if sum(other["name"] == c["name"] for other in courses) > 1
        }
        names = {
            c["id"]: (
                f"{c['name']} · {c['documents']} documents · {c['id'][:4]}"
                if c["name"] in duplicate_names
                else c["name"]
            )
            for c in courses
        }
        course_id = (
            st.selectbox("Current course", list(names), format_func=names.get) if courses else None
        )
        with (
            st.expander("Create a course", expanded=not courses),
            st.form("new_course", clear_on_submit=True),
        ):
            name = st.text_input(
                "Course name", max_chars=100, placeholder="e.g. Introduction to biology"
            )
            description = st.text_input("What are you working towards?", max_chars=1000)
            if st.form_submit_button("Create course"):
                store.create_course(name, description)
                flash("Course created. Choose it above and add your materials.")
        page = st.radio("Workspace", ["Library", "Assistant", "Cards", "Planner", "Settings"])
        st.divider()
        timezone = st.selectbox(
            "Your timezone",
            [
                "Asia/Seoul",
                "UTC",
                "Europe/London",
                "Europe/Paris",
                "America/New_York",
                "America/Los_Angeles",
                "Asia/Shanghai",
                "Asia/Kolkata",
                "Asia/Tokyo",
                "Australia/Sydney",
            ],
        )
        today = datetime.now(ZoneInfo(timezone)).date()
        st.caption(f"Study date: {today.isoformat()}")
        local_ready = chat_status() == "ready"
        choices = []
        if local_ready:
            choices.append("On this Mac · " + chat_model_name())
        if settings.ai_ready:
            choices.append(f"Configured provider · {settings.ai_model}")
        choices.append("Excerpts only")
        selected_engine = st.selectbox("Answer engine", choices)
        backend = None
        if selected_engine.startswith("On this Mac"):
            backend = local_backend()
            st.caption("Local generation · Your materials stay on this computer.")
        elif selected_engine.startswith("Configured provider"):
            st.caption(f"Provider: {settings.provider_host}")
            if st.checkbox(
                "Enable provider requests",
                key="ai_consent",
                help="Sends your question, selected excerpts and recent conversation to the configured provider.",
            ):
                backend = AIBackend(settings)
            else:
                st.caption("Provider requests are off. Answers currently use excerpt mode.")
        if not local_ready:
            with st.expander("Set up a local conversational model"):
                st.write(
                    "On Apple Silicon, prepare the 9B instruction model once (about 6 GB of public weights). It runs offline after setup."
                )
                st.code(
                    "uv sync --extra ml --extra local\nuv run --extra ml --extra local python scripts/prepare_chat_model.py"
                )
                st.caption(
                    f"Local model: {chat_status()}. Other computers can use a compatible local server through the provider settings."
                )
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    if page == "Settings":
        workspace_settings(store, course_id)
    elif not course_id:
        st.title("Start with a course")
        st.write(
            "Create a course in the sidebar, then add notes or try the original Learning Lab example. Each course keeps its own materials, conversations and study tools."
        )
    elif page == "Library":
        library(store, course_id)
    elif page == "Assistant":
        assistant(store, course_id, settings, backend)
    elif page == "Cards":
        cards_page(store, course_id, backend, today)
    elif page == "Planner":
        planner(store, course_id, today)


try:
    main()
except UserError as exc:
    st.error(str(exc))
