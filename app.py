"""Streamlit entry point. User data lives only behind authenticated Store calls."""

from __future__ import annotations

import json
import random
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

import streamlit as st

from modulemate.assistant import MODES, AIBackend, ProviderError, answer_question, generate_cards
from modulemate.db import AuthenticationError, Database, Store, UserError
from modulemate.ingest import SUPPORTED, parse_upload
from modulemate.local_models import ModelUnavailable, get_local_models, model_status
from modulemate.retrieval import prepare_index
from modulemate.search import overview, search
from modulemate.settings import ROOT, load_settings
from modulemate.study import calendar_export, cloze_cards, plan_tasks

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


def sign_in(db, settings):
    st.title("Your materials. Your study space.")
    st.write(
        "Turn course notes into answers you can check, cards you can review, and a plan you can follow."
    )
    a, b, c = st.columns(3)
    a.info(
        "**Bring your own sources**\n\nUpload documents or paste notes. Every course starts with an empty library."
    )
    b.info(
        "**Learn with evidence**\n\nAsk questions, connect ideas and practice recall with cited answers from your notes."
    )
    c.info(
        "**Build a study habit**\n\nMake recall cards, review what is due, and plan your next session."
    )
    st.caption(
        "No university materials are bundled. Local tools work without an AI key. Accounts belong to this installation."
    )
    action = st.radio("Account", ["Sign in", "Create account"], horizontal=True)
    with st.form("authentication"):
        username = st.text_input("Username", max_chars=40, autocomplete="username")
        password = st.text_input(
            "Password",
            type="password",
            max_chars=256,
            help="At least 12 characters. Use a unique passphrase.",
        )
        repeat = (
            st.text_input("Repeat password", type="password", max_chars=256)
            if action == "Create account"
            else None
        )
        submitted = st.form_submit_button(action, type="primary")
    if submitted:
        try:
            if action == "Create account":
                if password != repeat:
                    raise UserError("Passwords do not match.")
                token = db.register(username, password, allowed=settings.allow_registration)
            else:
                token = db.login(username, password)
            reset_session()
            st.session_state.token = token
            st.rerun()
        except UserError as exc:
            st.error(str(exc))
    st.caption(
        "Passwords are hashed. Content is stored on this server; its operator can access it. There is no password recovery service, so keep your passphrase safe."
    )


def source_panel(sources, key):
    if not sources:
        return
    with st.expander(
        f"Check sources · {len(sources)} {'passage' if len(sources) == 1 else 'passages'}"
    ):
        for i, passage in enumerate(sources, 1):
            st.text(f"[{i}] {passage['name']} · {passage['locator']}")
            st.text(passage["text"])
            st.divider()


def select_documents(documents, key):
    names = {d["id"]: d["name"] for d in documents}
    selected = st.multiselect(
        "Focus on documents (leave empty for all)", list(names), format_func=names.get, key=key
    )
    return selected or None


def library(store, course_id):
    st.header("Course library")
    st.write(
        "Add material you have permission to use. Search and study tools use only this course's library."
    )
    st.caption(
        "Extracted text is saved; original files are not retained. Images, handwriting and scanned pages need OCR first."
    )
    with (
        st.expander("Upload documents", expanded=True),
        st.form(f"upload_{course_id}", clear_on_submit=True),
    ):
        uploads = st.file_uploader(
            "PDF, DOCX, PPTX, TXT, Markdown or notebook",
            type=[s[1:] for s in sorted(SUPPORTED)],
            accept_multiple_files=True,
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
                        parsed = parse_upload(upload.name, upload.getvalue())
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
    for doc in docs:
        with st.expander(f"{plain(doc['name'])} · {doc['chunks']} passages"):
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
    # Permit paragraph/bullet/emphasis formatting, but neutralize links, images,
    # HTML and escape sequences from both source text and model output.
    for char in ["\\", "[", "]", "<", ">"]:
        text = text.replace(char, "\\" + char)
    st.markdown(text, unsafe_allow_html=False)


def assistant(store, course_id, settings, use_ai):
    st.header("Ask, understand, connect")
    docs = store.documents(course_id)
    if not docs:
        st.info("Add material in Library first. Your assistant starts with your sources.")
        return
    mode = st.selectbox("How would you like to study?", list(MODES), key=f"mode_{course_id}")
    doc_ids = select_documents(docs, f"assistant_docs_{course_id}")
    if mode == "Summarize":
        st.caption(
            "Summaries cover selected excerpts. Focus on one document or topic for more useful coverage."
        )
    if not use_ai:
        st.caption(
            "Local study assistant · Focused source-based answers, comparisons and recall prompts. No hosted AI requests."
        )
    status = model_status()
    if status == "ready":
        st.caption(
            "Local semantic models installed · Understands similar wording and ranks relevant evidence."
        )
    else:
        with st.expander("Improve matching with local machine learning"):
            st.write(
                "Install the local models once to match paraphrases and rank evidence. Model setup downloads public weights; your materials stay on this computer."
            )
            st.code("uv sync --extra ml\nuv run --extra ml python scripts/prepare_models.py")
            st.caption(f"Local model status: {status}. Word matching remains available.")
    if use_ai:
        st.caption(
            f"AI enabled · {settings.provider_host} · {settings.ai_model}. Check the evidence before relying on an answer."
        )
    messages = store.messages(course_id)
    if messages:
        with st.expander("Conversation options"):
            transcript = "\n\n".join(
                f"You: {m['question']}\n\nModuleMate: {m['answer']}" for m in messages
            )
            st.download_button(
                "Export displayed conversation", transcript, "conversation.txt", "text/plain"
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
            source_panel(message["sources"], message["id"])
    if messages:
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
            with st.spinner("Finding evidence and preparing your answer…"):
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
                    backend=AIBackend(settings) if use_ai else None,
                    engine=engine,
                )
            render_answer(result.text)
            if result.notice:
                st.caption(result.notice)
            source_panel([p.to_dict() for p in result.passages], "latest")
        st.rerun()


def cards_page(store, course_id, settings, use_ai, today):
    st.header("Recall beats recognition")
    st.write("Create focused cards, try the answer from memory, then check your source.")
    cards = store.cards(course_id)
    due = [c for c in cards if c["due"] <= today.isoformat()]
    counts = store.review_counts(course_id)
    a, b, c = st.columns(3)
    a.metric("Saved cards", len(cards))
    b.metric("Due today", len(due))
    c.metric("Reviews completed", sum(counts.values()))
    tool = st.radio(
        "Study tools", ["Review", "Create cards", "Practice quiz", "Manage cards"], horizontal=True
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
        st.text(card["front"])
        reveal_key = f"reveal_{card['id']}"
        if st.button("Reveal answer", key=f"show_{card['id']}"):
            st.session_state[reveal_key] = True
        if st.session_state.get(reveal_key):
            st.text(card["back"])
            if card["document_name"]:
                st.caption(f"{plain(card['document_name'])} · {plain(card['locator'])}")
            st.caption(
                "Again: tomorrow · Hard: a short interval · Good/Easy: increasing intervals. These are self-ratings, not an exam score."
            )
            for col, rating in zip(st.columns(4), ["Again", "Hard", "Good", "Easy"], strict=True):
                if col.button(rating, key=f"{rating}_{card['id']}", width="stretch"):
                    store.review_card(card["id"], rating, today=today)
                    st.session_state.pop(reveal_key, None)
                    st.rerun()
    elif tool == "Create cards":
        docs = store.documents(course_id)
        with st.expander("Write your own card", expanded=True):
            source_names = {
                None: "My own card (no source link)",
                **{d["id"]: d["name"] for d in docs},
            }
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
                    "\n\n".join(p["text"] for p in source_passages if p["locator"] == locator)[
                        :6000
                    ]
                )
            with st.form(f"manual_card_{course_id}", clear_on_submit=True):
                front = st.text_area("Question", max_chars=1000)
                back = st.text_area("Answer", max_chars=4000)
                if st.form_submit_button("Save card"):
                    created = store.add_card(
                        course_id, front, back, source_id, locator, today=today
                    )
                    flash("Card saved." if created else "That card already exists.")
        if not docs:
            st.info("Add material to create source-linked cards automatically.")
            return
        doc_ids = select_documents(docs, f"card_docs_{course_id}")
        topic = st.text_input("Optional topic for new cards", max_chars=500)
        st.caption(
            "AI drafts recall questions when enabled. Local mode makes fill-in-the-blank cards from source sentences. Review drafts before saving."
        )
        if st.button("Draft cards from material", type="primary"):
            passages = (
                search(store, course_id, topic, doc_ids)
                if topic.strip()
                else overview(store, course_id, doc_ids)
            )
            if not passages:
                st.info("No matching material. Try a different topic.")
            else:
                with st.spinner("Preparing card drafts…"):
                    try:
                        drafts = (
                            generate_cards(passages, AIBackend(settings))
                            if use_ai
                            else cloze_cards(passages)
                        )
                    except (ProviderError, ValueError, KeyError, TypeError):
                        drafts = cloze_cards(passages)
                        st.warning(
                            "AI drafts could not be verified. Showing local cloze drafts instead."
                        )
                st.session_state[f"draft_{course_id}"] = drafts
                if not drafts:
                    st.info("No suitable cards found. Try another document or write a card above.")
        drafts = st.session_state.get(f"draft_{course_id}", [])
        if drafts:
            with st.form(f"save_drafts_{course_id}"):
                selected = []
                for i, draft in enumerate(drafts):
                    selected.append(
                        st.checkbox(
                            f"Save draft {i + 1}", value=True, key=f"draft_select_{course_id}_{i}"
                        )
                    )
                    st.text(draft["front"])
                    st.text(draft["back"])
                    st.caption(draft["locator"])
                if st.form_submit_button("Save selected cards"):
                    saved = sum(
                        store.add_card(course_id, **draft, today=today)
                        for draft, select in zip(drafts, selected, strict=True)
                        if select
                    )
                    st.session_state.pop(f"draft_{course_id}", None)
                    flash(f"Saved {saved} new cards. Identical cards are skipped.")
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
        st.text(card["front"])
        st.text_area(
            "Your recall attempt (not saved)", key=f"quiz_attempt_{course_id}_{quiz['position']}"
        )
        if st.button("Check answer"):
            quiz["revealed"] = True
        if quiz["revealed"]:
            st.text(card["back"])
            if card["document_name"]:
                st.caption(f"{plain(card['document_name'])} · {plain(card['locator'])}")
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


def account(store, db, settings, course_id):
    st.header("Your data, your controls")
    st.write(
        "Your account is private within this app. The server operator can access its database; it is not encrypted end to end."
    )
    st.caption(
        "Uploaded originals are discarded after extraction. Notes, chat, cards, reviews and tasks persist in the local database. AI is off until you opt in for this session."
    )
    st.download_button(
        "Export all my data (JSON)",
        json.dumps(store.export(), ensure_ascii=False, indent=2),
        "modulemate-export.json",
        "application/json",
    )
    st.caption(
        "The export contains your content, not passwords or session tokens. Keep it private. JSON re-import is not currently supported."
    )
    with st.expander("Change password"), st.form("password_change", clear_on_submit=True):
        old = st.text_input("Current password", type="password", max_chars=256)
        new = st.text_input("New password", type="password", max_chars=256)
        repeat = st.text_input("Repeat new password", type="password", max_chars=256)
        if st.form_submit_button("Update password"):
            if new != repeat:
                st.error("New passwords do not match.")
            else:
                store.change_password(old, new)
                st.success("Password changed. Other sessions have been signed out.")
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
                token = st.session_state.token
                reset_session()
                st.session_state.token = token
                flash("Course deleted.")
    with st.expander("Delete my account"):
        st.warning(
            "This removes all of your data from the active database. Operator backups and copies already exported or sent to an AI provider are separate."
        )
        with st.form("delete_account"):
            password = st.text_input("Confirm your password", type="password", max_chars=256)
            confirmed = st.checkbox("Permanently delete my account and all my courses")
            if st.form_submit_button("Delete my account", disabled=not confirmed):
                store.delete_account(password)
                reset_session()
                st.rerun()


def main():
    settings = load_settings()
    db = Database(settings.data_dir / "modulemate.sqlite3")
    if not st.session_state.get("token"):
        sign_in(db, settings)
        return
    store = Store(db, st.session_state.token)
    try:
        profile = store.profile()
    except AuthenticationError:
        reset_session()
        st.rerun()
    with st.sidebar:
        st.title("ModuleMate")
        st.caption("A little structure. A lot of progress.")
        st.text(f"Signed in as {profile['username']}")
        courses = store.courses()
        names = {c["id"]: c["name"] for c in courses}
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
        page = st.radio("Workspace", ["Library", "Assistant", "Cards", "Planner", "Account"])
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
        if settings.ai_ready:
            st.caption(
                f"Optional AI provider: {settings.provider_host}\nModel: {settings.ai_model}"
            )
            use_ai = st.checkbox(
                "Enable AI for this session",
                key="ai_consent",
                help="Sends your question, up to 8 source excerpts and the last 3 conversation turns to this provider. Drafting and support checks send source excerpts and drafts to this same provider. Review its data policy first.",
            )
            st.caption(
                "Enabling AI shares selected text with this provider. Disable it to keep further requests local."
            )
        else:
            use_ai = False
            st.caption(
                "Local mode · No AI provider configured. All core study tools are available."
            )
        if st.button("Sign out"):
            db.logout(st.session_state.token)
            reset_session()
            st.rerun()
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    if page == "Account":
        account(store, db, settings, course_id)
    elif not course_id:
        st.title("Start with a course")
        st.write(
            "Create a course in the sidebar, then add notes or try the original Learning Lab example. Each course keeps its own materials, conversations and study tools."
        )
    elif page == "Library":
        library(store, course_id)
    elif page == "Assistant":
        assistant(store, course_id, settings, use_ai)
    elif page == "Cards":
        cards_page(store, course_id, settings, use_ai, today)
    elif page == "Planner":
        planner(store, course_id, today)


try:
    main()
except AuthenticationError as exc:
    st.error(str(exc))
    if st.button("Return to sign in"):
        reset_session()
        st.rerun()
except UserError as exc:
    st.error(str(exc))
