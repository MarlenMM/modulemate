"""M7 — Streamlit chat interface (FR8, NFR6), with S2, C3 and C4.

Design intent, tied to the "Ana" persona: she will not trust an answer she
cannot check. The citation the generator emits is left inline at the end of the
answer, and the passages the answer was built from are one click away in the
source panel — with their similarity scores *and* which retriever surfaced
them, so the grounding is inspectable rather than asserted.

Refusals are styled as a warning block so that "I don't know" is visually
unmistakable from an answer; a refusal that looks like an answer is the failure
mode the whole guardrail layer exists to prevent.
"""
from __future__ import annotations

import streamlit as st

import config
import feedback as feedback_mod
from pipeline import ModuleMate

st.set_page_config(page_title="ModuleMate", page_icon="🎓", layout="centered")


@st.cache_resource(show_spinner="Loading index, retrievers and models…")
def get_bot() -> ModuleMate:
    """Built once per server process, not per interaction."""
    return ModuleMate(config.load())


def render_sources(result, key: str) -> None:
    """S2/FR8 — expandable panel of retrieved passages and their scores."""
    if not result.passages:
        return
    with st.expander(f"Sources ({len(result.passages)} passages retrieved)"):
        st.caption(
            f"Gate score {result.top_score:.3f} vs threshold "
            f"{result.threshold:.2f} · intent `{result.intent.label}` "
            f"({result.intent.confidence:.0%})"
        )
        if result.was_rewritten:
            st.caption(f"Query rewritten to: *{result.rewritten_query}*")
        for p in result.passages:
            st.markdown(
                f"**{p.rank}. {p.source_document} — {p.slide_or_section}**  \n"
                f"cosine `{p.raw_score:.3f}`"
                + (f" · authority `×{p.authority}`" if p.authority != 1.0 else "")
                + f" · found by {p.found_by}"
            )
            st.text(p.text[:700] + ("…" if len(p.text) > 700 else ""))
            st.divider()


def render_feedback(result, key: str) -> None:
    """C3 — thumbs up/down, logged with the turn's retrieval diagnostics."""
    state_key = f"rated::{key}"
    if st.session_state.get(state_key):
        st.caption(f"Thanks — logged as {st.session_state[state_key]}.")
        return
    col_up, col_down, _ = st.columns([1, 1, 8])
    if col_up.button("👍", key=f"up::{key}", help="This answer was useful"):
        feedback_mod.record(result, "up")
        st.session_state[state_key] = "helpful"
        st.rerun()
    if col_down.button("👎", key=f"down::{key}", help="This answer was wrong or unhelpful"):
        feedback_mod.record(result, "down")
        st.session_state[state_key] = "not helpful"
        st.rerun()


def render_followups(result, key: str) -> None:
    """C4 — suggested next questions, grounded in the retrieved passages."""
    if not result.followups:
        return
    st.caption("You might also ask:")
    for i, q in enumerate(result.followups):
        if st.button(q, key=f"fu::{key}::{i}"):
            st.session_state.pending_question = q
            st.rerun()


def main() -> None:
    bot = get_bot()
    manifest = bot.manifest

    with st.sidebar:
        st.header("ModuleMate")
        st.caption("Course assistant for the HIT Global Summer School 2026.")
        st.divider()
        # R1 mitigation: corpus staleness is visible, not silent.
        st.metric("Indexed passages", manifest.get("n_chunks", "—"))
        st.metric("Source documents", manifest.get("n_documents", "—"))
        st.caption(f"Index built: {manifest.get('built_at', 'unknown')}")
        st.divider()
        st.caption(
            "ModuleMate answers **only** from material issued by the "
            "programme — lecture slides and PDFs, lab notebooks, assignment "
            "and project briefs, schedules and course announcements. Every "
            "answer names its source. If the materials don't cover your "
            "question, it will say so and point you to the tutor."
        )
        stats = feedback_mod.summarise()
        if stats.get("n"):
            st.divider()
            st.caption(
                f"Feedback so far: {stats['up']}👍 / {stats['down']}👎 "
                f"({stats['satisfaction']:.0%} helpful)"
            )
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    st.title("🎓 ModuleMate")
    st.caption(
        "Ask about any course on the programme — admin, concepts, "
        "or how an algorithm works."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay history.
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg.get("is_refusal"):
                st.warning(msg["content"], icon="🚫")
            else:
                st.markdown(msg["content"])
            result = msg.get("result")
            if result is not None:
                render_sources(result, key=str(i))
                render_feedback(result, key=str(i))
                render_followups(result, key=str(i))

    # A clicked follow-up behaves exactly like a typed question.
    prompt = st.chat_input("e.g. Is the 4000-word limit inclusive of references?")
    if not prompt:
        prompt = st.session_state.pop("pending_question", None)
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # S1 — short conversational memory drives the query rewriter.
    history: list[tuple[str, str]] = []
    pending_user: str | None = None
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            pending_user = msg["content"]
        elif pending_user is not None:
            history.append((pending_user, msg["content"]))
            pending_user = None

    with st.chat_message("assistant"):
        with st.spinner("Searching the course materials…"):
            result = bot.ask(prompt, history=history)

        if result.is_refusal:
            st.warning(result.answer, icon="🚫")
        else:
            st.markdown(result.answer)
        idx = str(len(st.session_state.messages))
        render_sources(result, key=idx)
        st.caption(f"Answered in {result.latency_seconds:.1f}s")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "is_refusal": result.is_refusal,
            "result": result,
        }
    )
    # Rerun so the feedback and follow-up controls render against a stable
    # message index rather than one that shifts as the turn completes.
    st.rerun()


if __name__ == "__main__":
    main()
