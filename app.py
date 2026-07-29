"""M7 — Streamlit chat interface (FR8, NFR6), with S2 source panel.

Design intent, tied to the "Ana" persona: she will not trust an answer she
cannot check. The citation the generator emits is left inline at the end of
the answer, and the passages the answer was built from are one click away in
the source panel with their similarity scores. Refusals are styled as a
warning block so that "I don't know" is visually unmistakable from an answer —
a refusal that looks like an answer is the failure mode the whole guardrail
layer exists to prevent.
"""
from __future__ import annotations

import streamlit as st

import config
from pipeline import ModuleMate

st.set_page_config(page_title="ModuleMate", page_icon="🎓", layout="centered")


@st.cache_resource(show_spinner="Loading index and models…")
def get_bot() -> ModuleMate:
    """Built once per server process, not per interaction."""
    return ModuleMate(config.load())


def render_sources(result) -> None:
    """S2/FR8 — expandable panel of retrieved passages and their scores."""
    if not result.passages:
        return
    label = f"Sources ({len(result.passages)} passages retrieved)"
    with st.expander(label, expanded=False):
        st.caption(
            f"Top similarity {result.top_score:.3f} · "
            f"refusal threshold {result.threshold:.2f} · "
            f"intent `{result.intent.label}` ({result.intent.confidence:.0%})"
        )
        if result.was_rewritten:
            st.caption(f"Query rewritten to: *{result.rewritten_query}*")
        for p in result.passages:
            st.markdown(
                f"**{p.rank}. {p.source_document} — {p.slide_or_section}** "
                f"· similarity `{p.score:.3f}`"
            )
            st.text(p.text[:700] + ("…" if len(p.text) > 700 else ""))
            st.divider()


def main() -> None:
    bot = get_bot()
    manifest = bot.manifest

    with st.sidebar:
        st.header("ModuleMate")
        st.caption(
            "Course assistant for the HIT Summer Course 2026 "
            "AI & Machine Learning module."
        )
        st.divider()
        # R1 mitigation: corpus staleness is visible, not silent.
        st.metric("Indexed passages", manifest.get("n_chunks", "—"))
        st.metric("Source documents", manifest.get("n_documents", "—"))
        st.caption(f"Index built: {manifest.get('built_at', 'unknown')}")
        st.divider()
        st.caption(
            "ModuleMate answers **only** from the module's own lecture slides, "
            "teaching schedule and assessment brief. Every answer names its "
            "source. If the materials don't cover your question, it will say "
            "so and point you to the tutor."
        )
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    st.title("🎓 ModuleMate")
    st.caption("Ask about the module — admin, concepts, or how an algorithm works.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay history.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("is_refusal"):
                st.warning(msg["content"], icon="🚫")
            else:
                st.markdown(msg["content"])
            if msg.get("result") is not None:
                render_sources(msg["result"])

    prompt = st.chat_input("e.g. Is the 4000-word limit inclusive of references?")
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
        with st.spinner("Searching the module materials…"):
            result = bot.ask(prompt, history=history)

        if result.is_refusal:
            st.warning(result.answer, icon="🚫")
        else:
            st.markdown(result.answer)
        render_sources(result)
        st.caption(f"Answered in {result.latency_seconds:.1f}s")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "is_refusal": result.is_refusal,
            "result": result,
        }
    )


if __name__ == "__main__":
    main()
