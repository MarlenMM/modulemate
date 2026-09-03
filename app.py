"""M7 — Streamlit chat interface (FR8, NFR6), with S2, C3 and C4.

Design intent, tied to the "Ana" persona: she will not trust an answer she
cannot check. So the interface is built around evidence rather than around
chat. Three things follow from that.

The answer is the only thing on the page allowed to be large. Everything the
machine measured — scores, intent, model, latency — is set in monospace and
demoted, so a reader can tell prose from evidence without reading either.

The refusal gate is drawn, not asserted. `ui.gate_meter` puts the best
passage's cosine on a scale with the calibrated threshold notched on it, and
it is shown on answers as well as refusals: a student who can see the margin
on a good answer has a reason to believe the refusal on a bad one.

Refusals say which of the three guardrails fired — router, similarity gate, or
the generator's own escape hatch — because "I don't know" is only useful when
you can tell whether the material is missing or the question was out of bounds.

The visual language (paper, one accent, one rhythm) lives in ui.py.
"""
from __future__ import annotations

import html

import streamlit as st

import config
import feedback as feedback_mod
import ui
from pipeline import ModuleMate

st.set_page_config(
    page_title="ModuleMate",
    page_icon=ui.favicon(),
    layout="wide",
    # "auto", not "expanded": on a narrow screen Streamlit overlays the
    # sidebar on the conversation, so forcing it open buries the answer on a
    # phone. Let it collapse itself and the layout stays readable at any width.
    initial_sidebar_state="auto",
)
st.markdown(ui.CSS, unsafe_allow_html=True)

# The stages pipeline.ask reports, in the order it reaches them.
STAGES = ["routing", "searching", "writing"]

# Openers taken from eval/gold_set.json, so every one of them is a question
# the system has been measured on rather than a demo that happens to work.
# One per answerable intent, plus one the guardrail is meant to stop: a
# student learns more from seeing the refusal than from being told about it.
OPENERS = [
    ("What is the word limit for the individual project report?", "admin"),
    ("What is the difference between supervised and unsupervised learning?", "a concept"),
    ("What are the steps for constructing a decision tree?", "code help"),
]
REFUSAL_OPENER = "Can you grant me an extension on my submission deadline?"


@st.cache_resource(show_spinner="Loading the index, retrievers and models…")
def get_bot() -> ModuleMate:
    """Built once per server process, not per interaction."""
    return ModuleMate(config.load())


def ask(question: str) -> None:
    """Queue a question. Clicking an opener is the same as typing one."""
    st.session_state.pending_question = question


# --------------------------------------------------------------------------
# Zero state
# --------------------------------------------------------------------------

def render_opening(manifest: dict) -> None:
    """What a student sees before they have asked anything.

    It answers the questions a first-time reader actually has — what this
    reads, what it does with a question, what it refuses — using the real
    corpus counts and the real calibrated threshold rather than claims.
    """
    cfg = config.load()
    threshold = cfg.get("retrieval.similarity_threshold")
    k = cfg.get("retrieval.k")
    n_chunks = manifest.get("n_chunks")
    n_docs = manifest.get("n_documents")

    st.markdown(
        '<p class="mm-lede">Ask about the summer school and get the slide '
        'the answer came from.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="mm-sub">ModuleMate reads {n_chunks:,} passages from '
        f'{n_docs} documents the programme issued — lecture decks and PDFs, '
        'lab notebooks, the assignment and project briefs, the teaching '
        'schedule and the course group chats. It quotes those and nothing '
        'else. It has no access to your submissions, your marks, or the '
        'internet.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="mm-label">What happens to your question</p>',
                unsafe_allow_html=True)
    st.markdown(
        "<ol class='mm-steps'>"
        "<li>A <b>router</b> sorts it into admin, concept, code help or "
        "out of scope. Out-of-scope questions stop here — nothing is "
        "searched and no model is called.</li>"
        f"<li>The rest go to <b>search</b>: dense vectors and exact-word "
        f"BM25 over the same {n_chunks:,} passages, fused, then re-ranked by "
        f"a cross-encoder. If the best of the top {k} scores below "
        f"<b>{threshold}</b> similarity, ModuleMate refuses instead of "
        "guessing.</li>"
        "<li>What survives is written into an answer that <b>names the "
        "document and the slide, page or cell</b>. The passages it used, "
        "their scores and which retriever found them are one click away "
        "under every answer.</li>"
        "</ol>",
        unsafe_allow_html=True,
    )

    st.markdown('<p class="mm-label">Start with one of these</p>',
                unsafe_allow_html=True)
    for question, kind in OPENERS:
        st.button(
            question,
            key=f"open::{question}",
            on_click=ask,
            args=(question,),
            help=f"A question about {kind}, from the evaluation set.",
        )

    st.markdown(
        '<p class="mm-label">And one it will turn down</p>',
        unsafe_allow_html=True,
    )
    st.button(
        REFUSAL_OPENER,
        key="open::refusal",
        on_click=ask,
        args=(REFUSAL_OPENER,),
        help="The router stops this one before anything is searched.",
    )
    st.markdown(
        '<p class="mm-note">ModuleMate will not grant extensions, mark work, '
        'write your report, or answer anything the course materials do not '
        'cover. It sends you to your tutor instead.</p>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Evidence under an answer
# --------------------------------------------------------------------------

def render_gate(result) -> None:
    """The handmade part: the refusal decision, drawn to scale.

    Skipped for router refusals, where no retrieval happened and there is
    therefore no score to plot — drawing an empty meter there would imply a
    measurement that was never taken.
    """
    if result.refusal_reason == "intent" or not result.threshold:
        return
    st.markdown(
        ui.gate_meter(result.top_score, result.threshold,
                      cleared=result.top_score >= result.threshold),
        unsafe_allow_html=True,
    )


def render_meta(result) -> None:
    """One monospace line of what this turn actually cost and used."""
    bits = [f"intent <b>{html.escape(result.intent.label)}</b> "
            f"{result.intent.confidence:.0%}"]
    if result.refusal_reason == "intent":
        bits += ["no search", "no model call"]
    else:
        bits.append(f"{len(result.passages)} passages")
        if result.error:
            bits.append("generation failed")
        elif result.model_used:
            bits.append(f"<b>{html.escape(result.model_used)}</b>")
    bits.append(f"{result.latency_seconds:.1f}s")
    st.markdown(
        '<p class="mm-meta">' + " &middot; ".join(bits) + "</p>",
        unsafe_allow_html=True,
    )


def render_sources(result, key: str) -> None:
    """S2/FR8 — the passages the answer was built from, with their scores.

    Deliberately flat: the passages sit on the page separated by space and a
    hairline, rather than each in its own bordered box inside the panel.
    """
    if not result.passages:
        return
    n = len(result.passages)
    if result.error:
        label = f"Read the {n} passages your question matched"
    elif result.is_refusal:
        label = f"Read the {n} closest passages it found anyway"
    else:
        label = f"Read the {n} passages this answer was built from"
    with st.expander(label):
        if result.was_rewritten:
            st.markdown(
                '<p class="mm-meta">Your follow-up was searched as '
                f'<b>{html.escape(result.rewritten_query)}</b></p>',
                unsafe_allow_html=True,
            )
        blocks = []
        for p in result.passages:
            scores = [f"cosine {p.raw_score:.3f}"]
            if p.authority != 1.0:
                scores.append(f"authority x{p.authority}")
            scores.append(html.escape(p.found_by))
            text = html.escape(p.text[:700]) + ("…" if len(p.text) > 700 else "")
            blocks.append(
                '<div class="mm-passage">'
                '<div class="mm-passage-head">'
                f'<span class="mm-rank">{p.rank}</span>'
                f'<span class="mm-passage-doc">{html.escape(p.source_document)}</span>'
                f'<span class="mm-passage-loc">{html.escape(p.slide_or_section)}</span>'
                "</div>"
                f'<p class="mm-passage-scores">{" &middot; ".join(scores)}</p>'
                f'<p class="mm-passage-text">{text}</p>'
                "</div>"
            )
        st.markdown("".join(blocks), unsafe_allow_html=True)


def render_feedback(result, key: str) -> None:
    """C3 — logged with the turn's retrieval diagnostics, for error analysis."""
    state_key = f"rated::{key}"
    if st.session_state.get(state_key):
        st.markdown(
            '<p class="mm-meta">Logged as '
            f'<b>{st.session_state[state_key]}</b> — thanks. It goes into the '
            'error-analysis log with this turn\'s scores.</p>',
            unsafe_allow_html=True,
        )
        return
    left, right, _ = st.columns([1, 1, 3])
    if left.button("Helpful", key=f"up::{key}"):
        feedback_mod.record(result, "up")
        st.session_state[state_key] = "helpful"
        st.rerun()
    if right.button("Not helpful", key=f"down::{key}",
                    help="Wrong, or it refused something it should have answered"):
        feedback_mod.record(result, "down")
        st.session_state[state_key] = "not helpful"
        st.rerun()


def render_followups(result, key: str) -> None:
    """C4 — next questions, grounded in the passages this turn retrieved."""
    if not result.followups:
        return
    st.markdown('<p class="mm-label">Ask next</p>', unsafe_allow_html=True)
    for i, q in enumerate(result.followups):
        st.button(q, key=f"fu::{key}::{i}", on_click=ask, args=(q,))


# --------------------------------------------------------------------------
# Turns
# --------------------------------------------------------------------------

REFUSAL_WHY = {
    "intent": (
        "Stopped by the intent router, before anything was searched and "
        "before any model was called."
    ),
    "threshold": (
        "Nothing in the indexed materials came close enough. The best "
        "passage scored under the calibrated gate, so ModuleMate refused "
        "rather than answer from a weak match."
    ),
    "model": (
        "The passages that came back were on the right topic but did not "
        "contain the answer, so the generator declined to fill the gap."
    ),
}


def render_question(text: str) -> None:
    with st.container():
        st.markdown(ui.turn_anchor("question"), unsafe_allow_html=True)
        st.markdown(ui.speaker("You"), unsafe_allow_html=True)
        st.markdown(
            f'<p class="mm-question">{html.escape(text)}</p>',
            unsafe_allow_html=True,
        )


def render_answer(result, key: str, *, rateable: bool = True) -> None:
    if result.error:
        kind = "error"
    elif result.is_refusal:
        kind = "refusal"
    else:
        kind = "answer"
    with st.container():
        st.markdown(ui.turn_anchor(kind), unsafe_allow_html=True)
        st.markdown(
            ui.speaker("ModuleMate", state=None if kind == "answer" else kind),
            unsafe_allow_html=True,
        )
        if result.error:
            # Retrieval succeeded and only the writing failed, so say exactly
            # that and leave the passages underneath: the turn is still worth
            # something to the student even with no answer on top of it.
            st.markdown(
                '<p class="mm-error">I found the material but could not write '
                'the answer \u2014 the generation service turned the request '
                'down. The passages below are the ones your question actually '
                'matched, so they are still worth reading. If this keeps '
                'happening, the model named in <code>config.yaml</code> may no '
                'longer be offered by the provider.</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="mm-error-detail">'
                + html.escape(result.error) + "</p>",
                unsafe_allow_html=True,
            )
        elif result.is_refusal:
            st.markdown(
                f'<p class="mm-refusal">{html.escape(result.answer)}</p>',
                unsafe_allow_html=True,
            )
            why = REFUSAL_WHY.get(result.refusal_reason)
            if why:
                st.markdown(f'<p class="mm-refusal-why">{why}</p>',
                            unsafe_allow_html=True)
        else:
            st.markdown(result.answer)
        render_gate(result)
        render_meta(result)
        render_sources(result, key)
        # Nothing was written, so there is nothing to rate or to follow up on.
        if rateable and not result.error:
            render_feedback(result, key)
            render_followups(result, key)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

def render_sidebar(manifest: dict) -> None:
    with st.sidebar:
        st.markdown(ui.wordmark(), unsafe_allow_html=True)
        st.markdown(
            '<p class="mm-side-note">Course assistant for the HIT Global '
            'Summer School 2026. It answers from the programme\'s own '
            'materials and names the source every time.</p>',
            unsafe_allow_html=True,
        )
        # R1 mitigation: corpus staleness is visible rather than silent, so a
        # student can see whether the index predates the material they mean.
        built = str(manifest.get("built_at", "unknown")).split(" ")[0]
        rows = {
            "Passages indexed": f"{manifest.get('n_chunks', 0):,}",
            "Source documents": str(manifest.get("n_documents", "—")),
            "Index built": built,
        }
        st.markdown(
            '<dl class="mm-stats">'
            + "".join(
                f'<div class="mm-stat"><dt>{k}</dt><dd>{v}</dd></div>'
                for k, v in rows.items()
            )
            + "</dl>",
            unsafe_allow_html=True,
        )

        stats = feedback_mod.summarise()
        if stats.get("n"):
            st.markdown('<p class="mm-label">Ratings so far</p>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<p class="mm-meta">{stats["n"]} rated &middot; '
                f'<b>{stats["up"]} helpful</b> &middot; '
                f'{stats["down"]} not</p>',
                unsafe_allow_html=True,
            )
        if st.session_state.get("messages"):
            st.button("Start over", key="reset", on_click=reset)


def reset() -> None:
    st.session_state.messages = []


# --------------------------------------------------------------------------
# Turn loop
# --------------------------------------------------------------------------

def conversation_history(messages) -> list[tuple[str, str]]:
    """S1 — the (question, answer) pairs the query rewriter reads."""
    history: list[tuple[str, str]] = []
    pending: str | None = None
    for msg in messages:
        if msg["role"] == "user":
            pending = msg["content"]
        elif pending is not None:
            history.append((pending, msg["content"]))
            pending = None
    return history


def main() -> None:
    bot = get_bot()
    render_sidebar(bot.manifest)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        render_opening(bot.manifest)

    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            render_question(msg["content"])
        else:
            render_answer(msg["result"], key=str(i))

    prompt = st.chat_input(
        "e.g. Is the 4000-word limit inclusive of references?"
    )
    if not prompt:
        prompt = st.session_state.pop("pending_question", None)
    if not prompt:
        return

    history = conversation_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_question(prompt)

    # The wait reports the stage the pipeline has actually reached, rather
    # than animating for its own sake. A router refusal never leaves the
    # first stage, which is itself the honest picture of what happened.
    slot = st.empty()

    def show_stage(name: str) -> None:
        slot.markdown(ui.stage_line(STAGES, STAGES.index(name)),
                      unsafe_allow_html=True)

    result = bot.ask(prompt, history=history, on_stage=show_stage,
                     degrade_on_error=True)
    slot.empty()

    render_answer(result, key=str(len(st.session_state.messages)),
                  rateable=False)
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "result": result}
    )
    # Rerun so the rating and follow-up controls render against a stable
    # message index rather than one that shifts as the turn completes.
    st.rerun()


if __name__ == "__main__":
    main()
