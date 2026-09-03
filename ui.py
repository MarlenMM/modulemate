"""Presentation layer for the chat interface — tokens, marks and primitives.

Kept apart from app.py so that the *flow* of a turn (ask, answer, cite, rate)
stays readable in one file while the decisions about how a turn should look
live in one place and are applied consistently.

Three decisions drive everything here.

1. Paper, not dashboard. ModuleMate quotes lecture slides and names its
   sources, so it is styled as a printed transcript — warm paper, one ink,
   a serif for the voice of the product and a monospace for anything the
   machine measured. Dark-on-light is the working default because the
   content is prose to be read, not telemetry to be scanned.

2. Colour carries meaning or it is not used. There is exactly one accent.
   The rule down the left of a turn is the state of that turn: grey for the
   student's question, teal for an answer that cleared the retrieval gate,
   amber for a refusal. Nothing is tinted for decoration.

3. The gate meter is the product. `gate_meter` draws the one number that
   decides whether a student gets an answer at all — the best passage's
   cosine against the calibrated threshold — as a scale with the threshold
   notched on it. It is shown on answers (how much margin there was) and on
   refusals (how far short the corpus fell), because "I don't know" is only
   trustworthy if you can see the reason.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

# The mark is the citation bracket the generator emits — [ • ] — which is the
# whole promise of the product in three strokes. Drawn by hand rather than
# taken from an icon set so that it survives being scaled to a 16px favicon.
MARK_SVG = """<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M9.4 3.6H5.2v16.8h4.2M14.6 3.6h4.2v16.8h-4.2"
        stroke="{ink}" stroke-width="2.1" stroke-linecap="round"
        stroke-linejoin="round"/>
  <circle cx="12" cy="12" r="2.7" fill="{accent}"/>
</svg>"""

_TOKENS = {
    "paper": "#FBF9F5",
    "paper_2": "#F4F0E8",
    "ink": "#1B1A17",
    "accent": "#1F5C50",
    "flag": "#8A5116",
}


def favicon() -> str | None:
    """Write the mark to disk and return the path Streamlit should use.

    Generated rather than committed as a binary so the mark and the palette
    can never drift apart: both are derived from the tokens above. Returns
    None if the file cannot be written, which costs a favicon and nothing
    else -- an unwritable directory should not stop the app from starting.
    """
    try:
        ASSETS.mkdir(exist_ok=True)
        path = ASSETS / "mark.svg"
        path.write_text(
            MARK_SVG.format(ink=_TOKENS["ink"], accent=_TOKENS["accent"]),
            encoding="utf-8",
        )
    except OSError:
        return None
    return str(path)


def wordmark() -> str:
    """The mark plus the product name, for the head of the sidebar."""
    svg = MARK_SVG.format(ink=_TOKENS["ink"], accent=_TOKENS["accent"])
    return f'<div class="mm-wordmark">{svg}<span>ModuleMate</span></div>'


# Streamlit ships its own generated class selectors (.st-emotion-cache-*) for
# paragraph type, and those out-specify a bare `.mm-thing` rule. Every custom
# element we emit goes through st.markdown, so every custom rule below is
# scoped to the markdown container -- written as `$md` and substituted in once,
# rather than repeated by hand seventy times.
_MD = '[data-testid="stMarkdownContainer"]'

_STYLESHEET = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  /* Four colours, four meanings, none decorative: grey for the student's
     question, teal for an answer that cleared the gate, amber for a refusal
     the guardrails made on purpose, brick for the backend being down. */
  --paper:      #FBF9F5;
  --paper-2:    #F4F0E8;
  --ink:        #1B1A17;
  --ink-2:      #57534B;
  --ink-3:      #8B857A;
  --rule:       #E2DCD0;
  --rule-strong:#D3CABA;   /* hairlines that have to be legible, e.g. the meter */
  --accent:     #1F5C50;
  --accent-2:   #2E7D6D;
  --accent-wash:#EAF1EE;
  --flag:       #8A5116;
  --flag-wash:  #F8EFE2;
  --alarm:      #8C2F26;
  --alarm-wash: #F8EAE7;

  --font-display: 'Newsreader', Georgia, 'Times New Roman', serif;
  --font-body: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;

  /* One rhythm, used everywhere. Nothing gets an ad-hoc margin. */
  --s1: .25rem; --s2: .5rem;  --s3: .75rem; --s4: 1rem;
  --s5: 1.5rem; --s6: 2.25rem; --s7: 3.5rem;
}

/* --- Remove the host framework's own furniture ------------------------- */
/* The default top gradient bar, deploy button and status widget advertise
   the tool rather than the product, so they go. */
[data-testid="stDecoration"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"],
#MainMenu, footer { display: none !important; }
[data-testid="stHeader"] { background: transparent; height: 0; }

/* --- Page ------------------------------------------------------------- */
[data-testid="stAppViewContainer"], [data-testid="stApp"] { background: var(--paper); }
html, body, [data-testid="stAppViewContainer"] * { font-family: var(--font-body); }

/* A real measure, anchored to a left edge rather than floated dead-centre. */
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 52rem;
  margin-left: 0;
  padding: var(--s7) var(--s5) 9rem var(--s7);
}
@media (max-width: 1100px) {
  /* The sidebar collapses to a floating toggle at this width, which sits over
     the top-left of the page -- so the content starts below it. */
  [data-testid="stMainBlockContainer"], .block-container {
    padding-left: var(--s5); padding-top: 4.25rem;
  }
}

/* --- Type system ------------------------------------------------------- */
/* Three levels: the answer, the labels above it, the data beneath it. */
$md p, $md li { font-size: .95rem; line-height: 1.62; color: var(--ink-2); }
$md h1 {
  font-family: var(--font-display); font-weight: 500; font-size: 2rem;
  letter-spacing: -.012em; line-height: 1.16; color: var(--ink);
  margin: 0 0 var(--s3);
}
$md h2 {
  font-family: var(--font-display); font-weight: 500; font-size: 1.3rem;
  letter-spacing: -.008em; color: var(--ink); margin: var(--s6) 0 var(--s3);
}
$md h3 {
  font-size: .72rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .1em; color: var(--ink-3); margin: var(--s5) 0 var(--s2);
}
$md strong { font-weight: 600; color: var(--ink); }
$md a { color: var(--accent); text-underline-offset: 3px; }
$md code {
  font-family: var(--font-mono); font-size: .82em; color: var(--accent);
  background: var(--accent-wash); padding: .08em .32em; border-radius: 2px;
}
hr { border-color: var(--rule); }

/* --- A turn ------------------------------------------------------------ */
/* The rule down the left says what kind of turn this is. That is the only
   job colour has in this interface. Streamlit gives its containers no stable
   class of their own, so the block is selected by what it contains. */
$md .mm-anchor { display: none; }

[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .mm-turn),
[data-testid="stVerticalBlock"]:has(> [data-testid="element-container"] .mm-turn) {
  border-left: 2px solid var(--rule);
  padding-left: var(--s5);
  margin: var(--s6) 0 0;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .mm-turn-answer),
[data-testid="stVerticalBlock"]:has(> [data-testid="element-container"] .mm-turn-answer) {
  border-left-color: var(--accent);
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .mm-turn-refusal),
[data-testid="stVerticalBlock"]:has(> [data-testid="element-container"] .mm-turn-refusal) {
  border-left-color: var(--flag);
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .mm-turn-error),
[data-testid="stVerticalBlock"]:has(> [data-testid="element-container"] .mm-turn-error) {
  border-left-color: var(--alarm);
}

$md .mm-speaker {
  font-family: var(--font-mono); font-size: .68rem; font-weight: 500;
  text-transform: uppercase; letter-spacing: .13em; color: var(--ink-3);
  margin: 0 0 var(--s2); line-height: 1.4;
}
$md .mm-speaker.is-refusal { color: var(--flag); }
$md .mm-speaker.is-error { color: var(--alarm); }

/* The student's own question, set as a question and not as a chat bubble. */
$md .mm-question {
  font-family: var(--font-display); font-size: 1.26rem; font-weight: 400;
  line-height: 1.4; color: var(--ink); margin: 0;
}

/* The answer is the one thing on the page that gets to be big. `:not([class])`
   picks out the generated prose specifically: every element this app writes
   itself carries a class, and the model's markdown does not, so the evidence
   lines underneath the answer keep their small monospace setting. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .mm-turn-answer) $md p:not([class]),
[data-testid="stVerticalBlock"]:has(> [data-testid="element-container"] .mm-turn-answer) $md p:not([class]),
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .mm-turn-answer) $md li:not([class]),
[data-testid="stVerticalBlock"]:has(> [data-testid="element-container"] .mm-turn-answer) $md li:not([class]) {
  font-size: 1.04rem; line-height: 1.68; color: var(--ink);
}

$md .mm-refusal { font-size: 1.02rem; line-height: 1.62; color: var(--ink); margin: 0; }
$md .mm-refusal-why {
  font-family: var(--font-body); font-size: .84rem; line-height: 1.5;
  color: var(--flag); background: var(--flag-wash);
  border-left: 2px solid var(--flag); padding: var(--s2) var(--s3);
  margin: var(--s3) 0 0;
}

/* An outage is not a refusal, so it does not borrow the refusal's colour.
   The passages retrieval already found are still shown underneath it. */
$md .mm-error { font-size: 1.02rem; line-height: 1.62; color: var(--ink); margin: 0; }
$md .mm-error-detail {
  font-family: var(--font-mono); font-size: .72rem; line-height: 1.55;
  color: var(--alarm); background: var(--alarm-wash);
  border-left: 2px solid var(--alarm); padding: var(--s2) var(--s3);
  margin: var(--s3) 0 0; overflow-wrap: anywhere;
}

/* --- Machine-measured facts ------------------------------------------- */
/* Anything the system counted is monospace, so a reader can tell at a glance
   what is prose and what is evidence. Prose asides use .mm-note instead. */
$md .mm-meta {
  font-family: var(--font-mono); font-size: .72rem; line-height: 1.55;
  color: var(--ink-3); margin: var(--s3) 0 0; letter-spacing: .01em;
}
$md .mm-meta b { font-weight: 500; color: var(--ink-2); }
$md .mm-note {
  font-size: .84rem; line-height: 1.6; color: var(--ink-3);
  max-width: 62ch; margin: var(--s3) 0 0;
}

/* --- The retrieval gate meter ----------------------------------------- */
$md .mm-gate { margin: var(--s4) 0 0; }
$md .mm-gate svg { display: block; }
$md .mm-gate-read {
  font-family: var(--font-mono); font-size: .72rem; color: var(--ink-3);
  margin: var(--s1) 0 0; letter-spacing: .01em;
}
$md .mm-gate-read b { font-weight: 500; color: var(--ink); }
$md .mm-gate-read .cleared { color: var(--accent-2); }
$md .mm-gate-read .short { color: var(--flag); }

/* --- Sources ----------------------------------------------------------- */
/* One container, not a container of containers: the passages sit directly on
   the paper, separated by space and a hairline rather than by nested boxes. */
[data-testid="stExpander"] { border: none; background: transparent; }
[data-testid="stExpander"] details {
  border: none; border-top: 1px solid var(--rule);
  background: transparent; border-radius: 0;
}
[data-testid="stExpander"] summary { padding: var(--s3) 0; }
[data-testid="stExpander"] summary p {
  font-family: var(--font-mono) !important; font-size: .7rem !important;
  font-weight: 500; text-transform: uppercase; letter-spacing: .1em;
  color: var(--ink-3) !important;
}
[data-testid="stExpander"] summary:hover p { color: var(--accent) !important; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding: 0 0 var(--s4); }

$md .mm-passage + .mm-passage {
  border-top: 1px solid var(--rule); padding-top: var(--s4); margin-top: var(--s4);
}
$md .mm-passage-head {
  display: flex; align-items: baseline; gap: var(--s2);
  margin: 0 0 var(--s2); flex-wrap: wrap;
}
$md .mm-rank {
  font-family: var(--font-mono); font-size: .66rem; font-weight: 500;
  color: var(--paper); background: var(--ink-3); border-radius: 2px;
  padding: .1rem .34rem; flex: none;
}
$md .mm-passage-doc { font-size: .84rem; font-weight: 600; color: var(--ink); }
$md .mm-passage-loc { font-size: .84rem; color: var(--ink-2); }
$md .mm-passage-scores {
  font-family: var(--font-mono); font-size: .68rem; color: var(--ink-3);
  margin: 0 0 var(--s2); letter-spacing: .01em;
}
$md .mm-passage-text {
  font-size: .86rem; line-height: 1.6; color: var(--ink-2);
  white-space: pre-wrap; margin: 0; padding-left: var(--s3);
  border-left: 1px solid var(--rule);
}

/* --- Zero state -------------------------------------------------------- */
$md .mm-lede {
  font-family: var(--font-display); font-size: 1.85rem; font-weight: 400;
  line-height: 1.26; letter-spacing: -.008em; color: var(--ink);
  max-width: 22ch; margin: 0 0 var(--s4);
}
$md .mm-sub { font-size: .95rem; line-height: 1.65; color: var(--ink-2); max-width: 62ch; margin: 0; }
$md .mm-label {
  font-family: var(--font-mono); font-size: .68rem; font-weight: 500;
  text-transform: uppercase; letter-spacing: .13em; color: var(--ink-3);
  margin: var(--s6) 0 var(--s3); line-height: 1.4;
}
$md .mm-steps { list-style: none; padding: 0; margin: 0; counter-reset: step; }
$md .mm-steps li {
  counter-increment: step; position: relative; padding-left: 2.1rem;
  margin: 0 0 var(--s3); font-size: .92rem; line-height: 1.6; color: var(--ink-2);
}
$md .mm-steps li::before {
  content: counter(step); position: absolute; left: 0; top: .1rem;
  font-family: var(--font-mono); font-size: .7rem; color: var(--ink-3);
  border: 1px solid var(--rule); border-radius: 50%;
  width: 1.3rem; height: 1.3rem; display: grid; place-items: center;
}
$md .mm-steps b { color: var(--ink); font-weight: 600; }

/* --- Buttons ----------------------------------------------------------- */
/* Text, a hairline and one accent -- no fills, no shadows, no icons. */
.stButton > button, [data-testid="stBaseButton-secondary"] {
  font-family: var(--font-body); font-weight: 500;
  color: var(--ink-2); background: transparent;
  border: 1px solid var(--rule); border-radius: 3px;
  padding: .42rem .8rem; transition: border-color .12s, color .12s;
  box-shadow: none;
}
.stButton > button:hover, [data-testid="stBaseButton-secondary"]:hover {
  border-color: var(--accent); color: var(--accent); background: transparent;
}
.stButton > button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.stButton > button p {
  font-size: .86rem !important; color: inherit !important; white-space: nowrap;
}

/* --- Live stage line ---------------------------------------------------- */
/* Motion reports state and nothing else: the stage the pipeline has actually
   reached. No spinner, no pulse, no progress theatre. */
$md .mm-stages {
  font-family: var(--font-mono); font-size: .72rem; letter-spacing: .06em;
  text-transform: uppercase; color: var(--ink-3); margin: var(--s5) 0 0;
}
$md .mm-stages .now { color: var(--accent); font-weight: 500; }
$md .mm-stages .todo { color: #CFC8BA; }

/* --- Sidebar ----------------------------------------------------------- */
[data-testid="stSidebar"] {
  background: var(--paper-2); border-right: 1px solid var(--rule);
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: var(--s7); }
[data-testid="stSidebarCollapseButton"] button { color: var(--ink-3); }
$md .mm-wordmark { display: flex; align-items: center; gap: .55rem; margin: 0 0 var(--s5); }
$md .mm-wordmark svg { width: 22px; height: 22px; flex: none; }
$md .mm-wordmark span {
  font-family: var(--font-display); font-size: 1.24rem; font-weight: 500;
  letter-spacing: -.01em; color: var(--ink);
}
$md .mm-side-note { font-size: .84rem; line-height: 1.58; color: var(--ink-2); margin: 0 0 var(--s5); }
$md .mm-stats { margin: 0; }
$md .mm-stat {
  display: flex; justify-content: space-between; gap: var(--s3);
  padding: .4rem 0; border-bottom: 1px solid var(--rule);
}
$md .mm-stat:first-child { border-top: 1px solid var(--rule); }
$md .mm-stat dt { font-size: .8rem; color: var(--ink-2); margin: 0; }
$md .mm-stat dd { font-family: var(--font-mono); font-size: .8rem; color: var(--ink); margin: 0; }

/* --- Composer ---------------------------------------------------------- */
[data-testid="stBottomBlockContainer"] {
  max-width: 52rem; margin-left: 0;
  padding-left: var(--s7); padding-right: var(--s5); background: var(--paper);
}
@media (max-width: 1100px) {
  [data-testid="stBottomBlockContainer"] { padding-left: var(--s5); }
}
[data-testid="stBottom"] > div { background: var(--paper); }
[data-testid="stChatInput"] {
  background: #FFFFFF; border: 1px solid var(--rule); border-radius: 3px;
}
[data-testid="stChatInput"]:focus-within { border-color: var(--accent); }
[data-testid="stChatInput"] textarea { font-size: .92rem; color: var(--ink); }
[data-testid="stChatInput"] textarea::placeholder { color: var(--ink-3); }
[data-testid="stChatInputSubmitButton"] { color: var(--ink-3); }
[data-testid="stChatInputSubmitButton"]:hover { color: var(--accent); }
</style>
"""

CSS = _STYLESHEET.replace("$md ", _MD + " ")

# The window the meter draws. In-scope questions bottom out near 0.46 and
# out-of-scope ones top out near 0.52 (see results/threshold_calibration.md),
# so a 0-0.8 scale puts the whole decision in the middle of the rule where it
# can be read, rather than squashing it against one end of a 0-1 axis.
_GATE_MIN, _GATE_MAX = 0.0, 0.80
_GATE_W, _GATE_PAD = 268.0, 5.0


def _gate_x(value: float) -> float:
    span = _GATE_MAX - _GATE_MIN
    frac = min(max((value - _GATE_MIN) / span, 0.0), 1.0)
    return _GATE_PAD + frac * (_GATE_W - 2 * _GATE_PAD)


def gate_meter(top_score: float, threshold: float, cleared: bool) -> str:
    """The refusal decision, drawn.

    A rule from 0 to 0.8 cosine, the calibrated threshold notched on it, and
    the best passage this question actually found marked against it. This is
    the number that decides whether a student is answered or refused, so it is
    shown either way rather than only when it is flattering.
    """
    x_score = _gate_x(top_score)
    x_gate = _gate_x(threshold)
    colour = "var(--accent)" if cleared else "var(--flag)"
    margin = abs(top_score - threshold)
    verdict = (
        f'<span class="cleared">cleared by {margin:.3f}</span>' if cleared
        else f'<span class="short">short by {margin:.3f}</span>'
    )
    return f"""<div class="mm-gate">
<svg width="{_GATE_W:.0f}" height="24" viewBox="0 0 {_GATE_W:.0f} 24"
     fill="none" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Best passage scored {top_score:.3f} against a gate of {threshold:.2f}">
  <line x1="{_GATE_PAD}" y1="14" x2="{_GATE_W - _GATE_PAD}" y2="14"
        stroke="var(--rule-strong)" stroke-width="1"/>
  <line x1="{_GATE_PAD}" y1="10" x2="{_GATE_PAD}" y2="18"
        stroke="var(--rule-strong)" stroke-width="1"/>
  <line x1="{_GATE_W - _GATE_PAD}" y1="10" x2="{_GATE_W - _GATE_PAD}" y2="18"
        stroke="var(--rule-strong)" stroke-width="1"/>
  <line x1="{_GATE_PAD}" y1="14" x2="{x_score:.1f}" y2="14"
        stroke="{colour}" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="{x_gate:.1f}" y1="3" x2="{x_gate:.1f}" y2="21"
        stroke="var(--ink-2)" stroke-width="1"/>
  <circle cx="{x_score:.1f}" cy="14" r="3.4" fill="{colour}"/>
</svg>
<p class="mm-gate-read">best passage <b>{top_score:.3f}</b> &middot;
gate {threshold:.2f} &middot; {verdict}</p>
</div>"""


def stage_line(stages: list[str], current: int) -> str:
    """The pipeline stage actually reached, for the wait between turns."""
    parts = []
    for i, name in enumerate(stages):
        cls = "done" if i < current else ("now" if i == current else "todo")
        parts.append(f'<span class="{cls}">{name}</span>')
    return '<p class="mm-stages">' + '<span class="todo"> &rarr; </span>'.join(parts) + "</p>"


def speaker(name: str, state: str | None = None) -> str:
    """The speaker label above a turn. `state` is "refusal", "error" or None."""
    cls = "mm-speaker" + (f" is-{state}" if state else "")
    return f'<p class="{cls}">{name}</p>'


def turn_anchor(kind: str) -> str:
    """Marks the block so CSS can colour the rule down its left edge.

    Streamlit gives its containers no stable class of their own, so the block
    is selected by what it contains (`:has`) rather than by a class we cannot
    set.
    """
    return f'<span class="mm-anchor mm-turn mm-turn-{kind}"></span>'
