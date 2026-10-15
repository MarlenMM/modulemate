"""Transparent dialogue planning and source-bound local answer composition."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache

import snowballstemmer

from modulemate.local_models import ModelUnavailable, normalized_vectors
from modulemate.search import CJK, tokens

SOCIAL = {
    "hello": "Hi! Ask me about a concept in your materials, compare two ideas, or ask me to quiz you.",
    "hi": "Hi! What would you like to work through in your materials?",
    "hey": "Hi! What would you like to work through in your materials?",
    "thanks": "You're welcome. We can go deeper, compare related ideas, or try a recall question.",
    "thank you": "You're welcome. We can go deeper, compare related ideas, or try a recall question.",
    "goodbye": "See you next time. Your conversation and study cards are saved in this course.",
    "help": "I can answer questions from your materials, compare two concepts, make a study outline, or guide recall practice. Try ‘What is …?’, ‘Compare … and …’, or ‘Quiz me on …’.",
    "what can you do": "I can answer questions from your materials, compare two concepts, make a study outline, or guide recall practice. Try ‘What is …?’, ‘Compare … and …’, or ‘Quiz me on …’.",
    "who are you": "I'm ModuleMate, your study assistant. I use the materials in this course and show the evidence behind my answers.",
}


def social_reply(question):
    value = re.sub(r"[^\w\s]", "", question.casefold()).strip()
    return SOCIAL.get(value)


def comparison_subjects(question):
    clean = question.strip().rstrip("?.!")
    patterns = [
        r"(?:difference(?:s)? between|compare)\s+(.+?)\s+(?:and|with|to)\s+(.+)",
        r"(.+?)\s+(?:versus|vs\.?)\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean, re.I)
        if match:
            subjects = [
                re.sub(r"^(?:please\s+)?(?:compare\s+)?", "", part, flags=re.I).strip()
                for part in match.groups()
            ]
            if all(1 <= len(s) <= 100 for s in subjects):
                return subjects
    return []


@dataclass
class Plan:
    question: str
    query: str
    mode: str
    topic: str = ""
    subjects: list[str] = field(default_factory=list)
    clarification: str = ""
    used_context: bool = False
    overview: bool = False
    context_ids: list[int] = field(default_factory=list)
    intent: str = "conceptual"
    review_attempt: bool = False


def plan_turn(question, mode, history):
    selected = mode
    if mode == "Auto":
        if re.search(
            r"\b(compare|versus|vs\.?|difference between|differences between)\b", question, re.I
        ):
            selected = "Compare"
        elif re.search(r"\b(summari[sz]e|summary|overview|outline)\b", question, re.I):
            selected = "Summarize"
        elif re.search(r"\b(quiz me|test me|hint|socratic|practice question)\b", question, re.I):
            selected = "Socratic tutor"
        else:
            selected = "Explain"
    plan = Plan(question, question, selected, subjects=comparison_subjects(question))
    referential = bool(
        re.search(
            r"\b(it|its|this|that|these|those|they|them|first one|second one|more|again)\b",
            question,
            re.I,
        )
    )
    followup = bool(
        re.fullmatch(
            r"(?:please )?(?:go deeper|give (?:me )?an example|another example|quiz me|test me|why|why not|how so|continue)[.!? ]*",
            question,
            re.I,
        )
    )
    # Avoid dragging the previous subject into a self-contained new question.
    meaningful = [
        t
        for t in tokens(question)
        if t
        not in {
            "more",
            "again",
            "simpler",
            "simply",
            "detail",
            "details",
            "example",
            "examples",
            "second",
            "first",
            "one",
            "those",
            "them",
            "they",
            "go",
            "deeper",
            "quiz",
            "test",
            "mean",
            "means",
            "work",
            "works",
            "shorter",
        }
    ]
    contextual = followup or (referential and len(meaningful) <= 2)
    grounded = [m for m in history if m.get("sources")]
    previous_mode = grounded[-1].get("metadata", {}).get("resolved_mode") if grounded else None
    if (
        mode == "Auto"
        and previous_mode == "Socratic tutor"
        and "?" not in question
        and re.match(
            r"^(?:I think|my answer|my explanation|it means|it is|I don't know|not sure|maybe)\b",
            question,
            re.I,
        )
    ):
        plan.review_attempt = True
        contextual = True
        selected = plan.mode = "Explain"
    if contextual:
        if not grounded:
            plan.clarification = (
                "Which concept do you mean? Name it or ask a first question about your materials."
            )
            return plan
        previous = grounded[-1]
        meta = previous.get("metadata", {})
        topic = meta.get("topic") or meta.get("query") or previous["question"]
        previous_subjects = meta.get("subjects", [])
        ordinal = re.search(r"\b(first|second)\s+(?:one|concept|idea|method)\b", question, re.I)
        if ordinal:
            index = 0 if ordinal[1].lower() == "first" else 1
            if len(previous_subjects) <= index:
                plan.clarification = (
                    "Which item do you mean? Name the concept so I can use the right evidence."
                )
                return plan
            topic = previous_subjects[index]
        if previous_subjects and not ordinal and re.search(r"\b(it|that|this)\b", question, re.I):
            plan.clarification = f"Do you mean {previous_subjects[0]} or {previous_subjects[1]}?"
            return plan
        plan.query = (
            topic[:1000]
            if ordinal or not meaningful or followup or plan.review_attempt
            else f"{topic[:1000]} — {question}"
        )
        plan.topic = topic[:1000]
        plan.used_context = True
        if not tokens(topic):
            plan.context_ids = [p["id"] for p in previous["sources"] if type(p.get("id")) is int]
        if selected == "Compare" and not plan.subjects:
            plan.subjects = previous_subjects
    else:
        plan.topic = re.sub(
            r"^(?:what (?:is|are)|explain|summari[sz]e|quiz me on|help me understand)\s+",
            "",
            question,
            flags=re.I,
        ).rstrip("?.!")[:1000]
    if selected == "Compare" and not plan.subjects:
        plan.clarification = "Which two ideas would you like to compare? Try ‘Compare X and Y’."
    overview_words = {
        "all",
        "everything",
        "main",
        "key",
        "points",
        "whole",
        "entire",
        "document",
        "documents",
        "library",
        "overview",
        "outline",
        "a",
        "an",
    }
    plan.overview = (
        selected == "Summarize"
        and not (set(tokens(question)) - overview_words)
        and not plan.used_context
    )
    return plan


@dataclass(frozen=True)
class Sentence:
    text: str
    source: int
    position: int
    heading: str = ""


def sentences(passages):
    result = []
    for source, passage in enumerate(passages, 1):
        heading = ""
        for block in re.split(r"\n\s*\n", passage.text):
            lines = block.strip().splitlines()
            while lines and re.match(r"^\s*#{1,6}\s+", lines[0]):
                heading = lines.pop(0).lstrip("# ").strip()
            if not lines:
                continue
            text = "\n".join(lines)
            text = re.sub(r"(?<!\n)\n(?!\s*[-*])", " ", text)
            for part in re.split(r"(?<=[.!?])\s+|(?<=[。！？])|\n(?=\s*[-*])", text):
                part = part.strip().lstrip("*- ").strip()
                if not (8 if CJK.search(part) else 20) <= len(part) <= 700 or len(tokens(part)) < 3:
                    continue
                # Skip English overlap fragments left by legacy chunk boundaries.
                if part[0].isascii() and part[0].islower() and len(part.split()) < 6:
                    continue
                result.append(Sentence(part, source, len(result), heading))
    return result


@lru_cache(maxsize=1)
def _stemmer():
    return snowballstemmer.stemmer("english")


def stems(text):
    words = tokens(text)
    return {_stemmer().stemWord(w) if w.isascii() and w.isalpha() else w for w in words}


def _lexical_scores(query, candidates):
    terms = stems(query)
    doc_terms = [stems(s.text) for s in candidates]
    idf = {t: math.log(1 + len(candidates) / (1 + sum(t in d for d in doc_terms))) for t in terms}
    return [sum(idf[t] for t in terms & d) / (1 + 0.015 * len(d)) for d in doc_terms]


def _detail_pattern(query):
    if re.search(r"시간|几点|时间", query):
        return r"오전|오후|\d|[一二三四五六七八九十]+点"
    if re.search(r"\b(deadline|due date|what date|what time)\b", query, re.I):
        return r"\b(deadline|due|\d{1,2}:\d{2}|20\d{2})\b"
    if re.search(r"\b(worth|weighting|grade|mark|percentage|percent)\b", query, re.I):
        return r"%|\b(percent|percentage|weight|mark|grade)\b"
    if re.search(r"\b(how long|length|word count|how many words)\b", query, re.I) and re.search(
        r"\b(report|write.up|essay|document|paper)\b", query, re.I
    ):
        return r"\b(words?|pages?|characters?)\b"
    if re.search(r"\b(how long|how much time|duration)\b", query, re.I):
        return r"\b(seconds?|minutes?|hours?|days?|weeks?)\b"
    if re.search(r"\b(operations?)\b", query, re.I):
        return r"\b(push|pop|enqueue|dequeue|insert|delete|operations?)\b"
    if re.search(r"\b(extension|extensions|extend)\b", query, re.I):
        return r"\b(extensions?|extend|permission)\b"
    if re.search(r"\b(late|after.{0,30}due|miss.{0,20}deadline)\b", query, re.I):
        return r"\b(late|penalt\w*|lose|deduct\w*)\b"
    if re.search(r"\b(after|following)\b", query, re.I):
        return r"\b(after|then|next)\b"
    if re.search(r"\b(reserve|reserved)\b", query, re.I):
        return r"\b(reserve|reserved)\b"
    return ""


def _order_conflict(query, text):
    newest = r"\b(newest|most recent(?:ly)?(?: inserted)?|last inserted|latest)\b"
    oldest = r"\b(oldest|earliest(?: inserted)?|first inserted)\b"
    q_new, q_old = bool(re.search(newest, query, re.I)), bool(re.search(oldest, query, re.I))
    s_new, s_old = bool(re.search(newest, text, re.I)), bool(re.search(oldest, text, re.I))
    return (q_new and not q_old and s_old and not s_new) or (
        q_old and not q_new and s_new and not s_old
    )


def rank_sentences(query, candidates, engine=None):
    if not candidates:
        return []
    lexical = _lexical_scores(query, candidates)
    ordered = sorted(range(len(candidates)), key=lambda i: (-lexical[i], candidates[i].position))[
        :48
    ]
    for source in dict.fromkeys(s.source for s in candidates):
        for i, sentence in enumerate(candidates):
            if sentence.source == source and i not in ordered:
                ordered.append(i)
                break
    ordered = ordered[:64]
    # Resolve pronouns for ranking by including the first source sentence. The
    # displayed evidence remains exact source text, never this constructed input.
    anchors = {
        source: next(s.text for s in candidates if s.source == source)
        for source in dict.fromkeys(s.source for s in candidates)
    }
    contextual = [
        (
            (anchors[candidates[i].source] + " ")
            if re.match(
                r"^(?:It|They|Their|This|That|These|Those|In|After|Before|Without|With)\b",
                candidates[i].text,
            )
            else ""
        )
        + candidates[i].text
        for i in ordered
    ]
    pattern = _detail_pattern(query)
    if engine and not CJK.search(query):
        try:
            values = engine.rerank(query[:2000], contextual)
            if len(values) != len(ordered) or not all(math.isfinite(v) for v in values):
                raise ModelUnavailable("Invalid sentence scores")
            vectors = normalized_vectors(
                engine.encode([query[:2000]] + contextual), len(ordered) + 1, engine.dimensions
            )
            similarities = vectors[1:] @ vectors[0]
            max_lexical = max(lexical) or 1
            combined = []
            for i, score, similarity in zip(ordered, values, similarities, strict=True):
                if similarity < 0.25 and score < -7 and lexical[i] == 0:
                    continue
                value = (
                    0.5 * float(similarity)
                    + 0.15 / (1 + math.exp(-float(score) / 3))
                    + 0.1 * lexical[i] / max_lexical
                    + 0.25 / candidates[i].source
                )
                if pattern:
                    value += 0.4 if re.search(pattern, candidates[i].text, re.I) else -0.15
                if _order_conflict(query, contextual[ordered.index(i)]):
                    value -= 0.5
                if re.search(
                    r"\b(newest|oldest|most recent|earliest|latest)\b", query, re.I
                ) and not re.search(
                    r"\b(newest|oldest|most recent|earliest|latest)\b", candidates[i].text, re.I
                ):
                    value -= 0.25
                combined.append((candidates[i], value))
            return sorted(combined, key=lambda x: (-x[1], x[0].position))
        except (ModelUnavailable, ValueError, RuntimeError):
            pass
    ranking = []
    for i in ordered:
        if lexical[i] > 0:
            value = lexical[i]
            if pattern and re.search(pattern, candidates[i].text, re.I):
                value += max(lexical)
            if _order_conflict(query, contextual[ordered.index(i)]):
                value *= 0.2
            ranking.append((candidates[i], value))
    return sorted(ranking, key=lambda x: (-x[1], x[0].position))


def select_distinct(ranked, count):
    chosen, seen = [], []
    for sentence, _score in ranked:
        words = set(tokens(sentence.text))
        if any(len(words & prior) / max(1, len(words | prior)) > 0.72 for prior in seen):
            continue
        chosen.append(sentence)
        seen.append(words)
        if len(chosen) == count:
            break
    return chosen


def _requested_detail(question, candidates):
    """Conservative slot checks: topical relevance is not an answer to a detail."""
    checks = [
        (
            r"\b(why|reason|cause)\b",
            r"\b(because|therefore|due to|caus\w*|reason\w*|so that|leads? to|result\w*|prevent\w*|reduc\w*)\b",
        ),
        (r"\b(example|examples)\b", r"\b(example|e\.g\.|such as|suppose|consider|for instance)\b"),
        (
            r"\b(who invented|inventor|who discovered)\b",
            r"\b(invent\w*|discover\w*|developed by|introduced by)\b",
        ),
        (
            r"\b(deadline|due date|what (?:date|time|day)|when (?:is|are|was|were|does|do|will))\b",
            r"\b(\d|one|two|three|four|five|six|seven|eight|nine|ten|day|days|week|weeks|month|months|year|years|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december|deadline|due)",
        ),
    ]
    for pattern, evidence in checks:
        if re.search(pattern, question, re.I) and not any(
            re.search(evidence, s.text, re.I) for s in candidates
        ):
            return False
    return True


def compose_local(plan, passages, engine=None):
    """Return prose, suggestions, and a limitation; content is extractive, not LLM."""
    pool = sentences(passages)
    if not pool:
        return (
            "I found related material, but no complete passage suitable for a short answer. Check the sources below or add clearer notes.",
            [],
            "No complete answer extracted.",
        )
    cite = lambda s: f"{s.text} [{s.source}]"  # noqa: E731
    if plan.mode == "Compare":
        blocks, used = [], set()
        for subject in plan.subjects:
            ranked = rank_sentences(subject, pool, engine)
            # A sentence about only the other subject must not fill this side.
            specific = [
                (s, score) for s, score in ranked if set(tokens(subject)) & set(tokens(s.text))
            ]
            chosen = select_distinct(specific, 2)
            fresh = [s for s in chosen if s.text not in used]
            if fresh:
                blocks.append(f"**{subject}**\n\n" + " ".join(cite(s) for s in fresh))
                used.update(s.text for s in fresh)
            elif chosen:
                blocks.append(
                    f"**{subject}**\n\nThe same passage discusses both ideas: " + cite(chosen[0])
                )
            else:
                blocks.append(
                    f"**{subject}**\n\nI couldn't find enough evidence for this side of the comparison."
                )
        return (
            "\n\n".join(blocks),
            [f"Explain {s}" for s in plan.subjects],
            "Side-by-side source statements; no unstated differences inferred.",
        )
    if plan.mode == "Summarize":
        if plan.overview:
            # Centroid similarity provides a lightweight extractive summarizer,
            # with round-robin source coverage so one document cannot dominate.
            from sklearn.feature_extraction.text import TfidfVectorizer

            try:
                matrix = TfidfVectorizer(stop_words="english").fit_transform(s.text for s in pool)
                centrality = (matrix @ matrix.mean(axis=0).T).A1
            except ValueError:
                centrality = [0] * len(pool)
            ranked = sorted(
                zip(pool, centrality, strict=True), key=lambda x: (-x[1], x[0].position)
            )
        else:
            ranked = rank_sentences(plan.query, pool, engine)
        chosen = []
        sources = list(dict.fromkeys((s.source, s.heading) for s, _ in ranked))
        for source, heading in sources:
            group = [
                (
                    s,
                    score
                    + 0.3
                    * len(set(tokens(s.heading)) & set(tokens(s.text)))
                    / max(1, len(tokens(s.heading))),
                )
                for s, score in ranked
                if s.source == source and s.heading == heading
            ]
            group.sort(key=lambda item: (-item[1], item[0].position))
            chosen.extend(select_distinct(group, 1))
            if len(chosen) == 6:
                break
        if len(chosen) < 6:
            chosen = select_distinct([(s, 1) for s in chosen] + ranked, 6)
        if not chosen:
            return (
                "I couldn't find enough evidence for a summary of that topic.",
                [],
                "Try a narrower topic or another document.",
            )
        return (
            "Here are the main points in the selected passages:\n\n"
            + "\n\n".join("- " + cite(s) for s in chosen),
            ["Quiz me on that"],
            "Extractive outline of selected passages, not a full-document summary.",
        )
    ranked = rank_sentences(plan.query, pool, engine)
    reader_notice = ""
    if (
        plan.mode == "Explain"
        and engine
        and hasattr(engine, "extract")
        and not CJK.search(plan.query)
    ):
        try:
            # Choose the relevant source before reading for an answer. QA logits
            # are not comparable confidence values across unrelated paragraphs.
            best_source = ranked[0][0].source if ranked else 0
            context = passages[best_source - 1].text if best_source else ""
            spans = engine.extract(plan.query, [context]) if context else []
            hints = []
            for span in spans:
                if (
                    type(span.source) is int
                    and span.source == 1
                    and 0 <= span.start < span.end <= len(context)
                    and context[span.start : span.end] == span.text
                ):
                    hints.append((best_source, " ".join(span.text.split())))
            adjusted = []
            for sentence, score in ranked:
                if any(
                    source == sentence.source and text in " ".join(sentence.text.split())
                    for source, text in hints
                ):
                    score += 0.18
                adjusted.append((sentence, score))
            ranked = sorted(adjusted, key=lambda item: (-item[1], item[0].position))
        except (ModelUnavailable, ValueError, TypeError, AttributeError):
            reader_notice = "Local answer extraction was unavailable. "
    chosen = select_distinct(ranked, 2 if plan.intent == "admin" else 3)
    if not chosen:
        return (
            "I found related material, but it doesn't give a clear answer to that question. Try naming the concept or add the missing material.",
            [],
            "Related sources are available below.",
        )
    if plan.review_attempt:
        return (
            "Check your explanation against this source statement:\n\n"
            + cite(chosen[0])
            + "\n\nWhat would you keep or change after comparing them?",
            [f"Explain {plan.topic[:300]}", "Quiz me on that"],
            "Source comparison for self-assessment; your explanation has not been automatically graded.",
        )
    if plan.mode == "Socratic tutor":
        # Ask for recall without disclosing the answer. This is not auto-grading.
        first = chosen[0]
        words = tokens(first.text)
        topic = first.heading or (" ".join(words[:3]) if not tokens(plan.topic) else plan.topic)
        return (
            (
                f"Let's work through it. Before opening the source, how would you explain the idea behind ‘{topic[:160]}’ in your own words?\n\n"
                f"Use the passage at [{first.source}] to check your explanation afterward. Look for what the concept does and any conditions the notes attach to it."
            ),
            [f"Explain {plan.topic[:300]}"],
            "Recall prompt based on your source; this does not grade your answer.",
        )
    if not _requested_detail(plan.question, chosen):
        return (
            (
                "The related notes don't establish the specific detail you're asking for. What they do say is:\n\n"
                + cite(chosen[0])
            ),
            [],
            "Partial evidence only; the requested detail is not confirmed.",
        )
    primary = chosen[0]
    support = [primary]
    source_sentences = [
        s
        for s in pool
        if s.source == primary.source and (not primary.heading or s.heading == primary.heading)
    ]
    if re.match(
        r"^(?:It|They|Their|This|That|These|Those|In (?:eukaryotic|these|those))\b", primary.text
    ):
        anchor = source_sentences[0]
        if anchor != primary:
            support.insert(0, anchor)
    # Questions often need an antecedent or a second condition from the SAME
    # short source paragraph. Preserve these instead of splicing unrelated facts.
    broad = bool(re.match(r"^(?:explain|describe|what (?:is|are)|define)\b", plan.question, re.I))
    if broad and not _detail_pattern(plan.question):
        for candidate in source_sentences:
            if len(support) >= 3:
                break
            if (
                candidate not in support
                and len(" ".join(s.text for s in support + [candidate])) <= 450
            ):
                support.append(candidate)
    return (
        " ".join(cite(s) for s in support),
        ["Quiz me on that", "Summarize that"],
        reader_notice
        + "Selected source sentences, composed locally. Open sources to check context.",
    )
