"""Question-aware context assembly rather than a fixed top-eight excerpt list."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from modulemate.dialogue import stems
from modulemate.evidence import source_notation_limits
from modulemate.requests import material_overview
from modulemate.retrieval import RetrievalIndex, retrieve, scoped_passages
from modulemate.search import Passage, search, tokens


@dataclass
class Context:
    passages: list[Passage] = field(default_factory=list)
    complete: bool = False
    total_passages: int = 0
    characters: int = 0
    method: str = "word matching"
    notice: str = ""
    catalog: list[dict] = field(default_factory=list)
    clarification: str = ""
    awaiting_document: bool = False
    lesson: dict = field(default_factory=dict)
    lesson_session: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    policy: str = "baseline"
    links: list[dict] = field(default_factory=list)
    reading_limits: list[dict] = field(default_factory=list)
    recovery_candidate: Context | None = field(default=None, repr=False)


def estimate_tokens(text):
    cjk = len(re.findall(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))
    return (len(text) - cjk) / 3 + cjk * 1.5


def document_mentioned(name, question):
    stem = Path(name).stem
    if name.casefold() in question.casefold() or (
        len(stem) >= 3 and re.search(r"(?<!\w)" + re.escape(stem) + r"(?!\w)", question, re.I)
    ):
        return True
    lecture = re.match(r"lec(?:ture)?[ _.-]*0*(\d+)(?!\d)", stem, re.I)
    return bool(
        lecture
        and re.search(r"\blec(?:ture)?\s*0*" + str(int(lecture[1])) + r"(?!\d)\b", question, re.I)
    )


def document_catalog(groups):
    result = []
    per_document_budget = 1800 / max(1, len(groups))
    for group in groups.values():
        outline, seen = [], set()
        for p in group:
            if "›" in p.locator:
                title = p.locator.rsplit("›", 1)[-1].strip()
            elif " · " in p.locator:
                title = p.locator.split(" · ", 1)[-1].strip()
            else:
                title = next(
                    (line.strip("# \t") for line in p.text.splitlines() if line.strip()), ""
                )[:90]
            if title and title.casefold() not in seen:
                outline.append({"location": p.locator, "heading": title})
                seen.add(title.casefold())
        average_cost = sum(
            estimate_tokens(json.dumps(e, ensure_ascii=False)) for e in outline
        ) / max(1, len(outline))
        limit = max(1, min(48, int(per_document_budget / max(1, average_cost))))
        if len(outline) > limit:
            positions = [round(i * (len(outline) - 1) / max(1, limit - 1)) for i in range(limit)]
            selected = [outline[i] for i in positions]
        else:
            selected = outline
        result.append(
            {
                "document": group[0].name,
                "passages": len(group),
                "outline": selected,
                "outline_complete": len(selected) == len(outline),
            }
        )
    return result[:12]


def topic_heading(passage):
    if "›" in passage.locator:
        return passage.locator.rsplit("›", 1)[-1].strip()
    if " · " in passage.locator:
        return passage.locator.split(" · ", 1)[-1].strip()
    return next((s.strip("# \t") for s in passage.text.splitlines() if s.strip()), "")[:90]


def referenced_sections(candidates, groups, store, course_id, question, engine=None):
    """Follow at most two explicit filenames already inside the allowed scope.

    This is an evidence graph over uploaded documents, never filesystem access,
    a URL fetch or an instruction executor. It helps when a relevant page points
    to a second file that a keyword filter or an overconfident scorer overlooked.
    """
    present = {p.document_id for p in candidates}
    names = Counter(group[0].name.casefold() for group in groups.values())
    links, passages = [], []
    for source in candidates[:8]:
        for document_id, group in groups.items():
            if document_id in present:
                continue
            name = group[0].name
            if names[name.casefold()] != 1:
                continue  # A filename alone cannot disambiguate two uploads.
            mention = re.search(
                r"(?<![\w.-])" + re.escape(name) + r"(?![\w-]|\.[\w-])", source.text, re.I
            )
            if mention is None or not re.search(
                r"\b(?:see|read|refer|consult|section|appendix)\b",
                source.text[max(0, mention.start() - 180) : mention.start()],
                re.I,
            ):
                continue
            selected = [
                p
                for p in group
                if len(topic_heading(p)) >= 5
                and topic_heading(p).casefold() in source.text.casefold()
            ]
            if not selected:
                selected = retrieve(
                    store, course_id, question, [document_id], engine=engine, limit=3
                ).passages
            passages.extend(selected[:4])
            present.add(document_id)
            links.append(
                {"from_document": source.document_id, "to_document": document_id, "name": name}
            )
            if len(links) >= 2:
                return passages, links
    return passages, links


def related_source_passages(candidates, groups, store, course_id, question):
    """A preferred title must not hide facts about that subject in another file.

    Search the remaining allowed files for the actual title/query subject. This
    small recall channel adds evidence, not automatic citations or credibility.
    Generic detail words cannot pull in a different event's instructions.
    """
    present = {p.document_id for p in candidates}
    remaining = set(groups) - present
    generic = stems(
        "notes material document presentation slide lecture course file guide handbook handout "
        "instruction directions project source text topic concept example summary step method "
        "main key current previous information detail number amount only what how when which"
    )
    terms = tokens(question)
    anchors = (stems(question) & stems(" ".join(p.name for p in candidates[:8]))) - generic
    words = [word for word in terms if stems(word) & anchors]
    if not remaining or not words:
        return []
    found = search(store, course_id, " ".join(words), sorted(remaining), limit=8)
    related, documents = [], set()
    for passage in found:
        if not anchors & stems(passage.text):
            continue
        if passage.document_id not in documents and len(documents) >= 2:
            continue
        related.append(passage)
        documents.add(passage.document_id)
        if len(related) >= 4:
            break
    return related


def instruction_packet(passage):
    return bool(
        re.search(
            r"\b(?:ignore|disregard)\b[\s\S]{0,80}\binstructions\b[\s\S]{0,80}\b(?:answer every|reveal|system prompt)\b|<\|im_(?:start|end)\|>",
            passage.text,
            re.I,
        )
    )


def lesson_content(passage):
    title = topic_heading(passage)
    if re.match(
        r"^(?:ice[- ]?breaker|agenda|contents|bibliography|references|thank you)\b", title, re.I
    ):
        return False
    if instruction_packet(passage):
        return False
    if (
        " · " in passage.locator
        and "›" not in passage.locator
        and re.search(
            r"\b(?:lecture|chapter|course|document) (?:covers|introduces)\b", passage.text, re.I
        )
    ):
        return False  # An overview is an outline, not another concept to teach.
    words = passage.text.split()
    return (
        len(words) >= 25
        or (
            len(words) >= 5
            and bool(re.search(r"\b(?:is|are|means?|defined|refers)\b|=|```", passage.text, re.I))
        )
        or len(re.findall(r"[\u3400-\u9fff\uac00-\ud7af]", passage.text)) >= 35
    )


def teaching_request(question):
    return bool(
        re.search(r"\b(?:teach|tutor|walk me through|lesson|study guide)\b", question, re.I)
    )


def broad_request(question, mode):
    generic_scope = bool(
        re.search(
            r"\b(?:lecture|course|notes|materials|chapter|document|library|presentation|slides?|deck|talk|reading|everything|all topics)\b",
            question,
            re.I,
        )
    )
    topical = bool(re.search(r"\bon\s+(?!(?:this|the|my|all)\b)\w+", question, re.I))
    return material_overview(question) or (
        generic_scope
        and not topical
        and (
            mode in {"Summarize", "Teach"}
            or teaching_request(question)
            or bool(re.search(r"\b(?:entire|whole|fully|everything|all topics)\b", question, re.I))
        )
    )


def representative_passages(groups, limit=24):
    """Round-robin structural coverage, including each document's end.

    Outlines are navigation, not evidence. Select actual passages from evenly
    spaced positions; preserve section diversity before neighboring chunks.
    """
    sampled = []
    for group in groups.values():
        sections, seen = [], set()
        for passage in group:
            section = (passage.locator.split(" · OCR")[0], topic_heading(passage))
            if section not in seen:
                sections.append(passage)
                seen.add(section)
        count = min(len(sections), max(2, limit // max(1, len(groups))))
        positions = [round(i * (len(sections) - 1) / max(1, count - 1)) for i in range(count)]
        # Establish beginning AND conclusion before filling middle sections.
        positions = list(dict.fromkeys([positions[0], positions[-1], *positions[1:-1]]))
        sampled.append([sections[i] for i in positions])
    return [
        group[i]
        for i in range(max((len(group) for group in sampled), default=0))
        for group in sampled
        if i < len(group)
    ][:limit]


def queries_for(question, plan, history, extra_queries=()):
    # Keep the actual question first: an inferred topic must never replace a correction.
    queries = [question, *extra_queries, plan.query]
    queries.extend(plan.subjects)
    queries.extend(
        part.strip()
        for part in re.split(r"[?;\n]|\b(?:also|and then)\b", question)
        if len(tokens(part)) >= 2
    )
    if plan.used_context and history:
        queries.extend([history[-1]["question"], history[-1].get("metadata", {}).get("topic", "")])
    return list(dict.fromkeys(q.strip() for q in queries if q.strip()))[:6]


def focused_documents(groups, question, plan, *, allow_topic_filter=True, prefer_titles=False):
    """Use distinctive query terms to keep unrelated documents out of context.

    This is a document-level precision step, not a hard-coded subject router.
    Retain the union for comparisons and multi-part questions. Generic request
    verbs and qualifiers cannot establish a document's topic.
    """
    if re.search(r"\bcurrent\b", question, re.I) and not re.search(
        r"\b(?:archiv\w*|retired|superseded|obsolete|previous|histor\w*|earlier)\b",
        question,
        re.I,
    ):
        # Explicitly retired files are evidence about a past version. Do not
        # insert their numbers into a question restricted to the current one.
        groups = {
            key: passages
            for key, passages in groups.items()
            if not re.search(
                r"\b(?:archiv\w*|retired|superseded|obsolete|previous)\b",
                passages[0].name,
                re.I,
            )
        }
    if (
        (not allow_topic_filter and not prefer_titles)
        or len(groups) < 2
        or re.search(
            r"\b(?:all|every|which|across)\b.*\b(?:files?|documents?|notes|materials)\b",
            question,
            re.I,
        )
    ):
        return set(groups)
    generic = stems(
        "need bring give show tell find answer question example simple simpler short sentence definition describe explain useful different same alone independently current previous earlier old new first second version number amount only just actually meant include information following provided selected document file note chapter lecture request help topic concept use mean contain refer please specific exact know based according state detail day year how many much any more most less really should register guide handbook handout instruction directions material source text summary step method"
    )
    terms = stems(question + " " + plan.query + " " + " ".join(plan.subjects)) - generic
    terms = {t for t in terms if len(t) >= 3 and not t.isdigit()}
    if (
        prefer_titles
        and not plan.subjects
        and not re.search(
            r"\b(?:and|also|compare|versus|across)\b|[?;]\s*\S", question.rstrip("? "), re.I
        )
    ):
        # A distinctive topic in an uploaded filename is stronger than a
        # generic body word such as 'measurements'. Keep all matching titles;
        # references may still bring in another allowed file afterwards.
        titled = {
            doc: terms & stems(Path(passages[0].name).stem) for doc, passages in groups.items()
        }
        maximum = max(1, len(groups) // 3)
        anchors = {
            term
            for matches in titled.values()
            for term in matches
            if sum(term in other for other in titled.values()) <= maximum
        }
        if anchors:
            return {doc for doc, matches in titled.items() if matches & anchors}
    if not allow_topic_filter:
        return set(groups)
    matches = {term: set() for term in terms}
    for document_id, passages in groups.items():
        words = stems(" ".join(p.name + " " + p.locator + " " + p.text for p in passages))
        for term in terms & words:
            matches[term].add(document_id)
    maximum = max(1, len(groups) // 3)
    anchors = [ids for ids in matches.values() if 0 < len(ids) <= maximum]
    return set().union(*anchors) if anchors else set(groups)


def build_context(
    store,
    course_id,
    question,
    plan,
    history,
    document_ids=None,
    *,
    engine=None,
    budget=30000,
    extra_queries=(),
    policy="baseline",
    recover=False,
):
    all_passages = scoped_passages(store, course_id, document_ids)
    # An explicit filename is stronger than similarity. This also lets a learner
    # answer a document clarification simply by typing its name.
    named = {p.document_id for p in all_passages if document_mentioned(p.name, question)}
    if named:
        all_passages = [p for p in all_passages if p.document_id in named]
        document_ids = sorted(named)
    deictic_lesson = (teaching_request(question) or plan.mode == "Teach") and bool(
        re.search(
            r"\b(?:this|the) (?:lecture|document|chapter|presentation|deck)\b", question, re.I
        )
    )
    previous_lesson = {}
    for message in reversed(history):
        meta = message.get("metadata", {})
        if "lesson_session" in meta:
            previous_lesson = meta["lesson_session"]
            break
        if meta.get("lesson"):
            previous_lesson = meta["lesson"]
            break
    available_docs = {p.document_id for p in all_passages}
    if (
        not isinstance(previous_lesson, dict)
        or not set(previous_lesson.get("document_ids", [])).issubset(available_docs)
        or re.search(
            r"\b(?:stop|end|cancel|exit) (?:the |this )?(?:lesson|tutoring)\b", question, re.I
        )
    ):
        previous_lesson = {}
    continuation = bool(
        previous_lesson
        and re.fullmatch(
            r"(?:please )?(?:continue(?: (?:the |this |our )?(?:lecture|lesson|teaching|explanation))?|go on|carry on|next(?: part| section| lesson)?)(?: please)?[.!? ]*",
            question,
            re.I,
        )
    )
    if continuation:
        lesson_scope = set(previous_lesson.get("document_ids", []))
        all_passages = [p for p in all_passages if p.document_id in lesson_scope]
        document_ids = sorted(lesson_scope)
    if deictic_lesson and not named and document_ids is None:
        candidates = {p.document_id for p in all_passages}
        previous_docs = (
            {p["document_id"] for p in history[-1].get("sources", [])} if history else set()
        )
        lecture_docs = {
            p.document_id for p in all_passages if re.search(r"(?:lecture|\blec\d)", p.name, re.I)
        }
        inferred = previous_docs if len(previous_docs) == 1 else lecture_docs
        if len(inferred) == 1:
            all_passages = [p for p in all_passages if p.document_id in inferred]
            document_ids = sorted(inferred)
        elif len(candidates) > 1:
            names = list(dict.fromkeys(p.name for p in all_passages))
            return Context(
                awaiting_document=True,
                clarification="Which lecture should we work through? Select a document above, or name it: "
                + ", ".join(names[:12])
                + ".",
            )
    result = Context(
        total_passages=len(all_passages), lesson_session=previous_lesson, policy=policy
    )
    groups = {}
    for passage in all_passages:
        groups.setdefault(passage.document_id, []).append(passage)
    version_scope = focused_documents(groups, question, plan, allow_topic_filter=False)
    if version_scope != set(groups):
        all_passages = [p for p in all_passages if p.document_id in version_scope]
        groups = {doc: group for doc, group in groups.items() if doc in version_scope}
        document_ids = sorted(groups)
        if all_passages:
            result.total_passages = len(all_passages)
    if (
        material_overview(question)
        and not named
        and document_ids is None
        and len(groups) > 1
        and not re.search(r"\b(?:all|across|each|every|compare|between)\b", question, re.I)
    ):
        # A singular deictic document can refer to the previous answer's one
        # document. Do not silently blend unrelated files into its 'main idea'.
        prior = {p["document_id"] for p in history[-1].get("sources", [])} if history else set()
        decks = {
            doc
            for doc, group in groups.items()
            if Path(group[0].name).suffix.casefold() == ".pptx"
            or re.search(r"\b(?:presentation|slides?|deck)\b", Path(group[0].name).stem, re.I)
        }
        inferred = (
            decks if re.search(r"\b(?:presentation|slides?|deck)\b", question, re.I) else prior
        )
        if len(inferred) == 1 and inferred.issubset(groups):
            groups = {doc: group for doc, group in groups.items() if doc in inferred}
            all_passages = [p for group in groups.values() for p in group]
            document_ids = sorted(inferred)
            result.total_passages = len(all_passages)
        else:
            result.clarification = (
                "Which material should I explain? Select a document above or name it: "
                + ", ".join(group[0].name for group in list(groups.values())[:12])
                + ". You can also ask for a connection across all selected materials."
            )
            return result
    result.catalog = document_catalog(groups)
    version_scope = focused_documents(groups, question, plan, allow_topic_filter=False)
    reference_groups = {key: group for key, group in groups.items() if key in version_scope}
    if not all_passages:
        return result
    resumed_selection = bool(
        named and history and history[-1].get("metadata", {}).get("awaiting_document")
    )
    broad = broad_request(question, plan.mode) or plan.overview or continuation or resumed_selection
    full_selection = policy == "full" and not broad
    lesson = (
        ((teaching_request(question) or plan.mode == "Teach") and broad)
        or continuation
        or resumed_selection
    )
    signature = hashlib.sha256(
        "|".join(f"{p.document_id}:{p.id}" for p in all_passages).encode()
    ).hexdigest()
    offset = (
        previous_lesson.get("next_offset", 0)
        if continuation and previous_lesson.get("signature") == signature
        else 0
    )
    if offset >= len(all_passages):
        result.clarification = "This lesson has reached the end of the selected material. Ask about a topic to revisit it, or ask for a practice question."
        return result
    if lesson:
        budget = min(budget, 8000)
    if not broad and not full_selection:
        budget = min(budget, 8500)
        if len(tokens(plan.query)) <= 3:
            budget = min(budget, 3500)
        short_library = policy == "adaptive" and sum(len(p.text) for p in all_passages) <= min(
            budget, 3500
        )
        focus = focused_documents(
            groups,
            question,
            plan,
            allow_topic_filter=not recover and not short_library,
            prefer_titles=policy == "adaptive" and not recover,
        )
        if focus != set(groups):
            all_passages = [p for p in all_passages if p.document_id in focus]
            document_ids = sorted(focus)
            groups = {doc: passages for doc, passages in groups.items() if doc in focus}
            result.catalog = document_catalog(groups)
    ranked = []
    complete_cost = sum(len(p.text) + len(p.name) + len(p.locator) + 80 for p in all_passages)
    token_budget = min(8500, max(1800, 10500 - estimate_tokens(question)))
    complete_tokens = sum(
        estimate_tokens(p.text) + (len(p.name) + len(p.locator) + 80) / 3 for p in all_passages
    )
    fits_complete = (
        complete_cost <= budget and complete_tokens <= token_budget and len(all_passages) <= 48
    )
    bound_context = bool(plan.context_ids and any(p.id in plan.context_ids for p in all_passages))
    queries = (
        []
        if full_selection or lesson or bound_context or (broad and fits_complete)
        else queries_for(question, plan, history, extra_queries)
    )
    if broad and not extra_queries:
        queries = queries[:1]
    index = (
        RetrievalIndex(store, course_id, document_ids, engine, queries)
        if engine and queries
        else None
    )
    for query in queries:
        found = retrieve(
            store,
            course_id,
            query,
            document_ids,
            engine=engine,
            limit=12 if recover else 10 if broad else 6,
            strategy="learned" if policy == "adaptive" else "hybrid",
            _index=index,
        )
        result.method = found.method
        if found.notice:
            result.notice = found.notice
        ranked.append(found.passages)
    # Round robin makes both halves of a question compete fairly for the budget.
    hits = []
    for i in range(10):
        for group in ranked:
            if i < len(group) and group[i] not in hits:
                hits.append(group[i])
    if lesson:
        candidates = all_passages[offset:]
    elif plan.context_ids:
        candidates = [p for p in all_passages if p.id in plan.context_ids]
    elif full_selection:
        candidates = all_passages
        result.method = "full selected documents"
    elif broad:
        # Full short lectures stay in source order. For a long library, sample
        # across documents/locations and explicitly disclose incomplete coverage.
        candidates = (
            all_passages if fits_complete else representative_passages(groups) + hits + all_passages
        )
    else:
        candidates = list(hits)
        # A pronoun often refers to the preceding answer's evidence, not to a
        # word shared with the latest question. Reuse only still-selected IDs.
        if plan.used_context and history:
            prior_ids = {p["id"] for p in history[-1].get("sources", [])}
            candidates = [p for p in all_passages if p.id in prior_ids] + candidates
        # Add neighbors from the SAME document: definitions, caveats, tables and
        # worked examples frequently straddle a chunk or page boundary.
        for hit in hits[:8]:
            group = groups[hit.document_id]
            position = next(i for i, p in enumerate(group) if p.id == hit.id)
            candidates.extend(group[max(0, position - 1) : position + 2])
        # Small selected libraries are cheaper and more reliable to read whole
        # than to discard potentially crucial context with a relevance threshold.
        if sum(len(p.text) for p in all_passages) <= min(budget, 3500):
            candidates = list(all_passages) if policy == "adaptive" else candidates + all_passages
            if policy == "adaptive":
                result.method = "full short selection"
    if policy == "adaptive" and not broad:
        linked, result.links = referenced_sections(
            candidates, reference_groups, store, course_id, question, engine
        )
        if linked:
            candidates = candidates[:4] + linked + candidates[4:]
            for p in linked:
                if p.document_id not in groups:
                    groups[p.document_id] = reference_groups[p.document_id]
                    all_passages.extend(reference_groups[p.document_id])
            result.catalog = document_catalog(groups)
            result.method += " + linked sections"
        companions = related_source_passages(
            candidates, reference_groups, store, course_id, question
        )
        if companions:
            candidates = candidates[:4] + companions + candidates[4:]
            for passage in companions:
                if passage.document_id not in groups:
                    groups[passage.document_id] = reference_groups[passage.document_id]
            result.catalog = document_catalog(groups)
            result.method += " + related source text"
    seen = set()
    estimated_tokens = 0
    consumed = offset
    targets = {}
    substantive = {p.id for p in all_passages if lesson_content(p)} if lesson else set()
    # Very short notes can still be useful. Preserve them when a document has
    # no longer conceptual sections, rather than treating length as authority.
    if lesson and not substantive:
        substantive = {
            p.id for p in all_passages if len(p.text.split()) >= 3 and not instruction_packet(p)
        }
    for position, passage in enumerate(candidates, offset):
        if passage.id in seen:
            continue
        target = (passage.document_id, topic_heading(passage))
        if lesson and passage.id not in substantive:
            consumed = position + 1
            continue
        if lesson and target not in targets and len(targets) >= 3:
            break
        cost = len(passage.text) + len(passage.name) + len(passage.locator) + 80
        token_cost = estimate_tokens(passage.text) + (cost - len(passage.text)) / 3
        if (
            result.characters + cost > budget
            or estimated_tokens + token_cost > token_budget
            or len(result.passages) >= (12 if lesson else 48)
        ):
            if lesson:
                break
            continue
        result.passages.append(passage)
        if lesson:
            targets.setdefault(target, []).append(len(result.passages))
            consumed = position + 1
        result.characters += cost
        estimated_tokens += token_cost
        seen.add(passage.id)
    result.complete = len(seen) == result.total_passages
    if not broad:
        used_documents = {p.document_id for p in result.passages}
        result.links = [link for link in result.links if link["to_document"] in used_documents]
        result.catalog = [
            item
            for item in result.catalog
            if any(groups[doc][0].name == item["document"] for doc in used_documents)
        ]
    if lesson:
        while consumed < len(all_passages) and all_passages[consumed].id not in substantive:
            consumed += 1
        result.lesson = {
            "document_ids": list(groups),
            "signature": signature,
            "start_offset": offset,
            "next_offset": consumed,
            "total_passages": len(all_passages),
            "complete": consumed == len(all_passages),
            "targets": [{"topic": title, "source_ids": ids} for (_, title), ids in targets.items()],
        }
        result.lesson_session = result.lesson
        if not result.passages:
            result.clarification = "There are no further substantive lesson sections in the selected text. You can revisit a topic or ask for a practice question."
    included = {p.document_id for p in result.passages}
    for notation in source_notation_limits(result.passages):
        result.reading_limits.append(
            {
                "document": notation["document"],
                "notation": True,
                "notices": [notation["location"] + ": " + notation["notice"]],
            }
        )
    for document in store.documents(course_id):
        if document["id"] in included:
            warnings = json.loads(document["warnings"])
            if warnings:
                result.reading_limits.append(
                    {
                        "document": document["name"],
                        "notices": [w[:600] for w in dict.fromkeys(warnings[:2] + warnings[-2:])],
                    }
                )
    return result
