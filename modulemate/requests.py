"""Subject-independent request cues, shared by context and answer generation.

These describe the requested operation, never whether a source is credible or a
question is allowed. Unrecognized phrasing still reaches the normal tutor.
"""

import re


def conceptual_reframing(question):
    return bool(
        re.search(
            r"\b(?:can|could|may) (?:I|we) (?:think of|understand|interpret|view|describe|explain|compare|connect|relate)\b"
            r"|\b(?:can|could|may) (?:I|we) (?:regard|see|treat|read|take|consider)\b.{0,80}\bas\b"
            r"|\b(?:can|could|may) (?:I|we) use\b.{0,60}\b(?:analogy|metaphor)\b",
            question,
            re.I,
        )
    )


def preparatory_method(question):
    """A request about doing preparatory work, not institutional authority."""
    return bool(
        re.search(
            r"\b(?:can|could|may|should) (?:I|we) (?:still )?"
            r"(?:prepare|draft|outline|set up|begin|start|make (?:useful )?progress)\b",
            question,
            re.I,
        )
    )


def synthesis_request(question):
    return conceptual_reframing(question) or bool(
        re.search(
            r"\b(?:main|central|core|key|big|overall|underlying|unifying)\s+"
            r"(?:ideas?|concepts?|messages?|points?|themes?|takeaways?|lessons?|arguments?|principles?)\b"
            r"|\b(?:big picture|take away|getting at|trying to (?:say|show|teach)|"
            r"theme|interpretation|interpret|analogy|analogies)\b"
            r"|\b(?:connect\w*|relationship|ties?)\b.{0,100}\b(?:between|together|ideas|concepts)\b",
            question,
            re.I,
        )
    )


def material_overview(question):
    """A whole-material explanation can be implicit in an ordinary question."""
    main = re.search(
        r"\b(?:main|central|core|key|big|overall|underlying|unifying)\s+"
        r"(?:ideas?|concepts?|messages?|points?|themes?|takeaways?|lessons?|arguments?|principles?)\b",
        question,
        re.I,
    )
    if main and re.fullmatch(
        r"\s*(?:(?:of|in|from|behind)\s+(?:here|this|that|these|it))?[.!? ]*",
        question[main.end() :],
        re.I,
    ):
        # Unqualified 'the main idea' / 'main concept from here' refers to the
        # selected material, including polite or misspelled lead-in verbs.
        return True
    if re.search(
        r"\b(?:what(?:'s| is| are)) (?:this|that|it|the (?:presentation|lecture|document|deck)) about\b"
        r"|\b(?:explain|help me understand) (?:this|the|my|our) (?:presentation|slides|lecture|document|deck|reading|materials)\b"
        r"|\bwhat should (?:I|we) (?:remember|take away|learn)\b",
        question,
        re.I,
    ):
        return True
    scope = bool(
        re.search(
            r"\b(?:lecture|course|notes|materials|chapter|document|library|presentation|"
            r"slides?|deck|talk|reading|images?|pictures?|figures?|screenshots?|everything|all topics)\b",
            question,
            re.I,
        )
    )
    return synthesis_request(question) and (
        scope
        or bool(
            re.fullmatch(
                r"\s*(?:please )?(?:what(?:'s| is| are)|explain|describe|give me|show me) "
                r"(?:the )?(?:main|central|core|key|big|overall|underlying|unifying) "
                r"(?:ideas?|concepts?|messages?|points?|themes?|takeaways?|lessons?|arguments?|principles?)"
                r"[.!? ]*",
                question,
                re.I,
            )
        )
    )
