"""Small JSON schemas shared by local constrained decoding and validation.

These constrain syntax, not truth. Exact evidence checks and bounded factual
review remain necessary after generation. Schemas never contain learner text.
"""


def _object(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


STRING = {"type": "string"}
STRINGS = {"type": "array", "items": STRING, "maxItems": 3}

# Answer first lets the model formulate its response before assigning evidence
# labels. Avoid large bounded-string regexes: inference already limits tokens,
# and the parser separately checks lengths, source IDs and exact quotations.
ANSWER_SCHEMA = _object(
    {
        "answer": STRING,
        "status": {"enum": ["answered", "insufficient", "clarify"]},
        "basis": {"enum": ["materials", "mixed", "input", "general", "none"]},
        "citations": {
            "type": "array",
            "items": _object(
                {"source_id": {"type": "integer", "minimum": 1, "maximum": 48}, "quote": STRING}
            ),
            "maxItems": 12,
        },
        "followups": STRINGS,
        "topic": STRING,
        "subjects": STRINGS,
        "intent": {"enum": ["conceptual", "admin", "code_help", "social"]},
    }
)

AUDIT_SCHEMA = _object(
    {
        "approved": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": _object(
                {
                    "quote": {"type": "string", "minLength": 12, "maxLength": 80},
                    "problem": {"type": "string", "maxLength": 100},
                    "correction": {"type": "string", "maxLength": 100},
                    "source_id": {"type": "integer", "minimum": 0, "maximum": 48},
                    "source_quote": {"type": "string", "maxLength": 100},
                }
            ),
            "maxItems": 1,
        },
    }
)

LESSON_SCHEMA = ANSWER_SCHEMA | {
    "properties": ANSWER_SCHEMA["properties"] | {"status": {"enum": ["answered"]}}
}

METADATA_SCHEMA = {
    **ANSWER_SCHEMA,
    "properties": {
        **{k: v for k, v in ANSWER_SCHEMA["properties"].items() if k not in {"answer", "status"}},
        "citations": {
            "type": "array",
            "items": _object({"evidence_id": {"type": "integer", "minimum": 1}}),
            "maxItems": 12,
        },
    },
    "required": [k for k in ANSWER_SCHEMA["required"] if k not in {"answer", "status"}],
}

CARD_SCHEMA = _object(
    {
        "cards": {
            "type": "array",
            "maxItems": 6,
            "items": _object(
                {
                    "front": STRING,
                    "evidence_id": {"type": "integer", "minimum": 1},
                }
            ),
        }
    }
)
SUPPORT_SCHEMA = _object({"supported": {"type": "boolean"}})
