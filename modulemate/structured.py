"""An MLX adapter for Outlines Core's grammar API, without disk/pickle caches.

The pinned Qwen tokenizers use reversible byte-level BPE. Keep vocabulary entries
as bytes: decoding individual partial UTF-8 tokens to strings would replace bytes
and corrupt the grammar's handling of Unicode text.
"""

from __future__ import annotations

import json


def byte_vocabulary(tokenizer):
    literal = {*range(33, 127), *range(161, 173), *range(174, 256)}
    escaped = [b for b in range(256) if b not in literal]
    decoder = {chr(b): b for b in literal} | {chr(256 + i): b for i, b in enumerate(escaped)}
    special = set(tokenizer.all_special_ids)
    vocabulary = {}
    for token, token_id in tokenizer.get_vocab().items():
        if token_id in special:
            continue
        raw = (
            bytes(decoder[c] for c in token) if all(c in decoder for c in token) else token.encode()
        )
        vocabulary.setdefault(raw, []).append(token_id)
    return vocabulary


class GuidedDecoding:
    def __init__(self, tokenizer):
        from outlines_core import Vocabulary

        decoder = json.loads(tokenizer.backend_tokenizer.decoder.__getstate__())
        if decoder.get("type") != "ByteLevel":
            raise ValueError("This adapter requires the pinned byte-level tokenizer")
        self.eos = tokenizer.eos_token_id
        self.vocabulary = Vocabulary(self.eos, byte_vocabulary(tokenizer))
        self.indexes = {}  # Generic schemas only, never learner text or prompts.

    def processor(self, schema):
        from outlines_core import Guide, Index
        from outlines_core.json_schema import build_regex_from_schema
        from outlines_core.kernels.mlx import (
            allocate_token_bitmask,
            apply_token_bitmask,
            fill_next_token_bitmask,
        )

        key = json.dumps(schema, sort_keys=True)
        if key not in self.indexes:
            regex = build_regex_from_schema(json.dumps(schema), whitespace_pattern=r"[ \t\n\r]*")
            self.indexes[key] = Index(regex, self.vocabulary)
        guide = Guide(self.indexes[key])
        first, mask = True, None

        def constrain(input_ids, logits):
            nonlocal first, mask
            if first:
                first = False  # The input prefix is the prompt, not generated JSON.
            else:
                token = int(input_ids[-1].item())
                if not guide.is_finished() or guide.accepts_tokens([token]):
                    guide.advance(token, return_tokens=False)
            if mask is None:
                mask = allocate_token_bitmask(logits.shape[-1])
            fill_next_token_bitmask(guide, mask)
            result = apply_token_bitmask(logits, mask)
            return result[0] if logits.ndim == 1 else result

        return constrain
