"""Local extractive reading with context-only spans and SQuAD2 no-answer scores."""

from dataclasses import dataclass

import numpy as np

from modulemate.local_models import ModelUnavailable


@dataclass(frozen=True)
class Span:
    source: int  # one-based, aligned with the supplied passages
    text: str
    start: int
    end: int
    margin: float  # answer logit sum minus no-answer logit sum, not a probability


class QuestionReader:
    def __init__(self, path):
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, use_fast=True
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, use_safetensors=True
        ).eval()

    def extract(self, question, texts):
        import torch

        if not texts:
            return []
        question_ids = self.tokenizer(question[:2000], add_special_tokens=False)["input_ids"][:64]
        question = self.tokenizer.decode(question_ids)
        encoded = self.tokenizer(
            [question] * len(texts),
            texts,
            truncation="only_second",
            max_length=384,
            stride=96,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
        )
        offsets = encoded.pop("offset_mapping").numpy()
        owners = encoded.pop("overflow_to_sample_mapping").tolist()
        best = {}
        for batch_start in range(0, len(owners), 8):
            with torch.inference_mode():
                output = self.model(
                    **{k: v[batch_start : batch_start + 8] for k, v in encoded.items()}
                )
            starts, ends = output.start_logits.numpy(), output.end_logits.numpy()
            if not np.isfinite(starts).all() or not np.isfinite(ends).all():
                raise ModelUnavailable("Invalid local answer scores.")
            for row in range(len(starts)):
                index = batch_start + row
                owner = owners[index]
                context = np.array(
                    [
                        seq == 1 and end > start
                        for seq, (start, end) in zip(
                            encoded.sequence_ids(index), offsets[index], strict=True
                        )
                    ]
                )
                # Question/special/padding tokens can never be answer positions.
                start_scores = np.where(context, starts[row], -np.inf)
                end_scores = np.where(context, ends[row], -np.inf)
                span_scores = start_scores[:, None] + end_scores[None, :]
                positions = np.arange(len(context))
                valid = (positions[:, None] <= positions[None, :]) & (
                    positions[None, :] - positions[:, None] < 40
                )
                span_scores = np.where(valid, span_scores, -np.inf)
                start_token, end_token = np.unravel_index(span_scores.argmax(), span_scores.shape)
                score = float(span_scores[start_token, end_token])
                null_score = float(starts[row, 0] + ends[row, 0])
                margin = score - null_score
                if not np.isfinite(margin) or margin <= 0:
                    continue
                start = int(offsets[index, start_token, 0])
                end = int(offsets[index, end_token, 1])
                text = texts[owner][start:end]
                if not text.strip() or len(text) > 700:
                    continue
                span = Span(owner + 1, text, start, end, margin)
                if owner not in best or margin > best[owner].margin:
                    best[owner] = span
        return sorted(best.values(), key=lambda s: (-s.margin, s.source))
