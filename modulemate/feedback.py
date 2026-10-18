"""Local uncertainty sampling for human review, never automatic self-training."""

import math


def review_queue(messages, limit=100):
    rows = []
    for message in messages:
        metadata = message.get("metadata", {})
        feedback = metadata.get("feedback", {})
        estimates = [
            p.get("estimate") for p in metadata.get("evidence_assessment", {}).get("parts", [])
        ]
        estimates = [
            float(p)
            for p in estimates
            if isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1
        ]
        entropy = sum(
            -p * math.log2(p) - (1 - p) * math.log2(1 - p) if 0 < p < 1 else 0 for p in estimates
        ) / max(1, len(estimates))
        negative = feedback.get("rating") in {"incorrect", "unsupported", "unclear"}
        failed = metadata.get("kind") == "error"
        priority = 3 * negative + 2 * failed + bool(metadata.get("citation_review")) + entropy
        if feedback.get("rating") == "helpful":
            priority -= 1
        rows.append(
            {
                "message_id": message["id"],
                "question": message["question"],
                "answer": message["answer"],
                "sources": message.get("sources", []),
                "feedback": feedback,
                "citation_review": metadata.get("citation_review", []),
                "label_status": "Human review required; user feedback is not verified ground truth",
                "priority": round(priority, 4),
                "uncertainty": round(entropy, 4),
                "model": metadata.get("model"),
                "evidence_assessment": metadata.get("evidence_assessment", {}),
            }
        )
    return sorted(rows, key=lambda row: (-row["priority"], row["message_id"]))[
        : max(0, min(limit, 200))
    ]
