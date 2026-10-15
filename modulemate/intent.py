"""Small supervised router trained only on original, non-user utterances."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


@dataclass(frozen=True)
class Intent:
    label: str
    confidence: float


@lru_cache(maxsize=1)
def classifier():
    records = json.loads((Path(__file__).parent / "intent_training.json").read_text())
    model = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                        (
                            "characters",
                            TfidfVectorizer(
                                analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(C=8, max_iter=500, random_state=42, class_weight="balanced"),
            ),
        ]
    )
    model.fit([r["text"] for r in records], [r["intent"] for r in records])
    return model


def classify(question):
    model = classifier()
    probabilities = model.predict_proba([question[:4000]])[0]
    best = probabilities.argmax()
    return Intent(str(model.classes_[best]), float(probabilities[best]))
