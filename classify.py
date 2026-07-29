"""M6 — Four-class intent router (FR3, O4).

This is the project's sole trained artefact, in line with the locked
"no training from scratch" constraint. Training data is ~18 hand-authored
utterances per class; features are the same 384-dim all-MiniLM-L6-v2
embeddings used for retrieval, so no second representation has to be
maintained and an utterance is embedded once per turn.

Classifier choice — multinomial logistic regression over nearest-centroid:
  * Nearest-centroid assumes each class is an isotropic blob around its mean.
    `admin/logistics` is not: "what is the deadline", "how is it weighted" and
    "how do I submit" sit in genuinely different regions of embedding space,
    and a single centroid sits in the middle of all three, nearest to none.
  * Logistic regression learns a discriminative boundary per class instead,
    and with L2 regularisation it stays stable at ~72 training examples.
  * It emits calibrated probabilities, which the UI shows as router confidence
    and which make low-confidence routing decisions inspectable.
Both were measured; the comparison is reported by `python classify.py`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.neighbors import NearestCentroid

import config
import index as index_mod

ADMIN = "admin/logistics"
CONCEPTUAL = "conceptual"
CODE_HELP = "code-help"
OUT_OF_SCOPE = "out-of-scope/social"

# --------------------------------------------------------------------------
# Hand-authored training utterances (~18 per class)
# --------------------------------------------------------------------------
TRAINING_UTTERANCES: dict[str, list[str]] = {
    ADMIN: [
        "What is the word limit for the individual project?",
        "How many words should the report be?",
        "When is the project due?",
        "What is the submission deadline?",
        "How much is the project worth?",
        "What percentage of my grade is this assignment?",
        "How is the project weighted in the overall assessment?",
        "What happens if I submit late?",
        "What is the penalty for late submission?",
        "Who do I submit my report to?",
        "How should I name my submission file?",
        "What file format should I submit in?",
        "How many pages should the report be?",
        "What is the marking scheme for the report?",
        "How many marks is the literature survey worth?",
        "Am I allowed to use AI for this assignment?",
        "Can I propose my own project topic?",
        "What topics can I choose from for the project?",
        "When does the course start and finish?",
        "What is covered in the teaching schedule this week?",
    ],
    CONCEPTUAL: [
        "What is machine learning?",
        "Explain the difference between supervised and unsupervised learning.",
        "What is a neural network?",
        "How does a decision tree work?",
        "What does overfitting mean?",
        "What is the difference between classification and regression?",
        "Can you explain what reinforcement learning is?",
        "What is knowledge representation in AI?",
        "What is the purpose of an activation function?",
        "Why do we need a hidden layer in a neural network?",
        "What is deep learning?",
        "What does uncertainty management mean in AI?",
        "Explain what a convolutional neural network is.",
        "What is the difference between AI and machine learning?",
        "What is a recurrent neural network used for?",
        "What is meant by generalisation in machine learning?",
        "How is propositional logic used in AI?",
        "What is the role of search in artificial intelligence?",
        "What are the main types of machine learning?",
        "What is a perceptron?",
    ],
    CODE_HELP: [
        "What are the steps of the gradient descent algorithm?",
        "How do I implement backpropagation?",
        "What is the formula for computing entropy in a decision tree?",
        "Which algorithm is used to train a multi-layer perceptron?",
        "What steps does the k-means algorithm follow?",
        "How is the sigmoid function calculated?",
        "What is the procedure for breadth-first search?",
        "Give me the steps of the ID3 algorithm.",
        "How do you calculate information gain?",
        "What is the update rule for the weights in a perceptron?",
        "How does the A star search algorithm work step by step?",
        "What is the equation for the loss function?",
        "How do I compute the output of a convolutional layer?",
        "What are the stages in the machine learning pipeline?",
        "Which method is used for pruning a decision tree?",
        "How is the learning rate applied during training?",
        "What is the algorithm for depth-first search?",
        "How do you calculate the gradient of the error?",
        "What steps are involved in training a CNN?",
        "What technique is used to avoid overfitting during training?",
    ],
    OUT_OF_SCOPE: [
        "Hello!",
        "Hi there, how are you?",
        "Good morning",
        "Thanks, that was helpful",
        "Thank you so much",
        "Goodbye",
        "What is the weather like today?",
        "Can you write me a poem?",
        "What is the capital of France?",
        "Tell me a joke",
        "Who won the football match last night?",
        "What should I have for dinner?",
        "Can you help me book a flight?",
        "What do you think about politics?",
        "Are you a real person?",
        "Can you do my assignment for me?",
        "What is my grade going to be?",
        "I am feeling stressed about my exams, can you help?",
        "Can you get me an extension on my deadline?",
        "What is the meaning of life?",
    ],
}


@dataclass
class IntentPrediction:
    label: str
    confidence: float
    scores: dict[str, float]

    @property
    def is_short_circuit(self) -> bool:
        return self.label == OUT_OF_SCOPE


def _training_matrix(cfg) -> tuple[np.ndarray, list[str], list[str]]:
    texts, labels = [], []
    for label, utterances in TRAINING_UTTERANCES.items():
        texts.extend(utterances)
        labels.extend([label] * len(utterances))
    vectors = index_mod.embed_texts(texts, cfg)
    return vectors, labels, texts


def train(cfg=None, report: bool = False) -> dict:
    """Fit the router and persist it. Returns training diagnostics."""
    cfg = cfg or config.load()
    seed = cfg.get("evaluation.seed", 42)
    vectors, labels, _ = _training_matrix(cfg)

    clf = LogisticRegression(
        max_iter=2000, C=4.0, random_state=seed, multi_class="multinomial"
    )
    clf.fit(vectors, labels)

    out_path = cfg.path("intents.model_path")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_path)

    diagnostics = {"n_train": len(labels), "model": "LogisticRegression(C=4.0)"}

    if report:
        # Honest comparison of the two candidate methods on the training pool.
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        y = np.array(labels)
        for name, candidate in [
            ("logistic regression", LogisticRegression(
                max_iter=2000, C=4.0, random_state=seed,
                multi_class="multinomial")),
            ("nearest centroid", NearestCentroid()),
        ]:
            pred = cross_val_predict(candidate, vectors, y, cv=cv)
            diagnostics[f"cv_accuracy::{name}"] = round(
                float((pred == y).mean()), 4
            )
    return diagnostics


class IntentClassifier:
    """Loads the persisted router; retrains automatically if absent."""

    def __init__(self, cfg=None):
        self.cfg = cfg or config.load()
        path = self.cfg.path("intents.model_path")
        if not path.exists():
            train(self.cfg)
        self.clf = joblib.load(path)
        self.short_circuit = set(self.cfg.get("intents.short_circuit", []))

    def predict(self, text: str) -> IntentPrediction:
        vector = index_mod.embed_texts([text], self.cfg)
        probs = self.clf.predict_proba(vector)[0]
        classes = list(self.clf.classes_)
        best = int(np.argmax(probs))
        return IntentPrediction(
            label=classes[best],
            confidence=float(probs[best]),
            scores={c: float(p) for c, p in zip(classes, probs)},
        )

    def should_short_circuit(self, label: str) -> bool:
        """True when the intent bypasses retrieval and generation (FR7)."""
        return label in self.short_circuit


if __name__ == "__main__":
    diag = train(report=True)
    print("Intent router trained")
    for key, value in diag.items():
        print(f"  {key:38s} {value}")

    clf = IntentClassifier()
    print("\nSpot checks:")
    for q in [
        "How many words should my report be?",
        "What is a convolutional neural network?",
        "What are the steps of gradient descent?",
        "Hey, what's up?",
    ]:
        p = clf.predict(q)
        print(f"  {p.label:22s} {p.confidence:.2f}  {q}")
