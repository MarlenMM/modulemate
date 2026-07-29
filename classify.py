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
        # Programme-level administration. The corpus now spans the whole
        # summer school, not one module, but these utterances originally
        # covered only AI & ML coursework. Measured consequence: "Where is the
        # HIT Global Summer School held?" routed to out-of-scope/social and was
        # refused outright, despite the programme brochure answering it.
        "Where is the summer school held?",
        "Which city and university host the programme?",
        "What are the dates of the summer school?",
        "What language are the lectures taught in?",
        "What English proficiency do I need for the programme?",
        "Who lectures the Intelligent Software Engineering course?",
        "How do I check in when I arrive?",
        "What accommodation is provided for students?",
        "Which projects can I choose from in the summer school?",
        "How are teams formed for the group project?",
        "When is the mid-term review?",
        "What do I need to submit for the code review experiments?",
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
        # Distance and similarity metrics. The module devotes a run of slides
        # to these (Euclidean, Manhattan, Hamming, edit distance, properties
        # of a dissimilarity function), but the original utterances above were
        # all "how do I / what are the steps of" procedural phrasings, so
        # property- and definition-shaped questions about metrics fell outside
        # every class and landed in out-of-scope with low confidence.
        "How is Euclidean distance computed between two points?",
        "What is the formula for Minkowski distance?",
        "How do you measure the similarity between two vectors?",
        "What is Hamming distance used for?",
        "Which distance metric should be used for binary features?",
        "What conditions must a distance function satisfy?",
        "How is the cost of an edit operation calculated?",
    ],
    # Two sub-populations share this class, and the split matters.
    #
    # (a) Chit-chat and general-knowledge turns. These are also topically far
    #     from the corpus, so the similarity gate catches them independently.
    #
    # (b) Policy refusals. §2.6 of the specification places pastoral matters,
    #     extenuating circumstances, academic appeals, grading and doing the
    #     student's work out of scope "by design". These are phrased around
    #     deadlines, marks and reports, so they retrieve corpus passages at
    #     high similarity (measured: 0.54 and 0.70) and the similarity gate
    #     structurally cannot reject them. The router is the only layer that
    #     can, which is why this class carries deliberate coverage of them.
    #     The first evaluation run under-represented (b) — three examples
    #     against twenty admin examples about deadlines and reports — and both
    #     policy questions in the gold set were wrongly answered as a result.
    OUT_OF_SCOPE: [
        # (a) social and general knowledge
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
        "What is the meaning of life?",
        # (b) policy refusals — pastoral / extenuating / appeals
        "I am feeling stressed about my exams, can you help?",
        "I was ill last week, can I claim extenuating circumstances?",
        "I want to appeal the mark I was given.",
        "My family situation is affecting my studies, what should I do?",
        "Can you excuse my absence from the lecture?",
        "I need special consideration because of my health.",
        "Can you change the deadline for me?",
        "Please give me more time to finish the assignment.",
        "Can you make an exception to the late penalty in my case?",
        # (b) policy refusals — doing the student's work / grading
        "Can you do my assignment for me?",
        "Write the literature review section for me.",
        "Please complete this project on my behalf.",
        "Draft my report introduction.",
        "What is my grade going to be?",
        "Can you mark my draft and tell me what score it would get?",
        "Give me the answers to the assignment questions.",
        "Solve this homework problem for me.",
    ],
}

# Held-out probes for the policy sub-population, used only to measure whether
# the router generalises beyond its training phrasings. Never trained on, and
# deliberately worded differently from both the training set and the gold set.
PROGRAMME_PROBES: list[str] = [
    "Whereabouts does the summer programme actually take place?",
    "Do I need to prove my English level before attending?",
    "What time should I arrive to register on the first day?",
]

METRIC_PROBES: list[str] = [
    "In what situation is Manhattan distance equivalent to Hamming distance?",
    "What characteristics does a good dissimilarity measure need?",
    "How would I work out the edit distance between two words?",
]

POLICY_PROBES: list[str] = [
    "Would it be possible to hand in a few days later than planned?",
    "I had a hospital appointment and missed the class, does that count?",
    "Could you produce the first draft of section three for me?",
    "Am I going to pass this module based on what I have done?",
    "Is there any way to get the penalty waived just this once?",
    "Could you put together the whole thing so I only have to proofread it?",
    "I disagree with how my work was assessed, can you overturn it?",
    "My internet was down so I could not submit, can you fix that for me?",
]


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
        self.min_confidence = self.cfg.get(
            "intents.short_circuit_min_confidence", 0.0
        )

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

    def should_short_circuit(self, label: str, confidence: float = 1.0) -> bool:
        """True when the intent bypasses retrieval and generation (FR7).

        Confidence-gated on purpose: a short-circuit refuses without ever
        retrieving, so an unconfident prediction must not trigger one. Below
        the cut-off the turn proceeds to retrieval and the similarity gate
        decides, which lets the two refusal layers check each other.
        """
        return label in self.short_circuit and confidence >= self.min_confidence


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

    # Generalisation check on unseen policy phrasings. Reported because the
    # out-of-scope training set was deliberately strengthened after the first
    # evaluation run; without this, the improved refusal rate on the gold set
    # could not be distinguished from having fitted the gold set.
    correct = 0
    print("\nHeld-out policy probes (never trained on, not in the gold set):")
    for q in POLICY_PROBES:
        p = clf.predict(q)
        ok = p.label == OUT_OF_SCOPE
        correct += ok
        print(f"  {'OK ' if ok else 'ERR'}  {p.label:22s} {p.confidence:.2f}  {q}")
    print(f"\n  policy generalisation: {correct}/{len(POLICY_PROBES)} "
          f"= {correct / len(POLICY_PROBES):.1%}")

    ok_metric = 0
    print("\nHeld-out distance-metric probes (never trained on):")
    for q in METRIC_PROBES:
        p = clf.predict(q)
        ok = p.label == CODE_HELP
        ok_metric += ok
        print(f"  {'OK ' if ok else 'ERR'}  {p.label:22s} {p.confidence:.2f}  {q}")
    print(f"\n  metric generalisation: {ok_metric}/{len(METRIC_PROBES)} "
          f"= {ok_metric / len(METRIC_PROBES):.1%}")

    ok_prog = 0
    print("\nHeld-out programme-admin probes (never trained on):")
    for q in PROGRAMME_PROBES:
        p = clf.predict(q)
        ok = p.label == ADMIN
        ok_prog += ok
        print(f"  {'OK ' if ok else 'ERR'}  {p.label:22s} {p.confidence:.2f}  {q}")
    print(f"\n  programme generalisation: {ok_prog}/{len(PROGRAMME_PROBES)} "
          f"= {ok_prog / len(PROGRAMME_PROBES):.1%}")
