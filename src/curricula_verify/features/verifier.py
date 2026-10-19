"""Local verifier model: TF-IDF + numeric features + logistic regression.

Honest scope: this is a heuristic linear baseline over engineered features,
NOT an LLM. It exists so the curriculum pipeline runs anywhere without an
external API.
"""

from __future__ import annotations

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MARKERS = [
    "ignore all previous", "ignore previous", "disregard the system",
    "developer mode", "no restrictions", "you are now dan", "pretend you have no rules",
    "social security", "home address", "medical records", "without their consent",
    "phishing", "passwords", "poison", "explosive device", "weapon",
]


def conversation_text(record: dict) -> str:
    """All user turns joined (assistant turns add little signal here)."""
    return " ".join(
        t["content"] for t in record["conversation"] if t["role"] == "user"
    )


def numeric_features(records: list[dict]) -> np.ndarray:
    feats = np.zeros((len(records), 4), dtype=np.float64)
    for i, r in enumerate(records):
        text = conversation_text(r)
        turns = [t for t in r["conversation"] if t["role"] == "user"]
        feats[i, 0] = len(turns)
        feats[i, 1] = sum(c.isupper() for c in text) / max(len(text), 1)
        feats[i, 2] = sum(text.lower().count(m) for m in MARKERS)
        words = text.split()
        feats[i, 3] = sum(1 for w in words if w.rstrip(".,!?").islower() and len(w) > 2) / max(len(words), 1)
    return feats


class VerifierModel:
    def __init__(self, tfidf_max_features: int = 4000, C: float = 1.0):
        self.vectorizer = TfidfVectorizer(max_features=tfidf_max_features,
                                          ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.clf = LogisticRegression(C=C, max_iter=2000)
        self._fitted = False

    def fit(self, records: list[dict]) -> "VerifierModel":
        texts = [conversation_text(r) for r in records]
        X_text = self.vectorizer.fit_transform(texts)
        X = hstack([X_text, csr_matrix(numeric_features(records))], format="csr")
        y = [1 if r["label"] == "unsafe" else 0 for r in records]
        if len(set(y)) < 2:
            raise ValueError("training data must contain both classes")
        self.clf.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, records: list[dict]) -> np.ndarray:
        """P(unsafe) per record."""
        if not self._fitted:
            raise RuntimeError("model not fitted")
        X_text = self.vectorizer.transform([conversation_text(r) for r in records])
        X = hstack([X_text, csr_matrix(numeric_features(records))], format="csr")
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, records: list[dict]) -> np.ndarray:
        return (self.predict_proba(records) >= 0.5).astype(int)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "VerifierModel":
        return joblib.load(path)
