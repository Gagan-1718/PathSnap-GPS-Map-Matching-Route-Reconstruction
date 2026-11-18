"""Candidate probability classifier."""

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:  # pragma: no cover - core use remains available without sklearn
    LogisticRegression = None


class CandidateClassifier:
    def __init__(self):
        self.model = LogisticRegression(max_iter=500) if LogisticRegression else None
        self.is_fitted = False

    def fit(self, features: Iterable, labels: Iterable):
        x = np.asarray(list(features), dtype=float)
        y = np.asarray(list(labels))
        if x.size == 0:
            raise ValueError("Cannot train on empty features")
        if self.model is None:
            raise RuntimeError("scikit-learn is required to train the classifier")
        if len(np.unique(y)) < 2:
            raise ValueError("Training requires both positive and negative examples")
        self.model.fit(x, y)
        self.is_fitted = True
        return self

    def predict_proba(self, features: Iterable) -> np.ndarray:
        x = np.asarray(list(features), dtype=float)
        if x.size == 0:
            return np.empty((0, 2))
        if self.model is None:
            raise RuntimeError("scikit-learn is required for prediction")
        return self.model.predict_proba(x)

    def positive_probabilities(self, features: Iterable) -> np.ndarray:
        features = list(features)
        if not features or not self.is_fitted:
            return np.asarray([1.0 / len(features)] * len(features))
        probabilities = self.predict_proba(features)
        if probabilities.shape[0] == 0:
            return np.asarray([])
        if hasattr(self.model, "classes_"):
            positive = np.where(self.model.classes_ == 1)[0]
            if len(positive):
                return probabilities[:, positive[0]]
        return probabilities[:, -1]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        self.is_fitted = True
        return self
