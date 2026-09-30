"""Tests for model module (NumPyTfidfVectorizer and NumPyMultinomialNB)."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from src.model import NumPyTfidfVectorizer, NumPyMultinomialNB
except ImportError:
    from custom_model import NumPyTfidfVectorizer, NumPyMultinomialNB


class TestNumPyTfidfVectorizer:
    def test_fit_transform_basic(self):
        corpus = [
            "hello world test",
            "spam message winner prize",
            "hello friend meeting"
        ]
        vec = NumPyTfidfVectorizer(max_features=10)
        X = vec.fit_transform(corpus)

        assert X.shape[0] == 3
        assert X.shape[1] <= 10
        # Check L2 norm equals 1 for non-empty documents
        norms = np.linalg.norm(X, axis=1)
        for norm in norms:
            assert np.isclose(norm, 1.0)

    def test_empty_corpus(self):
        vec = NumPyTfidfVectorizer()
        X = vec.fit_transform(["", ""])
        assert X.shape[0] == 2
        assert X.shape[1] == 0


class TestNumPyMultinomialNB:
    def test_fit_and_predict(self):
        X_train = np.array([
            [1.0, 0.0, 0.5],
            [0.8, 0.1, 0.0],
            [0.0, 1.0, 0.9],
            [0.1, 0.9, 1.0]
        ])
        y_train = np.array([0, 0, 1, 1])

        model = NumPyMultinomialNB(alpha=0.1)
        model.fit(X_train, y_train)

        preds = model.predict(X_train)
        assert len(preds) == 4
        assert np.array_equal(preds, y_train)

        probs = model.predict_proba(X_train)
        assert probs.shape == (4, 2)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
