"""
Pure NumPy implementations of TF-IDF Vectorizer and Multinomial Naive Bayes.

Provides 100% dependency-free, high-performance text vectorisation and classification
without requiring external C/C++ DLL extensions.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class NumPyTfidfVectorizer:
    """TF-IDF Vectoriser using NumPy for feature extraction and L2 normalisation."""

    def __init__(self, max_features: int = 4000, sublinear_tf: bool = True):
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.vocabulary_ = {}
        self.idf_ = None
        self.feature_names_ = []

    def fit(self, raw_documents):
        """Build vocabulary and compute IDF weights from corpus."""
        df_counts = {}
        for doc in raw_documents:
            if not doc:
                continue
            words = set(doc.split())
            for w in words:
                df_counts[w] = df_counts.get(w, 0) + 1

        # Select top max_features terms by document frequency
        sorted_terms = sorted(df_counts.items(), key=lambda x: x[1], reverse=True)[:self.max_features]
        self.vocabulary_ = {term: i for i, (term, _) in enumerate(sorted_terms)}
        self.feature_names_ = [term for term, _ in sorted_terms]

        # Calculate smooth IDF: log((1 + N) / (1 + df)) + 1.0
        n_docs = len(raw_documents)
        idfs = []
        for term, _ in sorted_terms:
            df = df_counts[term]
            idf = np.log((1 + n_docs) / (1 + df)) + 1.0
            idfs.append(idf)

        self.idf_ = np.array(idfs, dtype=np.float64)
        return self

    def transform(self, raw_documents):
        """Transform documents to L2-normalised TF-IDF feature matrix."""
        n_docs = len(raw_documents)
        n_features = len(self.vocabulary_)
        X = np.zeros((n_docs, n_features), dtype=np.float64)

        if n_features == 0:
            return X

        for i, doc in enumerate(raw_documents):
            if not doc:
                continue
            counts = {}
            for w in doc.split():
                if w in self.vocabulary_:
                    counts[w] = counts.get(w, 0) + 1
            for w, count in counts.items():
                idx = self.vocabulary_[w]
                X[i, idx] = (1.0 + np.log(count)) if self.sublinear_tf else float(count)

        # Apply IDF weighting
        X = X * self.idf_

        # L2 normalisation
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def fit_transform(self, raw_documents):
        """Fit vectoriser and return transformed document matrix."""
        self.fit(raw_documents)
        return self.transform(raw_documents)


class NumPyMultinomialNB:
    """Multinomial Naive Bayes classifier using NumPy log-probabilities."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.classes_ = np.array([0, 1])
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        """Fit Naive Bayes model using Laplace smoothing."""
        n_samples, n_features = X.shape
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.class_log_prior_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_log_prob_ = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            n_c = X_c.shape[0]
            self.class_log_prior_[idx] = np.log(n_c / n_samples)

            # Feature counts with Laplace smoothing
            feature_counts = X_c.sum(axis=0) + self.alpha
            total_count = feature_counts.sum()
            self.feature_log_prob_[idx] = np.log(feature_counts / total_count)

        return self

    def predict_log_proba(self, X):
        """Compute unnormalised log posterior probabilities."""
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_proba(self, X):
        """Compute class probabilities via Softmax / Log-Sum-Exp."""
        log_prob = self.predict_log_proba(X)
        max_log = np.max(log_prob, axis=1, keepdims=True)
        exp_prob = np.exp(log_prob - max_log)
        return exp_prob / np.sum(exp_prob, axis=1, keepdims=True)

    def predict(self, X):
        """Predict class labels for samples in X."""
        log_prob = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_prob, axis=1)]
