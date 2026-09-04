"""
Ordinal classification for the 3-class GOF/NNE/LOF label (idea #5).

The label is ordinal — LOF (0) < No-net-effect (1) < GOF (2) — but the
pipeline treats it as nominal multiclass. This module implements the standard
cumulative-link (a.k.a. Frank–Hall) decomposition: for K ordered classes, fit
K-1 binary classifiers on the cumulative threshold events

    y >= 1, y >= 2, ..., y >= K-1

and reconstruct the K-class probability vector by differencing the cumulative
probabilities. A plain logistic regression (class_weight="balanced") is the
binary base estimator, so there are no new dependencies (mord/spacecutter not
required).

The estimator is sklearn-compatible (fit/predict/predict_proba) so it slots
into the existing Optuna + nested-CV pipeline like any other model.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """
    Cumulative-link ordinal classifier backed by logistic regression.

    Parameters
    ----------
    C : float
        Inverse regularization strength for the binary logistic estimators.
    max_iter : int
        Max iterations for the binary logistic estimators.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def _binary(self) -> LogisticRegression:
        return LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight="balanced",
            random_state=self.random_state,
        )

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        if K < 2:
            raise ValueError("OrdinalClassifier needs at least 2 classes")

        # Sort classes by integer label (the label encoding already reflects the
        # LOF < NNE < GOF ordering: LABEL_MAPPING assigns 0/1/2 in that order).
        self.classes_ = np.sort(self.classes_)

        self._estimators_ = []
        for k in range(1, K):
            # y >= class_k  (cumulative probability of being at least class k)
            y_bin = (y >= self.classes_[k]).astype(int)
            est = self._binary()
            est.fit(X, y_bin)
            self._estimators_.append(est)

        return self

    def predict_proba(self, X):
        K = len(self.classes_)
        n = X.shape[0]

        # cum[k-1] = P(y >= class_k) for k = 1..K-1
        cum = [est.predict_proba(X)[:, 1] for est in self._estimators_]

        probs = np.zeros((n, K))
        probs[:, 0] = 1.0 - cum[0]                    # P(y == class_0)
        for k in range(1, K - 1):
            probs[:, k] = cum[k - 1] - cum[k]         # P(y == class_k)
        probs[:, K - 1] = cum[K - 2]                  # P(y == class_{K-1})

        # Clip tiny negative values from floating-point differencing, renormalize.
        probs = np.clip(probs, 0.0, None)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
