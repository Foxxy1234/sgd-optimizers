"""Soft-margin linear SVM trained with subgradient SGD.

Loss:  L(w, b) = 0.5 * ||w||^2 + C * mean_i max(0, 1 - y_i (w . x_i + b)),  y_i in {-1, +1}.

Hyperparameters in the interface: C, fit_intercept.
"""
from __future__ import annotations

import numpy as np


class SoftMarginSVM:
    def __init__(self, C: float = 1.0, fit_intercept: bool = True):
        self.C = C
        self.fit_intercept = fit_intercept

    def init_params(self, d: int, rng: np.random.Generator | None = None) -> np.ndarray:
        n_params = d + (1 if self.fit_intercept else 0)
        if rng is None:
            return np.zeros(n_params)
        return rng.standard_normal(n_params) * 0.01

    def _split(self, p: np.ndarray):
        if self.fit_intercept:
            return p[:-1], float(p[-1])
        return p, 0.0

    def margin(self, p: np.ndarray, X: np.ndarray) -> np.ndarray:
        w, b = self._split(p)
        return X @ w + b

    def loss(self, p: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        w, b = self._split(p)
        scores = X @ w + b
        hinge = np.maximum(0.0, 1.0 - y * scores)
        return 0.5 * float(w @ w) + self.C * float(hinge.mean())

    def grad(self, p: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        w, b = self._split(p)
        scores = X @ w + b
        active = (1.0 - y * scores) > 0  # margin violated
        # subgradient of mean hinge
        coef = -(y * active) / len(y)
        g_w = w + self.C * (X.T @ coef)
        if self.fit_intercept:
            g_b = self.C * float(coef.sum())
            return np.concatenate([g_w, [g_b]])
        return g_w

    def predict(self, p: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.sign(self.margin(p, X))
