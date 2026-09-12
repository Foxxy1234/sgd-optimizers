from __future__ import annotations

import numpy as np

from regularizers import NoReg, Regularizer


class LinearRegressionModel:
    def __init__(self, regularizer: Regularizer | None = None, fit_intercept: bool = True):
        self.regularizer = regularizer or NoReg()
        self.fit_intercept = fit_intercept

    @staticmethod
    def _augment(X: np.ndarray, fit_intercept: bool) -> np.ndarray:
        if not fit_intercept:
            return X
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def init_params(self, d: int, rng: np.random.Generator | None = None) -> np.ndarray:
        n_params = d + (1 if self.fit_intercept else 0)
        if rng is None:
            return np.zeros(n_params)
        return rng.standard_normal(n_params) * 0.01

    def predict(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        return self._augment(X, self.fit_intercept) @ w

    def loss(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        Xa = self._augment(X, self.fit_intercept)
        r = Xa @ w - y
        mse = 0.5 * float(r @ r) / len(y)
        w_reg = w[:-1] if self.fit_intercept else w
        return mse + self.regularizer.value(w_reg)

    def grad(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        Xa = self._augment(X, self.fit_intercept)
        r = Xa @ w - y
        g = (Xa.T @ r) / len(y)
        if self.fit_intercept:
            g_reg = self.regularizer.grad(w[:-1])
            g[:-1] += g_reg
        else:
            g += self.regularizer.grad(w)
        return g
