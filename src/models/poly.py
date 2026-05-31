"""Polynomial regression via feature expansion + LinearRegressionModel."""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np

from models.linear import LinearRegressionModel
from regularizers import Regularizer


def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Expand X (n, d) into all monomials of total degree <= `degree`, no bias term.

    For d=1 this gives [x, x^2, ..., x^degree]. For d>1 it gives all products
    x_{i1} * x_{i2} * ... * x_{ik} with k in 1..degree.
    """
    n, d = X.shape
    cols = []
    for k in range(1, degree + 1):
        for combo in combinations_with_replacement(range(d), k):
            cols.append(np.prod(X[:, combo], axis=1))
    return np.column_stack(cols) if cols else np.empty((n, 0))


class PolynomialRegression(LinearRegressionModel):
    """Linear model on polynomial features. degree=1 is plain linear."""

    def __init__(self, degree: int = 2, regularizer: Regularizer | None = None,
                 fit_intercept: bool = True):
        super().__init__(regularizer=regularizer, fit_intercept=fit_intercept)
        self.degree = degree

    def transform(self, X: np.ndarray) -> np.ndarray:
        return polynomial_features(X, self.degree)

    def predict(self, w, X):
        return super().predict(w, self.transform(X))

    def loss(self, w, X, y):
        return super().loss(w, self.transform(X), y)

    def grad(self, w, X, y):
        return super().grad(w, self.transform(X), y)
