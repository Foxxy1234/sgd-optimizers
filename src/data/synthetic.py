"""Synthetic regression datasets with controllable conditioning and noise."""
from __future__ import annotations

import numpy as np


def make_linear_regression(
    n: int = 1000,
    d: int = 10,
    noise: float = 0.1,
    cond: float = 1.0,
    seed: int = 0,
):
    """Generate (X, y, w_true, b_true) with y = X @ w_true + b_true + noise * eps.

    `cond` controls the condition number of X.T @ X by stretching feature singular
    values geometrically from 1 to `cond`. For cond=1 features are isotropic.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    if cond != 1.0:
        # SVD-shape: replace singular values with a geometric sequence in [1, cond].
        U, _, Vt = np.linalg.svd(X, full_matrices=False)
        s = np.geomspace(1.0, cond, num=min(n, d))
        X = U @ np.diag(s) @ Vt
    w_true = rng.standard_normal(d)
    b_true = float(rng.standard_normal())
    y = X @ w_true + b_true + noise * rng.standard_normal(n)
    return X, y, w_true, b_true


def make_quadratic_2d(n: int = 200, noise: float = 0.05, seed: int = 0):
    """Tiny 2-feature dataset for level-curve plots (d=2)."""
    return make_linear_regression(n=n, d=2, noise=noise, cond=1.0, seed=seed)
