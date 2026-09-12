from __future__ import annotations

import numpy as np


def make_linear_regression(
    n: int = 1000,
    d: int = 10,
    noise: float = 0.1,
    cond: float = 1.0,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    if cond != 1.0:
        U, _, Vt = np.linalg.svd(X, full_matrices=False)
        s = np.geomspace(1.0, cond, num=min(n, d))
        X = U @ np.diag(s) @ Vt
    w_true = rng.standard_normal(d)
    b_true = float(rng.standard_normal())
    y = X @ w_true + b_true + noise * rng.standard_normal(n)
    return X, y, w_true, b_true

