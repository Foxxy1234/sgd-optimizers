"""Numerical gradient via central finite differences."""
from __future__ import annotations

from typing import Callable

import numpy as np


def numerical_grad(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient. O(d) evaluations of f."""
    g = np.zeros_like(x, dtype=np.float64)
    e = np.zeros_like(x, dtype=np.float64)
    for i in range(x.size):
        e[i] = eps
        g[i] = (f(x + e) - f(x - e)) / (2.0 * eps)
        e[i] = 0.0
    return g


def epsilon_sweep(f, x, analytic_grad, eps_values):
    """Return array of ||numerical_grad(f, x, eps) - analytic_grad|| for each eps."""
    errs = []
    for eps in eps_values:
        gn = numerical_grad(f, x, eps=eps)
        errs.append(float(np.linalg.norm(gn - analytic_grad)))
    return np.array(errs)
