from __future__ import annotations

import numpy as np


class Regularizer:
    def value(self, w: np.ndarray) -> float:
        raise NotImplementedError

    def grad(self, w: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NoReg(Regularizer):
    def value(self, w):
        return 0.0

    def grad(self, w):
        return np.zeros_like(w)


class L2(Regularizer):
    def __init__(self, lam: float = 1e-3):
        self.lam = lam

    def value(self, w):
        return 0.5 * self.lam * float(w @ w)

    def grad(self, w):
        return self.lam * w


class L1(Regularizer):
    def __init__(self, lam: float = 1e-3):
        self.lam = lam

    def value(self, w):
        return self.lam * float(np.abs(w).sum())

    def grad(self, w):
        return self.lam * np.sign(w)


class ElasticNet(Regularizer):
    def __init__(self, lam: float = 1e-3, l1_ratio: float = 0.5):
        self.lam = lam
        self.l1_ratio = l1_ratio

    def value(self, w):
        return self.lam * (
            self.l1_ratio * float(np.abs(w).sum())
            + 0.5 * (1.0 - self.l1_ratio) * float(w @ w)
        )

    def grad(self, w):
        return self.lam * (self.l1_ratio * np.sign(w) + (1.0 - self.l1_ratio) * w)
