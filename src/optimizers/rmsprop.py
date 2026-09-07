from __future__ import annotations

import numpy as np

from optimizers.base import Optimizer


class RMSProp(Optimizer):
    name = "RMSProp"

    def __init__(self, lr: float = 0.01, alpha: float = 0.99, eps: float = 1e-8, scheduler=None):
        super().__init__(lr=lr, scheduler=scheduler)
        self.alpha = alpha
        self.eps = eps
        self.v: np.ndarray | None = None

    def reset(self):
        super().reset()
        self.v = None

    def step(self, w, g):
        if self.v is None:
            self.v = np.zeros_like(w)
        lr = self.current_lr()
        self.v = self.alpha * self.v + (1.0 - self.alpha) * g * g
        self.t += 1
        return w - lr * g / (np.sqrt(self.v) + self.eps)
