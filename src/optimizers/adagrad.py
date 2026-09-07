from __future__ import annotations

import numpy as np

from optimizers.base import Optimizer


class AdaGrad(Optimizer):
    name = "AdaGrad"

    def __init__(self, lr: float = 0.1, eps: float = 1e-8, scheduler=None):
        super().__init__(lr=lr, scheduler=scheduler)
        self.eps = eps
        self.g_sum_sq: np.ndarray | None = None

    def reset(self):
        super().reset()
        self.g_sum_sq = None

    def step(self, w, g):
        if self.g_sum_sq is None:
            self.g_sum_sq = np.zeros_like(w)
        lr = self.current_lr()
        self.g_sum_sq += g * g
        self.t += 1
        return w - lr * g / (np.sqrt(self.g_sum_sq) + self.eps)
