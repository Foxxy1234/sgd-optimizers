from __future__ import annotations

import numpy as np

from optimizers.base import Optimizer


class Momentum(Optimizer):
    name = "Momentum"

    def __init__(self, lr: float = 0.01, momentum: float = 0.9, scheduler=None):
        super().__init__(lr=lr, scheduler=scheduler)
        self.momentum = momentum
        self.v: np.ndarray | None = None

    def reset(self):
        super().reset()
        self.v = None

    def step(self, w, g):
        if self.v is None:
            self.v = np.zeros_like(w)
        lr = self.current_lr()
        self.v = self.momentum * self.v + g
        self.t += 1
        return w - lr * self.v
