"""Nesterov accelerated gradient (PyTorch flavor).

PyTorch's torch.optim.SGD(..., nesterov=True) implements:
    v <- momentum * v + g
    w <- w - lr * (g + momentum * v)
This is the formulation we use here for direct comparability.
"""
from __future__ import annotations

import numpy as np

from optimizers.base import Optimizer


class Nesterov(Optimizer):
    name = "Nesterov"

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
        return w - lr * (g + self.momentum * self.v)
