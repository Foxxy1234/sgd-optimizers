from __future__ import annotations

import numpy as np

from optimizers.base import Optimizer


class Adam(Optimizer):
    name = "Adam"

    def __init__(self, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, scheduler=None):
        super().__init__(lr=lr, scheduler=scheduler)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m: np.ndarray | None = None
        self.v: np.ndarray | None = None

    def reset(self):
        super().reset()
        self.m = None
        self.v = None

    def step(self, w, g):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
        lr = self.current_lr()
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * g * g
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        return w - lr * m_hat / (np.sqrt(v_hat) + self.eps)
