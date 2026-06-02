"""Lion optimizer (Chen et al., 2023): EvoLved Sign Momentum.

Algorithm (one momentum buffer; no second moment, half the memory of Adam):

    update = sign(beta1 * m_{t-1} + (1 - beta1) * g_t)
    w_t    = w_{t-1} - lr * (update + weight_decay * w_{t-1})
    m_t    = beta2 * m_{t-1} + (1 - beta2) * g_t

Notes:
- Uses TWO independent momentum coefficients: beta1 for the update direction
  (interpolates current grad and previous momentum), beta2 for storage.
- Update magnitude is +/- lr per coordinate (sign-based), so the effective step
  is far larger than Adam at the same lr; typical lr ~ Adam_lr / 3..10.
- Decoupled weight decay (AdamW-style).

Paper: "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023).
"""
from __future__ import annotations

import numpy as np

from optimizers.base import Optimizer


class Lion(Optimizer):
    name = "Lion"

    def __init__(self, lr: float = 1e-4, betas=(0.9, 0.99), weight_decay: float = 0.0,
                 scheduler=None):
        super().__init__(lr=lr, scheduler=scheduler)
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.m: np.ndarray | None = None

    def reset(self):
        super().reset()
        self.m = None

    def step(self, w, g):
        if self.m is None:
            self.m = np.zeros_like(w)
        lr = self.current_lr()
        update = np.sign(self.beta1 * self.m + (1.0 - self.beta1) * g)
        new_w = w - lr * (update + self.weight_decay * w)
        self.m = self.beta2 * self.m + (1.0 - self.beta2) * g
        self.t += 1
        return new_w
