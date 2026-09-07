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
