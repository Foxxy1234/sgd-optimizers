"""Vanilla mini-batch SGD."""
from __future__ import annotations

import numpy as np

from optimizers.base import Optimizer


class SGD(Optimizer):
    name = "SGD"

    def step(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        lr = self.current_lr()
        self.t += 1
        return w - lr * g
