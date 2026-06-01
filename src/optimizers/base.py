"""Base class for first-order optimizers."""
from __future__ import annotations

import numpy as np

from schedulers import ConstantLR


class Optimizer:
    """Stateful optimizer. Hyperparameters live on the instance.

    Usage:
        opt = SGD(lr=0.01)
        w = opt.step(w, grad)
    """

    name = "base"

    def __init__(self, lr: float = 0.01, scheduler=None):
        self.scheduler = scheduler if scheduler is not None else ConstantLR(lr)
        self.t = 0  # step counter

    def current_lr(self) -> float:
        return float(self.scheduler(self.t))

    def step(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        self.t = 0
