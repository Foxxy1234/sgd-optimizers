from __future__ import annotations

import numpy as np

from schedulers import ConstantLR, Scheduler


class Optimizer:
    name = "base"

    def __init__(self, lr: float = 0.01, scheduler: Scheduler | None = None):
        self.scheduler = scheduler if scheduler is not None else ConstantLR(lr)
        self.t = 0

    def current_lr(self) -> float:
        return float(self.scheduler(self.t))

    def step(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        self.t = 0
