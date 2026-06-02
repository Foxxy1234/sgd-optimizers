"""Learning rate schedulers. Each scheduler is a callable: step (int) -> lr (float)."""
from __future__ import annotations

import math


class ConstantLR:
    def __init__(self, lr: float):
        self.lr = lr

    def __call__(self, step: int) -> float:
        return self.lr


class StepDecay:
    """lr = lr0 * gamma ** (step // step_size)."""

    def __init__(self, lr0: float, step_size: int, gamma: float = 0.5):
        self.lr0 = lr0
        self.step_size = step_size
        self.gamma = gamma

    def __call__(self, step):
        return self.lr0 * (self.gamma ** (step // self.step_size))


class ExpDecay:
    """lr = lr0 * exp(-rate * step)."""

    def __init__(self, lr0: float, rate: float):
        self.lr0 = lr0
        self.rate = rate

    def __call__(self, step):
        return self.lr0 * math.exp(-self.rate * step)


class InvDecay:
    """lr = lr0 / (1 + rate * step)."""

    def __init__(self, lr0: float, rate: float):
        self.lr0 = lr0
        self.rate = rate

    def __call__(self, step):
        return self.lr0 / (1.0 + self.rate * step)


class CosineWarmup:
    """Linear warmup over `warmup` steps to lr0, then cosine decay to `lr_min` over `total`."""

    def __init__(self, lr0: float, total: int, warmup: int = 0, lr_min: float = 0.0):
        self.lr0 = lr0
        self.total = total
        self.warmup = warmup
        self.lr_min = lr_min

    def __call__(self, step):
        if step < self.warmup:
            return self.lr0 * (step + 1) / max(self.warmup, 1)
        progress = (step - self.warmup) / max(self.total - self.warmup, 1)
        progress = min(progress, 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_min + (self.lr0 - self.lr_min) * cos
