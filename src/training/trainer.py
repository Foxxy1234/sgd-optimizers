from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrainHistory:
    epoch_loss: list[float] = field(default_factory=list)
    epoch_time_s: list[float] = field(default_factory=list)
    peak_mem_bytes: int = 0
    opt_state_bytes: int = 0
    n_steps: int = 0


def optimizer_state_bytes(optimizer) -> int:
    return int(sum(v.nbytes for v in vars(optimizer).values() if isinstance(v, np.ndarray)))


def train(
    model,
    optimizer,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
    init_w: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, TrainHistory]:
    rng = np.random.default_rng(seed)
    n = len(X)
    w = init_w.copy() if init_w is not None else model.init_params(X.shape[1])
    optimizer.reset()

    hist = TrainHistory()
    tracemalloc.start()
    try:
        for _ in range(epochs):
            t0 = time.perf_counter()
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                Xb, yb = X[idx], y[idx]
                g = model.grad(w, Xb, yb)
                w = optimizer.step(w, g)
                hist.n_steps += 1
            hist.epoch_time_s.append(time.perf_counter() - t0)
            hist.epoch_loss.append(model.loss(w, X, y))
        _, peak = tracemalloc.get_traced_memory()
        hist.peak_mem_bytes = int(peak)
        hist.opt_state_bytes = optimizer_state_bytes(optimizer)
    finally:
        tracemalloc.stop()

    return w, hist
