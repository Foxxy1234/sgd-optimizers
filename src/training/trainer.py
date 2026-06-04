"""Mini-batch training loop with metric collection."""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrainHistory:
    epoch_loss: list[float] = field(default_factory=list)  # full-train loss per epoch
    step_loss: list[float] = field(default_factory=list)   # per-step (mini-batch) loss
    weights: list[np.ndarray] = field(default_factory=list)  # per-epoch snapshot (small models)
    epoch_time_s: list[float] = field(default_factory=list)
    peak_mem_bytes: int = 0
    n_grad_evals: int = 0
    n_steps: int = 0
    flops_estimate: int = 0


def estimate_flops_mse(n_batch: int, d_params: int) -> int:
    """Rough FLOP count for one mini-batch MSE gradient step.

    Forward: 2 * n * d (matvec) + n (sub) + n (residual squared sum for loss).
    Backward: 2 * d * n (X^T @ r) + d (divide).
    Optimizer: ~3 * d (a few elementwise ops on average).
    """
    return 4 * n_batch * d_params + 2 * n_batch + 4 * d_params


def train(
    model,
    optimizer,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 20,
    batch_size: int = 32,
    init_w: np.ndarray | None = None,
    seed: int = 0,
    track_weights: bool = False,
    on_step=None,
) -> tuple[np.ndarray, TrainHistory]:
    """Generic mini-batch trainer.

    Hyperparameters are explicit in the signature: epochs, batch_size, seed, etc.
    Optimizer carries its own lr/momentum/etc.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    w = init_w.copy() if init_w is not None else model.init_params(X.shape[1])
    optimizer.reset()

    hist = TrainHistory()
    tracemalloc.start()
    try:
        for epoch in range(epochs):
            t0 = time.perf_counter()
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                Xb, yb = X[idx], y[idx]
                g = model.grad(w, Xb, yb)
                w = optimizer.step(w, g)
                hist.n_grad_evals += 1
                hist.n_steps += 1
                hist.flops_estimate += estimate_flops_mse(len(idx), len(w))
                if on_step is not None:
                    on_step(w, g)
                # per-step loss on the same batch is cheap and informative
                hist.step_loss.append(model.loss(w, Xb, yb))
            hist.epoch_time_s.append(time.perf_counter() - t0)
            hist.epoch_loss.append(model.loss(w, X, y))
            if track_weights:
                hist.weights.append(w.copy())
        _, peak = tracemalloc.get_traced_memory()
        hist.peak_mem_bytes = int(peak)
    finally:
        tracemalloc.stop()

    return w, hist
