from __future__ import annotations

import time

import torch

from autodiff.forward import Dual, forward_grad
from autodiff.numeric import numerical_grad


def mse_loss_numpy(X, y):
    def f(w):
        r = X @ w - y
        return 0.5 * float(r @ r) / len(y)
    return f


def mse_loss_dual(X, y):
    def f(w):
        n, d = X.shape
        s = Dual(0.0)
        for i in range(n):
            dot = Dual(0.0)
            for j in range(d):
                dot = dot + X[i, j] * w[j]
            r = dot - y[i]
            s = s + r * r
        return s * (0.5 / n)
    return f


def analytic_mse_grad(X, y, w):
    r = X @ w - y
    return (X.T @ r) / len(y)


def time_numeric(X, y, w, eps=1e-6, repeat=1):
    f = mse_loss_numpy(X, y)
    g = None
    t0 = time.perf_counter()
    for _ in range(repeat):
        g = numerical_grad(f, w, eps=eps)
    return (time.perf_counter() - t0) / repeat, g


def time_forward(X, y, w, repeat=1):
    f = mse_loss_dual(X, y)
    g = None
    t0 = time.perf_counter()
    for _ in range(repeat):
        g = forward_grad(f, w)
    return (time.perf_counter() - t0) / repeat, g


def time_reverse_torch(X, y, w, repeat=1):
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    g = None
    t0 = time.perf_counter()
    for _ in range(repeat):
        wt = torch.tensor(w, dtype=torch.float64, requires_grad=True)
        r = Xt @ wt - yt
        loss = 0.5 * (r * r).mean()
        loss.backward()
        gt = wt.grad
        g = None if gt is None else gt.detach().numpy().copy()
    return (time.perf_counter() - t0) / repeat, g
