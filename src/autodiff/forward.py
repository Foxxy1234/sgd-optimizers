from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


class Dual:
    __slots__ = ("primal", "tangent")

    def __init__(self, primal: float, tangent: float = 0.0):
        self.primal = float(primal)
        self.tangent = float(tangent)

    def __add__(self, other):
        o = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.primal + o.primal, self.tangent + o.tangent)

    __radd__ = __add__

    def __sub__(self, other):
        o = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.primal - o.primal, self.tangent - o.tangent)

    def __rsub__(self, other):
        o = other if isinstance(other, Dual) else Dual(other)
        return Dual(o.primal - self.primal, o.tangent - self.tangent)

    def __mul__(self, other):
        o = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.primal * o.primal,
                    self.tangent * o.primal + self.primal * o.tangent)

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.primal / o.primal,
                    (self.tangent * o.primal - self.primal * o.tangent) / (o.primal ** 2))

    def __rtruediv__(self, other):
        o = other if isinstance(other, Dual) else Dual(other)
        return o.__truediv__(self)

    def __neg__(self):
        return Dual(-self.primal, -self.tangent)

    def __pow__(self, p: float):
        return Dual(self.primal ** p, p * (self.primal ** (p - 1)) * self.tangent)

    def __repr__(self):
        return f"Dual({self.primal}, ε·{self.tangent})"


def exp(x):
    if isinstance(x, Dual):
        e = math.exp(x.primal)
        return Dual(e, e * x.tangent)
    return math.exp(x)


def log(x):
    if isinstance(x, Dual):
        return Dual(math.log(x.primal), x.tangent / x.primal)
    return math.log(x)


def sqrt(x):
    if isinstance(x, Dual):
        s = math.sqrt(x.primal)
        return Dual(s, x.tangent / (2.0 * s))
    return math.sqrt(x)


def forward_grad(f: Callable[[np.ndarray], Dual], x: np.ndarray) -> np.ndarray:
    n = x.size
    g = np.zeros(n, dtype=np.float64)
    for i in range(n):
        duals = np.array([Dual(x[j], 1.0 if j == i else 0.0) for j in range(n)], dtype=object)
        out = f(duals)
        g[i] = out.tangent if isinstance(out, Dual) else 0.0
    return g
