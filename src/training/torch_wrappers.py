"""Wrappers around torch.optim with the same `optimizer.step(w, g)` interface as our own.

These allow apples-to-apples comparison in the same training loop. Hyperparameters
are passed through to the underlying torch optimizer.
"""
from __future__ import annotations

import numpy as np
import torch

from optimizers.base import Optimizer


class _TorchOptWrapper(Optimizer):
    """Generic adapter from `optimizer.step(w_np, g_np) -> w_np` to torch.optim."""

    torch_cls: type
    name = "torch"

    def __init__(self, lr: float, scheduler=None, **torch_kwargs):
        super().__init__(lr=lr, scheduler=scheduler)
        self.torch_kwargs = torch_kwargs
        self._param: torch.Tensor | None = None
        self._opt: torch.optim.Optimizer | None = None

    def _ensure(self, w: np.ndarray):
        if self._param is None or self._param.numel() != w.size:
            self._param = torch.tensor(w, dtype=torch.float64, requires_grad=True)
            self._opt = self.torch_cls([self._param], lr=self.current_lr(), **self.torch_kwargs)
        else:
            with torch.no_grad():
                self._param.copy_(torch.from_numpy(np.asarray(w, dtype=np.float64)))

    def step(self, w, g):
        self._ensure(w)
        # set lr from scheduler each step
        for pg in self._opt.param_groups:
            pg["lr"] = self.current_lr()
        self._opt.zero_grad()
        # write incoming numpy grad into .grad
        self._param.grad = torch.from_numpy(np.ascontiguousarray(g, dtype=np.float64))
        self._opt.step()
        self.t += 1
        return self._param.detach().numpy().copy()

    def reset(self):
        super().reset()
        self._param = None
        self._opt = None


class TorchSGD(_TorchOptWrapper):
    name = "torch.SGD"
    torch_cls = torch.optim.SGD


class TorchMomentum(_TorchOptWrapper):
    name = "torch.SGD(momentum)"
    torch_cls = torch.optim.SGD

    def __init__(self, lr=0.01, momentum=0.9, scheduler=None):
        super().__init__(lr=lr, scheduler=scheduler, momentum=momentum)


class TorchNesterov(_TorchOptWrapper):
    name = "torch.SGD(nesterov)"
    torch_cls = torch.optim.SGD

    def __init__(self, lr=0.01, momentum=0.9, scheduler=None):
        super().__init__(lr=lr, scheduler=scheduler, momentum=momentum, nesterov=True)


class TorchAdaGrad(_TorchOptWrapper):
    name = "torch.Adagrad"
    torch_cls = torch.optim.Adagrad


class TorchRMSProp(_TorchOptWrapper):
    name = "torch.RMSprop"
    torch_cls = torch.optim.RMSprop

    def __init__(self, lr=0.01, alpha=0.99, eps=1e-8, scheduler=None):
        super().__init__(lr=lr, scheduler=scheduler, alpha=alpha, eps=eps)


class TorchAdam(_TorchOptWrapper):
    name = "torch.Adam"
    torch_cls = torch.optim.Adam

    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, scheduler=None):
        super().__init__(lr=lr, scheduler=scheduler, betas=betas, eps=eps)


class TorchAdamW(_TorchOptWrapper):
    name = "torch.AdamW"
    torch_cls = torch.optim.AdamW

    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, scheduler=None):
        super().__init__(lr=lr, scheduler=scheduler, betas=betas, eps=eps, weight_decay=weight_decay)
