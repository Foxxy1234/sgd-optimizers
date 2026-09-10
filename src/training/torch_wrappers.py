from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from optimizers.base import Optimizer


class _TorchOptWrapper(Optimizer):
    torch_cls: Callable[..., torch.optim.Optimizer]
    name = "torch"

    def __init__(self, lr: float, scheduler=None, **torch_kwargs):
        super().__init__(lr=lr, scheduler=scheduler)
        self.torch_kwargs = torch_kwargs
        self._param: torch.Tensor | None = None
        self._opt: torch.optim.Optimizer | None = None

    def _ensure(self, w: np.ndarray) -> tuple[torch.Tensor, torch.optim.Optimizer]:
        if self._param is None or self._opt is None or self._param.numel() != w.size:
            self._param = torch.tensor(w, dtype=torch.float64, requires_grad=True)
            self._opt = self.torch_cls([self._param], lr=self.current_lr(), **self.torch_kwargs)
        else:
            with torch.no_grad():
                self._param.copy_(torch.from_numpy(np.asarray(w, dtype=np.float64)))
        return self._param, self._opt

    def step(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        param, opt = self._ensure(w)
        for pg in opt.param_groups:
            pg["lr"] = self.current_lr()
        opt.zero_grad()
        param.grad = torch.from_numpy(np.ascontiguousarray(g, dtype=np.float64))
        opt.step()
        self.t += 1
        return param.detach().numpy().copy()

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

    def __init__(self, lr=0.01, eps=1e-8, scheduler=None):
        super().__init__(lr=lr, scheduler=scheduler, eps=eps)


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
