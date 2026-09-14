from __future__ import annotations

from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np


@contextmanager
def figure(title=None, xlabel=None, ylabel=None, logx=False, logy=False,
           figsize=(8, 5), ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    main = axes if ncols == 1 else axes[0]
    yield axes
    if title:
        main.set_title(title)
    if xlabel:
        main.set_xlabel(xlabel)
    if ylabel:
        main.set_ylabel(ylabel)
    if logx:
        main.set_xscale("log")
    if logy:
        main.set_yscale("log")
    for ax in np.atleast_1d(axes):
        if ax.get_legend() is None and ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.tight_layout()
    plt.show()


def plot_contours(f, xlim, ylim, trajectories=None, optimum=None, levels=30,
                  resolution=80, **figure_kwargs) -> None:
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    gx, gy = np.meshgrid(xs, ys)
    z = np.array([f(np.array([a, b])) for a, b in zip(gx.ravel(), gy.ravel())])
    with figure(**figure_kwargs) as ax:
        cs = ax.contour(gx, gy, z.reshape(gx.shape), levels=levels,
                        cmap="viridis", alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=7)
        for name, points in (trajectories or {}).items():
            t = np.asarray(points)
            ax.plot(t[:, 0], t[:, 1], marker=".", label=name)
        if optimum is not None:
            ax.scatter(optimum[0], optimum[1], marker="*", s=200, color="red",
                       zorder=5, label="w*")
