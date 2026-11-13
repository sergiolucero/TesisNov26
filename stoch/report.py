"""Reporting: manifests, metrics and figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = [
    "ensure_dir",
    "save_json",
    "plot_layout",
    "plot_plume",
    "plot_convergence",
    "plot_distributions",
    "write_manifest",
]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def plot_layout(
    cfg: dict,
    grid,
    K: np.ndarray,
    mask: np.ndarray,
    wells: np.ndarray,
    compliance_xy,
    path: str | Path,
    title: str = "Design plume and well layout",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(
        K, origin="lower", extent=[0, grid.lx, 0, grid.ly],
        cmap="viridis", aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="K [m/d]")
    ax.contour(
        (np.arange(grid.nx) + 0.5) * grid.dx,
        (np.arange(grid.ny) + 0.5) * grid.dy,
        mask.astype(float), levels=[0.5], colors="w", linewidths=1.2,
    )
    ax.plot(wells[:, 0], wells[:, 1], "o", mfc="red", mec="k", ms=7,
            label=f"{wells.shape[0]} extraction wells")
    for k, (x, y) in enumerate(compliance_xy):
        ax.plot(x, y, "s", mfc="cyan", mec="k", ms=8,
                label="compliance well" if k == 0 else None)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_plume(
    grid,
    C: np.ndarray,
    wells: np.ndarray,
    compliance_xy,
    path: str | Path,
    creg: float = 1.0,
    title: str = "Final plume and wells",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(
        C, origin="lower", extent=[0, grid.lx, 0, grid.ly],
        cmap="magma", aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="C [mg/L]")
    ax.contour(
        (np.arange(grid.nx) + 0.5) * grid.dx,
        (np.arange(grid.ny) + 0.5) * grid.dy,
        C, levels=[creg], colors="w", linewidths=1.2,
    )
    ax.plot(wells[:, 0], wells[:, 1], "o", mfc="lime", mec="k", ms=7)
    for x, y in compliance_xy:
        ax.plot(x, y, "s", mfc="cyan", mec="k", ms=8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(history: list[dict], path: str | Path) -> None:
    if not history:
        return
    f = [h["f"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(np.arange(len(f)), f, "k.-")
    ax.set_xlabel("evaluation")
    ax.set_ylabel("penalised SAA objective")
    ax.set_title("Optimizer convergence")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distributions(ev, tau_cap: float, path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].hist(ev.G, bins=20, color="steelblue")
    axes[0].axvline(0.0, color="r", ls="--", label="compliance")
    axes[0].set_xlabel("G (max violation)")
    axes[0].set_ylabel("scenarios")
    axes[0].legend(fontsize=8)
    axes[1].hist(np.minimum(ev.tau, tau_cap), bins=20, color="darkorange")
    axes[1].set_xlabel("cleanup time [d]")
    axes[1].set_ylabel("scenarios")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_manifest(path: str | Path, cfg: dict, extra: dict | None = None) -> None:
    manifest = {
        "config": cfg,
        "extra": extra or {},
        "code": "TESIS/stoch (stages O-0 to O-4)",
    }
    save_json(path, manifest)
