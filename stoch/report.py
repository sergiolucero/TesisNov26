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
    "plot_initial_contamination",
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
    q: np.ndarray | None = None,
) -> None:
    """Final concentration field with wells, annotated by pumping rate.

    Passing ``q`` labels each well with its rate in m3/d and scales the marker
    so the active wells are identifiable at a glance. Without ``q`` every well
    is drawn identically, which hides the solution: the optimizer typically
    concentrates the whole budget on one or two wells, and a figure that draws
    nine identical dots makes that invisible.
    """
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
    if q is None:
        ax.plot(wells[:, 0], wells[:, 1], "o", mfc="lime", mec="k", ms=7,
                label="well")
    else:
        q = np.asarray(q, dtype=float).ravel()
        qmax = float(q.max()) if q.size and q.max() > 0 else 0.0
        for i, (x, y) in enumerate(wells):
            rate = float(q[i]) if i < q.size else 0.0
            if qmax > 0 and rate > 1e-9:
                # Size scales with the rate; every active well gets a label.
                size = 7.0 + 15.0 * (rate / qmax)
                ax.plot(x, y, "o", mfc="lime", mec="k", ms=size, zorder=5)
                ax.annotate(f"{rate:.1f}", (x, y), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=8,
                            color="white", fontweight="bold", zorder=6)
            else:
                ax.plot(x, y, "o", mfc="none", mec="lime", ms=7, mew=1.2,
                        zorder=5)
        active = int((q > 1e-9).sum()) if q.size else 0
        total = float(q.sum()) if q.size else 0.0
        ax.set_title(f"{title}\nq*: {active}/{len(wells)} wells active, "
                     f"total {total:.1f} m3/d (labels in m3/d)")
    for x, y in compliance_xy:
        ax.plot(x, y, "s", mfc="cyan", mec="k", ms=8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if q is None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_initial_contamination(
    grid,
    C: np.ndarray,
    wells: np.ndarray,
    path: str | Path,
    creg: float = 1.0,
    source_xy=None,
    compliance_xy=(),
    obs_xy=(),
    times: np.ndarray | None = None,
    C_compliance: np.ndarray | None = None,
    window_start: float | None = None,
    title: str = "Initial contamination (no remediation)",
) -> None:
    """The plume the optimizer inherits, before any well pumps.

    `C` is the contaminant field at the end of the horizon for ``q = 0`` -- what
    the compliance point would see if nothing were done.  The optional right
    panel adds the untreated breakthrough curve at the compliance point, with
    the compliance window shaded, which is what makes the numbers interpretable:
    it shows *when* the plume arrives, not just how much of it there is.

    Note that the concentration field is zero at ``t = 0`` by construction (the
    source injects over time), so "initial" here means "the incoming state",
    not the first time step.
    """
    two_panel = times is not None and C_compliance is not None
    if two_panel:
        # TransportResult.C_record holds one row per *step* while .times has
        # nsteps+1 entries (the initial state is excluded from the record), so
        # align them on the trailing end before plotting.
        times = np.asarray(times, dtype=float)[-len(C_compliance):]
    fig, axes = plt.subplots(
        1, 2 if two_panel else 1,
        figsize=(13.5 if two_panel else 9, 4.5),
        squeeze=False,
    )
    ax = axes[0][0]
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
    ax.plot(wells[:, 0], wells[:, 1], "o", mfc="none", mec="lime", ms=7,
            label=f"{wells.shape[0]} wells (off)")
    if source_xy is not None:
        ax.plot(*source_xy, marker="*", ms=16, mfc="yellow", mec="k",
                label="source")
    for k, (x, y) in enumerate(compliance_xy):
        ax.plot(x, y, "s", mfc="cyan", mec="k", ms=8,
                label="compliance well" if k == 0 else None)
    for k, (x, y) in enumerate(obs_xy):
        ax.plot(x, y, "^", mfc="deepskyblue", mec="k", ms=8,
                label="observation" if k == 0 else None)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{title}\nfinal field, q = 0")
    ax.legend(loc="upper right", fontsize=8)

    if two_panel:
        ax2 = axes[0][1]
        ax2.plot(times, C_compliance, "k-", lw=1.6)
        ax2.axhline(creg, color="r", ls="--", lw=1.0,
                    label=f"creg = {creg:g}")
        if window_start is not None and len(times):
            ax2.axvspan(window_start * times[-1], times[-1], color="r",
                        alpha=0.08, label="compliance window")
        ax2.set_xlabel("time [d]")
        ax2.set_ylabel("C at compliance point [mg/L]")
        ax2.set_title("Untreated breakthrough curve")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8)

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
