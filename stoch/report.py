"""Reporting: manifests, metrics and figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

__all__ = [
    "ensure_dir",
    "save_json",
    "plot_layout",
    "plot_plume",
    "plot_remediation",
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
    _draw_wells(ax, wells, q, off_label="well", ly=grid.ly)
    if q is not None:
        qa = np.asarray(q, dtype=float).ravel()
        active = int((qa > 1e-9).sum()) if qa.size else 0
        total = float(qa.sum()) if qa.size else 0.0
        ax.set_title(f"{title}\nq*: {active}/{len(wells)} wells active, "
                     f"total {total:.1f} m3/d (labels in m3/d)")
    else:
        ax.set_title(title)
    for x, y in compliance_xy:
        ax.plot(x, y, "s", mfc="cyan", mec="k", ms=8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_wells(ax, wells, q=None, off_label="wells (off)",
                ly: float | None = None) -> None:
    """Wells, annotated by pumping rate when ``q`` is given.

    Sizing and labelling matter: the optimizer concentrates the whole budget on
    one well, and identical dots make that solution look like any other layout.
    """
    if q is None:
        ax.plot(wells[:, 0], wells[:, 1], "o", mfc="none", mec="lime", ms=7,
                mew=1.2, zorder=5, label=off_label)
        return
    q = np.asarray(q, dtype=float).ravel()
    qmax = float(q.max()) if q.size and q.max() > 0 else 0.0
    for i, (x, y) in enumerate(wells):
        rate = float(q[i]) if i < q.size else 0.0
        if qmax > 0 and rate > 1e-9:
            ax.plot(x, y, "o", mfc="lime", mec="k", ms=7.0 + 15.0 * rate / qmax,
                    zorder=5)
            # Stroked text: the label sits on top of the plume, and plain
            # white disappears against the bright (high-concentration) core.
            # Wells near the top edge get the label underneath instead, where
            # it cannot collide with the panel title.
            above = ly is None or y < 0.85 * ly
            ax.annotate(f"{rate:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 10 if above else -14), ha="center",
                        fontsize=8, color="white", fontweight="bold", zorder=6,
                        path_effects=[pe.withStroke(linewidth=2.0,
                                                    foreground="black")])
        else:
            ax.plot(x, y, "o", mfc="none", mec="lime", ms=7, mew=1.2, zorder=5)


def _draw_points(ax, source_xy=None, compliance_xy=(), obs_xy=(),
                 legend: bool = True) -> None:
    """Source, compliance and observation markers, labelled once each."""
    if source_xy is not None:
        ax.plot(*source_xy, marker="*", ms=16, mfc="yellow", mec="k",
                zorder=6, label="source" if legend else None)
    for k, (x, y) in enumerate(compliance_xy):
        ax.plot(x, y, "s", mfc="cyan", mec="k", ms=8, zorder=6,
                label="compliance well" if (legend and k == 0) else None)
    for k, (x, y) in enumerate(obs_xy):
        ax.plot(x, y, "^", mfc="deepskyblue", mec="k", ms=8, zorder=6,
                label="observation" if (legend and k == 0) else None)


def _draw_field(ax, fig, grid, C, creg: float, title: str) -> float:
    """One concentration panel with its *own* colour scale.

    Independent scales are deliberate: a shared one makes the remediated panel
    look empty next to the untreated peak, and the contrast is the point. The
    peak is in the title and on the colorbar so the scaling stays honest.
    """
    peak = float(np.max(C)) if C.size else 0.0
    vmax = peak if peak > 0 else 1.0
    im = ax.imshow(C, origin="lower", extent=[0, grid.lx, 0, grid.ly],
                   cmap="magma", aspect="equal", vmin=0.0, vmax=vmax)
    fig.colorbar(im, ax=ax, label=f"C [mg/L], scale 0-{vmax:.3g}",
                 fraction=0.046)
    if peak > creg:
        # Above creg the contour exists; below it a level line would be empty.
        ax.contour((np.arange(grid.nx) + 0.5) * grid.dx,
                   (np.arange(grid.ny) + 0.5) * grid.dy, C,
                   levels=[creg], colors="w", linewidths=1.0)
    ax.set_title(f"{title}  (peak {peak:.3g} mg/L)")
    ax.set_xlabel("x [m]")
    return peak


def plot_remediation(
    grid,
    C_off: np.ndarray,
    C_on: np.ndarray,
    wells: np.ndarray,
    path: str | Path,
    creg: float = 1.0,
    q: np.ndarray | None = None,
    source_xy=None,
    compliance_xy=(),
    obs_xy=(),
    times: np.ndarray | None = None,
    C_comp_off: np.ndarray | None = None,
    C_comp_on: np.ndarray | None = None,
    window_start: float | None = None,
    scenario: int | None = None,
    title: str | None = None,
) -> dict:
    """What the plume does without remediation, and what ``q`` buys.

    Three panels, left to right: the end-of-horizon field with every well off,
    the same field under ``q`` (wells labelled in m3/d), and the breakthrough
    curves at the compliance point (untreated dashed, treated solid) against
    ``creg`` with the compliance window shaded. Together they answer the three
    questions the numbers in ``metrics.json`` raise: how big is the problem,
    what is left after pumping, and when does the plume arrive.

    Both callers share this one figure -- ``run_optimization.py`` renders it
    from the run it just finished, and ``SCRIPTS/plot_policy_plume.py``
    re-renders it from a stored ``policy.json`` without rerunning the
    optimization.

    Note the concentration field is zero at ``t = 0`` by construction (the
    source injects over time), so "untreated" means the incoming state at the
    end of the horizon, not the first time step.

    Returns a summary dict (peaks, compliance maxima, reduction) so callers can
    persist the numbers instead of re-deriving them from pixels.
    """
    curves = times is not None and C_comp_off is not None
    ncols = 3 if curves else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols + 1.5, 4.6),
                             squeeze=False)
    ax0, ax1 = axes[0][0], axes[0][1]

    peak_off = _draw_field(ax0, fig, grid, C_off, creg,
                           "No remediation")
    _draw_wells(ax0, wells, None, ly=grid.ly)
    _draw_points(ax0, source_xy, compliance_xy, obs_xy)
    ax0.set_ylabel("y [m]")
    ax0.legend(loc="upper right", fontsize=8)

    peak_on = _draw_field(ax1, fig, grid, C_on, creg,
                          "Under q*" if q is not None else "Under pumping")
    _draw_wells(ax1, wells, q if q is not None else np.zeros(len(wells)),
                ly=grid.ly)
    _draw_points(ax1, source_xy, compliance_xy, obs_xy, legend=False)

    reduction = 100.0 * (1.0 - peak_on / peak_off) if peak_off > 0 else None
    where = f"test scenario {scenario}: " if scenario is not None else ""
    red_txt = f"{reduction:.1f}%" if reduction is not None else "n/a"
    fig.suptitle(f"{where}peak {peak_off:.3g} -> {peak_on:.3g} mg/L "
                 f"({red_txt} lower); each field panel has its own colour scale")

    summary = {"peak_off": peak_off, "peak_on": peak_on,
               "peak_reduction_pct": reduction, "creg": float(creg)}

    if curves:
        ax2 = axes[0][2]
        # TransportResult.C_record has one row per step while .times has
        # nsteps+1 entries; align on the trailing end.
        t = np.asarray(times, dtype=float)[-len(C_comp_off):]
        ax2.plot(t, C_comp_off, "--", color="0.45", lw=1.6,
                 label="no remediation")
        if C_comp_on is not None:
            t_on = np.asarray(times, dtype=float)[-len(C_comp_on):]
            ax2.plot(t_on, C_comp_on, "k-", lw=1.8, label="under q*")
        ax2.axhline(creg, color="r", ls="--", lw=1.0, label=f"creg = {creg:g}")
        if window_start is not None and len(t):
            ax2.axvspan(window_start * t[-1], t[-1], color="r", alpha=0.08,
                        label="compliance window")
        ax2.set_xlabel("time [d]")
        ax2.set_ylabel("C at compliance point [mg/L]")
        ax2.set_title("At the compliance point")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8)
        summary["compliance_max_off"] = float(np.max(C_comp_off))
        if C_comp_on is not None:
            summary["compliance_max_on"] = float(np.max(C_comp_on))

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return summary


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
