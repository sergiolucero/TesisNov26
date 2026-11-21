#!/usr/bin/env python3
"""Generate the article figure set from the repository's own runs.

Every panel is computed from the production code paths (`run_optimization`'s
config helpers, `stoch.saa.Archive`, `physics.fv2d`) or from a finished run's
artifacts -- nothing is hand-drawn or hard-coded, except the compute-cost panel,
which plots measured numbers and says so in the caption.

Usage
-----
    python3 SCRIPTS/make_figures.py                # everything (~3 min, NumPy)
    python3 SCRIPTS/make_figures.py --only plume   # one figure
    python3 SCRIPTS/make_figures.py --out FIGURES

Figures
-------
fig1_1d_verification   1D reference vs analytic Ogata-Banks (the Phase 1 gate)
fig2_field_layout      conductivity realization, wells, source, compliance
fig3_plume             no pumping vs certified policy, fields + compliance trace
fig4_convergence       search trajectory of the two certified runs
fig5_chance            out-of-sample distributions of G and cleanup time
fig6_certificate       reliability vs sample size with Wilson intervals
fig8_tail_spread       tail vs proportion: which statistic transports
fig7_compute           measured cost per compass sweep by backend/card

Design notes: all figures are 300 dpi PNG with a fixed style (no LaTeX needed),
sized for a single or double column.  Colour-vision-safe palette; the compliance
level is always the same dashed line so panels can be read against each other.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stoch.saa import Archive  # noqa: E402

# Certified runs (see DOCS/Results.md).  Kept as constants because a figure that
# silently follows whatever run is newest would stop matching the paper.
RUN_F64 = "runs/colab_multi/20261118-114831/r0"
RUN_F32 = "runs/colab_multi/20261120-111849/r0"
CERT_F64 = "out/cert/f64_M400.json"
CERT_F32 = "out/cert/f32_M400.json"
TAIL_CAL = "runs/tail_calibration"    # one check_tail.py JSON per (policy, draw)
CONFIG = "configs/opt/hydrology_gpu.yaml"

# ---------------------------------------------------------------- house style
C_NOPUMP = "#1b6ca8"
C_POLICY = "#c1440e"
C_ACCENT = "#2a9d8f"
C_GATE = "#6a4c93"
COMPLIANCE_LS = (0, (5, 2))


def use_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


def _save(fig, out: Path, name: str) -> Path:
    path = out / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return path


def _load_cfg(path: str) -> dict:
    from stoch.config import load_config
    return load_config(ROOT / path)


def _run_opt_helpers():
    """Import the production builders from the script that defines them."""
    sys.path.insert(0, str(ROOT / "SCRIPTS"))
    import run_optimization as ro
    return ro


def _run_metrics(run_dir: Path) -> dict:
    """Read a finished run's metrics, unpacking ``artifacts.zip`` if needed.

    The downloaded run directory only holds the zip (the VM is already gone), so
    the figure script has to open it rather than assume an extracted tree.
    """
    for cand in (run_dir / "out/opt/full_opt/metrics.json",
                 run_dir / "artifacts/out/opt/full_opt/metrics.json"):
        if cand.exists():
            return json.loads(cand.read_text())
    zip_path = run_dir / "artifacts.zip"
    if zip_path.exists():
        import zipfile
        with zipfile.ZipFile(zip_path) as zf:
            name = next((n for n in zf.namelist()
                         if n.endswith("out/opt/full_opt/metrics.json")), None)
            if name is not None:
                return json.loads(zf.read(name))
    raise FileNotFoundError(f"no metrics.json under {run_dir} (and no artifacts.zip)")


# ------------------------------------------------------------------- figure 1
PE = 20.0
TAU_MAX = 0.5


def _load_pinn():
    """Load the archived PINN checkpoint, or None if it is not usable.

    The checkpoints of ``MLP`` and ``HardBCMLP`` share their ``net.*`` keys, so
    the class is inferred by test: whichever reproduces the analytic solution
    better is the one that produced the file.  That keeps the figure honest even
    if the checkpoint is later replaced by a soft-BC run.
    """
    path = ROOT / "out/pinn/model.pt"
    if not path.exists():
        return None, "no checkpoint"
    try:
        import torch
        sys.path.insert(0, str(ROOT / "SCRIPTS"))
        from solve_1d_pinn import HardBCMLP, MLP  # type: ignore
        from SCRIPTS.hydrology import u_reference  # type: ignore
    except Exception as exc:                      # torch missing, etc.
        return None, f"{type(exc).__name__}"

    # default dtype (float32): solve_1d_pinn's own metrics build float32 grids
    xi = torch.linspace(0.0, 1.0, 200).unsqueeze(-1)
    tau = torch.full_like(xi, TAU_MAX)
    ref = u_reference(xi.numpy().ravel(), np.full(200, TAU_MAX), PE)
    best = None
    for name, factory in (("hard-BC", lambda: HardBCMLP(64, 4, eps=1e-3)),
                          ("soft-BC", lambda: MLP(64, 4))):
        try:
            model = factory()
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            with torch.no_grad():
                pred = model(xi, tau).numpy().ravel()
            err = float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))
        except Exception:
            continue
        if best is None or err < best[2]:
            best = (model, name, err)
    if best is None:
        return None, "incompatible checkpoint"
    return best, ""


def fig_1d(out: Path) -> None:
    """1D verification: analytic reference, the trained surrogate, and the family.

    The surrogate panel is the point of the figure -- a reference curve alone
    proves nothing about the PINN.  Its error is computed here from the archived
    checkpoint, so the annotation cannot drift from the artifact.
    """
    from SCRIPTS.hydrology import C_reference, generate_scenario, u_reference  # type: ignore

    sc = generate_scenario(C0=1.0, v=0.1, D=0.5, L=100.0, T=500.0, nx=256, nt=256)

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.1))

    # (a) dimensional reference at three times
    ax = axes[0]
    for frac, colour in ((0.25, "#8ecae6"), (0.5, "#219ebc"), (1.0, "#023047")):
        t = frac * sc.T
        ax.plot(sc.x, C_reference(sc.x, t, sc.C0, sc.v, sc.D),
                color=colour, lw=1.8, label=f"t = {t:.0f} d")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("C / C$_0$")
    ax.set_title(f"(a) Ogata–Banks reference, Pe = {sc.Pe:.0f}", fontsize=9.5)
    ax.legend(loc="lower left")
    ax.set_xlim(0, 100)

    # (b) trained surrogate vs the reference, dimensionless
    ax = axes[1]
    xi = np.linspace(0.0, 1.0, 400)
    ref = u_reference(xi, np.full_like(xi, TAU_MAX), PE)
    ax.plot(xi, ref, "k-", lw=1.8, label="analytic")
    loaded, note = _load_pinn()
    if loaded is not None:
        model, kind, _ = loaded
        import torch
        with torch.no_grad():
            pred = model(torch.tensor(xi, dtype=torch.float32).unsqueeze(-1),
                         torch.full((xi.size, 1), TAU_MAX, dtype=torch.float32)
                         ).numpy().ravel()
        ax.plot(xi, pred, "--", color=C_POLICY, lw=1.6, label=f"PINN ({kind})")
        # quote the project's own space-time metric, not a single-slice one, so
        # the panel agrees with the number in DOCS/Results.md
        try:
            from solve_1d_pinn import region_errors  # type: ignore
            errs = region_errors(model, PE, TAU_MAX)   # dict of named regions
            full = float(errs["full"])
            near_corner = float(errs["tau,xi>0.005"])
            ax.set_title(f"(b) surrogate: rel-L2 {full:.1e} full,\n"
                         f"{near_corner:.1e} away from corner", fontsize=9.5)
        except Exception:
            ax.set_title(f"(b) surrogate at τ = {TAU_MAX}", fontsize=9.5)
    else:
        ax.set_title(f"(b) surrogate unavailable ({note})", fontsize=9.5)
    ax.set_xlabel("ξ = x/L")
    ax.set_ylabel("u")
    ax.legend(loc="lower left")

    # (c) the scenario family the surrogate must cover
    ax = axes[2]
    for Pe, colour in ((2.0, "#e07a5f"), (20.0, "#3d405b"), (200.0, "#81b29a")):
        ax.plot(xi, u_reference(xi, np.full_like(xi, TAU_MAX), Pe),
                color=colour, lw=1.8, label=f"Pe = {Pe:g}")
    ax.set_xlabel("ξ = x/L")
    ax.set_title("(c) scenario family to cover", fontsize=9.5)
    ax.legend(loc="lower left")
    fig.tight_layout()
    _save(fig, out, "fig1_1d_verification")


# ------------------------------------------------------------------- figure 2
def fig_field(out: Path, cfg: dict) -> None:
    ro = _run_opt_helpers()
    grid = ro.make_grid(cfg)
    kl = ro.make_kl(cfg, grid)
    gen = ro.ScenarioGenerator(
        grid, kl,
        random_source=bool(cfg["random_field"]["random_source"]),
        random_gradient=bool(cfg["random_field"]["random_gradient"]),
    )
    scen = gen.sample(int(cfg.get("seed", {}).get("train", 7000)))

    fig = plt.figure(figsize=(9.4, 4.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    k = np.asarray(scen.K, dtype=float)
    im = ax.imshow(k, origin="lower", extent=[0, grid.lx, 0, grid.ly],
                   cmap="viridis", aspect="equal",
                   norm=LogNorm(vmin=float(np.percentile(k, 2)),
                                vmax=float(np.percentile(k, 98))))
    fig.colorbar(im, ax=ax, label="K [m/d]  (log)", shrink=0.8)
    # The well layout is whatever the certified run actually used: re-deriving it
    # here would risk plotting a different physical problem (a mistake made once
    # already with a transposed grid).
    wells = np.asarray(_run_metrics(ROOT / RUN_F64)["stages"]["O0"]["well_xy"],
                       dtype=float)
    if wells.size:
        ax.plot(wells[:, 0], wells[:, 1], "o", ms=6, mfc="white", mec="k", mew=0.9,
                label="candidate wells")
    sx, sy = cfg["source"]["x"], cfg["source"]["y"]
    ax.plot([sx], [sy], marker="*", ms=15, color="#d62828", mec="k", mew=0.6,
            label="source")
    cx, cy = cfg["compliance"]["x"], cfg["compliance"]["y"]
    ax.plot([cx], [cy], marker="s", ms=7, mfc="none", mec="white", mew=1.6,
            label="compliance point")
    ox, oy = cfg["observation"]["x"], cfg["observation"]["y"]
    ax.plot([ox], [oy], marker="^", ms=6, mfc="none", mec="#f4a261", mew=1.4,
            label="observation well")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"(a) one conductivity realization\n"
                 f"σ = {cfg['random_field']['sigma']}, "
                 f"ℓ = {cfg['random_field']['ell_x']:.0f} m, "
                 f"{grid.nx}×{grid.ny} cells", fontsize=9.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              fontsize=7.5)

    # (b) KL variance retention: how many modes the field really needs
    ax = fig.add_subplot(gs[0, 1])
    frac = np.cumsum(np.asarray(kl.weights, dtype=float)) / float(kl.weights.sum())
    ax.plot(np.arange(1, frac.size + 1), frac, color=C_ACCENT, lw=1.8)
    ax.axhline(float(cfg["random_field"]["target_variance"]), color="k",
               ls=COMPLIANCE_LS, lw=1.0,
               label=f"target = {cfg['random_field']['target_variance']:.2f}")
    ax.axvline(kl.n_kl, color=C_POLICY, lw=1.2,
               label=f"$n_{{KL}}$ = {kl.n_kl}")
    ax.set_xscale("log")
    ax.set_xlabel("number of KL modes")
    ax.set_ylabel("retained variance fraction")
    ax.set_title("(b) Karhunen–Loève truncation")
    ax.legend(loc="lower right")
    _save(fig, out, "fig2_field_layout")


# ------------------------------------------------------------------- figure 3
def _archive(cfg: dict, ro, n_scen: int, seed_offset: int):
    grid = ro.make_grid(cfg)
    kh = ro.make_kl(cfg, grid)
    aq = ro.base_aquifer(cfg, grid)
    gen = ro.ScenarioGenerator(
        grid, kh,
        random_source=bool(cfg["random_field"]["random_source"]),
        random_gradient=bool(cfg["random_field"]["random_gradient"]),
    )
    base = int(cfg.get("seed", {}).get("test", 700000))
    scen = [gen.sample(base + seed_offset + i) for i in range(n_scen)]
    fwd = ro.make_forward(cfg)
    return grid, aq, scen, fwd


def fig_plume(out: Path, cfg: dict) -> None:
    """The physical result: what pumping actually does to the plume."""
    ro = _run_opt_helpers()
    metrics = _run_metrics(ROOT / RUN_F64)
    q_star = np.asarray(metrics["stages"]["opt"]["q_star"], dtype=float)
    wells = np.asarray(metrics["stages"]["O0"]["well_xy"], dtype=float)

    # Same 40 test scenarios used by fig5, so the two figures cannot disagree.
    n_scen = 40
    grid, aq, scen, fwd = _archive(cfg, ro, n_scen, 0)
    arch = Archive(grid, aq, wells, scen, fwd)

    snapshots = {}
    for label, q in (("off", np.zeros_like(q_star)), ("on", q_star)):
        ev = arch.evaluate(q, ro.make_cost(cfg))
        snapshots[label] = ev.details[0]  # scenario 0, same realization for both

    # details store the state as a flat (n,) vector; the figure wants (ny, nx)
    for d in (snapshots["off"], snapshots["on"]):
        d["C_final"] = np.asarray(d["C_final"], dtype=float).reshape(grid.ny, grid.nx)
    off, on = snapshots["off"], snapshots["on"]
    creg = float(cfg["compliance"]["creg"])
    vmax = float(max(np.max(off["C_final"]), np.max(on["C_final"]), creg))
    norm = LogNorm(vmin=max(vmax * 1e-4, 1e-4), vmax=vmax)

    fig = plt.figure(figsize=(9.6, 3.8))
    gs = fig.add_gridspec(1, 2, wspace=0.22)
    j_test = float(metrics["stages"]["opt"]["test"]["mean_J"])
    fields = [("(a) no pumping", off["C_final"], C_NOPUMP),
              (f"(b) certified policy, J = {j_test:,.0f}", on["C_final"], C_POLICY)]
    im = None
    for j, (title, field, _c) in enumerate(fields):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(field, origin="lower", norm=norm,
                       extent=[0, grid.lx, 0, grid.ly], cmap="magma", aspect="equal")
        ax.contour((np.arange(grid.nx) + 0.5) * grid.dx,
                   (np.arange(grid.ny) + 0.5) * grid.dy,
                   field, levels=[creg], colors="cyan", linewidths=1.3,
                   linestyles="--")
        ax.plot(wells[:, 0], wells[:, 1], "o", ms=5, mfc="none", mec="white", mew=1.0)
        active = q_star > 1e-9
        if active.any():
            ax.plot(wells[active, 0], wells[active, 1], "o", ms=5.5, mfc="white",
                    mec=C_ACCENT, mew=1.6)
        ax.plot([cfg["source"]["x"]], [cfg["source"]["y"]], marker="*", ms=13,
                color="#ffd166", mec="k", mew=0.5)
        ax.plot([cfg["compliance"]["x"]], [cfg["compliance"]["y"]], marker="s",
                ms=6, mfc="none", mec="cyan", mew=1.5)
        ax.set_xlabel("x [m]")
        if j == 0:
            ax.set_ylabel("y [m]")
        ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=fig.axes, label="C [mg/L]", shrink=0.8)

    # (c) compliance-point traces and (d) peak-concentration effect
    fig2, axs = plt.subplots(1, 2, figsize=(9.0, 3.0))
    for label, colour, det in (("no pumping", C_NOPUMP, off), ("certified", C_POLICY, on)):
        trace = np.asarray(det["C_comp"], dtype=float).ravel()
        if trace.size:
            tt = np.arange(trace.size) * float(cfg["time"]["dt"])
            axs[0].plot(tt, trace, color=colour, lw=1.6, label=label)
    axs[0].axhline(creg, color="k", ls=COMPLIANCE_LS, lw=1.0,
                   label=f"limit {creg:g} mg/L")
    axs[0].set_xlabel("t [d]")
    axs[0].set_ylabel("C at compliance point [mg/L]")
    axs[0].set_title("(c) compliance-point concentration, scenario 0")
    axs[0].legend()

    # (d) peak concentration over the held-out set, before/after
    peaks = {"off": [], "on": []}
    for q, key in ((np.zeros_like(q_star), "off"), (q_star, "on")):
        ev = arch.evaluate(q, ro.make_cost(cfg))
        peaks[key] = [float(np.max(np.asarray(d["C_final"], dtype=float)))
                      for d in ev.details]
    axs[1].boxplot([peaks["off"], peaks["on"]], widths=0.55,
                   labels=["no pumping", "certified"],
                   showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#e9ecef", edgecolor="0.3"),
                   medianprops=dict(color="k"))
    # No compliance line here on purpose: the 1 mg/L limit applies at the
    # compliance point, while pumping *concentrates* mass at the wells, so the
    # field maximum can legitimately exceed the limit under a compliant policy.
    axs[1].annotate("limit applies at the\ncompliance point, not the\nfield maximum",
                    xy=(0.03, 0.06), xycoords="axes fraction", fontsize=7,
                    color="0.3")
    axs[1].set_ylabel("peak concentration [mg/L]")
    axs[1].set_yscale("log")
    axs[1].set_title(f"(d) field peak over {len(peaks['on'])} test scenarios")
    fig2.tight_layout()
    _save(fig, out, "fig3_plume")
    _save(fig2, out, "fig3b_compliance")


# ------------------------------------------------------------------- figure 4
def _parse_history(path: Path) -> dict:
    """Pull (evals, best, step) out of a run's remote log."""
    rows = []
    if not path.exists():
        return {"evals": [], "best": [], "step": []}
    for line in path.read_text().splitlines():
        m = re.search(r"iter\s+(\d+)\s+evals=\s*(\d+)\s+step=([\d.]+)\s+best=([\d.eE+]+)", line)
        if m:
            rows.append((int(m.group(2)), float(m.group(4)), float(m.group(3))))
    return {"evals": [r[0] for r in rows], "best": [r[1] for r in rows],
            "step": [r[2] for r in rows]}


def fig_convergence(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    styles = ((RUN_F64, C_POLICY, "float64 (4.1 h)", "-", "o", 3.4),
              (RUN_F32, C_ACCENT, "float32 mixed (1.7 h)", "--", "^", 3.0))
    for run, colour, label, ls, marker, ms in styles:
        h = _parse_history(ROOT / run / "history.md")
        if not h["evals"]:
            continue
        best = np.asarray(h["best"]) / 1e3
        # solid+open vs dashed+filled so the two near-identical curves stay visible
        ax.plot(h["evals"], best, ls=ls, marker=marker, ms=ms, lw=1.4,
                color=colour, mfc="white" if ls == "-" else colour,
                label=f"{label}: {(best[-1] * 1e3):,.0f}")
    ax.set_xlabel("objective evaluations")
    ax.set_ylabel("best penalised objective  [$10^3$]")
    ax.set_title("Pattern search: identical till ~250 evaluations")
    ax.annotate("precision noise changes\nacceptance, not physics",
                xy=(300, 460), xytext=(150, 520), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.35"))
    ax.legend()
    fig.tight_layout()
    _save(fig, out, "fig4_convergence")


# ------------------------------------------------------------------- figure 5
def fig_chance(out: Path, cfg: dict) -> None:
    """The chance constraint itself: the tail that the CVaR penalty prices."""
    ro = _run_opt_helpers()
    metrics = _run_metrics(ROOT / RUN_F64)
    q_star = np.asarray(metrics["stages"]["opt"]["q_star"], dtype=float)
    wells = np.asarray(metrics["stages"]["O0"]["well_xy"], dtype=float)
    grid, aq, scen, fwd = _archive(cfg, ro, 40, 0)
    arch = Archive(grid, aq, wells, scen, fwd)
    cost = ro.make_cost(cfg)
    ev = arch.evaluate(q_star, cost)

    G = np.asarray(ev.G, dtype=float).ravel()
    tau = np.asarray(ev.tau, dtype=float).ravel()
    alpha = float(cfg["saa"]["alpha"])
    k = max(int(np.ceil(alpha * G.size)), 1)
    cvar = float(np.mean(np.sort(G)[-k:]))
    viol = int((G > 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    ax = axes[0]
    bins = np.linspace(G.min(), max(G.max(), 1e-9), 22)
    ax.hist(G, bins=bins, color="#ced4da", edgecolor="0.35", linewidth=0.5)
    viol_g = G[G > 0]
    if viol_g.size:
        ax.hist(viol_g, bins=bins, color=C_POLICY, alpha=0.75)
    ax.axvline(0.0, color="k", ls=COMPLIANCE_LS, lw=1.2, label="G = 0 (limit)")
    ax.axvline(cvar, color=C_POLICY, lw=2.2, ls=(0, (1, 1.2)),
               label=f"CVaR at {alpha:.0%} = {cvar:.2f} (priced)")
    ax.axvline(np.sort(G)[-k], color=C_GATE, lw=1.2, ls=":",
               label=f"worst rung of the {alpha:.0%} tail (k = {k})")
    ax.set_ylim(0, min(G.size, float(np.max(np.histogram(G, bins=bins)[0])) * 1.12))
    ax.set_xlabel("G  (worst compliance violation in window) [mg/L]")
    ax.set_ylabel("scenarios")
    ax.set_title(f"(a) {viol}/{G.size} violations, "
                 f"reliability {1 - viol / G.size:.3f}")

    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[1]
    tau_cap = float(fwd.nsteps) * float(fwd.dt)
    ax.hist(np.minimum(tau, tau_cap), bins=22, color="#ced4da",
            edgecolor="0.35", linewidth=0.5)
    ax.set_xlabel("cleanup time [d]")
    ax.set_ylabel("scenarios")
    ax.set_title(f"(b) median cleanup {np.median(tau):.0f} d")
    fig.suptitle(f"Out-of-sample statistics at the certified policy "
                 f"($M$ = {G.size})", y=1.02, fontsize=9.5)
    fig.tight_layout()
    _save(fig, out, "fig5_chance")


# ------------------------------------------------------------------- figure 6
def fig_certificate(out: Path) -> None:
    """Reliability vs sample size -- the figure that justifies M = 400."""
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for path, colour, label in ((CERT_F64, C_POLICY, "float64 policy"),
                                (CERT_F32, C_ACCENT, "float32 policy")):
        p = ROOT / path
        if not p.exists():
            print(f"  (skipping {label}: {path} missing)")
            continue
        d = json.loads(p.read_text())
        ladder = [e for e in (d.get("ladder") or []) if e["M"] >= 16]
        if not ladder:
            continue
        M = np.array([e["M"] for e in ladder], dtype=float)
        rel = np.array([e["reliability"] for e in ladder], dtype=float)
        lo = np.array([e["wilson"][0] for e in ladder], dtype=float)
        hi = np.array([e["wilson"][1] for e in ladder], dtype=float)
        ax.fill_between(M, lo, hi, color=colour, alpha=0.16, lw=0)
        # annotate the widest-evidence rung: this is the claim the paper makes
        ax.annotate(f"{lo[-1]:.3f}–{hi[-1]:.3f}",
                    xy=(M[-1], rel[-1]), xytext=(-6, 8),
                    textcoords="offset points", ha="right", fontsize=7.5,
                    color=colour)
        ax.plot(M, rel, "o-", ms=3.2, lw=1.4, color=colour, label=label)
        err = np.vstack([np.maximum(rel - lo, 0.0), np.maximum(hi - rel, 0.0)])
        ax.errorbar(M, rel, yerr=err, fmt="none",
                    ecolor=colour, elinewidth=0.8, capsize=2, alpha=0.8)
    ax.axhline(0.94, color="k", ls=COMPLIANCE_LS, lw=1.1,
               label="gate  $1-\\alpha-1\\%$")
    ax.set_xscale("log")
    ax.set_xlabel("out-of-sample scenarios M")
    ax.set_ylabel("empirical reliability")
    ax.set_title("Reliability and its 95% Wilson interval")
    ax.set_ylim(0.935, 1.004)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, out, "fig6_certificate")



# ------------------------------------------------------------------- figure 8
def fig_tail(out: Path) -> None:
    """Two statistics of the same four draws, on very different footings.

    Panel (a) is CVaR-5% with its standard error -- the average of ~20 of the
    most variable draws in each sample -- and panel (b) is reliability with its
    Wilson interval, estimated from all 400.  The tail moves; the proportion does
    not.  That is why the thesis certifies with reliability and only quotes the
    tail with an error bar.
    """
    cal = ROOT / TAIL_CAL
    if not cal.is_dir():
        print(f"  (skipping tail: {TAIL_CAL} missing)")
        return
    draws = []
    for f in sorted(cal.glob("20261120-111849_seed*.json")):
        d = json.loads(f.read_text())
        draws.append((int(d["seed_base"]), d))
    if not draws:
        print(f"  (skipping tail: no accepted-policy draws in {TAIL_CAL})")
        return
    x = np.arange(len(draws), dtype=float)
    labels = [f"{b}" for b, _ in draws]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    ax = axes[0]
    val = np.array([d["cvar"] for _, d in draws])
    se = np.array([d["tail"]["tail_se"] for _, d in draws])
    ax.errorbar(x, val, yerr=se, fmt="o", ms=5, lw=1.4, capsize=4,
                color=C_POLICY, label="CVaR-5%(G) $\\pm$ 1 s.e.")
    ax.axhline(0.0, color="k", ls=COMPLIANCE_LS, lw=1.0,
               label="constraint value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("fresh draw (seed base)")
    ax.set_ylabel("CVaR-5%(G)")
    ax.set_title("(a) the tail is a 20-draw average")
    ax.set_ylim(-0.15, max(val + se) * 1.25)
    ax.legend(loc="upper right", fontsize=7.5)

    ax = axes[1]
    rel = np.array([d["reliability"] for _, d in draws])
    lo = np.array([d["wilson"][0] for _, d in draws])
    hi = np.array([d["wilson"][1] for _, d in draws])
    ax.errorbar(x, rel, yerr=np.vstack([rel - lo, hi - rel]), fmt="o", ms=5,
                lw=1.4, capsize=4, color=C_ACCENT,
                label="reliability, 95% Wilson")
    ax.axhline(0.94, color=C_GATE, ls=COMPLIANCE_LS, lw=1.1,
               label="gate  $1-\\alpha-1\\%$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("fresh draw (seed base)")
    ax.set_ylabel("empirical reliability")
    ax.set_title("(b) the proportion is a 400-draw estimate")
    ax.set_ylim(min(0.935, float(lo.min()) - 0.005), 1.0)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.95)

    fig.tight_layout()
    _save(fig, out, "fig8_tail_spread")


# ------------------------------------------------------------------- figure 7
def fig_compute(out: Path) -> None:
    """Measured cost, not a model: numbers from DOCS/GPU.md's benchmark tables.

    The two groups are deliberately kept apart: mixing geometries in one bar
    chart would compare a 3200-cell float32 sweep against a 15,360-cell float64
    one and read as a 450x speed-up that does not exist.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    # (a) grouped by geometry so every comparison is like-for-like
    ax = axes[0]
    groups = [
        ("80×40, M = 8\nfloat32", [("NumPy\n(1 core)", 69.0, "#adb5bd"),
                                    ("T4 batched", 1.46, C_ACCENT)]),
        ("160×96, M = 100\nfloat64", [("NumPy\n(1 core)", 662.0, "#adb5bd"),
                                      ("T4 batched", 680.0, C_POLICY),
                                      ("A100 batched", 106.0, C_GATE)]),
    ]
    width = 0.36
    for gi, (_glabel, bars) in enumerate(groups):
        for bi, (label, value, colour) in enumerate(bars):
            x = gi + (bi - (len(bars) - 1) / 2) * width
            ax.bar(x, value, width=width * 0.94, color=colour,
                   label=label if gi == 0 or bi else None)
            ax.text(x, value * 1.18, f"{value:g}s", ha="center", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylabel("seconds per 18-candidate sweep")
    ax.set_ylim(1.0, 6000.0)   # headroom for the value labels above the tall bars
    ax.set_title("(a) one compass sweep, like-for-like")
    handles = {"NumPy (1 core)": "#adb5bd", "T4 batched": C_ACCENT,
               "A100 batched": C_GATE}
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in handles.values()],
              list(handles), fontsize=7.5, loc="lower right", ncol=3,
              columnspacing=0.9, handlelength=1.1)

    # (b) batching scaling at fixed geometry/dtype
    ax = axes[1]
    S = np.array([1, 18, 72, 288])
    ms = np.array([176.7, 10.16, 8.12, 7.63])
    ax.plot(S, ms, "o-", color=C_ACCENT, lw=1.5, ms=4,
            label="T4 float32, measured")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("candidates per batched call, S")
    ax.set_ylabel("ms per candidate·scenario")
    ax.set_title("(b) batching is the win, not the card")
    ax.annotate("17× from batching\n(S: 1 → 18)",
                xy=(18, 10.16), xytext=(2.0, 30),
                fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    ax.legend(loc="lower left")
    fig.tight_layout()
    _save(fig, out, "fig7_compute")


# ---------------------------------------------------------------------- main
FIGS = {
    "1d": ("fig1_1d_verification", lambda out, cfg: fig_1d(out)),
    "field": ("fig2_field_layout", fig_field),
    "plume": ("fig3_plume", fig_plume),
    "convergence": ("fig4_convergence", lambda out, cfg: fig_convergence(out)),
    "chance": ("fig5_chance", fig_chance),
    "certificate": ("fig6_certificate", lambda out, cfg: fig_certificate(out)),
    "compute": ("fig7_compute", lambda out, cfg: fig_compute(out)),
    "tail": ("fig8_tail_spread", lambda out, cfg: fig_tail(out)),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="FIGURES", help="output directory")
    ap.add_argument("--config", default=CONFIG, help="optimization config")
    ap.add_argument("--only", nargs="*", choices=sorted(FIGS),
                    help="generate only these figures")
    args = ap.parse_args(argv)

    use_style()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg(args.config)

    names = args.only or list(FIGS)
    for name in names:
        title, fn = FIGS[name]
        print(f"[fig] {title}")
        try:
            fn(out, cfg)
        except Exception as exc:  # keep going: one broken panel must not lose the rest
            print(f"  !! {title} failed: {type(exc).__name__}: {exc}")
    print(f"[fig] done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
