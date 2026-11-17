#!/usr/bin/env python3
"""Run the 2D flow/transport PDE and the pump-and-treat optimization.

Two modes, usable separately or together:

``forward``
    Solve the steady Darcy flow PDE and the transient advection--dispersion
    equation (ADE) on a heterogeneous, log-normal conductivity field for a
    fixed extraction-rate vector.  Writes the head/plume fields, a
    breakthrough curve, a mass-balance check and a JSON metric file.

``optimize``
    Design the well layout from the mean plume, draw training/test scenario
    archives, and solve the sample-average approximation (SAA) problem with
    pattern search.  Writes the optimal policy, out-of-sample metrics, gates
    and figures.

Examples
--------
    python run_2d.py --quick --mode all
    python run_2d.py --mode forward --seed 7 --rate 0.4
    python run_2d.py --mode optimize --config configs/opt/benchmark.yaml

The numerical kernels live in ``physics/fv2d.py`` and ``stoch/``; this file is
the reproducible entry point that binds them together.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# The entry points live in SCRIPTS/; make the repository root importable so
# `physics`/`stoch` resolve no matter where this script is launched from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics.fv2d import FlowModel, flow_responses, transport_run
from run_optimization import (
    base_aquifer,
    design_plume,
    make_cost,
    make_forward,
    make_grid,
    make_kl,
)
from stoch.objective import scenario_objective
from stoch.optimize import project_capacity
from stoch.random_fields import ScenarioGenerator
from stoch.report import ensure_dir, plot_layout, plot_plume, save_json


# ---------------------------------------------------------------------------
# Forward problem: -div(K grad h) = sources, then R dC/dt + ...
# ---------------------------------------------------------------------------
def run_forward(
    cfg: dict,
    outdir: Path,
    seed: int = 0,
    rate_fraction: float = 0.3,
    make_plots: bool = True,
) -> dict:
    """Solve one 2D flow + transport realisation and persist diagnostics."""
    outdir = ensure_dir(outdir)
    grid = make_grid(cfg)
    kl = make_kl(cfg, grid)
    gen = ScenarioGenerator(
        grid,
        kl,
        random_source=bool(cfg["random_field"]["random_source"]),
        random_gradient=bool(cfg["random_field"]["random_gradient"]),
    )
    fwd = make_forward(cfg)
    cost = make_cost(cfg)

    # Well layout from the mean no-pumping plume of a few design scenarios.
    Cmean, mask, layout = design_plume(cfg, grid, kl, gen, fwd)

    # One random conductivity realisation -> steady flow PDE.
    scen = gen.sample(seed)
    aq = base_aquifer(cfg, grid, scen.K)
    flow = FlowModel(grid, aq, h_left=fwd.grad * grid.lx, h_right=0.0)
    resp = flow_responses(flow, layout.wells, fwd.eps_well)

    q = np.full(layout.n_wells, rate_fraction * float(cfg["wells"]["q_max"]))
    q = project_capacity(q, float(cfg["wells"]["q_max"]),
                         float(cfg["wells"]["Q_max"]))
    h = resp.head(q)

    # PDE residual of the assembled flow system (should be at round-off).
    rhs = flow.rhs_const - q @ resp.weights
    flow_residual = float(np.linalg.norm(flow.A @ h - rhs)
                          / max(np.linalg.norm(rhs), 1e-30))

    # Transient transport PDE, storing every field for the figures.
    comp_cells = grid.indices(
        [p[0] for p in fwd.compliance_xy], [p[1] for p in fwd.compliance_xy]
    )
    obs_cells = grid.indices(
        [p[0] for p in fwd.obs_xy], [p[1] for p in fwd.obs_xy]
    )
    res = transport_run(
        grid,
        aq,
        flow,
        resp,
        q,
        fwd.dt,
        fwd.nsteps,
        source_idx=grid.index(*fwd.source_xy),
        source_rate=fwd.source_q * fwd.source_c,
        source_steps=fwd.source_steps,
        record_cells=comp_cells,
        store_fields=True,
    )
    obj = scenario_objective(
        grid, aq, res, q, h[obs_cells], fwd.dt, fwd.T, cost
    )

    C_final = res.C_final.reshape(grid.ny, grid.nx)
    mass = res.mass
    balance = (mass["total"][-1] - mass["total"][0]
               - (mass["source"][-1] - mass["well"][-1]
                  - mass["outflow"][-1] - mass["decay"][-1]))
    denom = max(abs(mass["source"][-1]), abs(mass["total"][0]), 1e-30)

    metrics = {
        "seed": int(seed),
        "grid": {"nx": grid.nx, "ny": grid.ny, "dx": grid.dx, "dy": grid.dy},
        "kl": kl.to_dict(),
        "K": {
            "min": float(scen.K.min()),
            "max": float(scen.K.max()),
            "mean": float(scen.K.mean()),
            "clamp_fraction": float(scen.clamp_fraction),
        },
        "wells": layout.wells.tolist(),
        "q": q.tolist(),
        "total_pumping": float(q.sum()),
        "flow": {
            "residual_rel": flow_residual,
            "head_min": float(h.min()),
            "head_max": float(h.max()),
        },
        "transport": {
            "peak_C": float(C_final.max()),
            "mean_C": float(C_final.mean()),
            "final_compliance_C": float(res.C_record[-1].max()),
            "compliance_limit": float(cost.creg),
        },
        "objective": {
            "J": float(obj["J"]),
            "J_pump": float(obj["J_pump"]),
            "J_treat": float(obj["J_treat"]),
            "J_res": float(obj["J_res"]),
            "J_time": float(obj["J_time"]),
            "G": float(obj["G"]),
            "tau_c": float(obj["tau_c"]),
            "compliant": bool(obj["compliance_prob"]),
        },
        "mass_balance": {
            "relative_error": float(balance / denom),
            "source": float(mass["source"][-1]),
            "well": float(mass["well"][-1]),
            "outflow": float(mass["outflow"][-1]),
            "decay": float(mass["decay"][-1]),
            "stored": float(mass["total"][-1] - mass["total"][0]),
        },
    }
    save_json(outdir / "forward_metrics.json", metrics)

    np.savez_compressed(
        outdir / "forward_fields.npz",
        K=scen.K,
        head=h.reshape(grid.ny, grid.nx),
        C_final=C_final,
        C_all=res.C_fields,
        C_compliance=res.C_record,
        times=res.times,
        wells=layout.wells,
        q=q,
    )

    if make_plots:
        plot_layout(
            cfg, grid, scen.K, mask, layout.wells, fwd.compliance_xy,
            outdir / "layout.png",
            title=f"Conductivity, design plume and {layout.n_wells} wells "
                  f"(seed {seed})",
        )
        plot_plume(
            grid, C_final, layout.wells, fwd.compliance_xy,
            outdir / "plume_final.png", creg=cost.creg,
            title="Final plume after transport PDE solve",
        )
        _plot_flow(grid, h.reshape(grid.ny, grid.nx), flow, resp, q,
                   layout.wells, outdir / "flow_field.png")
        _plot_breakthrough(res, cost.creg, outdir / "breakthrough.png")
        _plot_mass(mass, outdir / "mass_balance.png")

    return metrics


def _plot_flow(grid, h, flow, resp, q, wells, path):
    """Head contours with a subsampled seepage-velocity quiver."""
    vx, vy = flow.velocity(resp.head(q))
    vxc = 0.5 * (vx[:, :-1] + vx[:, 1:])
    vyc = 0.5 * (vy[:-1, :] + vy[1:, :])
    X, Y = grid.centres()
    speed = np.hypot(vxc, vyc)
    step = max(1, min(grid.nx, grid.ny) // 20)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cs = ax.contourf(X, Y, h, levels=20, cmap="Blues_r")
    fig.colorbar(cs, ax=ax, label="head h [m]")
    ax.quiver(
        X[::step, ::step], Y[::step, ::step],
        vxc[::step, ::step], vyc[::step, ::step],
        speed[::step, ::step], cmap="inferno", scale_units="xy",
        angles="xy", width=0.0025,
    )
    ax.plot(wells[:, 0], wells[:, 1], "o", mfc="red", mec="k", ms=7)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Steady Darcy flow: head and seepage velocity")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_breakthrough(res, creg, path):
    t = res.times[1:]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for k in range(res.C_record.shape[1]):
        ax.plot(t, res.C_record[:, k], label=f"compliance cell {k}")
    ax.axhline(creg, color="r", ls="--", lw=1.2, label="regulatory limit")
    ax.set_xlabel("time [d]")
    ax.set_ylabel("concentration [mg/L]")
    ax.set_title("Compliance-point breakthrough")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_mass(mass, path):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    t = np.arange(mass["total"].size)
    ax.plot(t, mass["total"] - mass["total"][0], label="stored")
    ax.plot(t, mass["source"], label="injected")
    ax.plot(t, mass["well"], label="extracted")
    ax.plot(t, mass["outflow"], label="outflow")
    if np.any(mass["decay"]):
        ax.plot(t, mass["decay"], label="decayed")
    ax.set_xlabel("time step")
    ax.set_ylabel("cumulative mass")
    ax.set_title("Transport mass budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Optimization: SAA over conductivity scenarios
# ---------------------------------------------------------------------------
def run_optimize(cfg: dict, outdir: Path, make_plots: bool = True) -> dict:
    """Run the tested SAA pipeline (stages O-0 ... optimization)."""
    from run_optimization import run as run_pipeline

    outdir = ensure_dir(outdir)
    cfg = dict(cfg)
    cfg["plot"] = bool(make_plots)
    return run_pipeline(cfg, outdir, {"all"})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_forward(m: dict) -> None:
    print(f"  q            = {np.round(m['q'], 2).tolist()} "
          f"(total {m['total_pumping']:.1f} m3/d)")
    print(f"  flow residual= {m['flow']['residual_rel']:.2e}")
    print(f"  peak C       = {m['transport']['peak_C']:.3f} mg/L")
    print(f"  J            = {m['objective']['J']:.3f} "
          f"(G={m['objective']['G']:.3f}, tau={m['objective']['tau_c']:.0f} d)")
    print(f"  mass balance = {m['mass_balance']['relative_error']:.2e}")


def _print_optimize(m: dict) -> None:
    o = m["stages"].get("opt")
    if o is None:
        print("  optimization stage did not run")
        return
    print(f"  q*           = {np.round(o['q_star'], 2).tolist()} "
          f"(total {o['total_pumping']:.1f} m3/d)")
    print(f"  train J      = {o['train']['mean_J']:.2f}")
    print(f"  test J       = {o['test']['mean_J']:.2f}")
    print(f"  reliability  = {o['test']['reliability']:.3f}")
    print(f"  cleanup time = {o['test']['mean_tau']:.0f} d")
    failed = [k for k, v in m.get("gates", {}).items()
              if isinstance(v, bool) and not v]
    print(f"  gates        = {'all passed' if not failed else failed}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mode", choices=["forward", "optimize", "all"],
                    default="all")
    ap.add_argument("--config", default=str(ROOT / "configs/opt/benchmark.yaml"))
    ap.add_argument("--quick", action="store_true",
                    help="small grid/scenario counts for a smoke run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="conductivity seed for the forward realisation")
    ap.add_argument("--rate", type=float, default=0.3,
                    help="forward pumping rate as a fraction of wells.q_max")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    from stoch.config import load_config

    cfg = load_config(args.config, quick=args.quick)
    tag = "quick" if args.quick else "full"
    root = Path(args.out) if args.out else (ROOT / "out" / "run_2d" / tag)
    root = ensure_dir(root)
    make_plots = not args.no_plot
    t0 = time.time()
    summary = {"mode": args.mode, "quick": bool(args.quick)}

    if args.mode in ("forward", "all"):
        print(f"\n[forward] 2D flow + transport PDE -> {root / 'forward'}")
        fm = run_forward(cfg, root / "forward", seed=args.seed,
                         rate_fraction=args.rate, make_plots=make_plots)
        summary["forward"] = fm
        _print_forward(fm)

    if args.mode in ("optimize", "all"):
        print(f"\n[optimize] SAA pump-and-treat -> {root / 'optimize'}")
        om = run_optimize(cfg, root / "optimize", make_plots=make_plots)
        summary["optimization"] = {
            "gates": om.get("gates", {}),
            "opt": om.get("stages", {}).get("opt", {}),
            "wall_clock_s": om.get("wall_clock_s"),
        }
        _print_optimize(om)

    summary["wall_clock_s"] = time.time() - t0
    save_json(root / "summary.json", summary)
    print(f"\nsummary written to {root / 'summary.json'}"
          f"  ({summary['wall_clock_s']:.1f} s)")


if __name__ == "__main__":
    main()
