#!/usr/bin/env python3
"""End-to-end driver for the stochastic pump-and-treat optimization.

Stages follow ``optimization_plan.md``:

  O-0  deterministic forward check, design plume and well layout
  O-1  random-field archive (KL, log-normal K, reproducibility)
  O-2  objective, cleanup time, SAA convergence
  O-3  discrete-adjoint gradient check
  O-4  chance-constraint reformulations on a toy ray
  opt  SAA optimization (pattern search or SLSQP+adjoint) and test certificate

Examples
--------
    python run_optimization.py --quick
    python run_optimization.py --config configs/opt/benchmark.yaml --stage opt
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

from physics.fv2d import (
    Aquifer,
    FlowModel,
    Grid,
    flow_responses,
    transport_run,
)
from physics.references import thiem_drawdown
from stoch.adjoint import check_gradient
from stoch.chance import compare_reformulations
from stoch.config import load_config
from stoch.objective import CostConfig
from stoch.optimize import SAAProblem, solve_saa
from stoch.random_fields import KLExpansion, ScenarioGenerator, save_archive
from stoch.report import (
    ensure_dir,
    plot_convergence,
    plot_distributions,
    plot_layout,
    plot_plume,
    save_json,
    write_manifest,
)
from stoch.saa import Archive, ForwardConfig
from stoch.wells import place_wells, plume_mask


def log(msg: str) -> None:
    """Timestamped, flushed progress line (long runs must never look dead)."""
    print(f"[opt] {time.strftime('%H:%M:%S')} {msg}", flush=True)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def make_grid(cfg: dict) -> Grid:
    d = cfg["domain"]
    return Grid(d["nx"], d["ny"], d["lx"], d["ly"])


def make_forward(cfg: dict) -> ForwardConfig:
    c, s, t = cfg["compliance"], cfg["source"], cfg["time"]
    o = cfg["observation"]
    return ForwardConfig(
        dt=float(t["dt"]),
        nsteps=int(t["nsteps"]),
        grad=float(cfg["flow"]["grad"]),
        source_xy=(float(s["x"]), float(s["y"])),
        source_q=float(s["q"]),
        source_c=float(s["c"]),
        source_days=float(s["days"]),
        compliance_xy=tuple(zip(c["x"], c["y"])),
        obs_xy=tuple(zip(o["x"], o["y"])),
        eps_well=float(cfg["wells"]["eps"]),
    )


def make_cost(cfg: dict) -> CostConfig:
    c = cfg["cost"]
    comp = cfg["compliance"]
    obs = cfg["observation"]
    saa = cfg["saa"]
    return CostConfig(
        c_e=float(c["c_e"]), c_w=float(c["c_w"]), c_m=float(c["c_m"]),
        c_tau=float(c["c_tau"]), c_fail=float(c["c_fail"]),
        creg=float(comp["creg"]), window_start=float(comp["window_start"]),
        h_min=float(obs["h_min"]), time_mode=c["time_mode"],
        eps_time=float(c["eps_time"]),
        alpha=float(saa["alpha"]), alpha_h=float(saa["alpha_h"]),
    )


def make_kl(cfg: dict, grid: Grid) -> KLExpansion:
    r = cfg["random_field"]
    a = cfg["aquifer"]
    return KLExpansion(
        grid, sigma=r["sigma"], ell_x=r["ell_x"], ell_y=r["ell_y"],
        n_kl=r.get("n_kl", 0), k0=a["k0"],
        target_variance=float(r.get("target_variance", 0.95)),
    )


def base_aquifer(cfg: dict, grid: Grid, K=None) -> Aquifer:
    a = cfg["aquifer"]
    if K is None:
        K = np.full((grid.ny, grid.nx), float(a["k0"]))
    return Aquifer(
        K=K, phi=float(a["phi"]), R=float(a["R"]), b=float(a["b"]),
        alpha_l=float(a["alpha_l"]), alpha_t=float(a["alpha_t"]),
        decay=float(a["decay"]),
    )


def no_pump_field(cfg, grid, aq, fwd, well_xy, q=0.0) -> np.ndarray:
    flow = FlowModel(grid, aq, h_left=fwd.grad * grid.lx, h_right=0.0)
    resp = flow_responses(flow, well_xy, fwd.eps_well)
    res = transport_run(
        grid, aq, flow, resp, np.zeros(len(well_xy)), fwd.dt, fwd.nsteps,
        source_idx=grid.index(*fwd.source_xy),
        source_rate=fwd.source_q * fwd.source_c,
        source_steps=fwd.source_steps,
    )
    return res.C_final


def design_plume(
    cfg, grid, kl, gen, fwd
) -> tuple[np.ndarray, np.ndarray, object]:
    """Mean no-pumping plume over a few design scenarios -> mask -> wells."""
    n_design = 5
    seeds = [9000 + k for k in range(n_design)]
    dummy = np.array([[grid.lx / 2.0, grid.ly / 2.0]])
    fields = []
    for seed in seeds:
        s = gen.sample(seed)
        aq = base_aquifer(cfg, grid, s.K)
        fields.append(no_pump_field(cfg, grid, aq, fwd, dummy))
    Cmean = np.mean(fields, axis=0).reshape(grid.ny, grid.nx)
    mask = plume_mask(grid, Cmean, float(cfg["wells"]["plume_threshold"]))
    layout = place_wells(
        grid, mask, int(cfg["wells"]["n"]),
        method=cfg["wells"]["method"], seed=int(cfg["optim"]["seed"]),
    )
    return Cmean, mask, layout


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_o0(cfg, outdir, grid, grid_kwargs):
    """Deterministic forward checks and the well layout."""
    a = cfg["aquifer"]
    fwd = make_forward(cfg)

    # (a) Thiem shape check on a large, all-Dirichlet domain
    q0 = 20.0
    g2 = Grid(100, 100, 1000.0, 1000.0)
    aq2 = Aquifer(K=np.full((g2.ny, g2.nx), float(a["k0"])), b=float(a["b"]))
    flow2 = FlowModel(g2, aq2, h_left=0.0, h_right=0.0, bc_y="dirichlet")
    resp2 = flow_responses(flow2, [(500.0, 500.0)], eps_cells=0.3)
    h2 = resp2.head([q0]).reshape(g2.ny, g2.nx)
    X, Y = g2.centres()
    r = np.sqrt((X - 500.0) ** 2 + (Y - 500.0) ** 2)
    sel = (r >= 25.0) & (r <= 100.0)
    r1 = 30.0
    hn = h2[sel]
    rr = r[sel]
    # linear interpolation of the numerical solution at r1 (radial average)
    order = np.argsort(r.ravel())
    h_at_r1 = float(np.interp(r1, r.ravel()[order], h2.ravel()[order]))
    dnum = hn - h_at_r1
    th = thiem_drawdown(q0, float(a["k0"]) * float(a["b"]), rr, r_ref=1.0)
    th1 = thiem_drawdown(q0, float(a["k0"]) * float(a["b"]), r1, r_ref=1.0)
    dth = th - th1
    thiem_err = float(np.max(np.abs(dnum - dth)) / np.max(np.abs(dth)))

    # (b) uniform-flow velocity check
    aq = base_aquifer(cfg, grid)
    flow = FlowModel(grid, aq, h_left=fwd.grad * grid.lx, h_right=0.0)
    resp0 = flow_responses(flow, [(grid.lx / 2, grid.ly / 2)], fwd.eps_well)
    vx, vy = flow.velocity(resp0.h0)
    v_exact = float(a["k0"]) * fwd.grad / float(a["phi"])
    vel_err = float(np.max(np.abs(vx[:, 1:-1] - v_exact)) / abs(v_exact))

    return {
        "thiem_shape_error": thiem_err,
        "velocity_error": vel_err,
        "q_thiem": q0,
    }


def stage_o1(cfg, outdir, grid, aq, gen, fwd, layout):
    seeds_tr = [int(cfg["seed"]["train"]) + i for i in range(int(cfg["saa"]["M_train"]))]
    seeds_te = [int(cfg["seed"]["test"]) + i for i in range(int(cfg["saa"]["M_test"]))]
    train = [gen.sample(s) for s in seeds_tr]
    test = [gen.sample(s) for s in seeds_te]

    sc = ensure_dir(Path(outdir) / "scenarios")
    save_archive(sc / "train", train, gen, extra={"split": "train"})
    save_archive(sc / "test", test, gen, extra={"split": "test"})

    # statistical checks by Monte Carlo
    rng = np.random.default_rng(12345)
    n = 300
    Gs = np.stack([gen.kl.log_field(gen.kl.sample_coefficients(rng)) for _ in range(n)])
    mean_G = float(Gs.mean())
    var_G = float(Gs.var())
    se_var = float(var_G * np.sqrt(2.0 / (n - 1)))
    clamps = []
    for _ in range(30):
        s = gen.sample(int(rng.integers(1 << 30)))
        clamps.append(s.clamp_fraction)
    # reproducibility
    s_a = gen.sample(4242)
    s_b = gen.sample(4242)
    repro = bool(np.array_equal(s_a.K, s_b.K) and np.array_equal(s_a.coeffs, s_b.coeffs))
    # covariance at lags (from the KL, exact for the truncated expansion)
    ell = gen.kl
    cov_lag = {}
    for lag in [0, 10, 20, 40]:
        cov = ell.sigma ** 2 * np.exp(-lag / ell.ell_x) * np.exp(-0.0 / ell.ell_y)
        cov_lag[str(lag)] = float(cov)

    return {
        "retained_variance": ell.retained,
        "n_kl": ell.n_kl,
        "sigma": ell.sigma,
        "ell_x": ell.ell_x,
        "ell_y": ell.ell_y,
        "mean_G": mean_G,
        "variance_G": var_G,
        "variance_se": se_var,
        "variance_target": ell.sigma ** 2,
        "clamp_fraction_mean": float(np.mean(clamps)),
        "clamp_fraction_max": float(np.max(clamps)),
        "reproducible": repro,
        "covariance_lag": cov_lag,
        "n_train": len(train),
        "n_test": len(test),
    }


def stage_o2(cfg, outdir, train, test, layout, cost, fwd):
    q_uniform = np.full(layout.n_wells, 0.1 * float(cfg["wells"]["q_max"]))
    q_none = np.zeros(layout.n_wells)

    ev_tr = train.evaluate(q_uniform, cost)
    ev_te = test.evaluate(q_uniform, cost)
    ev_none = test.evaluate(q_none, cost)

    # SAA convergence at fixed q from the per-scenario sample
    J = ev_te.J
    sizes = [m for m in (8, 16, 32, 64, 128, 256) if m <= J.size]
    if not sizes or sizes[-1] != J.size:
        sizes.append(J.size)
    conv = []
    for m in sizes:
        sub = J[:m]
        conv.append({
            "M": int(m),
            "mean_J": float(sub.mean()),
            "se_J": float(sub.std(ddof=1) / np.sqrt(m)) if m > 1 else 0.0,
            "reliability": float(np.mean(ev_te.G[:m] <= 0.0)),
        })

    mass_err = float(max(d["mass_balance_error"] for d in ev_te.details))
    return {
        "uniform_q": q_uniform.tolist(),
        "train": ev_tr.summary(),
        "test": ev_te.summary(),
        "no_pump_test": ev_none.summary(),
        "max_abs_mass_balance_error": abs(mass_err),
        "saa_convergence": conv,
    }, ev_te


def stage_o3(cfg, outdir, train, cost):
    q = np.full(train.well_xy.shape[0], 0.3 * float(cfg["wells"]["q_max"]))
    direction = np.ones_like(q) / q.size
    res = check_gradient(train, q, cost, scenario_index=0, delta=1e-4,
                         direction=direction)
    res.pop("gradient", None)
    return res


def stage_o4(cfg, outdir, train, cost):
    nw = train.well_xy.shape[0]
    qmax = float(cfg["wells"]["q_max"])
    Qmax = float(cfg["wells"]["Q_max"])
    alpha = float(cfg["saa"]["alpha"])

    tvals = np.linspace(0.0, 1.0, 11)
    q_grid = []
    for t in tvals:
        q = np.full(nw, t * qmax)
        s = q.sum()
        if s > Qmax:
            q *= Qmax / s
        q_grid.append(q)

    cache = {}

    def evaluate(q):
        key = q.tobytes()
        if key not in cache:
            ev = train.evaluate(q, cost)
            cache[key] = {"J": ev.mean_J, "G": ev.G}
        return cache[key]

    try:
        verdict = compare_reformulations(evaluate, q_grid, alpha=alpha,
                                         eps_list=(0.1, 0.01, 0.001))
        return verdict.to_dict()
    except RuntimeError as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run(cfg: dict, outdir: Path, stages: set[str]) -> dict:
    t0 = time.time()
    outdir = ensure_dir(outdir)
    ensure_dir(outdir / "figures")
    log(f"start: out={outdir} stages={sorted(stages)}")
    grid = make_grid(cfg)
    aq = base_aquifer(cfg, grid)
    kl = make_kl(cfg, grid)
    gen = ScenarioGenerator(
        grid, kl,
        random_source=bool(cfg["random_field"]["random_source"]),
        random_gradient=bool(cfg["random_field"]["random_gradient"]),
    )
    fwd = make_forward(cfg)
    cost = make_cost(cfg)

    metrics: dict = {"config": cfg, "stages": {}}
    write_manifest(outdir / "run_config.json", cfg)

    # ---- O0: design plume and wells -----------------------------------
    t_stage = time.time()
    log("stage O0: deterministic forward checks and well layout")
    Cmean, mask, layout = design_plume(cfg, grid, kl, gen, fwd)
    o0 = stage_o0(cfg, outdir, grid, {})
    metrics["stages"]["O0"] = o0
    log(f"stage O0 done in {time.time() - t_stage:.1f}s "
        f"({layout.n_wells} wells, {layout.method})")

    # layout figure uses the mean design field
    Kmean = np.mean([gen.sample(9000 + k).K for k in range(5)], axis=0)
    if cfg.get("plot", True):
        plot_layout(
            cfg, grid, Kmean, mask, layout.wells, fwd.compliance_xy,
            outdir / "figures" / "well_layout.png",
            title=f"Design plume, {layout.n_wells} wells ({layout.method})",
        )
    metrics["stages"]["O0"].update({
        "n_wells": layout.n_wells,
        "well_xy": layout.wells.tolist(),
        "method": layout.method,
        "plume_area": float(mask.sum() * grid.dx * grid.dy),
        "well_spacing": float(layout.spacing),
        "covering_radius": float(layout.covering_radius),
        "coverage_at_spacing_over_sqrt3": float(layout.coverage),
    })

    # ---- archives ------------------------------------------------------
    seeds_tr = [int(cfg["seed"]["train"]) + i for i in range(int(cfg["saa"]["M_train"]))]
    seeds_te = [int(cfg["seed"]["test"]) + i for i in range(int(cfg["saa"]["M_test"]))]
    train_scen = [gen.sample(s) for s in seeds_tr]
    test_scen = [gen.sample(s) for s in seeds_te]

    if "o1" in stages or "all" in stages:
        t_stage = time.time()
        log("stage O1: random-field archive and statistics")
        metrics["stages"]["O1"] = stage_o1(cfg, outdir, grid, aq, gen, fwd, layout)
        log(f"stage O1 done in {time.time() - t_stage:.1f}s")

    train = Archive(grid, aq, layout.wells, train_scen, fwd)
    test = Archive(grid, aq, layout.wells, test_scen, fwd)
    log(f"archives ready: {len(train_scen)} train / {len(test_scen)} test scenarios")

    if "o2" in stages or "all" in stages:
        t_stage = time.time()
        log("stage O2: objective, cleanup time, SAA convergence")
        o2, ev_te = stage_o2(cfg, outdir, train, test, layout, cost, fwd)
        metrics["stages"]["O2"] = o2
        log(f"stage O2 done in {time.time() - t_stage:.1f}s")

    if "o3" in stages or "all" in stages:
        t_stage = time.time()
        log("stage O3: discrete-adjoint gradient check")
        metrics["stages"]["O3"] = stage_o3(cfg, outdir, train, cost)
        log(f"stage O3 done in {time.time() - t_stage:.1f}s")

    if "o4" in stages or "all" in stages:
        t_stage = time.time()
        log("stage O4: chance-constraint reformulations")
        metrics["stages"]["O4"] = stage_o4(cfg, outdir, train, cost)
        log(f"stage O4 done in {time.time() - t_stage:.1f}s")

    # ---- SAA optimization ---------------------------------------------
    if "opt" in stages or "all" in stages:
        opt = cfg["optim"]
        t_stage = time.time()
        log(f"stage opt: method={opt['method']} starts={opt['n_starts']} "
            f"max_iter={opt['max_iter']} max_evals={opt.get('max_evals', 2000)}")

        def opt_progress(ev: dict) -> None:
            phase = ev.get("phase")
            if phase == "start":
                log(f"opt start {ev['start']}/{ev['n_starts']}")
            elif phase == "search-start":
                log(f"  search: f={ev['best']:.6g} step={ev['step']:.3g}")
            elif phase == "iter":
                log(f"  iter {ev['iteration']:3d} evals={ev['evals']:5d} "
                    f"step={ev['step']:.3g} best={ev['best']:.6g} "
                    f"elapsed={ev['elapsed']:.1f}s")
            elif phase == "eval":
                log(f"  eval {ev['evals']:5d} f={ev['best']:.6g} "
                    f"elapsed={ev['elapsed']:.1f}s")
            elif phase == "done":
                log(f"  start {ev['start']}/{ev['n_starts']} done: "
                    f"f={ev['fun']:.6g} nit={ev['nit']}")

        problem = SAAProblem(
            train, cost,
            q_max=float(cfg["wells"]["q_max"]),
            Q_max=float(cfg["wells"]["Q_max"]),
            alpha=float(cfg["saa"]["alpha"]),
            alpha_h=float(cfg["saa"]["alpha_h"]),
            chance=opt["chance"],
            rho_g=float(opt["rho_g"]),
            rho_h=float(opt["rho_h"]),
        )
        result = solve_saa(
            problem,
            method=opt["method"],
            n_starts=int(opt["n_starts"]),
            max_iter=int(opt["max_iter"]),
            max_evals=int(opt.get("max_evals", 2000)),
            seed=int(opt["seed"]),
            progress=opt_progress,
        )
        log(f"stage opt done in {time.time() - t_stage:.1f}s")
        q_star = result.q
        ev_tr = train.evaluate(q_star, cost, keep=True)
        ev_te = test.evaluate(q_star, cost, keep=True)

        # baselines
        q_none = np.zeros(layout.n_wells)
        q_uni = np.full(layout.n_wells, 0.5 * float(cfg["wells"]["q_max"]))
        ev_none = test.evaluate(q_none, cost)
        ev_uni = test.evaluate(q_uni, cost)

        metrics["stages"]["opt"] = {
            "method": result.method,
            "n_evaluations": int(problem.n_eval),
            "train": ev_tr.summary(),
            "test": ev_te.summary(),
            "q_star": q_star.tolist(),
            "total_pumping": float(q_star.sum()),
            "active_wells": int(np.sum(q_star > 1e-6)),
            "test_gap_vs_train": float(
                (ev_te.mean_J - ev_tr.mean_J) / max(abs(ev_tr.mean_J), 1e-12)
            ),
            "baseline_no_pump": ev_none.summary(),
            "baseline_uniform_half": ev_uni.summary(),
            "max_mass_balance_error_test": float(
                max(abs(d["mass_balance_error"]) for d in ev_te.details)
            ),
        }
        save_json(outdir / "policy.json", {
            "q": q_star.tolist(),
            "wells": layout.wells.tolist(),
            "chance": opt["chance"],
            "alpha": float(cfg["saa"]["alpha"]),
        })
        if cfg.get("plot", True):
            plot_convergence(result.history, outdir / "figures" / "convergence.png")
            plot_distributions(ev_te, fwd.T, outdir / "figures" / "test_distributions.png")
            C0 = ev_te.details[0]["C_final"].reshape(grid.ny, grid.nx)
            plot_plume(
                grid, C0, layout.wells, fwd.compliance_xy,
                outdir / "figures" / "plume_scenario0.png",
                creg=cost.creg,
                title="Final plume under q* (test scenario 0)",
            )

    # ---- gates ---------------------------------------------------------
    gates = {}
    s = metrics["stages"]
    if "O0" in s:
        gates["O0_thiem_shape_error<=1e-2"] = s["O0"]["thiem_shape_error"] <= 1e-2
        gates["O0_velocity_error<=1e-2"] = s["O0"]["velocity_error"] <= 1e-2
        gates["O0_thiem_shape_error"] = s["O0"]["thiem_shape_error"]
    if "O1" in s:
        gates["O1_retained_variance>=0.95"] = s["O1"]["retained_variance"] >= 0.95
        gates["O1_clamp_fraction<0.01"] = s["O1"]["clamp_fraction_max"] < 0.01
        gates["O1_reproducible"] = s["O1"]["reproducible"]
    if "O2" in s:
        gates["O2_rel_se_J<=1%"] = s["O2"]["test"]["rel_se_J"] <= 0.01
        gates["O2_mass_balance<=1%"] = s["O2"]["max_abs_mass_balance_error"] <= 0.01
    if "O3" in s:
        gates["O3_adjoint_rel_error<=1e-4"] = s["O3"]["rel_error"] <= 1e-4
    if "O4" in s and "error" not in s["O4"]:
        gates["O4_cvar_not_better_than_exact"] = s["O4"]["cvar_J"] >= s["O4"]["exact_J"] - 1e-9
        gates["O4_sigmoid_exact_feasible"] = s["O4"]["sigmoid_exact_feasible"]
    if "opt" in s:
        alpha = float(cfg["saa"]["alpha"])
        gates["final_test_reliability>=1-alpha-1%"] = (
            s["opt"]["test"]["reliability"] >= 1.0 - alpha - 0.01
        )
        gates["final_gap<=5%"] = abs(s["opt"]["test_gap_vs_train"]) <= 0.05
        gates["final_mass_balance<=1%"] = s["opt"]["max_mass_balance_error_test"] <= 0.01
    metrics["gates"] = gates
    metrics["wall_clock_s"] = time.time() - t0
    save_json(outdir / "metrics.json", metrics)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/opt/benchmark.yaml"))
    ap.add_argument("--quick", action="store_true",
                    help="small grid/scenario counts for a smoke run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--nx", type=int, default=None,
                    help="Override domain.nx from the config (applied after --quick)")
    ap.add_argument("--ny", type=int, default=None,
                    help="Override domain.ny from the config (applied after --quick)")
    ap.add_argument("--stage", default="all",
                    choices=["all", "gates", "o0", "o1", "o2", "o3", "o4", "opt"])
    args = ap.parse_args()

    stage_map = {"all": {"all"}, "gates": {"o1", "o2", "o3", "o4"},
                 "o0": {"o0"}, "o1": {"o1"}, "o2": {"o2"},
                 "o3": {"o3"}, "o4": {"o4"}, "opt": {"opt"}}
    stages = stage_map[args.stage]
    cfg = load_config(args.config, quick=args.quick)
    if args.nx is not None or args.ny is not None:
        domain = cfg.setdefault("domain", {})
        if args.nx is not None:
            domain["nx"] = int(args.nx)
        if args.ny is not None:
            domain["ny"] = int(args.ny)
        print(f"[opt] grid override: nx={domain['nx']} ny={domain['ny']}")
    tag = "quick" if args.quick else "full"
    out = Path(args.out) if args.out else (ROOT / "out" / "opt" / f"benchmark_{tag}")
    metrics = run(cfg, out, stages)

    print("\n=== gates ===")
    for k, v in metrics["gates"].items():
        print(f"  {k}: {v}")
    if "opt" in metrics["stages"]:
        o = metrics["stages"]["opt"]
        print(f"\nq* = {np.round(o['q_star'], 2).tolist()}  (total {o['total_pumping']:.1f} m3/d)")
        print(f"train J = {o['train']['mean_J']:.2f}   test J = {o['test']['mean_J']:.2f}")
        print(f"test reliability = {o['test']['reliability']:.3f}   "
              f"cleanup time = {o['test']['mean_tau']:.0f} d")
    print(f"\nresults written to {out}  ({metrics['wall_clock_s']:.1f} s)")


if __name__ == "__main__":
    main()
