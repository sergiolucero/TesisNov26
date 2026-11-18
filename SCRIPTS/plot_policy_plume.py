#!/usr/bin/env python3
"""plot_policy_plume.py — render the final plume under a stored policy.

Rebuilds the scenario archive deterministically from the config, applies the
``q`` saved in ``policy.json``, and draws the untreated field next to the
remediated one, with every well annotated by its pumping rate, plus the
breakthrough curves at the compliance point. The figure itself comes from
``stoch.report.plot_remediation`` -- the same function ``run_optimization.py``
calls at the end of a run -- so a stored policy re-renders byte-comparable
figures without rerunning the optimization.

Run from the repository root:

    python3 SCRIPTS/plot_policy_plume.py --run out/opt_run
    python3 SCRIPTS/plot_policy_plume.py --policy p.json --config configs/opt/benchmark.yaml

Exit status: 0 ok, 2 bad input.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")

from physics.fv2d import transport_run
from stoch.config import load_config
from stoch.random_fields import ScenarioGenerator
from stoch.report import plot_plume, plot_remediation
from stoch.saa import Archive

# Reuse the driver's builders so the scenario is identical to the one the
# optimizer saw (same grid, flow, wells, seeds).
sys.path.insert(0, HERE)
from run_optimization import make_cost, make_forward, make_grid, make_kl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None,
                    help="run directory containing policy.json and run_config.json")
    ap.add_argument("--policy", default=None, help="explicit policy.json path")
    ap.add_argument("--config", default=None,
                    help="config YAML (default: taken from the run manifest)")
    ap.add_argument("--outdir", default=None,
                    help="where to write figures (default: <run>/figures)")
    ap.add_argument("--scenario", type=int, default=0,
                    help="test-scenario index to draw (default: 0)")
    ap.add_argument("--all", action="store_true",
                    help="also sweep every test scenario and report the "
                         "median/IQR of peak reduction and compliance "
                         "(the honest form of a single-scenario figure)")
    args = ap.parse_args()

    if not args.run and not args.policy:
        print("error: pass --run or --policy", file=sys.stderr)
        return 2

    run = args.run
    policy_path = args.policy or os.path.join(run, "policy.json")
    if not os.path.isfile(policy_path):
        print(f"error: policy not found: {policy_path}", file=sys.stderr)
        return 2
    with open(policy_path, encoding="utf-8") as f:
        policy = json.load(f)
    q = np.asarray(policy["q"], dtype=float)
    wells = np.asarray(policy["wells"], dtype=float)

    cfg_path = args.config
    if cfg_path is None and run:
        man = os.path.join(run, "run_config.json")
        if os.path.isfile(man):
            with open(man, encoding="utf-8") as f:
                m = json.load(f)
            cfg_path = (m.get("extra") or {}).get("config_path") \
                or (m.get("config") or {}).get("_path")
    if cfg_path is None:
        cfg_path = os.path.join(ROOT, "configs", "opt", "benchmark.yaml")
    cfg = load_config(cfg_path)
    print(f"[plume] config: {cfg_path}")

    outdir = args.outdir or (os.path.join(run, "figures") if run else ".")
    os.makedirs(outdir, exist_ok=True)

    grid = make_grid(cfg)
    fwd = make_forward(cfg)
    cost = make_cost(cfg)
    kl = make_kl(cfg, grid)

    # Same deterministic archive as the optimizer: the train/test split and
    # seeds come straight from the config.
    rf = cfg["random_field"]
    gen = ScenarioGenerator(
        grid, kl,
        random_source=bool(rf["random_source"]),
        random_gradient=bool(rf["random_gradient"]),
    )
    # Same seed lists as the driver: <seed>+i for i in range(M), so this
    # reproduces the exact test scenarios the optimizer was scored on.
    n_te = int(cfg["saa"]["M_test"])
    seeds_te = [int(cfg["seed"]["test"]) + i for i in range(n_te)]
    test_sc = [gen.sample(s) for s in seeds_te]
    # Wells come from the policy so the figure matches the solved problem
    # exactly, even if place_wells would pick a different layout now.
    archive = Archive(grid, _aquifer(cfg), wells, test_sc, fwd)
    test_problems = archive.problems
    idx = min(args.scenario, len(test_problems) - 1)
    prob = test_problems[idx]
    print(f"[plume] scenario {idx} of {len(test_problems)} test scenarios")

    t0 = time.time()
    res_on = transport_run(
        grid, prob.flow.aquifer, prob.flow, prob.resp, q, fwd.dt, fwd.nsteps,
        source_idx=prob.source_idx, source_rate=prob.source_rate,
        source_steps=fwd.source_steps, record_cells=archive.compliance_cells,
        store_fields=True,
    )
    res_off = transport_run(
        grid, prob.flow.aquifer, prob.flow, prob.resp, np.zeros_like(q),
        fwd.dt, fwd.nsteps, source_idx=prob.source_idx,
        source_rate=prob.source_rate, source_steps=fwd.source_steps,
        record_cells=archive.compliance_cells, store_fields=True,
    )
    print(f"[plume] two transport solves in {time.time() - t0:.1f}s")

    C_on = res_on.C_final.reshape(grid.ny, grid.nx)
    C_off = res_off.C_final.reshape(grid.ny, grid.nx)
    peak_on, peak_off = float(C_on.max()), float(C_off.max())
    comp_on = float(res_on.C_record[:, 0].max())
    comp_off = float(res_off.C_record[:, 0].max())

    p1 = os.path.join(outdir, "plume_qstar.png")
    plot_plume(grid, C_on, wells, fwd.compliance_xy, p1, creg=cost.creg,
               title=f"Final plume under q* (test scenario {idx})", q=q)
    p2 = os.path.join(outdir, "plume_nopump.png")
    plot_plume(grid, C_off, wells, fwd.compliance_xy, p2, creg=cost.creg,
               title=f"Final plume, all wells off (test scenario {idx})",
               q=np.zeros_like(q))

    # The figure the driver also produces, from a stored policy: untreated vs
    # remediated fields plus the breakthrough curves at the compliance point.
    p3 = os.path.join(outdir, "contamination.png")
    summary = plot_remediation(
        grid, C_off, C_on, wells, p3, creg=cost.creg, q=q,
        source_xy=fwd.source_xy, compliance_xy=fwd.compliance_xy,
        obs_xy=fwd.obs_xy, times=res_off.times,
        C_comp_off=res_off.C_record[:, 0], C_comp_on=res_on.C_record[:, 0],
        window_start=float(cfg["compliance"]["window_start"]),
        scenario=idx,
    )

    # plot_remediation already returns peak_off/peak_on/peak_reduction_pct/
    # compliance_max_*/creg -- the same numbers the driver records in
    # metrics.json -- so only the policy itself is added here.
    summary.update({
        "scenario": idx,
        "q": q.tolist(),
        "wells": wells.tolist(),
        "active_wells": int((q > 1e-9).sum()),
        "total_rate_m3_d": float(q.sum()),
    })
    sp = os.path.join(outdir, "plume_summary.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"[plume] q* = {np.array2string(q, precision=2)} m3/d "
          f"({summary['active_wells']} active, total "
          f"{summary['total_rate_m3_d']:.1f} m3/d)")
    print(f"[plume] peak C: off {peak_off:.4f} -> on {peak_on:.4f} mg/L "
          f"({summary['peak_reduction_pct']:.1f}% reduction)")
    print(f"[plume] compliance max: off {comp_off:.4f} -> on {comp_on:.4f} mg/L "
          f"(limit {cost.creg})")
    for p in (p1, p2, p3, sp):
        print(f"[plume] -> {p}")

    if args.all:
        _distribution(archive, grid, fwd, q, cost, outdir)
    return 0


def _stat(v: np.ndarray) -> dict:
    """Median/IQR/min/max with linear interpolation (numpy default method)."""
    v = np.sort(np.asarray(v, dtype=float))
    n = len(v)

    def qf(p: float) -> float:
        pos = (n - 1) * p
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        f = pos - lo
        return float(v[lo] * (1 - f) + v[hi] * f)

    return {"median": qf(0.5), "q1": qf(0.25), "q3": qf(0.75),
            "min": float(v[0]), "max": float(v[-1])}


def _distribution(archive, grid, fwd, q, cost, outdir) -> None:
    """Sweep every test scenario: peak reduction and compliance, as medians.

    A single-scenario figure invites quoting a single-scenario number. This is
    the multi-realisation form: two transport solves per test scenario.
    """
    n = len(archive.problems)
    print(f"[plume] --all: {n} test scenarios, 2 solves each")
    t0 = time.time()
    on, off, comp_on, comp_off = [], [], [], []
    for i, prob in enumerate(archive.problems):
        r_on = transport_run(
            grid, prob.flow.aquifer, prob.flow, prob.resp, q, fwd.dt,
            fwd.nsteps, source_idx=prob.source_idx,
            source_rate=prob.source_rate, source_steps=fwd.source_steps,
            record_cells=archive.compliance_cells, store_fields=True)
        r_off = transport_run(
            grid, prob.flow.aquifer, prob.flow, prob.resp, np.zeros_like(q),
            fwd.dt, fwd.nsteps, source_idx=prob.source_idx,
            source_rate=prob.source_rate, source_steps=fwd.source_steps,
            record_cells=archive.compliance_cells, store_fields=True)
        on.append(float(r_on.C_final.max()))
        off.append(float(r_off.C_final.max()))
        comp_on.append(float(r_on.C_record[:, 0].max()))
        comp_off.append(float(r_off.C_record[:, 0].max()))
        if (i + 1) % 20 == 0:
            print(f"[plume]   {i + 1}/{n} scenarios ({time.time() - t0:.0f}s)")

    on = np.asarray(on)
    off = np.asarray(off)
    comp_on = np.asarray(comp_on)
    comp_off = np.asarray(comp_off)
    red = 100.0 * (1.0 - on / off)
    out = {
        "n_scenarios": int(n),
        "creg": float(cost.creg),
        "peak_off": _stat(off),
        "peak_on": _stat(on),
        "reduction_pct": _stat(red),
        "compliance_off": _stat(comp_off),
        "compliance_on": _stat(comp_on),
        "n_compliant_off": int((comp_off <= cost.creg).sum()),
        "n_compliant_on": int((comp_on <= cost.creg).sum()),
    }
    dp = os.path.join(outdir, "plume_distribution.json")
    with open(dp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"[plume] over {n} test scenarios:")
    print(f"[plume]   peak  off median {out['peak_off']['median']:.3f} "
          f"[IQR {out['peak_off']['q1']:.3f}, {out['peak_off']['q3']:.3f}]")
    print(f"[plume]   peak   on median {out['peak_on']['median']:.3f} "
          f"[IQR {out['peak_on']['q1']:.3f}, {out['peak_on']['q3']:.3f}]")
    print(f"[plume]   reduction median {out['reduction_pct']['median']:.1f}% "
          f"[IQR {out['reduction_pct']['q1']:.1f}, "
          f"{out['reduction_pct']['q3']:.1f}%] "
          f"worst {out['reduction_pct']['min']:.1f}%")
    print(f"[plume]   compliant at {cost.creg} mg/L: "
          f"{out['n_compliant_off']}/{n} off -> {out['n_compliant_on']}/{n} on q*")
    print(f"[plume] -> {dp}")
    print(f"[plume] distribution sweep done in {time.time() - t0:.0f}s")


def _aquifer(cfg: dict):
    """Build the base Aquifer from the config (K is overridden per scenario)."""
    from physics.fv2d import Aquifer
    a = cfg["aquifer"]
    return Aquifer(K=float(a["k0"]), phi=float(a["phi"]), R=float(a["R"]),
                   b=float(a["b"]), alpha_l=float(a["alpha_l"]),
                   alpha_t=float(a["alpha_t"]), decay=float(a.get("decay", 0.0)))


if __name__ == "__main__":
    raise SystemExit(main())
