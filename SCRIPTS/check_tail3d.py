#!/usr/bin/env python3
"""Independent certificate for a finished 3D run: re-evaluate its policy on FRESH fields.

The 2D lesson, restated for 3D: a run's own `test` row is evaluated at the same
`q*` on scenarios drawn from the same generator the search used, and it is the
row the search was steered by.  That is a consistency check, not a certificate.
A certificate needs (a) fields the optimizer never saw and (b) enough of them
that the binomial interval is narrow.

So this script rebuilds the run's problem *through the run's own constructors*
(`Grid3D`/`Aquifer3D`/`KL3D`/`ForwardConfig3D`/`CostConfig3D`/`sample_scenarios`/
`Archive3D`, exactly as `run()` does) from `<run>/run_config.json`, reads `q*`
and the well layout from `<run>/policy.json`, and evaluates that fixed policy on
`--m-scen` scenarios starting at `--seed-base`.

    python3 SCRIPTS/check_tail3d.py --run out3d_colab/plume3d_size_mid --m-scen 128

Reported quantity: **reliability with a 95% Wilson interval**, never the CVaR
value -- the tail mean is an average over a handful of extreme draws and cannot
resolve policy differences (DOCS/Results.md, "The tail does not transport").
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import full3D_colab_run as f3d  # noqa: E402


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def wilson_interval(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (k violations of n)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def reliability_ci(violations: int, n: int) -> tuple[float, float]:
    """Wilson interval for RELIABILITY = (n - violations) / n.

    Deliberately on the compliant count: the gate is a lower bound on how often
    the policy complies, and inverting a violation interval would put the
    bounds on the wrong side of the proportion.
    """
    return wilson_interval(n - violations, n)


def verdict(violations: int, n: int, gate: float) -> str:
    """PASS only when the interval's LOWER bound clears the gate; FAIL when the
    upper bound is below it; otherwise the sample cannot decide."""
    lo, hi = reliability_ci(violations, n)
    if lo >= gate:
        return "PASS"
    if hi < gate:
        return "FAIL"
    return "INCONCLUSIVE"


# --------------------------------------------------------------------------- #
# reconstruction
# --------------------------------------------------------------------------- #
def load_run(run_dir: Path) -> tuple[dict, dict, dict | None]:
    cfg_path = run_dir / "run_config.json"
    pol_path = run_dir / "policy.json"
    met_path = run_dir / "metrics.json"
    if not cfg_path.exists():
        raise SystemExit(f"no run_config.json under {run_dir}")
    cfg = json.loads(cfg_path.read_text())
    if pol_path.exists():
        pol = json.loads(pol_path.read_text())
    elif met_path.exists():                       # fall back to the metrics
        met = json.loads(met_path.read_text())
        pol = {"q": met["opt"]["q_star"], "wells": met["wells"]}
    else:
        raise SystemExit(f"neither policy.json nor metrics.json under {run_dir}")
    met = json.loads(met_path.read_text()) if met_path.exists() else None
    return cfg, pol, met


def rebuild(cfg: dict, wells) -> tuple[object, object, object, object, object]:
    """Recreate (grid, aquifer, kl, fwd, cost) exactly as run() does."""
    grid = f3d.Grid3D(cfg["nx"], cfg["ny"], cfg["nz"],
                      cfg["lx"], cfg["ly"], cfg["lz"])
    aquifer = f3d.Aquifer3D(
        K=np.full((grid.nz, grid.ny, grid.nx), cfg["k0"]),
        kv_ratio=cfg["kv_ratio"], phi=cfg["phi"], R=cfg["R"],
        alpha_l=cfg["alpha_l"], alpha_th=cfg["alpha_th"],
        alpha_tv=cfg["alpha_tv"], decay=cfg["decay"])
    kl = f3d.KL3D(grid, cfg["sigma"], cfg["ell_x"], cfg["ell_y"], cfg["ell_z"],
                  n_kl=cfg["n_kl"], k0=cfg["k0"], kv_ratio=cfg["kv_ratio"],
                  target_variance=cfg["target_variance"])
    fwd = f3d.ForwardConfig3D(
        dt=cfg["dt"], nsteps=cfg["nsteps"], grad=cfg["grad"],
        source_xyz=tuple(cfg["source_xyz"]), source_q=cfg["source_q"],
        source_c=cfg["source_c"], source_days=cfg["source_days"],
        compliance_xyz=[tuple(p) for p in cfg["compliance_xyz"]],
        obs_xyz=[tuple(p) for p in cfg["obs_xyz"]], eps_well=cfg["eps_well"])
    cost = f3d.CostConfig3D(
        c_e=cfg["c_e"], c_w=cfg["c_w"], c_m=cfg["c_m"], c_tau=cfg["c_tau"],
        c_fail=cfg["c_fail"], creg=cfg["creg"],
        window_start=cfg["window_start"], h_min=cfg["h_min"],
        eps_time=cfg["eps_time"], alpha=cfg["alpha"], alpha_h=cfg["alpha_h"])
    return grid, aquifer, kl, fwd, cost


def evaluate(cfg, wells, q, seeds, grid, aquifer, kl, fwd, cost):
    """Mean J / mean G / CVaR-5%(G) / violations for a fixed policy on `seeds`."""
    scen = f3d.sample_scenarios(grid, kl, seeds)
    arch = f3d.Archive3D(grid, aquifer, wells, scen, fwd)
    ev = arch.evaluate(np.asarray(q, dtype=float), cost)
    J = np.asarray(ev.J, dtype=float).ravel()
    G = np.asarray(ev.G, dtype=float).ravel()
    tau = np.asarray(ev.tau, dtype=float).ravel()
    alpha = cfg["alpha"]
    # empirical CVaR by the Rockafellar-Uryasev rule; degenerate when
    # alpha * M < 1 (fewer than one draw in the tail).
    m = G.size
    if alpha * m < 1.0:
        cvar = float(G.max())
    else:
        srt = np.sort(G)[::-1]
        k = int(math.floor(alpha * m))
        cvar = float(srt[:k].mean()) if k >= 1 else float(G.max())
    return {
        "M": int(m), "J": float(J.mean()), "mean_G": float(G.mean()),
        "cvar_G": cvar, "violations": int(np.sum(G > 0)),
        "reliability": float(np.mean(G <= 0)),
        "mean_tau_days": float(tau.mean()),
        "max_tau_days": float(tau.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="run directory holding run_config.json/policy.json/metrics.json")
    ap.add_argument("--m-scen", type=int, default=128,
                    help="scenarios in the certificate draw (default 128)")
    ap.add_argument("--seed-base", type=int, default=900000,
                    help="first scenario seed; must be disjoint from the run's "
                         "seed_train/seed_test (default 900000)")
    ap.add_argument("--ladder", action="store_true",
                    help="also report the estimate at increasing sample sizes")
    ap.add_argument("--json", default=None, help="write the numbers to this JSON path")
    ap.add_argument("--alpha", type=float, default=None,
                    help="override the run's chance level (default: from run_config)")
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg, pol, met = load_run(run_dir)
    if args.alpha is not None:
        cfg["alpha"] = args.alpha
    q = np.asarray(pol["q"], dtype=float)
    wells = np.asarray(pol["wells"], dtype=float)

    used = ((cfg.get("seed_train", 0), cfg["m_train"]),
            (cfg.get("seed_test", 0), cfg["m_test"]))
    lo_s, hi_s = args.seed_base, args.seed_base + args.m_scen
    clash = any(lo_s < u + m and u < hi_s for u, m in used)
    if clash:
        print(f"WARNING: seeds {args.seed_base}..{args.seed_base + args.m_scen - 1} "
              f"overlap the run's own train/test seeds {used} -> NOT out of sample")

    print(f"run        : {run_dir}")
    print(f"grid       : {cfg['nx']}x{cfg['ny']}x{cfg['nz']} = "
          f"{cfg['nx'] * cfg['ny'] * cfg['nz']} cells, nsteps={cfg['nsteps']}")
    print(f"policy     : q*={np.round(q, 2).tolist()} "
          f"(total {q.sum():.2f} m3/d, {int((q > 1e-9).sum())} active of {q.size})")
    print(f"draw       : M={args.m_scen} scenarios, seeds "
          f"{args.seed_base}..{args.seed_base + args.m_scen - 1}")
    print(f"gate       : reliability >= {1 - cfg['alpha'] - 0.01:.3f}")

    if met is not None:
        opt = met.get("opt", {})
        tr, te = opt.get("train", {}), opt.get("test", {})
        print("\nreported by the run (same q*, its own scenario draw):")
        for name, row in (("train", tr), ("test", te)):
            if row:
                print(f"  {name:5s}: J={row.get('mean_J', float('nan')):.1f} "
                      f"reliability={row.get('reliability', float('nan')):.3f} "
                      f"cvar_G={row.get('cvar_G', float('nan')):+.3f} "
                      f"M={row.get('M', '?')}")

    grid, aquifer, kl, fwd, cost = rebuild(cfg, wells)
    ladder = sorted({m for m in (16, 24, 40, 64, 100, args.m_scen) if m <= args.m_scen})
    if not args.ladder:
        ladder = [args.m_scen]

    # One archive of the full draw; the ladder sub-slices its first M scenarios,
    # exactly as check_tail.py does, so the smaller rows are nested subsets.
    t0 = time.time()
    seeds = [args.seed_base + i for i in range(args.m_scen)]
    res = evaluate(cfg, wells, q, seeds, grid, aquifer, kl, fwd, cost)
    elapsed = time.time() - t0

    lo, hi = reliability_ci(res["violations"], res["M"])
    print(f"\nfresh draw (M={res['M']}, {elapsed:.0f}s):")
    print(f"  J          = {res['J']:.1f}")
    print(f"  mean G     = {res['mean_G']:+.3f}")
    print(f"  CVaR-{cfg['alpha']:.0%}(G)  = {res['cvar_G']:+.3f}   "
          "(report only against the constraint, never as a ranking)")
    print(f"  violations = {res['violations']}/{res['M']}")
    print(f"  reliability= {res['reliability']:.4f}  "
          f"95% CI [{lo:.4f}, {hi:.4f}]  -> "
          f"{verdict(res['violations'], res['M'], 1 - cfg['alpha'] - 0.01)}")
    print(f"  mean tau   = {res['mean_tau_days']:.1f} d "
          f"(max {res['max_tau_days']:.0f} d)")

    if args.ladder:
        print("\nsample-size ladder (nested sub-draws):")
        for m in ladder:
            r = evaluate(cfg, wells, q, seeds[:m], grid, aquifer, kl, fwd, cost)
            l, h = reliability_ci(r["violations"], r["M"])
            print(f"  M={r['M']:4d}  viol {r['violations']:3d}  "
                  f"rel {r['reliability']:.3f} [{l:.3f}, {h:.3f}]  "
                  f"cvar {r['cvar_G']:+.3f}  J {r['J']:.0f}")

    if args.json:
        out = {"run": str(run_dir), "q": q.tolist(), "alpha": cfg["alpha"],
               "seed_base": args.seed_base, "gate": 1 - cfg["alpha"] - 0.01,
               "fresh": res, "wilson": list(reliability_ci(res["violations"],
                                                          res["M"])),
               "verdict": verdict(res["violations"], res["M"],
                                  1 - cfg["alpha"] - 0.01)}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
