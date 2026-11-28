#!/usr/bin/env python3
"""Re-score candidate policies on the exact NumPy/splu reference.

This settles the 32x16x8 attribution question from the paper-preset runs.  At
the *same* 9 wells the torch float32 optimizer converged to a single south-edge
well (``q = [0,...,0, 24.96, 0]``) while the numpy optimizer converged to four
central wells (``q = [12.5,12.5,12.5,12.5,0,0,0,0,0]``).  Both candidates are
re-scored here on one NumPy ``splu`` archive so we can tell whether the torch
optimum is a genuine lower-``f`` local minimum (an SAA overfitting effect) or a
float32 mis-evaluation.

    python SCRIPTS/rescore_paper_candidates.py --run out3d_colab/run_paper_torch
    python SCRIPTS/rescore_paper_candidates.py --run <dir> --archive train --max-m 16
    python SCRIPTS/rescore_paper_candidates.py --run <dir> --archive train \
        --q 0,0,0,0,0,0,0,24.96,0 --q 12.5,12.5,12.5,12.5,0,0,0,0,0

Nothing is re-optimized.  Each candidate is scored serially on the same
archive; scenario seeds come from ``run_config.json`` (``seed_train`` /
``seed_test``), so the KL fields are identical to the original run's.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import full3D_colab_run as f3d  # noqa: E402

# Canonical candidates at 32x16x8 (well order = policy.json).
DEFAULT_CANDIDATES = {
    "4well_50 (numpy paper)": [12.5, 12.5, 12.5, 12.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    "1well_25 (torch paper)": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24.96, 0.0],
}


def rebuild(run_dir: Path):
    """Grid/aquifer/KL/fwd/cost/wells from run_config.json + policy.json.

    Mirrors ``run_torch``'s preamble (same constructors, same seeds) and
    ``SCRIPTS/verify_papergpu.py``.
    """
    cfg = json.loads((run_dir / "run_config.json").read_text())
    pol = json.loads((run_dir / "policy.json").read_text())

    grid = f3d.Grid3D(cfg["nx"], cfg["ny"], cfg["nz"], cfg["lx"], cfg["ly"], cfg["lz"])
    aquifer = f3d.Aquifer3D(
        K=np.full((grid.nz, grid.ny, grid.nx), cfg["k0"]),
        kv_ratio=cfg["kv_ratio"], phi=cfg["phi"], R=cfg["R"],
        alpha_l=cfg["alpha_l"], alpha_th=cfg["alpha_th"], alpha_tv=cfg["alpha_tv"],
        decay=cfg["decay"],
    )
    kl = f3d.KL3D(grid, cfg["sigma"], cfg["ell_x"], cfg["ell_y"], cfg["ell_z"],
                  n_kl=cfg["n_kl"], k0=cfg["k0"], kv_ratio=cfg["kv_ratio"],
                  target_variance=cfg["target_variance"])
    fwd = f3d.ForwardConfig3D(
        dt=cfg["dt"], nsteps=cfg["nsteps"], grad=cfg["grad"],
        source_xyz=tuple(cfg["source_xyz"]),
        source_q=cfg["source_q"], source_c=cfg["source_c"],
        source_days=cfg["source_days"],
        compliance_xyz=[tuple(p) for p in cfg["compliance_xyz"]],
        obs_xyz=[tuple(p) for p in cfg["obs_xyz"]],
        eps_well=cfg["eps_well"],
    )
    cost = f3d.CostConfig3D(
        c_e=cfg["c_e"], c_w=cfg["c_w"], c_m=cfg["c_m"], c_tau=cfg["c_tau"],
        c_fail=cfg["c_fail"], creg=cfg["creg"], window_start=cfg["window_start"],
        h_min=cfg["h_min"], eps_time=cfg["eps_time"],
        alpha=cfg["alpha"], alpha_h=cfg["alpha_h"],
    )
    wells = np.atleast_2d(np.asarray(pol["wells"], dtype=float))
    return cfg, grid, aquifer, kl, fwd, cost, wells


def penalized(ev: f3d.SAAEvaluation3D, alpha: float, alpha_h: float,
              rho_g: float, rho_h: float) -> float:
    """The exact objective used by ``SAAProblem3D.objective``."""
    cg = f3d.cvar(ev.G, alpha)
    ch = f3d.cvar(-ev.H, alpha_h)
    return float(ev.mean_J + rho_g * max(cg, 0.0) ** 2
                 + rho_h * max(ch, 0.0) ** 2)


def score_archive(grid, aquifer, kl, fwd, cost, wells, candidates, seeds,
                  label, max_m, rho_g, rho_h, q_max, Q_max):
    seeds = seeds[:max_m] if max_m else seeds
    scenarios = f3d.sample_scenarios(grid, kl, seeds)
    f3d.log(f"{label}: building splu archive for {len(scenarios)} scenarios ...")
    t0 = time.time()
    archive = f3d.Archive3D(grid, aquifer, wells, scenarios, fwd)
    f3d.log(f"{label}: archive ready in {time.time() - t0:.1f}s "
            f"(KL n_kl={kl.n_kl}, retained={kl.retained:.4f})")

    problem = f3d.SAAProblem3D(
        archive, cost,
        q_max=q_max, Q_max=Q_max,
        alpha=cost.alpha, alpha_h=cost.alpha_h, rho_g=rho_g, rho_h=rho_h,
    )

    out = {}
    print(f"\n== {label} archive: numpy/splu re-score ==")
    print(f"{'candidate':<28} {'mean_J':>12} {'reliab':>8} {'cvar_G':>9} "
          f"{'mean_tau':>9} {'f (penalized)':>14}")
    for name, q in candidates.items():
        q = np.asarray(q, dtype=float)
        ev = problem.evaluate(q)
        s = ev.summary()
        f = penalized(ev, cost.alpha, cost.alpha_h, rho_g, rho_h)
        print(f"{name:<28} {s['mean_J']:12.2f} {s['reliability']:8.4f} "
              f"{s['cvar_G']:9.4f} {s['mean_tau']:9.1f} {f:14.2f}")
        out[name] = {"q": q.tolist(), "f": f, **s}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run", default="out3d_colab/run_paper_torch",
                    help="finished run directory with run_config.json + policy.json")
    ap.add_argument("--archive", choices=["train", "test", "both"], default="both")
    ap.add_argument("--max-m", type=int, default=0,
                    help="cap scenarios (0 = all) for a quick smoke check")
    ap.add_argument("--q", action="append", default=None, metavar="CSV",
                    help="extra candidate q-vector (comma-separated); may repeat")
    ap.add_argument("--out", default=None,
                    help="write results to this JSON file (default: <run>/rescore.json)")
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    cfg, grid, aquifer, kl, fwd, cost, wells = rebuild(run_dir)
    rho_g = float(cfg.get("rho_g", 1e6))
    rho_h = float(cfg.get("rho_h", 1e6))
    q_max = float(cfg.get("q_max", 50.0))
    Q_max = float(cfg.get("Q_max", 200.0))

    candidates = dict(DEFAULT_CANDIDATES)
    if args.q:
        for i, csv in enumerate(args.q):
            vec = [float(x) for x in csv.split(",")]
            candidates[f"custom_{i}: {vec}"] = vec

    f3d.log(f"rescore {run_dir}: grid {grid.nx}x{grid.ny}x{grid.nz} "
            f"({grid.n} cells), n_wells={wells.shape[0]}, "
            f"rho_g={rho_g:g} rho_h={rho_h:g}")
    f3d.log(f"wells (x,y,z_top,z_bot):\n"
            + "\n".join(f"  {w.tolist()}" for w in wells))

    results = {}

    if args.archive in ("train", "both"):
        seeds = [cfg["seed_train"] + i for i in range(cfg["m_train"])]
        results["train"] = score_archive(
            grid, aquifer, kl, fwd, cost, wells, candidates, seeds,
            "train", args.max_m, rho_g, rho_h, q_max, Q_max)

    if args.archive in ("test", "both"):
        seeds = [cfg["seed_test"] + i for i in range(cfg["m_test"])]
        results["test"] = score_archive(
            grid, aquifer, kl, fwd, cost, wells, candidates, seeds,
            "test", args.max_m, rho_g, rho_h, q_max, Q_max)

    # verdict
    if "train" in results:
        best = min(results["train"].items(), key=lambda kv: kv[1]["f"])
        print(f"\nbest on train: {best[0]}  f={best[1]['f']:.2f} "
              f"(reliab {best[1]['reliability']:.4f})")
    if "test" in results:
        best = min(results["test"].items(), key=lambda kv: kv[1]["f"])
        print(f"best on test:  {best[0]}  f={best[1]['f']:.2f} "
              f"(reliab {best[1]['reliability']:.4f})")

    out = args.out or str(run_dir / "rescore.json")
    Path(out).write_text(json.dumps(results, indent=2))
    f3d.log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
