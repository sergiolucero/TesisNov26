#!/usr/bin/env python3
"""Re-evaluate a finished 3D run's policy on the exact NumPy/splu reference.

The batched torch backend is verified on tiny grids in
``tests/test_fv3d_torch_colab.py``, but the headline ``paperGPU`` result was
produced entirely in float32 on a T4.  This script rebuilds the run through the
run's own constructors (as ``SCRIPTS/movie3d_from_policy.py`` does) and scores
the *same* ``q*`` with the validated ``splu`` path, so the float32 torch numbers
can be checked against an exact reference at the full 64x32x16 grid.

    python3 SCRIPTS/verify_papergpu.py --run out3d_colab/run_paperGPU
    python3 SCRIPTS/verify_papergpu.py --run <dir> --archive train --max-m 8

Nothing is re-optimized; only ``q*`` is scored.  The scenario seeds are rebuilt
from ``run_config.json`` (``seed_train``/``seed_test``), so the KL fields are
identical to the torch run's.
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


def rebuild(run_dir: Path):
    """Grid/aquifer/KL/fwd/cost from run_config.json, mirroring run_torch."""
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
    q = np.asarray(pol["q"], dtype=float).ravel()
    return cfg, pol, grid, aquifer, kl, fwd, cost, wells, q


def score_archive(grid, aquifer, kl, fwd, cost, wells, q, seeds, label, max_m):
    """Build the NumPy archive for ``seeds`` and score ``q``."""
    seeds = seeds[:max_m] if max_m else seeds
    scenarios = f3d.sample_scenarios(grid, kl, seeds)
    f3d.log(f"{label}: building splu archive for {len(scenarios)} scenarios ...")
    t0 = time.time()
    archive = f3d.Archive3D(grid, aquifer, wells, scenarios, fwd)
    f3d.log(f"{label}: archive ready in {time.time() - t0:.1f}s "
            f"(KL n_kl={kl.n_kl}, retained={kl.retained:.4f})")

    ev = archive.evaluate(q, cost)
    return ev


def compare(label, np_summary, torch_summary):
    """Print a per-key comparison between the NumPy and torch summaries."""
    keys = ["mean_J", "se_J", "mean_G", "reliability", "violation_prob",
            "mean_H", "mean_tau", "failure_fraction", "J_pump", "J_treat",
            "J_res", "J_time", "max_mass_balance_error"]
    rows = []
    for k in keys:
        if k not in np_summary or k not in torch_summary:
            continue
        a, b = np_summary[k], torch_summary[k]
        if isinstance(a, float) and isinstance(b, float):
            denom = max(abs(b), 1e-30)
            rel = (a - b) / denom
            rows.append((k, a, b, rel))
    f3d.log(f"== {label}: numpy(splu) vs torch(float32) ==")
    print(f"{'quantity':<24} {'numpy':>16} {'torch32':>16} {'rel err':>12}")
    for k, a, b, rel in rows:
        print(f"{k:<24} {a:16.6g} {b:16.6g} {rel:12.3e}")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="out3d_colab/run_paperGPU",
                    help="finished run directory")
    ap.add_argument("--archive", choices=["train", "test", "both"], default="both")
    ap.add_argument("--max-m", type=int, default=0,
                    help="cap scenarios (0 = all) for a quick smoke check")
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    cfg, pol, grid, aquifer, kl, fwd, cost, wells, q = rebuild(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())

    f3d.log(f"verify {run_dir}: grid {grid.nx}x{grid.ny}x{grid.nz} "
            f"({grid.n} cells), n_wells={wells.shape[0]}")
    f3d.log(f"q* = {np.round(q, 3).tolist()} (sum {q.sum():.3f} m3/d)")

    results = {}

    if args.archive in ("train", "both"):
        seeds = [cfg["seed_train"] + i for i in range(cfg["m_train"])]
        ev = score_archive(grid, aquifer, kl, fwd, cost, wells, q, seeds,
                           "train", args.max_m)
        s = ev.summary()
        results["train"] = s
        compare("train", s, metrics["opt"]["train"])

    if args.archive in ("test", "both"):
        seeds = [cfg["seed_test"] + i for i in range(cfg["m_test"])]
        ev = score_archive(grid, aquifer, kl, fwd, cost, wells, q, seeds,
                           "test", args.max_m)
        s = ev.summary()
        results["test"] = s
        compare("test", s, metrics["opt"]["test"])

    out = run_dir / "verify_splu.json"
    out.write_text(json.dumps(results, indent=2))
    f3d.log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
