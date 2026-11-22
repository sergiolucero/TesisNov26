#!/usr/bin/env python3
"""Render the plume movie for an already-finished 3D optimization run.

``full3D_colab_run.run`` renders its movie as part of the run; this script does
the same thing from the artifacts alone, for a run whose movie is missing (older
run, a crashed renderer, or a run started before the movie existed):

    python3 SCRIPTS/movie3d_from_policy.py --run out3d_colab/run_paper
    python3 SCRIPTS/movie3d_from_policy.py --run <dir> --frames 32 --scenario 3

Nothing is re-optimized.  The policy is read from ``policy.json``/``metrics.json``,
the scenario set is rebuilt from ``run_config.json`` through the run's own
constructors, and only two transport solves are performed (untreated and ``q*``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SCRIPTS"))


def read_run(run_dir: Path) -> tuple[dict, dict]:
    """Config and policy of a finished run (tolerating a missing policy.json)."""
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"{run_dir} has no run_config.json")
    cfg = json.loads(cfg_path.read_text())
    pol_path = run_dir / "policy.json"
    if pol_path.exists():
        pol = json.loads(pol_path.read_text())
    else:
        metrics = json.loads((run_dir / "metrics.json").read_text())
        pol = {"q": metrics["opt"]["q_star"], "wells": metrics.get("wells")}
    return cfg, pol


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="finished run directory")
    ap.add_argument("--out", default=None, help="output dir (default <run>/movies)")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--scenario", type=int, default=0,
                    help="which held-out scenario to illustrate")
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--stride", type=int, default=1,
                    help="spatial coarsening for large grids (2 or 3 is plenty)")
    args = ap.parse_args(argv)

    import full3D_colab_run as f3d

    run_dir = Path(args.run)
    cfg, pol = read_run(run_dir)
    out = Path(args.out) if args.out else run_dir / "movies"

    # Rebuild through the run's own constructors: hand-rolled rebuilds are how a
    # diagnostic ends up describing a different problem than the run did.
    grid = f3d.Grid3D(cfg["nx"], cfg["ny"], cfg["nz"], cfg["lx"], cfg["ly"], cfg["lz"])
    aquifer = f3d.Aquifer3D(
        K=np.full((grid.nz, grid.ny, grid.nx), cfg["k0"]), kv_ratio=cfg["kv_ratio"],
        phi=cfg["phi"], R=cfg["R"], alpha_l=cfg["alpha_l"],
        alpha_th=cfg["alpha_th"], alpha_tv=cfg["alpha_tv"], decay=cfg["decay"])
    kl = f3d.KL3D(grid, cfg["sigma"], cfg["ell_x"], cfg["ell_y"], cfg["ell_z"],
                  n_kl=cfg["n_kl"], k0=cfg["k0"], kv_ratio=cfg["kv_ratio"],
                  target_variance=cfg["target_variance"])
    fwd = f3d.ForwardConfig3D(
        dt=cfg["dt"], nsteps=cfg["nsteps"], grad=cfg["grad"],
        source_xyz=tuple(cfg["source_xyz"]), source_q=cfg["source_q"],
        source_c=cfg["source_c"], source_days=cfg["source_days"],
        compliance_xyz=[tuple(p) for p in cfg["compliance_xyz"]],
        obs_xyz=[tuple(p) for p in cfg["obs_xyz"]], eps_well=cfg["eps_well"])
    cost = f3d.CostConfig3D(creg=cfg["creg"], alpha=cfg["alpha"],
                            alpha_h=cfg["alpha_h"])
    wells = np.atleast_2d(np.asarray(pol.get("wells") or
                                     json.loads((run_dir / "metrics.json").read_text())["wells"],
                                     dtype=float))
    q = np.asarray(pol["q"], dtype=float)

    scenarios = f3d.sample_scenarios(grid, kl, [cfg["seed_test"] + args.scenario])
    archive = f3d.Archive3D(grid, aquifer, wells, scenarios, fwd)
    info = f3d.make_plume_movie(out, grid, archive, wells, q, fwd, cost=cost,
                                n_frames=args.frames, scenario_index=0,
                                stride_xy=args.stride,
                                stride_z=int(cfg.get("movie_stride_z", 1)))
    print(json.dumps(info, indent=2))
    return 0 if not info.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
