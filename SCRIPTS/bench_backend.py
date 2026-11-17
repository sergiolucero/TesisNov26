#!/usr/bin/env python3
"""Measure whether the GPU backend actually pays off for this workload.

This is the "should I pay for Colab Pro?" script.  It runs the same
configuration through both scenario archives and reports, side by side:

1. **Cost of one SAA evaluation** -- the unit the pattern search spends its
   budget on (``M`` scenarios, one candidate rate vector).
2. **Cost of one compass sweep** -- ``2 * n_wells`` candidates, which the
   batched backend does in a single call.
3. **Agreement** -- the relative difference in the objective, so a speedup
   cannot be bought with a wrong answer.
4. **Projection** -- the estimated wall clock of ``optim.max_evals``
   evaluations, which is what the full optimization stage actually costs.

Usage
-----
    # on a Colab GPU runtime, full benchmark sizing
    !python SCRIPTS/bench_backend.py --device cuda --dtype float32

    # CPU-only sanity check of the same geometry (no GPU needed)
    python SCRIPTS/bench_backend.py --device cpu --dtype float64 --quick

The script never claims a speedup it did not measure: if the GPU is slower
(small grids are launch-bound), it says so.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from physics.fv2d import Aquifer, Grid
from physics.fv2d_torch import resolve_device, torch_dtype
from stoch.config import load_config
from stoch.objective import CostConfig
from stoch.optimize import project_capacity
from stoch.random_fields import KLExpansion, ScenarioGenerator
from stoch.saa import Archive, ForwardConfig
from stoch.saa_torch import TorchArchive
from stoch.wells import place_wells


def log(msg: str) -> None:
    print(f"[bench] {msg}", flush=True)


def timed(fn, repeats: int = 3) -> tuple[float, object]:
    """Best-of-``repeats`` wall clock (best-of is the least noisy estimator)."""
    best = float("inf")
    out = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def build_inputs(cfg: dict, n_scenarios: int, seed: int):
    d, a = cfg["domain"], cfg["aquifer"]
    grid = Grid(int(d["nx"]), int(d["ny"]), float(d["lx"]), float(d["ly"]))
    aquifer = Aquifer(
        K=np.full((grid.ny, grid.nx), float(a["k0"])), phi=float(a["phi"]),
        R=float(a["R"]), b=float(a["b"]),
        alpha_l=float(a["alpha_l"]), alpha_t=float(a["alpha_t"]),
        decay=float(a["decay"]),
    )
    r = cfg["random_field"]
    kl = KLExpansion(grid, sigma=float(r["sigma"]), ell_x=float(r["ell_x"]),
                     ell_y=float(r["ell_y"]), n_kl=int(r.get("n_kl", 0)),
                     k0=float(a["k0"]),
                     target_variance=float(r.get("target_variance", 0.95)))
    gen = ScenarioGenerator(grid, kl)
    scenarios = [gen.sample(seed + i) for i in range(n_scenarios)]

    c, s, t, o = cfg["compliance"], cfg["source"], cfg["time"], cfg["observation"]
    fwd = ForwardConfig(
        dt=float(t["dt"]), nsteps=int(t["nsteps"]), grad=float(cfg["flow"]["grad"]),
        source_xy=(float(s["x"]), float(s["y"])), source_q=float(s["q"]),
        source_c=float(s["c"]), source_days=float(s["days"]),
        compliance_xy=tuple(zip(c["x"], c["y"])),
        obs_xy=tuple(zip(o["x"], o["y"])),
        eps_well=float(cfg["wells"]["eps"]),
    )
    cost = CostConfig(
        c_e=float(cfg["cost"]["c_e"]), c_w=float(cfg["cost"]["c_w"]),
        c_m=float(cfg["cost"]["c_m"]), c_tau=float(cfg["cost"]["c_tau"]),
        c_fail=float(cfg["cost"]["c_fail"]),
        creg=float(c["creg"]), window_start=float(c["window_start"]),
        h_min=float(o["h_min"]), time_mode=cfg["cost"]["time_mode"],
        eps_time=float(cfg["cost"]["eps_time"]),
        alpha=float(cfg["saa"]["alpha"]), alpha_h=float(cfg["saa"]["alpha_h"]),
    )
    wells = place_wells(grid, np.ones((grid.ny, grid.nx), dtype=bool),
                        int(cfg["wells"]["n"]), method=cfg["wells"]["method"],
                        seed=int(cfg["optim"]["seed"]))
    return grid, aquifer, wells.wells, scenarios, fwd, cost


def compass_set(q0: np.ndarray, step: float, q_max: float, Q_max: float) -> np.ndarray:
    """The pattern search's compass set: q +/- step * e_i, capacity-projected."""
    cands = []
    for i in range(q0.size):
        for sgn in (1.0, -1.0):
            cand = q0.copy()
            cand[i] += sgn * step
            cands.append(project_capacity(cand, q_max, Q_max))
    return np.stack(cands)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "opt" / "benchmark.yaml"))
    ap.add_argument("--quick", action="store_true", help="shrink grid/scenarios")
    ap.add_argument("--device", default="auto", help="torch device for the batched backend")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--scenarios", type=int, default=None,
                    help="scenarios in the archive (default: saa.M_train)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-numpy", action="store_true",
                    help="only benchmark the batched backend (large grids)")
    ap.add_argument("--out", default=None, help="write the report as JSON here")
    args = ap.parse_args()

    cfg = load_config(args.config, quick=args.quick)
    n_scen = args.scenarios or int(cfg["saa"]["M_train"])
    grid, aquifer, wells, scenarios, fwd, cost = build_inputs(cfg, n_scen, 1000)
    q_max = float(cfg["wells"]["q_max"])
    Q_max = float(cfg["wells"]["Q_max"])
    q0 = project_capacity(np.full(wells.shape[0], 0.5 * q_max), q_max, Q_max)
    compass = compass_set(q0, 0.25 * q_max, q_max, Q_max)
    max_evals = int(cfg["optim"].get("max_evals", 2000))

    device = resolve_device(args.device)
    log(f"config     : {Path(args.config).name}{' (--quick)' if args.quick else ''}")
    log(f"grid       : {grid.nx}x{grid.ny} = {grid.n:,} cells, {fwd.nsteps} steps, "
        f"{wells.shape[0]} wells, {n_scen} scenarios")
    log(f"numpy ref  : {'skipped' if args.skip_numpy else 'SciPy splu / float64'}")
    log(f"torch      : {device} / {torch_dtype(args.dtype)}")
    if device.type == "cuda":
        import torch

        props = torch.cuda.get_device_properties(device)
        log(f"gpu        : {props.name}, {props.total_memory / 1e9:.1f} GB")
    else:
        log("gpu        : none -- the batched backend runs on CPU, where it is "
            "expected to be slower than SciPy")

    report: dict = {
        "config": str(Path(args.config).resolve()),
        "grid": [grid.nx, grid.ny],
        "nsteps": fwd.nsteps,
        "n_wells": int(wells.shape[0]),
        "scenarios": n_scen,
        "device": str(device),
        "dtype": args.dtype,
        "max_evals": max_evals,
    }

    # ---- NumPy reference --------------------------------------------------
    J_ref = None
    if not args.skip_numpy:
        t_arch, archive_np = timed(lambda: Archive(grid, aquifer, wells, scenarios, fwd),
                                   repeats=1)
        t_eval, ev_np = timed(lambda: archive_np.evaluate(q0, cost), args.repeats)
        J_ref = float(ev_np.mean_J)
        report["numpy"] = {
            "archive_build_s": round(t_arch, 3),
            "eval_s": round(t_eval, 4),
            "mean_J": J_ref,
            "mass_balance_error": float(max(abs(d["mass_balance_error"])
                                            for d in ev_np.details)),
        }
        log(f"numpy      : archive {t_arch:.2f}s | one evaluation {t_eval * 1e3:.1f} ms "
            f"| mean_J {J_ref:.6g}")

    # ---- batched backend --------------------------------------------------
    t_arch_t, archive_t = timed(
        lambda: TorchArchive(grid, aquifer, wells, scenarios, fwd,
                             device=device, dtype=args.dtype), repeats=1)
    t_eval_t, ev_t = timed(lambda: archive_t.evaluate(q0, cost), args.repeats)
    J_t = float(ev_t.mean_J)
    t_compass, _ = timed(lambda: archive_t.evaluate_many(compass, cost), args.repeats)

    info = archive_t.describe()
    report["torch"] = {
        "archive_build_s": round(t_arch_t, 3),
        "eval_s": round(t_eval_t, 4),
        "compass_s": round(t_compass, 4),
        "compass_candidates": int(compass.shape[0]),
        "mean_J": J_t,
        "mass_balance_error": float(max(abs(d["mass_balance_error"])
                                        for d in ev_t.details)),
        "flow_krylov": info["flow_krylov"],
        "transport_krylov": None,
    }
    log(f"torch      : archive {t_arch_t:.2f}s | one evaluation {t_eval_t * 1e3:.1f} ms "
        f"| compass of {compass.shape[0]} {t_compass * 1e3:.1f} ms | mean_J {J_t:.6g}")
    log(f"             flow Krylov {info['flow_krylov']}")

    # ---- agreement --------------------------------------------------------
    if J_ref is not None:
        rel = abs(J_t - J_ref) / max(abs(J_ref), 1e-30)
        report["agreement"] = {"mean_J_rel_diff": rel,
                               "mass_balance_numpy": report["numpy"]["mass_balance_error"],
                               "mass_balance_torch": report["torch"]["mass_balance_error"]}
        ok = rel < (1e-6 if args.dtype == "float64" else 1e-3)
        log(f"agreement  : |dJ|/J = {rel:.2e} "
            f"({'OK' if ok else 'MISMATCH -- investigate before trusting timings'})")

    # ---- projection -------------------------------------------------------
    cost_per_eval_np = report.get("numpy", {}).get("eval_s")
    # The batched search spends ceil(evals / compass_size) sweeps; a sequential
    # search spends one evaluation per candidate.  Both are projected with the
    # measured per-unit cost, which is the honest comparison.
    sweeps = max(int(np.ceil(max_evals / compass.shape[0])), 1)
    proj = {
        "torch_compass_projected_s": round(t_compass * sweeps, 1),
        "torch_sequential_projected_s": round(t_eval_t * max_evals, 1),
    }
    if cost_per_eval_np is not None:
        proj["numpy_projected_s"] = round(cost_per_eval_np * max_evals, 1)
    report["projection"] = proj
    log(f"projection : {max_evals} evaluations -> "
        + " | ".join(f"{k.replace('_projected_s', '')} {v:.0f}s" for k, v in proj.items()))

    # ---- verdict ----------------------------------------------------------
    batch_vs_np = None
    if cost_per_eval_np:
        batch_vs_np = cost_per_eval_np / max(t_eval_t, 1e-9)
        verdict = (
            f"batched backend is {batch_vs_np:.2f}x the NumPy evaluation cost"
        )
    else:
        verdict = "NumPy reference skipped; no comparison available"
    if device.type == "cuda" and batch_vs_np and batch_vs_np < 1.0:
        verdict += " -- the GPU pays for itself on this workload"
    elif device.type == "cuda":
        verdict += (" -- the GPU does not pay for itself at this size; "
                    "increase nx*ny, nsteps or saa.M_train")
    report["verdict"] = verdict
    log(f"verdict    : {verdict}")

    if args.out:
        import json

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        log(f"report     : {args.out}")


if __name__ == "__main__":
    main()
