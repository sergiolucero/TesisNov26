#!/usr/bin/env python3
"""probe_krylov.py -- diagnostic for the batched torch transport solver.

Why this exists
---------------
A paid A100/T4 run at 160x96 in float32 came back with ``nan`` objectives for a
handful of compass candidates (``objective is not finite for candidate(s)
[10, 12, 13, 17]``).  The transport report said "did not converge, worst
residual 0.00e+00, 81 iterations, stalled=True" -- a NaN residual, which Python
``max()`` silently swallows (``max(0.0, nan) == 0.0``), so the log looked
harmless.  This script reproduces that on demand and localises it.

What it prints
--------------
1. how many Krylov iterations the compass batch needs, and whether the
   concentration field stays finite;
2. which candidates produce a non-finite objective;
3. for the first bad candidate, the divergence onset: the same solve with
   ``maxiter`` = 5, 10, 20, 40, 80, 160, ... printing the residual and the
   largest magnitude in the field, which separates "tolerance unreachable"
   (residual plateaus) from "solver breakdown" (field blows up).

Usage
-----
    python3 SCRIPTS/probe_krylov.py --config configs/opt/hydrology_gpu.yaml \
        --device cuda --dtype float32 --m-scen 8          # on a GPU VM
    python3 SCRIPTS/probe_krylov.py --m-scen 2 --nsteps 20 --nx 80 --ny 48
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stoch.config import load_config                        # noqa: E402
from stoch.optimize import project_capacity                  # noqa: E402
from stoch.saa_torch import TorchArchive                     # noqa: E402
from stoch.wells import place_wells                          # noqa: E402


def build(args) -> dict:
    """Build the same objects run_optimization.py builds, from a config.

    The construction is delegated to `run_optimization` on purpose: this script
    exists to diagnose what a *run* does, so it must not re-derive the grid, the
    KL basis, the wells or the cost.  (Re-deriving it once produced a transposed
    grid here -- `Grid` is (nx, ny, lx, ly) -- which quietly changed dx/dy and
    n_kl and made the comparison meaningless.)
    """
    import copy

    import run_optimization as ropt

    cfg = load_config(args.config)
    cfg = copy.deepcopy(cfg)
    if args.nx or args.ny:
        cfg["domain"]["nx"] = int(args.nx or cfg["domain"]["nx"])
        cfg["domain"]["ny"] = int(args.ny or cfg["domain"]["ny"])
    if args.nsteps:
        cfg["time"]["nsteps"] = int(args.nsteps)
    grid = ropt.make_grid(cfg)
    aq = ropt.base_aquifer(cfg, grid)
    kl = ropt.make_kl(cfg, grid)
    gen = ropt.ScenarioGenerator(
        grid, kl,
        random_source=bool(cfg["random_field"]["random_source"]),
        random_gradient=bool(cfg["random_field"]["random_gradient"]),
    )
    scen = [gen.sample(int(cfg["seed"]["train"]) + i) for i in range(args.m_scen)]
    wells = place_wells(grid, np.ones((grid.ny, grid.nx), bool),
                        int(cfg["wells"]["n"]), method=cfg["wells"]["method"],
                        seed=0).wells
    return {"cfg": cfg, "grid": grid, "aq": aq, "scen": scen, "wells": wells,
            "fwd": ropt.make_forward(cfg), "cost": ropt.make_cost(cfg),
            "w": cfg["wells"], "n_kl": kl.n_kl}


def compass(q0: np.ndarray, q_max: float, Q_max: float, step: float) -> np.ndarray:
    cands = [q0]
    for i in range(len(q0)):
        for sign in (+1.0, -1.0):
            q = q0.copy()
            q[i] += sign * step
            cands.append(project_capacity(q, q_max, Q_max))
    return np.array(cands)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "opt" / "hydrology_gpu.yaml"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--m-scen", type=int, default=8)
    ap.add_argument("--nsteps", type=int, default=None)
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--maxiter", type=int, default=None)
    ap.add_argument("--step-frac", type=float, default=0.25,
                    help="compass step as a fraction of q_max (default 0.25)")
    ap.add_argument("--probe", action="store_true", default=True,
                    help="run the maxiter ladder on the first non-finite candidate")
    args = ap.parse_args()

    import torch

    b = build(args)
    grid, w = b["grid"], b["w"]
    arch = TorchArchive(grid, b["aq"], b["wells"], b["scen"], b["fwd"],
                        device=args.device, dtype=args.dtype, rtol=args.rtol,
                        maxiter=args.maxiter or 4000)
    print(f"grid {grid.nx}x{grid.ny}  M={arch.M}  n_kl={b['n_kl']}  "
          f"device={arch.device} dtype={args.dtype} "
          f"rtol={arch.flow.rtol:g} maxiter={arch.maxiter} "
          f"nsteps={b['fwd'].nsteps}")
    print("flow:", arch.resp.report.as_dict())

    q_max, Q_max = float(w["q_max"]), float(w["Q_max"])
    q0 = project_capacity(np.full(len(b["wells"]), 0.5 * q_max), q_max, Q_max)
    cands = compass(q0, q_max, Q_max, args.step_frac * q_max)

    res = arch.transport(cands)
    rep = res.report
    print(f"transport: iters/solve={rep.iterations_per_solve:.1f} "
          f"worst={rep.worst_iterations} res={rep.worst_residual:.3e} "
          f"converged={rep.converged} stalled={rep.stalled} "
          f"finite_field={bool(torch.isfinite(res.C_final).all())} "
          f"|C|max={float(res.C_final.abs().max()):.4g}")

    ev = arch.evaluate_many(cands, b["cost"])
    J = ev.J.detach().cpu().numpy()
    ok = [bool(np.all(np.isfinite(row))) for row in J]
    print(f"finite objectives: {sum(ok)}/{len(ok)}")
    bad = [i for i, k in enumerate(ok) if not k]
    if bad:
        print(f"BAD candidates {bad}")
        for i in bad[:4]:
            print(f"  candidate {i}: q={np.round(cands[i], 3)}")

    if bad and args.probe:
        i = bad[0]
        q = cands[i:i + 1]
        print(f"\nmaxiter ladder for candidate {i} (S=1, M={arch.M}):")
        Q = torch.as_tensor(q, dtype=arch.dtype, device=arch.device)
        saved = arch.maxiter
        for mi in (5, 10, 20, 40, 80, 160, 320, 640):
            arch.maxiter = mi
            r = arch.transport(Q)
            rr = r.report
            print(f"  maxiter={mi:5d} iters={rr.iterations_per_solve:7.1f} "
                  f"res={rr.worst_residual:.3e} conv={rr.converged} "
                  f"stalled={rr.stalled} finite={bool(torch.isfinite(r.C_final).all())} "
                  f"|C|max={float(r.C_final.abs().max()):.4g}")
        arch.maxiter = saved
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
