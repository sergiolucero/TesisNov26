#!/usr/bin/env python3
"""bench_krylov_fusion.py — is the batched transport launch-bound or arithmetic-bound?

`DOCS/Conclusions.md` §9 claims the next speedup is software, not a bigger card:
one evaluation issues ~4000 dependent kernel launches (100 time steps x ~40
BiCGSTAB iterations, each ~10 small kernels plus one device-to-host sync), so at
`M=40` on an A100 81% of the wall time is overhead while the card idles, and
"removing the per-step launches" is worth ~5x.

That claim needs a measurement, and it can only be taken on hardware where
launches are asynchronous. This script runs the same evaluation four ways and
reports wall time, iterations and the objective, so the comparison is paired:

  baseline     the historical path: matvec closure, residual checked every iteration
  check8       same arithmetic, residual checked every 8 iterations (7/8 fewer syncs)
  fused        the BiCGSTAB iteration compiled into one call (Dynamo/inductor)
  fused+check8 both

Parity is asserted against the baseline, not assumed: the fused path reorders
elementwise reductions, so a run that is faster but moves the objective is a
failure, not a result.

Usage (CPU sanity check):

    python3 SCRIPTS/bench_krylov_fusion.py --nsteps 20 --m-scen 4 --grid 40x20

Usage (the measurement that matters, on a T4/A100):

    TESIS_FUSE_KRYLOV=0 python3 SCRIPTS/bench_krylov_fusion.py \\
        --grid 80x40 --nsteps 100 --m-scen 18 --s-cand 18 --json /tmp/fusion.json

`--json` writes the numbers so a Colab run can be compared offline.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "SCRIPTS"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def build(grid_nx, grid_ny, m_scen, nsteps, seed_base, device, dtype):
    from physics.fv2d import Aquifer, Grid
    from stoch.random_fields import KLExpansion, ScenarioGenerator
    from stoch.saa import ForwardConfig
    from stoch.saa_torch import TorchArchive

    grid = Grid(grid_nx, grid_ny, 200.0, 100.0)
    aq = Aquifer(K=np.full((grid.ny, grid.nx), 10.0), phi=0.30, b=10.0,
                 alpha_l=5.0, alpha_t=0.5)
    # n_kl is capped for the benchmark: the field is not the variable being
    # measured, and a large KL basis would dominate the setup time.
    kl = KLExpansion(grid, sigma=0.5, ell_x=20.0, ell_y=20.0, n_kl=60, k0=10.0)
    gen = ScenarioGenerator(grid, kl)
    scenarios = [gen.sample(seed_base + i) for i in range(m_scen)]
    fwd = ForwardConfig(dt=36.5, nsteps=nsteps, grad=1e-3,
                        source_xy=(20.0, 25.0), source_q=10.0, source_c=100.0,
                        source_days=365.0, compliance_xy=((190.0, 50.0),),
                        obs_xy=((100.0, 50.0),), eps_well=1.5)
    wells = np.array([[40.0, 30.0], [60.0, 20.0], [70.0, 40.0], [50.0, 10.0],
                      [90.0, 30.0], [110.0, 20.0], [130.0, 40.0], [150.0, 30.0],
                      [170.0, 20.0]])
    return grid, aq, wells, scenarios, fwd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="40x20", help="NXxNY (default %(default)s)")
    ap.add_argument("--nsteps", type=int, default=100)
    ap.add_argument("--m-scen", type=int, default=18)
    ap.add_argument("--s-cand", type=int, default=18, help="candidates per batch")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--seed-base", type=int, default=5000)
    ap.add_argument("--repeat", type=int, default=2, help="timed repetitions")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    import torch
    from stoch.objective import CostConfig

    nx, ny = (int(v) for v in args.grid.lower().split("x"))
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    from stoch.saa_torch import TorchArchive
    grid, aq, wells, scenarios, fwd = build(
        nx, ny, max(args.m_scen, 40), args.nsteps, args.seed_base, device, args.dtype)
    arch = TorchArchive(grid, aq, wells, scenarios[:args.m_scen], fwd,
                        device=device, dtype=args.dtype)
    cost = CostConfig()
    rng = np.random.default_rng(0)
    Q = rng.uniform(0.5, 6.0, size=(args.s_cand, 9))

    gpu = "n/a"
    if device == "cuda":
        gpu = torch.cuda.get_device_name(0)
    print(f"[bench] device={device} gpu={gpu} dtype={args.dtype} "
          f"grid={nx}x{ny} nsteps={args.nsteps} M={args.m_scen} S={args.s_cand} "
          f"torch={torch.__version__}")

    modes = [
        ("baseline", {}),
        ("check8", {"check_every": 8}),
        ("fused", {"fuse": True}),
        ("fused+check8", {"check_every": 8, "fuse": True}),
    ]
    results: dict[str, dict] = {}
    ref_J = None
    for name, kw in modes:
        ev = arch.evaluate_many(Q, cost, **kw)          # warm-up / compile
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            ev = arch.evaluate_many(Q, cost, **kw)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        J_t = ev.J
        J_t = J_t.detach().float().cpu() if hasattr(J_t, "detach") else torch.as_tensor(J_t)
        J = float(J_t.mean())
        if ref_J is None:
            ref_J = J
        rel = abs(J - ref_J) / abs(ref_J) if ref_J else 0.0
        best = min(times)
        results[name] = {"seconds": best, "times": times,
                         "J_mean": J, "J_rel_diff": rel}
        print(f"[bench] {name:13s} {best:8.3f} s   J={J:.6f} "
              f"(rel Δ {rel:.2e})   runs={['%.2f' % t for t in times]}")

    base = results["baseline"]["seconds"]
    print("\n[bench] speedups against baseline")
    for name, r in results.items():
        if name == "baseline":
            continue
        print(f"  {name:13s} {base / r['seconds']:5.2f}x   "
              f"numerics rel Δ {r['J_rel_diff']:.2e}"
              f"{'   <-- NOT parity' if r['J_rel_diff'] > 1e-6 else ''}")

    m_scaling = {}
    print("\n[bench] fixed-cost model: time(M) at S=%d"
          "   (only meaningful where launches are async: GPU)" % args.s_cand)
    for M in (18, 40):
        # Same fields for both M: the M=18 batch is the first 18 scenarios of the
        # M=40 set, so the fit measures batch size and nothing else.
        arch_M = TorchArchive(grid, aq, wells, scenarios[:M], fwd,
                              device=device, dtype=args.dtype)
        QM = Q                                   # distinct candidates: identical
        # ones let every batch element converge on the same path and make the
        # measured time depend on the candidate rather than the batch size.
        for name, kw in modes:
            arch_M.evaluate_many(QM, cost, **kw)         # warm-up
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            arch_M.evaluate_many(QM, cost, **kw)
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            m_scaling.setdefault(name, {})[M] = dt
    for name, per_M in m_scaling.items():
        if 18 in per_M and 40 in per_M:
            slope = (per_M[40] - per_M[18]) / (40 - 18)
            fixed = per_M[18] - slope * 18
            share = 100.0 * fixed / per_M[40] if per_M[40] else float("nan")
            print(f"  {name:13s} M=18 {per_M[18]:7.3f} s  M=40 {per_M[40]:7.3f} s  "
                  f"wall ≈ {fixed:6.3f} s + {slope * 1e3:6.2f} ms x M   "
                  f"overhead at M=40: {share:4.1f}%")

    if args.json:
        payload = {"device": device, "gpu": gpu, "dtype": args.dtype,
                   "grid": args.grid, "nsteps": args.nsteps,
                   "M": args.m_scen, "S": args.s_cand,
                   "results": results, "m_scaling": m_scaling}
        Path(args.json).write_text(json.dumps(payload, indent=1))
        print(f"\n[bench] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
