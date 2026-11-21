#!/usr/bin/env python3
"""Re-evaluate a finished run's policy on a fresh scenario set (NumPy reference).

Why this exists
---------------
A run's `metrics.json` reports the train/test summaries that the *run* computed.
This script recomputes them independently, from the reported well layout and
`q*`, on the NumPy reference backend, so a disagreement is visible instead of
being taken on trust.  It also prints the empirical-CVaR tail at several sample
sizes, which is how a chance constraint should be judged (see DOCS/GPU.md s7).

Usage
-----
    python3 SCRIPTS/check_tail.py --metrics runs/colab_multi/<run>/r0/artifacts.zip
    python3 SCRIPTS/check_tail.py --run out/opt/full_opt --m-scen 100

`--metrics` accepts either the artifacts zip or an extracted run directory; the
script looks for `metrics.json` underneath it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SCRIPTS"))

from stoch.config import load_config  # noqa: E402
from stoch.objective import cvar_and_weights  # noqa: E402
from stoch.saa import Archive  # noqa: E402


def find_metrics(path: Path) -> Path:
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.endswith("metrics.json"):
                    return path  # caller reads it through the zip
    if path.is_file():
        return path
    hits = sorted(path.rglob("metrics.json"))
    if not hits:
        raise SystemExit(f"no metrics.json under {path}")
    return hits[0]


def load_metrics(path: Path) -> dict:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.endswith("metrics.json"))
            return json.loads(zf.read(name))
    return json.loads(Path(path).read_text())


def build(config: str, nsteps: int | None, m_scen: int) -> dict:
    """Rebuild the run's scenario set with the run's own constructors.

    `run_optimization` owns the mapping config -> grid/KL/aquifer/forward/cost,
    so verification imports it rather than re-deriving it.  A re-derivation here
    once transposed the grid (`Grid` is (nx, ny, lx, ly)) and produced a
    different KL basis, i.e. a different set of scenarios, which looked like a
    reproducibility bug in the runs.
    """
    import copy

    import run_optimization as ropt

    cfg = copy.deepcopy(load_config(config))
    if nsteps:
        cfg["time"]["nsteps"] = int(nsteps)
    grid = ropt.make_grid(cfg)
    aq = ropt.base_aquifer(cfg, grid)
    kl = ropt.make_kl(cfg, grid)
    gen = ropt.ScenarioGenerator(
        grid, kl,
        random_source=bool(cfg["random_field"]["random_source"]),
        random_gradient=bool(cfg["random_field"]["random_gradient"]),
    )
    scen = [gen.sample(int(cfg["seed"]["train"]) + i) for i in range(m_scen)]
    return {"grid": grid, "aq": aq, "scen": scen, "fwd": ropt.make_forward(cfg),
            "cost": ropt.make_cost(cfg), "n_kl": kl.n_kl}


def wilson_interval(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (k successes out of n).

    A plain ``k/n`` hides how much a reliability figure depends on how many
    scenarios were drawn: 39/40 and 97/100 are both "about 0.975" but the first
    is consistent with a true reliability anywhere in ~0.87-0.995.  Reporting the
    interval is what makes a certificate checkable, and it is the honest answer to
    "is 0.975 above the 0.94 gate?" when the sample is small.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", "--run", dest="metrics", required=True,
                    help="artifacts zip, run directory, or metrics.json")
    ap.add_argument("--config", default=None,
                    help="opt YAML (default: the run's own config if resolvable)")
    ap.add_argument("--m-scen", type=int, default=100,
                    help="scenarios to re-evaluate on (default 100)")
    ap.add_argument("--nsteps", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--ladder", action="store_true",
                    help="walk M=16..(--m-scen) in fixed steps instead of a few points")
    ap.add_argument("--json", default=None, help="write the summary to this JSON path")
    args = ap.parse_args()

    path = find_metrics(Path(args.metrics))
    met = load_metrics(path)
    cfg_path = args.config
    if cfg_path is None:
        cfg_path = str(ROOT / "configs/opt/hydrology_gpu.yaml")
        print(f"[check_tail] config not given, assuming {cfg_path}")
    o0 = met["stages"].get("O0") or {}
    wells = np.array(o0["well_xy"], float)
    opt = met["stages"]["opt"]
    q = np.array(opt["q_star"], float)

    b = build(cfg_path, args.nsteps, args.m_scen)
    arch = Archive(b["grid"], b["aq"], wells, b["scen"], b["fwd"])
    print(f"[check_tail] grid {b['grid'].ny}x{b['grid'].nx} n_kl {b['n_kl']} "
          f"wells {wells.shape[0]} scenarios {len(b['scen'])}")
    print(f"[check_tail] q* {np.round(q, 3).tolist()}  total {q.sum():.2f}")

    ev = arch.evaluate(q, b["cost"], keep=False)
    G = np.array([d["G"] for d in ev.details])
    cvar, _ = cvar_and_weights(G, args.alpha)
    n_ok = int((G <= 0).sum())
    lo, hi = wilson_interval(n_ok, len(G))
    gate = 1.0 - args.alpha - 0.01
    print(f"[check_tail] recomputed : J {ev.mean_J:.1f}  meanG {G.mean():.3f}  "
          f"CVaR{args.alpha:.0%} {cvar:.3f}  reliability {n_ok}/{len(G)} = "
          f"{n_ok / len(G):.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
          f"gate {gate:.3f} -> {'PASS' if lo >= gate else ('FAIL' if hi < gate else 'INCONCLUSIVE')}")
    for tag in ("train", "test"):
        s = opt.get(tag) or {}
        if s:
            print(f"[check_tail] reported {tag:5s}: J {s['mean_J']:.1f}  "
                  f"meanG {s['mean_G']:.3f}  CVaR {s['cvar_G']:.3f}  "
                  f"reliability {s['reliability']:.3f}  (M={s['M']})")
    print("[check_tail] tail vs sample size (reliability 95% Wilson interval):")
    ladder = ([16, 20, 40, 60, 100, 150, 200, 300, args.m_scen] if args.ladder
              else [16, 20, 40, 60, 100, 200, 400, args.m_scen])
    seen = set()
    summary = []
    for m in ladder:
        if m > len(G) or m in seen:
            continue
        seen.add(m)
        c, _ = cvar_and_weights(G[:m], args.alpha)
        k = int((G[:m] <= 0).sum())
        lo_m, hi_m = wilson_interval(k, m)
        print(f"    M={m:4d}  CVaR{args.alpha:.0%} {c:8.3f}  "
              f"violations {m - k:3d}/{m}  reliability {k / m:.3f} "
              f"[{lo_m:.3f}, {hi_m:.3f}]")
        summary.append({"M": m, "cvar": float(c), "violations": m - k,
                        "reliability": k / m, "wilson": [lo_m, hi_m]})
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"config": cfg_path, "q_star": q.tolist(), "alpha": args.alpha,
             "J": float(ev.mean_J), "mean_G": float(G.mean()), "cvar": float(cvar),
             "reliability": n_ok / len(G), "wilson": [lo, hi], "ladder": summary},
            indent=2))
        print(f"[check_tail] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
