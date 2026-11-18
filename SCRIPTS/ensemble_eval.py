#!/usr/bin/env python3
"""ensemble_eval.py — evaluate an ensemble average of trained PINN models.

Why this exists
---------------
``DOCS/Conclusions.md`` §1: at fixed hyperparameters the 1D PINN's relative L2
error spans a ~7x range across seeds, so a single run is not a result. One of
the candidate variance reductions in ``DOCS/Actions.md`` is an *ensemble
average at eval time*: train N models from different seeds and average their
predictions. Averaging independent, unbiased predictors reduces the variance of
the estimate by ~1/N, so it directly attacks the seed spread without touching
the training loss.

This script takes N saved model state_dicts (from
``solve_1d_pinn.py --save-model``), evaluates each on the fixed reference grid,
and reports (a) every member's relative L2 error and (b) the ensemble mean's
error. If the ensemble beats the *median* member by more than the member IQR,
the averaging is doing real work; if not, the members are too correlated for a
plain mean to help.

Usage
-----
    python SCRIPTS/ensemble_eval.py --models m0.pt m1.pt m2.pt --hard-bc --eps 1e-3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# This script lives in SCRIPTS/ and imports its neighbours (hydrology,
# solve_1d_pinn) by bare name; make that work whether it is run directly or
# loaded by the test suite via importlib.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import torch

from hydrology import u_reference
from solve_1d_pinn import HardBCMLP, MLP, pick_device


def predict_grid(model, pe: float, tau_max: float, nx: int, nt: int,
                 device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_pred, u_ref) on the fixed (nt, nx) grid."""
    dtype = torch.get_default_dtype()
    xi = torch.linspace(0.0, 1.0, nx, device=device, dtype=dtype)
    tau = torch.linspace(0.0, tau_max, nt, device=device, dtype=dtype)
    Xi, Tau = torch.meshgrid(xi, tau, indexing="xy")
    with torch.no_grad():
        u_pred = model(Xi.reshape(-1, 1), Tau.reshape(-1, 1)).reshape(nt, nx)
        u_pred = u_pred.detach().cpu().numpy()
    u_ref = u_reference(Xi.detach().cpu().numpy(), Tau.detach().cpu().numpy(), pe)
    return u_pred, u_ref


def ensemble_rel_l2(predictions: list[np.ndarray],
                    u_ref: np.ndarray) -> tuple[list[float], float]:
    """Per-member relative L2 errors and the ensemble-mean error.

    ``predictions`` is a list of ``(nt, nx)`` arrays, one per member.
    """
    denom = np.linalg.norm(u_ref)
    per_member = [float(np.linalg.norm(p - u_ref) / denom) for p in predictions]
    stack = np.stack(predictions, axis=0)
    ensemble = float(np.linalg.norm(stack.mean(axis=0) - u_ref) / denom)
    return per_member, ensemble


def median_iqr(values: list[float]) -> dict[str, float]:
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return {}

    def q(p: float) -> float:
        pos = (n - 1) * p
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return vals[lo] * (1 - frac) + vals[hi] * frac

    return {"median": q(0.5), "q1": q(0.25), "q3": q(0.75),
            "iqr": q(0.75) - q(0.25), "min": vals[0], "max": vals[-1]}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate a PINN ensemble average vs its members.")
    ap.add_argument("--models", nargs="+", required=True,
                    help="paths to saved model state_dicts")
    ap.add_argument("--hard-bc", action="store_true",
                    help="models are HardBCMLP (default: soft MLP)")
    ap.add_argument("--eps", type=float, default=1e-3,
                    help="smoothing width for HardBCMLP")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--pe", type=float, default=20.0)
    ap.add_argument("--tau-max", type=float, default=0.5)
    ap.add_argument("--nx-test", type=int, default=256)
    ap.add_argument("--nt-test", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default=None,
                    help="write a summary JSON here (default: stdout only)")
    args = ap.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(0)

    def build():
        if args.hard_bc:
            return HardBCMLP(hidden=args.hidden, layers=args.layers, eps=args.eps)
        return MLP(hidden=args.hidden, layers=args.layers)

    predictions: list[np.ndarray] = []
    u_ref = None
    for i, path in enumerate(args.models):
        model = build().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        u_pred, u_ref = predict_grid(model, args.pe, args.tau_max,
                                     args.nx_test, args.nt_test, device)
        predictions.append(u_pred)
        print(f"[ensemble] member {i} ({path}): shape {u_pred.shape}")

    per_member, ensemble = ensemble_rel_l2(predictions, u_ref)
    s = median_iqr(per_member)

    print(f"[ensemble] members (n={len(predictions)}): "
          + " ".join(f"{e:.4e}" for e in per_member))
    print(f"[ensemble] member median {s['median']:.4e} | IQR {s['iqr']:.4e} | "
          f"min {s['min']:.4e} | max {s['max']:.4e}")
    print(f"[ensemble] ensemble-mean rel L2 = {ensemble:.4e}")

    verdict = ""
    if ensemble < s["median"] - s["iqr"]:
        verdict = "ensemble beats the median member by more than the IQR"
    elif ensemble < s["median"]:
        verdict = "ensemble beats the median member, but within the IQR"
    else:
        verdict = "ensemble does not beat the median member"
    print(f"[ensemble] verdict: {verdict}")

    if args.outdir:
        import os
        os.makedirs(args.outdir, exist_ok=True)
        payload = {
            "members": per_member,
            "member_stats": s,
            "ensemble_rel_l2": ensemble,
            "verdict": verdict,
        }
        with open(os.path.join(args.outdir, "ensemble.json"), "w",
                  encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"[ensemble] summary -> {os.path.join(args.outdir, 'ensemble.json')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
