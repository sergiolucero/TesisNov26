"""
inspect_residual.py
===================

Phase 2 residual inspection for the 1D ADE PINN.

Plots the absolute PDE residual

    R(xi, tau) = u_tau + u_xi - (1/Pe) u_xixi

on a uniform (xi, tau) grid, both at *random initialization* and *after
training* (loaded from a checkpoint).  This answers the plan's Phase 2
question: are there massive residual spikes at the boundaries / inlet
("injection well"), or is the residual spread uniformly over the interior?

Run:
    # fresh-init residual only (no checkpoint)
    python inspect_residual.py --hard-bc

    # init + trained residual for the best hard-BC run
    python inspect_residual.py --hard-bc --model out/pinn_hardbc_exp/model.pt

Author: TESIS (groundwater remediation PINN/FNO debugging)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from solve_1d_pinn import MLP, HardBCMLP, pde_residual


# --------------------------------------------------------------------------
# Residual evaluation
# --------------------------------------------------------------------------
def residual_grid(model, Pe, tau_max, nx=256, nt=256, chunk=4096):
    """Return (Xi, Tau, u, R) on a uniform grid. R is |residual| (detached).

    The double backward pass (u_xixi) is evaluated in *chunks* so the autograd
    graph never holds all nx*nt points at once.  On a box with ~1 GB RAM and no
    swap, evaluating the full 256x256 grid in one shot can trigger the OOM
    killer.
    """
    xi = torch.linspace(0.0, 1.0, nx)
    tau = torch.linspace(0.0, tau_max, nt)
    Xi, Tau = torch.meshgrid(xi, tau, indexing="xy")  # (nt, nx)

    xi_flat = Xi.reshape(-1, 1)
    tau_flat = Tau.reshape(-1, 1)
    npts = xi_flat.shape[0]
    U = np.empty(npts, dtype=np.float32)
    R = np.empty(npts, dtype=np.float32)

    for start in range(0, npts, chunk):
        end = min(start + chunk, npts)
        xi_f = xi_flat[start:end].requires_grad_(True)
        tau_f = tau_flat[start:end].requires_grad_(True)
        u = model(xi_f, tau_f)
        r = pde_residual(u, xi_f, tau_f, Pe)
        U[start:end] = u.detach().reshape(-1).numpy()
        R[start:end] = r.detach().reshape(-1).numpy()

    return Xi.numpy(), Tau.numpy(), U.reshape(nt, nx), R.reshape(nt, nx)


def residual_stats(Xi, Tau, R, name=""):
    """Print |R| summary: max location and interior vs boundary breakdown."""
    print(f"[resid] --- {name} ---")
    absR = np.abs(R)
    imax = np.unravel_index(np.argmax(absR), absR.shape)
    print(f"[resid]   max |R| = {absR[imax]:.3e} at "
          f"(xi,tau) = ({Xi[imax]:.4f}, {Tau[imax]:.4f})")
    print(f"[resid]   mean |R| (full)      = {absR.mean():.3e}")

    # interior: away from IC, inlet, outlet by a thin band
    interior = (Tau > 0.02) & (Xi > 0.02) & (Xi < 0.98)
    inlet_band = (Tau > 0.0) & (Xi < 0.02)
    ic_band = (Tau < 0.02) & (Xi > 0.02)
    corner = (Tau < 0.02) & (Xi < 0.02)
    print(f"[resid]   mean |R| (interior)  = {absR[interior].mean():.3e}")
    print(f"[resid]   mean |R| (inlet)     = {absR[inlet_band].mean():.3e}")
    print(f"[resid]   mean |R| (IC band)   = {absR[ic_band].mean():.3e}")
    print(f"[resid]   mean |R| (corner)    = {absR[corner].mean():.3e}")
    return absR


def plot_residual(Xi, Tau, R, title, path, tag):
    """Log-scale heatmap of |R| + profiles of |R| vs xi at selected tau."""
    absR = np.abs(R)
    # LogNorm with a floor: avoid log(0) and clip tiny values to vmin.
    vmin = max(float(absR[absR > 0].min()), 1e-10)
    vmax = float(absR.max())
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im = axes[0].pcolormesh(Xi, Tau, absR, shading="nearest",
                            cmap="inferno", norm=norm)
    axes[0].set_xlabel(r"$\xi$")
    axes[0].set_ylabel(r"$\tau$")
    axes[0].set_title(r"$|R(\xi,\tau)|$ (log scale)")
    fig.colorbar(im, ax=axes[0], label=r"$|R|$")

    nt, nx = Xi.shape
    for n in np.linspace(0, nt - 1, 6).astype(int):
        axes[1].semilogy(Xi[0], absR[n] + 1e-16,
                         label=rf"$\tau={Tau[n, 0]:.2f}$")
    axes[1].set_xlabel(r"$\xi$")
    axes[1].set_ylabel(r"$|R|$")
    axes[1].set_title(r"$|R|$ vs $\xi$ at fixed $\tau$")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")

    fig.suptitle(title)
    fig.tight_layout()
    out = os.path.join(path, f"residual_{tag}.png")
    os.makedirs(path, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[resid] figure -> {out}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Inspect 1D ADE PDE residual")
    ap.add_argument("--pe", type=float, default=20.0)
    ap.add_argument("--tau-max", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hard-bc", action="store_true",
                    help="use the hard-constrained ansatz")
    ap.add_argument("--eps", type=float, default=1e-3,
                    help="inlet smoothing width for hard BC")
    ap.add_argument("--model", default=None,
                    help="trained checkpoint (state_dict) to also inspect")
    ap.add_argument("--no-init", action="store_true",
                    help="skip the random-initialization plot")
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--nt", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="out/residual")
    args = ap.parse_args()

    # Flush prints immediately so output survives a SIGTERM/timeout kill.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    def build_model():
        if args.hard_bc:
            return HardBCMLP(hidden=args.hidden, layers=args.layers, eps=args.eps)
        return MLP(hidden=args.hidden, layers=args.layers)

    # 1) Random initialization
    if not args.no_init:
        model = build_model()
        Xi, Tau, u, R = residual_grid(model, args.pe, args.tau_max,
                                      nx=args.nx, nt=args.nt)
        print(f"[resid] INIT (random weights), u range "
              f"[{u.min():.3f}, {u.max():.3f}]")
        residual_stats(Xi, Tau, R, "initialization")
        plot_residual(Xi, Tau, R, "PDE residual at random initialization",
                      args.outdir, "init")

    # 2) Trained checkpoint
    if args.model:
        if not os.path.exists(args.model):
            raise SystemExit(f"[resid] checkpoint not found: {args.model}")
        model = build_model()
        state = torch.load(args.model, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        Xi, Tau, u, R = residual_grid(model, args.pe, args.tau_max,
                                      nx=args.nx, nt=args.nt)
        print(f"[resid] TRAINED ({args.model}), u range "
              f"[{u.min():.3f}, {u.max():.3f}]")
        residual_stats(Xi, Tau, R, "trained")
        plot_residual(Xi, Tau, R, "PDE residual after training",
                      args.outdir, "trained")


if __name__ == "__main__":
    main()
