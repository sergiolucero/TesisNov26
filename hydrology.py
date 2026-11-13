"""
hydrology.py
============

Phase 1 sandbox: synthetic 1D advection--dispersion "hydrologies".

Generates the analytical (Ogata--Banks 1961) solution of the 1D
advection--dispersion equation (ADE) for a homogeneous, isotropic, non-reactive
aquifer, and packs scenarios into supervised datasets for the PINN baseline
check and the FNO operator.

Governing equation (dimensional):
    dC/dt + v dC/dx = D d2C/dx2
Dimensionless (xi = x/L, tau = t v/L, u = C/C0, Pe = vL/D):
    du/dtau + du/dxi = (1/Pe) d2u/dxi2
Reference solution:
    u(xi,tau) = 1/2 [ erfc((xi-tau)*sqrt(Pe)/(2 sqrt(tau)))
                    + exp(Pe*xi) erfc((xi+tau)*sqrt(Pe)/(2 sqrt(tau))) ]

Run as a script to generate and save one scenario plus a dataset of scenarios.

    python hydrology.py --scenario out/scenario.npz --dataset out/dataset.npz

Author: TESIS (groundwater remediation PINN/FNO debugging)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
from scipy.special import erfc

__all__ = [
    "u_reference",
    "C_reference",
    "make_grid",
    "generate_scenario",
    "generate_dataset",
    "save_scenarios",
    "load_scenarios",
    "Scenario",
]


@dataclass
class Scenario:
    """One transport scenario (a single 'hydrology')."""

    C0: float          # inlet concentration [M/L^3]
    v: float           # pore velocity [L/T]
    D: float           # hydrodynamic dispersion [L^2/T]
    L: float           # domain length [L]
    T: float           # final time [T]
    nx: int            # spatial grid points
    nt: int            # time grid points
    x: np.ndarray      # spatial grid [L], shape (nx,)
    t: np.ndarray      # time grid [T], shape (nt,)
    C: np.ndarray      # concentration field, shape (nt, nx)

    # -- derived quantities ------------------------------------------------
    @property
    def Pe(self) -> float:
        return self.v * self.L / self.D

    @property
    def tau_max(self) -> float:
        return self.T * self.v / self.L

    @property
    def X(self) -> np.ndarray:
        return self.x[None, :] * np.ones_like(self.t)[:, None]  # (nt, nx)

    @property
    def Tgrid(self) -> np.ndarray:
        return self.t[:, None] * np.ones_like(self.x)[None, :]  # (nt, nx)

    def final_snapshot(self) -> np.ndarray:
        """Final-time concentration profile C(x, T), shape (nx,)."""
        return self.C[-1, :].copy()

    def fno_sample(self) -> Dict[str, np.ndarray]:
        """Supervised FNO sample: parameter input -> concentration field."""
        return {
            "params": np.array([self.v, self.D, self.C0], dtype=np.float64),
            "x": self.x.copy(),
            "t": self.t.copy(),
            "C": self.C.copy(),
        }


# --------------------------------------------------------------------------
# Analytical solution
# --------------------------------------------------------------------------
def u_reference(xi: np.ndarray, tau: np.ndarray, Pe: float) -> np.ndarray:
    """Dimensionless Ogata--Banks solution, broadcastable over (xi, tau).

    Handles the tau = 0 limit (u = 0 for xi > 0, u = 1 at xi = 0) without
    overflow by clamping the square root.
    """
    xi = np.asarray(xi, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    Xi, Tau = np.broadcast_arrays(xi, tau)

    eps = 1e-12
    s = 2.0 * np.sqrt(np.maximum(Tau, eps))
    a = (Xi - Tau) * np.sqrt(Pe) / s
    b = (Xi + Tau) * np.sqrt(Pe) / s
    return 0.5 * (erfc(a) + np.exp(Pe * Xi) * erfc(b))


def C_reference(x: np.ndarray, t: np.ndarray, C0: float, v: float,
                D: float) -> np.ndarray:
    """Dimensional Ogata--Banks solution, broadcastable over (x, t)."""
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    X, T = np.broadcast_arrays(x, t)

    eps = 1e-12
    s = 2.0 * np.sqrt(D * np.maximum(T, eps))
    a = (X - v * T) / s
    b = (X + v * T) / s
    return 0.5 * C0 * (erfc(a) + np.exp(v * X / D) * erfc(b))


# --------------------------------------------------------------------------
# Grid and scenario generation
# --------------------------------------------------------------------------
def make_grid(L: float, T: float, nx: int, nt: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x, t) grids; x in [0, L], t in [0, T]."""
    x = np.linspace(0.0, L, nx)
    t = np.linspace(0.0, T, nt)
    return x, t


def generate_scenario(C0: float, v: float, D: float, L: float, T: float,
                      nx: int = 256, nt: int = 256) -> Scenario:
    """Generate the concentration field for one homogeneous 1D ADE scenario."""
    x, t = make_grid(L, T, nx, nt)
    X, Tgrid = np.meshgrid(x, t, indexing="xy")  # (nt, nx)
    C = C_reference(X, Tgrid, C0, v, D)
    return Scenario(C0=C0, v=v, D=D, L=L, T=T, nx=nx, nt=nt, x=x, t=t, C=C)


def generate_dataset(n_scenarios: int, L: float, T: float, C0: float = 1.0,
                     v_range: Tuple[float, float] = (0.05, 0.5),
                     D_range: Tuple[float, float] = (0.1, 2.0),
                     nx: int = 128, nt: int = 128,
                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray, list]:
    """Generate a family of scenarios (a set of synthetic 'hydrologies').

    Returns
    -------
    params : (N, 3) array of (v, D, C0)
    fields : (N, nt, nx) concentration fields
    meta   : list of Scenario objects
    """
    rng = np.random.default_rng(seed)
    params = np.empty((n_scenarios, 3))
    fields = np.empty((n_scenarios, nt, nx))
    meta = []
    for i in range(n_scenarios):
        v = float(rng.uniform(*v_range))
        D = float(rng.uniform(*D_range))
        sc = generate_scenario(C0, v, D, L, T, nx, nt)
        params[i] = (v, D, C0)
        fields[i] = sc.C
        meta.append(sc)
    return params, fields, meta


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def save_scenarios(path: str, scenarios: list) -> None:
    """Save a list of Scenario objects to a compressed .npz archive."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {}
    for i, sc in enumerate(scenarios):
        payload[f"sc{i}_C0"] = np.array(sc.C0)
        payload[f"sc{i}_v"] = np.array(sc.v)
        payload[f"sc{i}_D"] = np.array(sc.D)
        payload[f"sc{i}_L"] = np.array(sc.L)
        payload[f"sc{i}_T"] = np.array(sc.T)
        payload[f"sc{i}_x"] = sc.x
        payload[f"sc{i}_t"] = sc.t
        payload[f"sc{i}_C"] = sc.C
    payload["n_scenarios"] = np.array(len(scenarios))
    np.savez_compressed(path, **payload)
    print(f"[hydrology] saved {len(scenarios)} scenario(s) -> {path}")


def load_scenarios(path: str) -> list:
    """Load scenarios written by :func:`save_scenarios`."""
    data = np.load(path, allow_pickle=False)
    n = int(data["n_scenarios"])
    out = []
    for i in range(n):
        out.append(Scenario(
            C0=float(data[f"sc{i}_C0"]),
            v=float(data[f"sc{i}_v"]),
            D=float(data[f"sc{i}_D"]),
            L=float(data[f"sc{i}_L"]),
            T=float(data[f"sc{i}_T"]),
            nx=len(data[f"sc{i}_x"]),
            nt=len(data[f"sc{i}_t"]),
            x=data[f"sc{i}_x"],
            t=data[f"sc{i}_t"],
            C=data[f"sc{i}_C"],
        ))
    return out


# --------------------------------------------------------------------------
# Demo / CLI
# --------------------------------------------------------------------------
def _demo_plots(sc: Scenario, outdir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    xi = sc.x / sc.L
    tau = sc.t * sc.v / sc.L

    # Heatmap of the full field
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.pcolormesh(xi, tau, sc.C, shading="nearest", cmap="viridis")
    fig.colorbar(im, ax=ax, label=r"$u = C/C_0$")
    ax.set_xlabel(r"$\xi = x/L$")
    ax.set_ylabel(r"$\tau = t\,v/L$")
    ax.set_title(f"Ogata--Banks field, Pe = {sc.Pe:.1f}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "field.png"), dpi=150)
    plt.close(fig)

    # Snapshots at several times
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in np.linspace(0, sc.nt - 1, 6).astype(int):
        ax.plot(xi, sc.C[n] / sc.C0, label=rf"$\tau={tau[n]:.2f}$")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$u$")
    ax.set_title("Concentration profiles")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "profiles.png"), dpi=150)
    plt.close(fig)
    print(f"[hydrology] figures written to {outdir}/")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate 1D ADE synthetic hydrologies")
    ap.add_argument("--scenario", default="out/scenario.npz",
                    help="path for the single-scenario archive")
    ap.add_argument("--dataset", default="out/dataset.npz",
                    help="path for the multi-scenario dataset archive")
    ap.add_argument("--figdir", default="figs", help="directory for demo figures")
    ap.add_argument("--L", type=float, default=100.0)
    ap.add_argument("--T", type=float, default=500.0)  # tau_max = 0.5 for v=0.1
    ap.add_argument("--v", type=float, default=0.1)
    ap.add_argument("--D", type=float, default=0.5)
    ap.add_argument("--C0", type=float, default=1.0)
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--nt", type=int, default=256)
    ap.add_argument("--n-scenarios", type=int, default=32)
    args = ap.parse_args()

    sc = generate_scenario(args.C0, args.v, args.D, args.L, args.T,
                           args.nx, args.nt)
    print(f"[hydrology] Pe = {sc.Pe:.2f}, tau_max = {sc.tau_max:.3f}, "
          f"max u = {sc.C.max()/sc.C0:.4f}")
    save_scenarios(args.scenario, [sc])
    _demo_plots(sc, args.figdir)

    params, fields, meta = generate_dataset(
        args.n_scenarios, args.L, args.T, C0=args.C0, nx=128, nt=128, seed=0)
    save_scenarios(args.dataset, meta)
    print(f"[hydrology] dataset params shape {params.shape}, "
          f"fields shape {fields.shape}")
    print(f"[hydrology] Pe range = "
          f"[{meta[0].Pe:.1f}, {meta[-1].Pe:.1f}]")


if __name__ == "__main__":
    main()
