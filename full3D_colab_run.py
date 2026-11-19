#!/usr/bin/env python3
"""full3D_colab_run.py — 3D full stochastic pump-and-treat optimization, sized for Colab.

Self-contained on purpose: it needs **numpy + scipy only** (both ship in Google
Colab), so it can be uploaded as a single file and run without the rest of this
repository.  It implements the whole stochastic pipeline of
``SCRIPTS/run_optimization.py`` in three dimensions:

    O-0  deterministic forward check + well layout (partially penetrating)
    O-1  random-field archive (separable 3D Karhunen-Loève, log-normal K)
    O-2  objective / cleanup time / SAA convergence
    opt  chance-constrained SAA optimization (pattern search, multi-start)
    out-of-sample certificate on the held-out test archive

"Full" means the complete stochastic run — random conductivity fields, the
sample-average objective with CVaR chance constraints, the outer optimization
and the test certificate — not a single deterministic forward solve.

Physics (finite volume, first order)
------------------------------------
    -div(K grad h) = W                       steady Darcy flow (TPFA)
    d(phi R C)/dt + div(v C) - div(D grad C) + lam R phi C = S
with the diagonal part of the Bear dispersion tensor,
    D_xx = a_L vx^2/|v| + a_TH vy^2/|v| + a_TV vz^2/|v|,
    D_yy = a_TH vx^2/|v| + a_L vy^2/|v| + a_TV vz^2/|v|,
    D_zz = a_TV vx^2/|v| + a_TV vy^2/|v| + a_L vz^2/|v|,
and horizontal conductivity ``K_h`` with vertical anisotropy ``kv_ratio``
(``K_v = kv_ratio * K_h``).  Wells are partially penetrating: a Gaussian
footprint in (x, y) times a uniform screened interval in z.  This is the 3D
extension of ``physics/fv2d.py``; cross terms of the dispersion tensor and
density coupling are deliberately left out (see ``DOCS/3D_plan.md`` §1).

Smart dimensions
----------------
The script never assumes a fixed grid.  ``--preset smoke|quick|full|paper``
picks a starting size, then a one-evaluation calibration measures *this*
runtime's cost and clamps the optimization budget so the run finishes inside
``--budget`` minutes.  The presets:

    smoke   nx=12 ny=6  nz=3   nsteps=20  M=6/12   ~seconds
    quick   nx=20 ny=10 nz=5   nsteps=40  M=12/24  ~a minute
    full    nx=24 ny=12 nz=6   nsteps=60  M=24/48  ~minutes   (default)
    paper   nx=32 ny=16 nz=8   nsteps=100 M=64/128 ~tens of minutes

``--nx/--ny/--nz`` override the grid; ``--budget`` (minutes) is the hard wall
clock the optimizer is allowed to spend.  The vertical dimension is kept
coarse (``nz << nx``) because vertical transport is diffusive at the chosen
``alpha_tv`` and a handful of layers already captures the screen contrast.

Usage
-----
    # Colab cell (CPU is enough — the workload is small sparse LU, not GEMM):
    #   !pip install -q numpy scipy
    #   !python -u full3D_colab_run.py --preset full --budget 20 --zip

    python full3D_colab_run.py --preset smoke            # smoke test
    python full3D_colab_run.py                           # full default
    python full3D_colab_run.py --preset paper --budget 60
    python full3D_colab_run.py --preset full --nx 28 --ny 14 --nz 7

Outputs (under ``out3d_colab/<run_id>/``): ``metrics.json`` (gates + policy),
``policy.json``, ``run_config.json``, figures (if matplotlib is present), and
an optional ``colab3d_results.zip`` with ``--zip``.

Exit status: 0 on success, 2 on a usage/preflight error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
EPS = 1e-30


def log(msg: str) -> None:
    print(f"[3d] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def cvar(x, alpha: float) -> float:
    """Empirical CVaR at level ``1-alpha`` (Rockafellar–Uryasev)."""
    x = np.asarray(x, dtype=float).ravel()
    M = x.size
    if M == 0:
        return 0.0
    xs = np.sort(x)[::-1]
    aM = alpha * M
    k = int(np.floor(aM))
    if k <= 0:
        return float(xs[0])
    if k >= M:
        return float(xs.mean())
    return float((xs[:k].sum() + (aM - k) * xs[k]) / aM)


# ---------------------------------------------------------------------------
# 3D geometry and aquifer
# ---------------------------------------------------------------------------
@dataclass
class Grid3D:
    """Rectangular cell-centred 3D grid.  Fields use shape ``(nz, ny, nx)``."""

    nx: int
    ny: int
    nz: int
    lx: float
    ly: float
    lz: float
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 2 or self.nz < 1:
            raise ValueError("grid needs at least 2x2x1 cells")
        if self.lx <= 0 or self.ly <= 0 or self.lz <= 0:
            raise ValueError("domain lengths must be positive")

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def dz(self) -> float:
        return self.lz / self.nz

    @property
    def n(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def cell_volume(self) -> float:
        return self.dx * self.dy * self.dz

    def index(self, x, y, z) -> int:
        i = int(np.floor((float(x) - self.x0) / self.dx))
        j = int(np.floor((float(y) - self.y0) / self.dy))
        k = int(np.floor((float(z) - self.z0) / self.dz))
        i = min(max(i, 0), self.nx - 1)
        j = min(max(j, 0), self.ny - 1)
        k = min(max(k, 0), self.nz - 1)
        return k * self.nx * self.ny + j * self.nx + i

    def indices(self, xs, ys, zs) -> np.ndarray:
        xs = np.atleast_1d(np.asarray(xs, dtype=float))
        ys = np.atleast_1d(np.asarray(ys, dtype=float))
        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        i = np.clip(np.floor((xs - self.x0) / self.dx).astype(int), 0, self.nx - 1)
        j = np.clip(np.floor((ys - self.y0) / self.dy).astype(int), 0, self.ny - 1)
        k = np.clip(np.floor((zs - self.z0) / self.dz).astype(int), 0, self.nz - 1)
        return k * self.nx * self.ny + j * self.nx + i

    def cell_centre(self, idx: int) -> tuple[float, float, float]:
        k, j, i = np.unravel_index(int(idx), (self.nz, self.ny, self.nx))
        return (
            self.x0 + (i + 0.5) * self.dx,
            self.y0 + (j + 0.5) * self.dy,
            self.z0 + (k + 0.5) * self.dz,
        )

    def centres(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.x0 + (np.arange(self.nx) + 0.5) * self.dx
        y = self.y0 + (np.arange(self.ny) + 0.5) * self.dy
        z = self.z0 + (np.arange(self.nz) + 0.5) * self.dz
        return np.meshgrid(x, y, z, indexing="ij")


@dataclass
class Aquifer3D:
    """3D aquifer.  ``K`` is the *horizontal* conductivity, shape ``(nz,ny,nx)``."""

    K: np.ndarray
    kv_ratio: float = 1.0
    phi: float = 0.30
    R: float = 1.0
    alpha_l: float = 5.0
    alpha_th: float = 0.5
    alpha_tv: float = 0.05
    decay: float = 0.0

    def __post_init__(self) -> None:
        if self.K.ndim != 3:
            raise ValueError("K must have shape (nz, ny, nx)")
        self.phi = float(self.phi)


# ---------------------------------------------------------------------------
# Separable 3D Karhunen–Loève log-conductivity
# ---------------------------------------------------------------------------
def _eig_1d(L: float, ell: float, n: int):
    """Discrete 1D Neumann eigenpairs of the exponential covariance.

    The integral operator ``(T f)_i = sum_j exp(-|x_i-x_j|/ell) f_j dx`` is
    diagonalised directly (midpoint quadrature), so retained variance is
    measured against the actual discretised field.
    """
    x = (np.arange(n) + 0.5) * L / n
    dx = L / n
    C = np.exp(-np.abs(x[:, None] - x[None, :]) / ell)
    nu, V = np.linalg.eigh(C * dx)
    order = np.argsort(nu)[::-1]
    nu = np.clip(nu[order], 0.0, None)
    phi = V[:, order].T / np.sqrt(dx)
    return nu, phi


class KL3D:
    """Log-normal K via a separable 3D KL expansion (Kronecker eigenmodes)."""

    def __init__(self, grid, sigma, ell_x, ell_y, ell_z, n_kl=0, k0=10.0,
                 kv_ratio=1.0, target_variance=0.95):
        self.grid = grid
        self.sigma = sigma
        self.k0 = k0
        self.kv_ratio = kv_ratio
        nu_x, phi_x = _eig_1d(grid.lx, ell_x, grid.nx)
        nu_y, phi_y = _eig_1d(grid.ly, ell_y, grid.ny)
        nu_z, phi_z = _eig_1d(grid.lz, ell_z, grid.nz)
        # 3D eigenvalues are products of the 1D eigenvalues (separable covariance)
        lam = (nu_x[:, None, None] * nu_y[None, :, None] * nu_z[None, None, :])
        total = float(lam.sum())
        flat = np.sort(lam.ravel())[::-1]
        cum = np.cumsum(flat) / total
        if n_kl is None or n_kl <= 0:
            n_kl = int(np.searchsorted(cum, target_variance) + 1)
        n_kl = int(min(n_kl, flat.size, 4096))
        order = np.argsort(lam.ravel())[::-1][:n_kl]
        ix, iy, iz = np.unravel_index(order, lam.shape)
        self.nu_x, self.phi_x = nu_x, phi_x
        self.nu_y, self.phi_y = nu_y, phi_y
        self.nu_z, self.phi_z = nu_z, phi_z
        self.ix, self.iy, self.iz = ix, iy, iz
        self.weights = lam[ix, iy, iz]
        self.retained = float(self.weights.sum() / total)
        self.n_kl = n_kl
        self.scale = self.sigma * np.sqrt(self.weights)
        self._px = phi_x[ix, :]  # (n_kl, nx)
        self._py = phi_y[iy, :]  # (n_kl, ny)
        self._pz = phi_z[iz, :]  # (n_kl, nz)
        self.k_min = self.k0 * np.exp(-3.5 * self.sigma)
        self.k_max = self.k0 * np.exp(3.5 * self.sigma)

    def log_field(self, coeffs: np.ndarray) -> np.ndarray:
        c = self.scale * np.asarray(coeffs, dtype=float).ravel()
        if c.size != self.n_kl:
            raise ValueError("wrong number of KL coefficients")
        # G[z, y, x] = sum_l c_l * pz[l,z] * py[l,y] * px[l,x]
        return np.einsum("l,lz,ly,lx->zyx", c, self._pz, self._py, self._px)

    def conductivity(self, coeffs: np.ndarray):
        G = self.log_field(coeffs)
        K = self.k0 * np.exp(G)
        clamped = np.clip(K, self.k_min, self.k_max)
        frac = float(np.mean(clamped != K))
        return clamped, frac


@dataclass
class Scenario3D:
    seed: int
    coeffs: np.ndarray
    K: np.ndarray
    clamp_fraction: float = 0.0
    source_scale: float = 1.0
    gradient_scale: float = 1.0


def sample_scenarios(grid, kl, seeds, random_source=False, random_gradient=False):
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        coeffs = rng.standard_normal(kl.n_kl)
        K, frac = kl.conductivity(coeffs)
        ss = float(np.exp(rng.normal(0.0, 0.2))) if random_source else 1.0
        gs = float(np.exp(rng.normal(0.0, 0.1))) if random_gradient else 1.0
        out.append(Scenario3D(seed, coeffs, K, frac, ss, gs))
    return out


# ---------------------------------------------------------------------------
# Wells (partially penetrating)
# ---------------------------------------------------------------------------
def gaussian_well_weights_3d(grid, x, y, z_top, z_bot, eps_cells=1.5,
                             truncate=4.0) -> np.ndarray:
    """Normalised well kernel: Gaussian in (x,y), uniform over the z-screen."""
    X, Y, Z = grid.centres()
    sx = max(eps_cells, 0.25) * grid.dx
    sy = max(eps_cells, 0.25) * grid.dy
    w = np.exp(-(((X - x) ** 2) / (2 * sx ** 2) + ((Y - y) ** 2) / (2 * sy ** 2)))
    w[w < np.exp(-0.5 * truncate ** 2)] = 0.0
    in_screen = (Z >= z_bot) & (Z <= z_top)
    w = w * in_screen
    s = w.sum()
    if s <= 0:
        w = np.zeros(grid.n)
        w[grid.index(x, y, 0.5 * (z_top + z_bot))] = 1.0
        return w
    return (w / s).ravel()


def _hex_points(centroid, spacing, n):
    """Hexagonal lattice points around ``centroid``, closest ``n`` returned."""
    lo, hi = -3.0, 3.0
    ii = np.arange(lo, hi + 1)
    jj = np.arange(lo, hi + 1)
    I, J = np.meshgrid(ii, jj)
    pts = np.column_stack([
        centroid[0] + spacing * (I.ravel() + 0.5 * J.ravel()),
        centroid[1] + spacing * (np.sqrt(3.0) / 2.0) * J.ravel(),
    ])
    d = np.linalg.norm(pts - np.asarray(centroid)[None, :], axis=1)
    return pts[np.argsort(d)[:n]]


def place_wells_3d(grid, plume_xy, n_wells, z_top, z_bot, seed=0) -> np.ndarray:
    """Hex well layout clipped to the domain; fixed screened interval in z."""
    plume_xy = np.atleast_2d(np.asarray(plume_xy, dtype=float))
    centroid = plume_xy.mean(axis=0)
    area = max(len(plume_xy), 1) * grid.dx * grid.dy
    spacing = np.sqrt(2.0 * area / (np.sqrt(3.0) * max(n_wells, 1)))
    pts = _hex_points(centroid, spacing, n_wells)
    lo_x, hi_x = grid.x0 + 0.5 * grid.dx, grid.x0 + grid.lx - 0.5 * grid.dx
    lo_y, hi_y = grid.y0 + 0.5 * grid.dy, grid.y0 + grid.ly - 0.5 * grid.dy
    pts[:, 0] = np.clip(pts[:, 0], lo_x, hi_x)
    pts[:, 1] = np.clip(pts[:, 1], lo_y, hi_y)
    return np.column_stack([pts, np.full(n_wells, z_top), np.full(n_wells, z_bot)])


# ---------------------------------------------------------------------------
# Steady 3D Darcy flow (TPFA)
# ---------------------------------------------------------------------------
class FlowModel3D:
    """Steady Darcy flow with linear well superposition (one LU reuse)."""

    def __init__(self, grid, aquifer, h_left=0.0, h_right=0.0) -> None:
        if aquifer.K.shape != (grid.nz, grid.ny, grid.nx):
            raise ValueError("K shape does not match the grid")
        self.grid = grid
        self.aquifer = aquifer
        self.h_left = float(h_left)
        self.h_right = float(h_right)
        self.Kh = np.asarray(aquifer.K, dtype=float)
        self.Kz = self.Kh * aquifer.kv_ratio
        self.phi = aquifer.phi
        self.A, self.rhs_const = self._build_matrix()
        self._lu = splu(self.A.tocsc())

    def _build_matrix(self):
        g = self.grid
        nz, ny, nx, n = g.nz, g.ny, g.nx, g.n
        idx = np.arange(n).reshape(nz, ny, nx)
        dx, dy, dz = g.dx, g.dy, g.dz
        diag = np.zeros(n)
        rows, cols, vals = [], [], []

        def hmean(a, b):
            return 2.0 * a * b / (a + b)

        # interior x faces
        Kf = hmean(self.Kh[:, :, :-1], self.Kh[:, :, 1:])
        cx = Kf * (dy * dz) / dx
        L, R = idx[:, :, :-1].ravel(), idx[:, :, 1:].ravel()
        diag[L] += cx.ravel()
        diag[R] += cx.ravel()
        rows += [R, L]
        cols += [L, R]
        vals += [-cx.ravel(), -cx.ravel()]

        # interior y faces
        Kf = hmean(self.Kh[:, :-1, :], self.Kh[:, 1:, :])
        cy = Kf * (dx * dz) / dy
        U, D = idx[:, :-1, :].ravel(), idx[:, 1:, :].ravel()
        diag[U] += cy.ravel()
        diag[D] += cy.ravel()
        rows += [D, U]
        cols += [U, D]
        vals += [-cy.ravel(), -cy.ravel()]

        # interior z faces (vertical anisotropy)
        Kf = hmean(self.Kz[:-1], self.Kz[1:])
        cz = Kf * (dx * dy) / dz
        B, T = idx[:-1].ravel(), idx[1:].ravel()
        diag[B] += cz.ravel()
        diag[T] += cz.ravel()
        rows += [T, B]
        cols += [B, T]
        vals += [-cz.ravel(), -cz.ravel()]

        # Dirichlet x boundaries (half-cell distance)
        rhs = np.zeros(n)
        cxL = 2.0 * self.Kh[:, :, 0] * (dy * dz) / dx
        cxR = 2.0 * self.Kh[:, :, -1] * (dy * dz) / dx
        diag[idx[:, :, 0].ravel()] += cxL.ravel()
        rhs[idx[:, :, 0].ravel()] += (cxL * self.h_left).ravel()
        diag[idx[:, :, -1].ravel()] += cxR.ravel()
        rhs[idx[:, :, -1].ravel()] += (cxR * self.h_right).ravel()

        rows.append(np.arange(n))
        cols.append(np.arange(n))
        vals.append(diag)
        A = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        ).tocsr()
        return A, rhs

    def solve(self, f=None) -> np.ndarray:
        rhs = self.rhs_const.copy()
        if f is not None:
            rhs = rhs + np.asarray(f, dtype=float).ravel()
        return self._lu.solve(rhs)

    def solve_perturbation(self, f: np.ndarray) -> np.ndarray:
        return self._lu.solve(np.asarray(f, dtype=float).ravel())

    def release_factorization(self) -> None:
        self._lu = None
        self.A = None

    def velocity(self, h: np.ndarray):
        """Face-centred seepage velocity ``v = -K grad(h) / phi``."""
        g = self.grid
        nz, ny, nx = g.nz, g.ny, g.nx
        h2 = np.asarray(h, dtype=float).reshape(nz, ny, nx)
        phi = self.phi

        def hmean(a, b):
            return 2.0 * a * b / (a + b)

        vx = np.zeros((nz, ny, nx + 1))
        Kf = hmean(self.Kh[:, :, :-1], self.Kh[:, :, 1:])
        vx[:, :, 1:nx] = -Kf * (h2[:, :, 1:] - h2[:, :, :-1]) / g.dx / phi
        vx[:, :, 0] = -self.Kh[:, :, 0] * (h2[:, :, 0] - self.h_left) / (0.5 * g.dx) / phi
        vx[:, :, nx] = -self.Kh[:, :, -1] * (self.h_right - h2[:, :, -1]) / (0.5 * g.dx) / phi

        vy = np.zeros((nz, ny + 1, nx))
        Kf = hmean(self.Kh[:, :-1, :], self.Kh[:, 1:, :])
        vy[:, 1:ny, :] = -Kf * (h2[:, 1:, :] - h2[:, :-1, :]) / g.dy / phi
        # top/bottom y faces: no-flow -> zero flux

        vz = np.zeros((nz + 1, ny, nx))
        Kf = hmean(self.Kz[:-1], self.Kz[1:])
        vz[1:nz, :, :] = -Kf * (h2[1:, :, :] - h2[:-1, :, :]) / g.dz / phi
        # top/bottom z faces: no-flow -> zero flux

        return vx, vy, vz


@dataclass
class FlowResponses3D:
    h0: np.ndarray
    hw: np.ndarray
    weights: np.ndarray
    well_specs: np.ndarray

    def head(self, q: Sequence[float]) -> np.ndarray:
        q = np.asarray(q, dtype=float).ravel()
        if q.size != self.hw.shape[0]:
            raise ValueError("q has the wrong length")
        return self.h0 + q @ self.hw


def flow_responses_3d(flow: FlowModel3D, well_specs, eps_cells=1.5) -> FlowResponses3D:
    well_specs = np.atleast_2d(np.asarray(well_specs, dtype=float))
    weights = np.vstack([
        gaussian_well_weights_3d(flow.grid, x, y, zt, zb, eps_cells)
        for x, y, zt, zb in well_specs
    ])
    h0 = flow.solve()
    hw = np.vstack([flow.solve_perturbation(-weights[i]) for i in range(weights.shape[0])])
    return FlowResponses3D(h0, hw, weights, well_specs)


# ---------------------------------------------------------------------------
# Transport operators
# ---------------------------------------------------------------------------
def _face_contribs(u, d, a, pu, pd, diag, rows, cols, vals):
    """Upwind contributions across a face; ``a`` is the signed volume flux.

    Positive ``a`` flows from cell ``u`` (upwind) into cell ``d`` (downwind).
    """
    a = np.asarray(a, dtype=float).ravel()
    u = np.asarray(u).ravel()
    d = np.asarray(d).ravel()
    pu = np.asarray(pu, dtype=float).ravel()
    pd = np.asarray(pd, dtype=float).ravel()
    pos = a >= 0.0
    if np.any(pos):
        diag[u[pos]] += pu[pos] * a[pos]
        rows.append(d[pos])
        cols.append(u[pos])
        vals.append(-pu[pos] * a[pos])
    neg = ~pos
    if np.any(neg):
        diag[d[neg]] += pd[neg] * (-a[neg])
        rows.append(u[neg])
        cols.append(d[neg])
        vals.append(-pd[neg] * (-a[neg]))


def advection_operator_3d(grid, vx, vy, vz, phi) -> sp.csr_matrix:
    """First-order upwind advection ``div(phi v C)`` in 3D."""
    g = grid
    nz, ny, nx, n = g.nz, g.ny, g.nx, g.n
    idx = np.arange(n).reshape(nz, ny, nx)
    p = np.full(n, float(phi))
    P = p[idx]
    diag = np.zeros(n)
    rows, cols, vals = [], [], []

    # interior x faces
    a = vx[:, :, 1:nx] * (g.dy * g.dz)
    _face_contribs(idx[:, :, :-1], idx[:, :, 1:], a,
                   P[:, :, :-1], P[:, :, 1:], diag, rows, cols, vals)
    # interior y faces
    a = vy[:, 1:ny, :] * (g.dx * g.dz)
    _face_contribs(idx[:, :-1, :], idx[:, 1:, :], a,
                   P[:, :-1, :], P[:, 1:, :], diag, rows, cols, vals)
    # interior z faces
    a = vz[1:nz, :, :] * (g.dx * g.dy)
    _face_contribs(idx[:-1], idx[1:], a,
                   P[:-1], P[1:], diag, rows, cols, vals)

    # x-boundary outflow (inflow has C = 0)
    v = vx[:, :, 0]
    m = v < 0.0
    if np.any(m):
        c = idx[:, :, 0][m]
        diag[c] += -P[:, :, 0][m] * v[m] * (g.dy * g.dz)
    v = vx[:, :, -1]
    m = v > 0.0
    if np.any(m):
        c = idx[:, :, -1][m]
        diag[c] += P[:, :, -1][m] * v[m] * (g.dy * g.dz)

    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    return sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()


def face_dispersion_3d(grid, aquifer, vx, vy, vz):
    """Face conductances of the (diagonal) Bear dispersion tensor."""
    g = grid
    nz, ny, nx = g.nz, g.ny, g.nx
    vxc = 0.5 * (vx[:, :, :-1] + vx[:, :, 1:])
    vyc = 0.5 * (vy[:, :-1, :] + vy[:, 1:, :])
    vzc = 0.5 * (vz[:-1, :, :] + vz[1:, :, :])
    speed = np.sqrt(vxc ** 2 + vyc ** 2 + vzc ** 2)
    speed = np.maximum(speed, 1e-12)
    al, ath, atv = aquifer.alpha_l, aquifer.alpha_th, aquifer.alpha_tv
    Dxx = al * vxc ** 2 / speed + ath * vyc ** 2 / speed + atv * vzc ** 2 / speed
    Dyy = ath * vxc ** 2 / speed + al * vyc ** 2 / speed + atv * vzc ** 2 / speed
    Dzz = atv * vxc ** 2 / speed + atv * vyc ** 2 / speed + al * vzc ** 2 / speed
    phi = aquifer.phi
    gx = phi * 0.5 * (Dxx[:, :, :-1] + Dxx[:, :, 1:]) * (g.dy * g.dz) / g.dx
    gy = phi * 0.5 * (Dyy[:, :-1, :] + Dyy[:, 1:, :]) * (g.dx * g.dz) / g.dy
    gz = phi * 0.5 * (Dzz[:-1, :, :] + Dzz[1:, :, :]) * (g.dx * g.dy) / g.dz
    return gx, gy, gz


def dispersion_operator_3d(grid, gx, gy, gz) -> sp.csr_matrix:
    g = grid
    nz, ny, nx, n = g.nz, g.ny, g.nx, g.n
    idx = np.arange(n).reshape(nz, ny, nx)
    diag = np.zeros(n)
    rows, cols, vals = [], [], []

    L, R = idx[:, :, :-1].ravel(), idx[:, :, 1:].ravel()
    gg = gx.ravel()
    diag[L] += gg
    diag[R] += gg
    rows += [R, L]
    cols += [L, R]
    vals += [-gg, -gg]

    U, D = idx[:, :-1, :].ravel(), idx[:, 1:, :].ravel()
    gg = gy.ravel()
    diag[U] += gg
    diag[D] += gg
    rows += [D, U]
    cols += [U, D]
    vals += [-gg, -gg]

    B, T = idx[:-1].ravel(), idx[1:].ravel()
    gg = gz.ravel()
    diag[B] += gg
    diag[T] += gg
    rows += [T, B]
    cols += [B, T]
    vals += [-gg, -gg]

    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    return sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()


@dataclass
class TransportResult3D:
    times: np.ndarray
    C_well: np.ndarray
    C_record: Optional[np.ndarray]
    record_cells: Optional[np.ndarray]
    C_final: np.ndarray
    mass: dict


def transport_run_3d(grid, aquifer, flow, resp, q, dt, nsteps,
                     source_idx=None, source_rate=0.0, source_steps=None,
                     c_init=None, record_cells=None,
                     disp_ref=None) -> TransportResult3D:
    q = np.asarray(q, dtype=float).ravel()
    nw = q.size
    if resp.hw.shape[0] != nw:
        raise ValueError("q length does not match the number of wells")
    g = grid
    nz, ny, nx, n = g.nz, g.ny, g.nx, g.n
    phi = aquifer.phi
    V = g.cell_volume
    st = aquifer.R * phi * V / dt
    decay = aquifer.decay * aquifer.R * phi * V

    if disp_ref is None:
        vx0, vy0, vz0 = flow.velocity(resp.h0)
    else:
        vx0, vy0, vz0 = disp_ref
    gx, gy, gz = face_dispersion_3d(grid, aquifer, vx0, vy0, vz0)
    Adisp = dispersion_operator_3d(grid, gx, gy, gz)

    h = resp.head(q)
    vx, vy, vz = flow.velocity(h)
    Aadv = advection_operator_3d(grid, vx, vy, vz, phi)
    sink = q @ resp.weights
    A = (Aadv + Adisp + sp.diags(st + decay + sink)).tocsc()
    lu = splu(A)

    C = np.zeros(n) if c_init is None else np.asarray(c_init, dtype=float).ravel().copy()
    if C.size != n:
        raise ValueError("c_init has the wrong size")

    if record_cells is not None:
        record_cells = np.atleast_1d(np.asarray(record_cells, dtype=int))
    C_rec = np.zeros((nsteps, 0 if record_cells is None else record_cells.size))
    C_well = np.zeros((nsteps, nw))

    src = np.zeros(n)
    if source_idx is not None and source_rate != 0.0:
        src[int(source_idx)] = float(source_rate)

    mass_total = np.zeros(nsteps + 1)
    mass_source = np.zeros(nsteps + 1)
    mass_well = np.zeros(nsteps + 1)
    mass_out = np.zeros(nsteps + 1)
    mass_decay = np.zeros(nsteps + 1)
    mass_total[0] = float(np.sum(phi * aquifer.R * C) * V)

    for k in range(nsteps):
        active = src if (source_steps is None or k < source_steps) else 0.0
        C = lu.solve(st * C + active)
        C_well[k] = resp.weights @ C
        if record_cells is not None:
            C_rec[k] = C[record_cells]

        Cc = C.reshape(nz, ny, nx)
        mass_total[k + 1] = float(np.sum(phi * aquifer.R * C) * V)
        mass_source[k + 1] = mass_source[k] + float(np.sum(active)) * dt
        mass_well[k + 1] = mass_well[k] + float(np.dot(C, sink)) * dt
        out = 0.0
        out += float(np.sum(np.maximum(-vx[:, :, 0], 0.0) * phi * Cc[:, :, 0])) * (g.dy * g.dz)
        out += float(np.sum(np.maximum(vx[:, :, -1], 0.0) * phi * Cc[:, :, -1])) * (g.dy * g.dz)
        mass_out[k + 1] = mass_out[k] + out * dt
        mass_decay[k + 1] = mass_decay[k] + float(np.sum(decay * C)) * dt

    mass = {
        "total": mass_total,
        "source": mass_source,
        "well": mass_well,
        "outflow": mass_out,
        "decay": mass_decay,
    }
    return TransportResult3D(
        times=np.arange(nsteps + 1) * dt,
        C_well=C_well,
        C_record=C_rec if record_cells is not None else None,
        record_cells=record_cells,
        C_final=C.copy(),
        mass=mass,
    )


# ---------------------------------------------------------------------------
# Objective (cleanup time + cost)
# ---------------------------------------------------------------------------
@dataclass
class CostConfig3D:
    c_e: float = 1.0
    c_w: float = 1.0
    c_m: float = 1.0
    c_tau: float = 1.0
    c_fail: float = 10.0
    creg: float = 1.0
    window_start: float = 0.8
    h_min: float = -10.0
    eps_time: float = 0.05
    alpha: float = 0.05
    alpha_h: float = 0.01


def cleanup_time(v: np.ndarray, dt: float, T: float) -> tuple[float, bool]:
    v = np.asarray(v, dtype=float)
    idx = np.nonzero(v > 0.0)[0]
    if idx.size == 0:
        return 0.0, False
    last = int(idx[-1])
    return float((last + 1) * dt), bool(last == v.size - 1)


def soft_last_violation(v: np.ndarray, dt: float, eps: float):
    v = np.asarray(v, dtype=float).ravel()
    nt = v.size
    s = sigmoid(v / eps)
    ds = s * (1.0 - s) / eps
    t = (np.arange(nt) + 1) * dt
    R = np.ones(nt + 1)
    for n in range(nt - 1, -1, -1):
        R[n] = R[n + 1] * (1.0 - s[n])
    tau = float(np.sum(t * s * R[1:]))
    return tau


def scenario_objective_3d(grid, aquifer, res, q, h_obs, dt, T, cfg) -> dict:
    q = np.asarray(q, dtype=float).ravel()
    C_comp = res.C_record
    if C_comp is None:
        raise ValueError("transport result has no compliance record")
    nt = C_comp.shape[0]
    v_t = np.max(C_comp - cfg.creg, axis=1)
    i0 = int(np.floor(cfg.window_start * nt))
    G = float(np.max(v_t[i0:])) if i0 < nt else 0.0
    H = float(np.min(h_obs - cfg.h_min))
    tau_c, failed = cleanup_time(v_t, dt, T)

    J_pump = cfg.c_e * float(np.sum(q)) * T
    J_treat = cfg.c_w * float(np.sum(res.C_well * q[None, :]) * dt)
    J_res = cfg.c_m * float(np.sum(aquifer.R * aquifer.phi * res.C_final) * grid.cell_volume)

    eps = max(cfg.eps_time * cfg.creg, 1e-12)
    tau_soft = soft_last_violation(v_t, dt, eps)
    zmax = float(np.max(v_t) / eps)
    J_time = cfg.c_tau * tau_soft + cfg.c_fail * float(sigmoid(zmax))

    J = J_pump + J_treat + J_res + J_time
    m = res.mass
    bal = (m["total"][-1] - m["total"][0]
           - (m["source"][-1] - m["well"][-1] - m["outflow"][-1] - m["decay"][-1]))
    denom = max(abs(m["source"][-1]), abs(m["total"][0]), 1e-30)
    mass_err = float(bal / denom)
    return {
        "J": J, "J_pump": J_pump, "J_treat": J_treat, "J_res": J_res, "J_time": J_time,
        "G": G, "H": H, "tau_c": tau_c, "failed": bool(failed),
        "v_t": v_t, "C_comp": C_comp, "C_well": res.C_well, "C_final": res.C_final,
        "mass_balance_error": mass_err,
    }


# ---------------------------------------------------------------------------
# Scenario archive (SAA)
# ---------------------------------------------------------------------------
@dataclass
class ForwardConfig3D:
    dt: float = 36.5
    nsteps: int = 60
    grad: float = 1e-3
    source_xyz: tuple = (30.0, 50.0, 10.0)
    source_q: float = 10.0
    source_c: float = 100.0
    source_days: float = 365.0
    compliance_xyz: Sequence = ((185.0, 50.0, 10.0),)
    obs_xyz: Sequence = ((100.0, 50.0, 10.0),)
    eps_well: float = 1.5

    @property
    def T(self) -> float:
        return self.dt * self.nsteps

    @property
    def source_steps(self) -> int:
        return max(int(round(self.source_days / self.dt)), 1)


@dataclass
class ScenarioProblem3D:
    scenario: Scenario3D
    flow: FlowModel3D
    resp: FlowResponses3D
    source_idx: int
    source_rate: float


class Archive3D:
    def __init__(self, grid, aquifer, well_specs, scenarios, fwd: ForwardConfig3D):
        self.grid = grid
        self.aquifer = aquifer
        self.well_specs = np.atleast_2d(np.asarray(well_specs, dtype=float))
        self.fwd = fwd
        self.scenarios = list(scenarios)
        self.compliance_cells = grid.indices(
            [p[0] for p in fwd.compliance_xyz],
            [p[1] for p in fwd.compliance_xyz],
            [p[2] for p in fwd.compliance_xyz],
        )
        self.obs_cells = grid.indices(
            [p[0] for p in fwd.obs_xyz],
            [p[1] for p in fwd.obs_xyz],
            [p[2] for p in fwd.obs_xyz],
        )
        self.source_idx = grid.index(*fwd.source_xyz)
        self.problems: list[ScenarioProblem3D] = []
        self._build()

    def _build(self):
        for s in self.scenarios:
            aq = Aquifer3D(
                K=s.K, kv_ratio=self.aquifer.kv_ratio, phi=self.aquifer.phi,
                R=self.aquifer.R, alpha_l=self.aquifer.alpha_l,
                alpha_th=self.aquifer.alpha_th, alpha_tv=self.aquifer.alpha_tv,
                decay=self.aquifer.decay,
            )
            h_left = self.fwd.grad * self.grid.lx * s.gradient_scale
            flow = FlowModel3D(self.grid, aq, h_left=h_left, h_right=0.0)
            resp = flow_responses_3d(flow, self.well_specs, self.fwd.eps_well)
            flow.release_factorization()
            rate = self.fwd.source_q * self.fwd.source_c * s.source_scale
            self.problems.append(ScenarioProblem3D(s, flow, resp, self.source_idx, rate))

    @property
    def n_wells(self) -> int:
        return int(self.well_specs.shape[0])

    def evaluate(self, q, cost: CostConfig3D, keep: bool = False) -> "SAAEvaluation3D":
        q = np.asarray(q, dtype=float).ravel()
        if q.size != self.n_wells:
            raise ValueError(f"q has length {q.size}, expected {self.n_wells}")
        fwd = self.fwd
        J = np.zeros(len(self.problems))
        G = np.zeros_like(J)
        H = np.zeros_like(J)
        tau = np.zeros_like(J)
        failed = np.zeros_like(J, dtype=bool)
        details = []
        for m, prob in enumerate(self.problems):
            h = prob.resp.head(q)
            res = transport_run_3d(
                self.grid, prob.flow.aquifer, prob.flow, prob.resp, q,
                fwd.dt, fwd.nsteps,
                source_idx=prob.source_idx, source_rate=prob.source_rate,
                source_steps=fwd.source_steps,
                record_cells=self.compliance_cells,
            )
            d = scenario_objective_3d(
                self.grid, prob.flow.aquifer, res, q, h[self.obs_cells],
                fwd.dt, fwd.T, cost,
            )
            J[m] = d["J"]
            G[m] = d["G"]
            H[m] = d["H"]
            tau[m] = d["tau_c"]
            failed[m] = d["failed"]
            details.append(d)
        return SAAEvaluation3D(q, J, G, H, tau, failed, details)


@dataclass
class SAAEvaluation3D:
    q: np.ndarray
    J: np.ndarray
    G: np.ndarray
    H: np.ndarray
    tau: np.ndarray
    failed: np.ndarray
    details: list

    @property
    def M(self) -> int:
        return self.J.size

    @property
    def mean_J(self) -> float:
        return float(np.mean(self.J))

    @property
    def se_J(self) -> float:
        return float(np.std(self.J, ddof=1) / np.sqrt(self.M)) if self.M > 1 else 0.0

    @property
    def reliability(self) -> float:
        return float(np.mean(self.G <= 0.0))

    @property
    def mean_tau(self) -> float:
        return float(np.mean(self.tau))

    def summary(self) -> dict:
        return {
            "M": int(self.M),
            "mean_J": self.mean_J,
            "se_J": self.se_J,
            "rel_se_J": self.se_J / max(abs(self.mean_J), 1e-12),
            "mean_G": float(np.mean(self.G)),
            "cvar_G": cvar(self.G, 0.05),
            "reliability": self.reliability,
            "violation_prob": 1.0 - self.reliability,
            "mean_H": float(np.mean(self.H)),
            "cvar_H": cvar(-self.H, 0.01),
            "mean_tau": self.mean_tau,
            "median_tau": float(np.median(self.tau)),
            "p90_tau": float(np.quantile(self.tau, 0.9)),
            "failure_fraction": float(np.mean(self.failed)),
            "J_pump": float(np.mean([d["J_pump"] for d in self.details])),
            "J_treat": float(np.mean([d["J_treat"] for d in self.details])),
            "J_res": float(np.mean([d["J_res"] for d in self.details])),
            "J_time": float(np.mean([d["J_time"] for d in self.details])),
            "max_mass_balance_error": float(
                max(abs(d["mass_balance_error"]) for d in self.details)
            ),
        }


# ---------------------------------------------------------------------------
# Optimization (pattern search + multi-start)
# ---------------------------------------------------------------------------
def project_capacity(q, q_max=50.0, Q_max=200.0) -> np.ndarray:
    q = np.clip(np.asarray(q, dtype=float), 0.0, q_max)
    s = q.sum()
    if s > Q_max:
        q = q * (Q_max / s)
    return q


class SAAProblem3D:
    def __init__(self, archive: Archive3D, cost: CostConfig3D, q_max=50.0,
                 Q_max=200.0, alpha=0.05, alpha_h=0.01, rho_g=1e6, rho_h=1e6):
        self.archive = archive
        self.cost = cost
        self.q_max = q_max
        self.Q_max = Q_max
        self.alpha = alpha
        self.alpha_h = alpha_h
        self.rho_g = rho_g
        self.rho_h = rho_h
        self.n_wells = archive.n_wells
        self.n_eval = 0

    def evaluate(self, q, keep: bool = False) -> SAAEvaluation3D:
        q = project_capacity(q, self.q_max, self.Q_max)
        ev = self.archive.evaluate(q, self.cost, keep=keep)
        self.n_eval += 1
        return ev

    def objective(self, q) -> float:
        ev = self.evaluate(q)
        cg = cvar(ev.G, self.alpha)
        ch = cvar(-ev.H, self.alpha_h)
        return float(ev.mean_J + self.rho_g * max(cg, 0.0) ** 2
                     + self.rho_h * max(ch, 0.0) ** 2)


def pattern_search(problem: SAAProblem3D, q0, step0=None, tol=1e-3,
                   max_iter=80, max_evals=2000):
    t0 = time.time()
    q = project_capacity(q0, problem.q_max, problem.Q_max)
    best = problem.objective(q)
    step = step0 if step0 is not None else 0.25 * problem.q_max
    hist = [{"eval": 1, "f": best, "q": q.tolist()}]
    nit = 0
    evals = 1
    while step > tol and nit < max_iter and evals < max_evals:
        improved = False
        for i in range(problem.n_wells):
            for sgn in (1.0, -1.0):
                cand = q.copy()
                cand[i] += sgn * step
                cand = project_capacity(cand, problem.q_max, problem.Q_max)
                if np.allclose(cand, q):
                    continue
                fc = problem.objective(cand)
                evals += 1
                if fc < best - 1e-12:
                    q, best = cand, fc
                    improved = True
                    hist.append({"eval": evals, "f": best, "q": q.tolist()})
                    break
            if evals >= max_evals:
                break
        nit += 1
        if not improved:
            step *= 0.5
    return q, best, nit, hist


def multi_start(problem: SAAProblem3D, n_starts=2, seed=0, max_iter=60,
                max_evals=2000):
    q_max, n = problem.q_max, problem.n_wells
    starts = [np.zeros(n), np.full(n, 0.5 * q_max)]
    if n_starts > 2:
        rng = np.random.default_rng(seed)
        starts += [rng.uniform(0.0, 0.7 * q_max, n) for _ in range(n_starts - 2)]
    best_q, best_f, best_nit, best_hist = None, np.inf, 0, []
    for s, q0 in enumerate(starts[:n_starts]):
        log(f"  multi-start {s + 1}/{min(n_starts, len(starts))} "
            f"(q0 sum={q0.sum():.1f} m3/d)")
        q, f, nit, hist = pattern_search(problem, q0, max_iter=max_iter,
                                         max_evals=max_evals)
        log(f"    -> f={f:.4g} nit={nit} evals={problem.n_eval}")
        if f < best_f - 1e-12:
            best_q, best_f, best_nit, best_hist = q, f, nit, hist
    return best_q, best_f, best_nit, best_hist


# ---------------------------------------------------------------------------
# Smart sizing + resource detection
# ---------------------------------------------------------------------------
PRESETS = {
    "smoke": dict(nx=12, ny=6, nz=3, nsteps=20, m_train=6, m_test=12,
                  max_evals=20, n_starts=1, n_wells=4),
    "quick": dict(nx=20, ny=10, nz=5, nsteps=40, m_train=12, m_test=24,
                  max_evals=60, n_starts=1, n_wells=6),
    "full": dict(nx=24, ny=12, nz=6, nsteps=80, m_train=24, m_test=48,
                 max_evals=120, n_starts=2, n_wells=6),
    "paper": dict(nx=32, ny=16, nz=8, nsteps=100, m_train=64, m_test=128,
                  max_evals=400, n_starts=3, n_wells=9),
}


def detect_resources() -> dict:
    cpu = os.cpu_count() or 2
    mem_gb = None
    try:
        import psutil  # type: ignore
        mem_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    in_colab = "COLAB_GPU" in os.environ or "google.colab" in sys.modules
    return {"cpu": cpu, "mem_gb": mem_gb, "colab": in_colab}


def autotune_budget(archive: Archive3D, cost: CostConfig3D, n_wells: int,
                    max_evals: int, n_starts: int, budget_s: float):
    """Measure one real evaluation and clamp ``max_evals`` to the budget."""
    t0 = time.time()
    archive.evaluate(np.zeros(n_wells), cost)
    per_eval = time.time() - t0
    est = max_evals * n_starts * per_eval
    new_evals = max_evals
    if est > budget_s:
        new_evals = max(4, int(max_evals * budget_s / est))
    log(f"calibration: {per_eval:.3f}s per full-archive evaluation "
        f"(est. {est:.0f}s for {max_evals}x{n_starts} evals)")
    if new_evals < max_evals:
        log(f"budget {budget_s:.0f}s clamps max_evals {max_evals} -> {new_evals}")
    return new_evals, per_eval


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def no_pump_field(grid, aquifer_base, scenario, fwd, dummy_well):
    aq = Aquifer3D(
        K=scenario.K, kv_ratio=aquifer_base.kv_ratio, phi=aquifer_base.phi,
        R=aquifer_base.R, alpha_l=aquifer_base.alpha_l,
        alpha_th=aquifer_base.alpha_th, alpha_tv=aquifer_base.alpha_tv,
        decay=aquifer_base.decay,
    )
    flow = FlowModel3D(grid, aq, h_left=fwd.grad * grid.lx * scenario.gradient_scale,
                       h_right=0.0)
    resp = flow_responses_3d(flow, dummy_well, fwd.eps_well)
    flow.release_factorization()
    res = transport_run_3d(
        grid, aq, flow, resp, np.zeros(1), fwd.dt, fwd.nsteps,
        source_idx=grid.index(*fwd.source_xyz),
        source_rate=fwd.source_q * fwd.source_c * scenario.source_scale,
        source_steps=fwd.source_steps,
    )
    return res.C_final


def run(cfg: dict, outdir: Path, budget_s: float, plots: bool, zip_results: bool) -> dict:
    t0 = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)
    log(f"start out={outdir}")

    # ---- grid / aquifer / KL -------------------------------------------
    grid = Grid3D(cfg["nx"], cfg["ny"], cfg["nz"], cfg["lx"], cfg["ly"], cfg["lz"])
    log(f"grid {grid.nx}x{grid.ny}x{grid.nz} = {grid.n} cells "
        f"({grid.dx:.1f} x {grid.dy:.1f} x {grid.dz:.1f} m)")
    aquifer = Aquifer3D(
        K=np.full((grid.nz, grid.ny, grid.nx), cfg["k0"]),
        kv_ratio=cfg["kv_ratio"], phi=cfg["phi"], R=cfg["R"],
        alpha_l=cfg["alpha_l"], alpha_th=cfg["alpha_th"], alpha_tv=cfg["alpha_tv"],
        decay=cfg["decay"],
    )
    kl = KL3D(grid, cfg["sigma"], cfg["ell_x"], cfg["ell_y"], cfg["ell_z"],
              n_kl=cfg["n_kl"], k0=cfg["k0"], kv_ratio=cfg["kv_ratio"],
              target_variance=cfg["target_variance"])
    log(f"KL: n_kl={kl.n_kl} retained_variance={kl.retained:.3f} "
        f"(sigma={kl.sigma}, ell=({cfg['ell_x']},{cfg['ell_y']},{cfg['ell_z']}))")

    fwd = ForwardConfig3D(
        dt=cfg["dt"], nsteps=cfg["nsteps"], grad=cfg["grad"],
        source_xyz=tuple(cfg["source_xyz"]),
        source_q=cfg["source_q"], source_c=cfg["source_c"],
        source_days=cfg["source_days"],
        compliance_xyz=[tuple(p) for p in cfg["compliance_xyz"]],
        obs_xyz=[tuple(p) for p in cfg["obs_xyz"]],
        eps_well=cfg["eps_well"],
    )
    cost = CostConfig3D(
        c_e=cfg["c_e"], c_w=cfg["c_w"], c_m=cfg["c_m"], c_tau=cfg["c_tau"],
        c_fail=cfg["c_fail"], creg=cfg["creg"], window_start=cfg["window_start"],
        h_min=cfg["h_min"], eps_time=cfg["eps_time"],
        alpha=cfg["alpha"], alpha_h=cfg["alpha_h"],
    )

    # ---- well layout from the mean no-pumping plume --------------------
    z_mid = grid.lz / 2.0
    half = 0.5 * cfg["z_screen_frac"] * grid.lz
    z_top, z_bot = z_mid + half, z_mid - half
    dummy_well = np.array([[grid.lx / 2, grid.ly / 2, z_top, z_bot]])
    log("design plume (mean no-pumping field over 5 scenarios)")
    fields = []
    for k in range(5):
        s = sample_scenarios(grid, kl, [9000 + k])[0]
        fields.append(no_pump_field(grid, aquifer, s, fwd, dummy_well))
    Cmean = np.mean(fields, axis=0).reshape(grid.nz, grid.ny, grid.nx)
    # depth-averaged 2D design plume
    C2d = Cmean.mean(axis=0)
    threshold = 0.05 * float(C2d.max())
    mask = C2d >= threshold
    if mask.sum() < 4:
        mask = np.zeros_like(mask, dtype=bool)
        j, i = np.unravel_index(np.argmax(C2d), C2d.shape)
        mask[max(j - 1, 0):j + 2, max(i - 1, 0):i + 2] = True
    plume_xy = np.column_stack([
        grid.x0 + (np.nonzero(mask)[1] + 0.5) * grid.dx,
        grid.y0 + (np.nonzero(mask)[0] + 0.5) * grid.dy,
    ])
    well_specs = place_wells_3d(grid, plume_xy, cfg["n_wells"], z_top, z_bot,
                                seed=cfg["seed"])
    log(f"wells: {well_specs.shape[0]} partially penetrating "
        f"(screen z in [{z_bot:.1f}, {z_top:.1f}] m)")

    # ---- scenario archives --------------------------------------------
    seeds_tr = [cfg["seed_train"] + i for i in range(cfg["m_train"])]
    seeds_te = [cfg["seed_test"] + i for i in range(cfg["m_test"])]
    train_scen = sample_scenarios(grid, kl, seeds_tr)
    test_scen = sample_scenarios(grid, kl, seeds_te)
    t_arch = time.time()
    train = Archive3D(grid, aquifer, well_specs, train_scen, fwd)
    test = Archive3D(grid, aquifer, well_specs, test_scen, fwd)
    log(f"archives ready: {len(train_scen)} train / {len(test_scen)} test "
        f"scenarios in {time.time() - t_arch:.1f}s")

    # reproducibility check (same seed -> same field)
    sa = sample_scenarios(grid, kl, [4242])[0]
    sb = sample_scenarios(grid, kl, [4242])[0]
    reproducible = bool(np.array_equal(sa.K, sb.K) and np.array_equal(sa.coeffs, sb.coeffs))

    # ---- SAA convergence at a fixed uniform q --------------------------
    q_uniform = np.full(train.n_wells, 0.1 * cfg["q_max"])
    ev_tr_uni = train.evaluate(q_uniform, cost)
    ev_te_uni = test.evaluate(q_uniform, cost)
    Js = ev_te_uni.J
    sizes = [m for m in (4, 8, 16, 32) if m <= Js.size]
    if not sizes or sizes[-1] != Js.size:
        sizes.append(Js.size)
    conv = [{"M": int(m), "mean_J": float(Js[:m].mean()),
             "se_J": float(Js[:m].std(ddof=1) / np.sqrt(m)) if m > 1 else 0.0}
            for m in sizes]

    # ---- smart budget calibration -------------------------------------
    max_evals, per_eval = autotune_budget(
        train, cost, train.n_wells, cfg["max_evals"], cfg["n_starts"], budget_s)

    # ---- optimization --------------------------------------------------
    log(f"optimize: pattern search, {cfg['n_starts']} starts, "
        f"max_evals={max_evals} (per start budget)")
    problem = SAAProblem3D(
        train, cost, q_max=cfg["q_max"], Q_max=cfg["Q_max"],
        alpha=cfg["alpha"], alpha_h=cfg["alpha_h"],
        rho_g=cfg["rho_g"], rho_h=cfg["rho_h"],
    )
    t_opt = time.time()
    q_star, f_star, nit, hist = multi_start(
        problem, n_starts=cfg["n_starts"], seed=cfg["seed"],
        max_iter=cfg["max_iter"], max_evals=max_evals,
    )
    log(f"optimization done in {time.time() - t_opt:.1f}s "
        f"(f*={f_star:.4g}, {problem.n_eval} evaluations)")

    ev_tr = train.evaluate(q_star, cost)
    ev_te = test.evaluate(q_star, cost)
    q_none = np.zeros(train.n_wells)
    ev_none = test.evaluate(q_none, cost)

    # ---- gates ---------------------------------------------------------
    alpha = cfg["alpha"]

    def _penalized(ev):
        cg = cvar(ev.G, alpha)
        ch = cvar(-ev.H, cfg["alpha_h"])
        return ev.mean_J + cfg["rho_g"] * max(cg, 0.0) ** 2 \
               + cfg["rho_h"] * max(ch, 0.0) ** 2

    gates = {
        "KL_retained_variance>=0.95": kl.retained >= 0.95,
        "reproducible": reproducible,
        "mass_balance<=1%": ev_te.summary()["max_mass_balance_error"] <= 0.01,
        "test_reliability>=1-alpha-1%": ev_te.reliability >= 1.0 - alpha - 0.01,
        "optimization_improved_vs_no_pump": _penalized(ev_te) < _penalized(ev_none),
        "no_pump_reliability": float(ev_none.reliability),
    }
    metrics = {
        "config": cfg,
        "grid": {"nx": grid.nx, "ny": grid.ny, "nz": grid.nz, "n": grid.n},
        "resources": detect_resources(),
        "kl": {"n_kl": kl.n_kl, "retained_variance": kl.retained},
        "reproducible": reproducible,
        "saa_convergence": conv,
        "uniform_q": ev_te_uni.summary(),
        "opt": {
            "f_star": float(f_star),
            "q_star": q_star.tolist(),
            "total_pumping": float(q_star.sum()),
            "active_wells": int(np.sum(q_star > 1e-6)),
            "n_evaluations": int(problem.n_eval),
            "nit": nit,
            "per_eval_s": per_eval,
            "train": ev_tr.summary(),
            "test": ev_te.summary(),
            "no_pump_test": ev_none.summary(),
            "test_gap_vs_train": float(
                (ev_te.mean_J - ev_tr.mean_J) / max(abs(ev_tr.mean_J), 1e-12)
            ),
        },
        "wells": well_specs.tolist(),
        "gates": gates,
        "wall_clock_s": time.time() - t0,
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (outdir / "policy.json").write_text(json.dumps({
        "q": q_star.tolist(), "wells": well_specs.tolist(),
        "alpha": alpha, "chance": "cvar",
    }, indent=2))
    (outdir / "run_config.json").write_text(json.dumps(cfg, indent=2))

    # ---- figures (optional, guarded) -----------------------------------
    if plots:
        _make_plots(outdir, grid, Cmean, C2d, mask, well_specs, q_star,
                    hist, ev_te, fwd, cost, z_top, z_bot)

    # ---- summary -------------------------------------------------------
    o = metrics["opt"]
    print("\n=== 3D stochastic run: gates ===")
    for k, v in gates.items():
        print(f"  {k}: {v}")
    print(f"\nq* = {np.round(q_star, 2).tolist()} m3/d "
          f"(total {o['total_pumping']:.1f} m3/d, {o['active_wells']} active)")
    print(f"train J = {o['train']['mean_J']:.2f}   test J = {o['test']['mean_J']:.2f}")
    print(f"test reliability = {o['test']['reliability']:.3f}   "
          f"mean cleanup = {o['test']['mean_tau']:.0f} d")
    print(f"results -> {outdir}  ({metrics['wall_clock_s']:.1f}s)")

    if zip_results:
        _zip_results(outdir)
    return metrics


def _make_plots(outdir, grid, Cmean, C2d, mask, well_specs, q_star, hist,
                ev_te, fwd, cost, z_top, z_bot):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(f"matplotlib unavailable, skipping figures ({exc})")
        return

    X, Y = np.meshgrid(
        grid.x0 + (np.arange(grid.nx) + 0.5) * grid.dx,
        grid.y0 + (np.arange(grid.ny) + 0.5) * grid.dy,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    pc = ax.pcolormesh(X, Y, C2d, shading="nearest", cmap="viridis")
    ax.scatter(well_specs[:, 0], well_specs[:, 1], marker="v", c="r", s=40,
               label="wells")
    ax.set_title("Depth-averaged design plume and well layout")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(pc, ax=ax, label="mean C (mg/L)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "well_layout.png", dpi=110)
    plt.close(fig)

    if hist:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([h["eval"] for h in hist], [h["f"] for h in hist], "o-")
        ax.set_xlabel("evaluations")
        ax.set_ylabel("best penalised objective")
        ax.set_title("pattern-search convergence (best start)")
        fig.tight_layout()
        fig.savefig(outdir / "figures" / "convergence.png", dpi=110)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ev_te.J, bins=min(20, max(5, ev_te.M // 5)), alpha=0.7)
    ax.axvline(ev_te.mean_J, color="r", ls="--", label="mean")
    ax.set_xlabel("scenario objective J")
    ax.set_title("test distribution under q*")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "test_distribution.png", dpi=110)
    plt.close(fig)


def _zip_results(outdir: Path):
    zpath = outdir / "colab3d_results.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in outdir.rglob("*"):
            if f.is_file() and f.name != "colab3d_results.zip":
                zf.write(f, f.relative_to(outdir))
    log(f"zipped -> {zpath}")


# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------
def build_config(preset: str, overrides: dict) -> dict:
    p = dict(PRESETS[preset])
    z_mid = 20.0 / 2.0
    cfg = {
        # grid / domain
        "nx": p["nx"], "ny": p["ny"], "nz": p["nz"],
        "lx": 200.0, "ly": 100.0, "lz": 20.0,
        # aquifer / flow
        "k0": 10.0, "kv_ratio": 0.1, "phi": 0.30, "R": 1.0,
        "grad": 1e-3,
        "alpha_l": 5.0, "alpha_th": 0.5, "alpha_tv": 0.05, "decay": 0.0,
        # random field
        "sigma": 0.5, "ell_x": 20.0, "ell_y": 20.0, "ell_z": 6.0,
        "n_kl": 0, "target_variance": 0.95,
        # wells
        "n_wells": p["n_wells"], "q_max": 50.0, "Q_max": 200.0,
        "eps_well": 1.5, "z_screen_frac": 0.3,
        # source / compliance / obs
        "source_xyz": [30.0, 50.0, z_mid], "source_q": 10.0, "source_c": 100.0,
        "source_days": 365.0,
        "compliance_xyz": [[185.0, 50.0, z_mid]],
        "obs_xyz": [[100.0, 50.0, z_mid]],
        # time
        "dt": 36.5, "nsteps": p["nsteps"],
        # SAA / cost / optim
        "m_train": p["m_train"], "m_test": p["m_test"],
        "alpha": 0.05, "alpha_h": 0.01,
        "c_e": 1.0, "c_w": 1.0, "c_m": 1.0, "c_tau": 1.0, "c_fail": 10.0,
        "creg": 1.0, "window_start": 0.8, "h_min": -10.0, "eps_time": 0.05,
        "rho_g": 1e6, "rho_h": 1e6,
        "max_evals": p["max_evals"], "n_starts": p["n_starts"],
        "max_iter": 60,
        "seed": 0, "seed_train": 1000, "seed_test": 500000,
    }
    cfg.update(overrides)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), default="full")
    ap.add_argument("--budget", type=float, default=20.0,
                    help="optimizer wall-clock budget in minutes (default 20)")
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--nz", type=int, default=None)
    ap.add_argument("--n-starts", type=int, default=None,
                    help="override the preset's number of pattern-search starts")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--no-plots", dest="plots", action="store_false",
                    help="skip matplotlib figures")
    ap.add_argument("--zip", dest="zip_results", action="store_true",
                    help="write colab3d_results.zip for download")
    args = ap.parse_args()

    res = detect_resources()
    log(f"resources: cpu={res['cpu']} mem={res.get('mem_gb')} colab={res['colab']}")

    overrides = {}
    if args.nx is not None:
        overrides["nx"] = args.nx
    if args.ny is not None:
        overrides["ny"] = args.ny
    if args.nz is not None:
        overrides["nz"] = args.nz
    if args.n_starts is not None:
        overrides["n_starts"] = args.n_starts
    cfg = build_config(args.preset, overrides)

    n_cells = cfg["nx"] * cfg["ny"] * cfg["nz"]
    if n_cells > 200_000:
        log(f"warning: {n_cells} cells is large for a sparse LU; "
            "consider a smaller --nz or --preset")
    if cfg["nz"] >= cfg["ny"]:
        log("warning: nz >= ny; vertical layers are meant to be coarse (nz << nx, ny)")

    out = Path(args.out) if args.out else Path("out3d_colab") / f"run_{args.preset}"
    run(cfg, out, budget_s=args.budget * 60.0, plots=args.plots,
        zip_results=args.zip_results)


if __name__ == "__main__":
    main()
