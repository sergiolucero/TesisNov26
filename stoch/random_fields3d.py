"""Log-normal 3D conductivity via a separable Karhunen--Loève expansion.

The covariance is

    C(x,x') = sigma^2 exp(-|x-x'|/ell_x) exp(-|y-y'|/ell_y) exp(-|z-z'|/ell_z)

so the 3D eigenpairs are products of the 1D Neumann eigenpairs already used by
:mod:`stoch.random_fields`.  Modes are retained in descending variance until
``target_variance`` of the discretised field is represented.  Coefficients,
not the realised fields, are the source of truth for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from physics.fv3d import Grid3D
from stoch.random_fields import _eig_1d

__all__ = ["KLExpansion3D", "Scenario3D", "ScenarioGenerator3D"]


@dataclass
class KLExpansion3D:
    grid: Grid3D
    sigma: float = 0.5
    ell_x: float = 20.0
    ell_y: float = 20.0
    ell_z: float = 8.0
    n_kl: int = 0
    k0: float = 10.0
    target_variance: float = 0.95
    k_min: Optional[float] = None
    k_max: Optional[float] = None

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("sigma must be nonnegative")
        nu_x, phi_x = _eig_1d(self.grid.lx, self.ell_x, self.grid.nx)
        nu_y, phi_y = _eig_1d(self.grid.ly, self.ell_y, self.grid.ny)
        nu_z, phi_z = _eig_1d(self.grid.lz, self.ell_z, self.grid.nz)
        lam = np.einsum("i,j,k->ijk", nu_x, nu_y, nu_z)
        total = float(lam.sum())
        flat = np.sort(lam.ravel())[::-1]
        cum = np.cumsum(flat) / total
        if self.n_kl is None or self.n_kl <= 0:
            self.n_kl = int(np.searchsorted(cum, self.target_variance) + 1)
        self.n_kl = int(min(self.n_kl, flat.size))
        order = np.argsort(lam.ravel())[::-1][: self.n_kl]
        ix, iy, iz = np.unravel_index(order, lam.shape)
        self.nu_x, self.phi_x = nu_x, phi_x
        self.nu_y, self.phi_y = nu_y, phi_y
        self.nu_z, self.phi_z = nu_z, phi_z
        self.ix, self.iy, self.iz = ix, iy, iz
        self.weights = lam[ix, iy, iz]
        self.retained = float(self.weights.sum() / total)
        self.scale = self.sigma * np.sqrt(self.weights)
        self._px = phi_x[ix, :]
        self._py = phi_y[iy, :]
        self._pz = phi_z[iz, :]
        if self.k_min is None:
            self.k_min = self.k0 * np.exp(-3.5 * self.sigma)
        if self.k_max is None:
            self.k_max = self.k0 * np.exp(3.5 * self.sigma)

    def sample_coefficients(self, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(self.n_kl)

    def log_field(self, coeffs: np.ndarray) -> np.ndarray:
        c = self.scale * np.asarray(coeffs, dtype=float).ravel()
        if c.size != self.n_kl:
            raise ValueError("wrong number of KL coefficients")
        # (ny, nx, nz) matches Grid3D / Aquifer3D
        return np.einsum("kx,ky,kz,k->yxz", self._px, self._py, self._pz, c,
                         optimize=True)

    def conductivity(self, coeffs: np.ndarray) -> tuple[np.ndarray, float]:
        K = self.k0 * np.exp(self.log_field(coeffs))
        clamped = np.clip(K, self.k_min, self.k_max)
        frac = float(np.mean(clamped != K))
        return clamped, frac

    def to_dict(self) -> dict:
        return {
            "sigma": float(self.sigma),
            "ell_x": float(self.ell_x),
            "ell_y": float(self.ell_y),
            "ell_z": float(self.ell_z),
            "n_kl": int(self.n_kl),
            "target_variance": float(self.target_variance),
            "k0": float(self.k0),
            "k_min": float(self.k_min),
            "k_max": float(self.k_max),
            "retained_variance": float(self.retained),
        }


@dataclass
class Scenario3D:
    seed: int
    coeffs: np.ndarray
    K: np.ndarray
    clamp_fraction: float = 0.0
    source_scale: float = 1.0
    gradient_scale: float = 1.0


@dataclass
class ScenarioGenerator3D:
    grid: Grid3D
    kl: KLExpansion3D
    random_source: bool = False
    random_gradient: bool = False
    source_sigma: float = 0.2
    gradient_sigma: float = 0.1

    def sample(self, seed: int) -> Scenario3D:
        rng = np.random.default_rng(seed)
        coeffs = self.kl.sample_coefficients(rng)
        K, frac = self.kl.conductivity(coeffs)
        ss = gs = 1.0
        if self.random_source:
            ss = float(np.exp(rng.normal(0.0, self.source_sigma)))
        if self.random_gradient:
            gs = float(np.exp(rng.normal(0.0, self.gradient_sigma)))
        return Scenario3D(seed=seed, coeffs=coeffs, K=K,
                          clamp_fraction=frac, source_scale=ss,
                          gradient_scale=gs)
