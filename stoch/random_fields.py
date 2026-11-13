"""Log-normal conductivity via a Karhunen--Lo\\`eve expansion.

The covariance is separable exponential,

    C(x, x') = sigma^2 exp(-|x-x'|/ell_x) exp(-|y-y'|/ell_y),

whose 1D eigenpairs on ``[0, L]`` with reflecting (Neumann) boundaries are

    nu_n = (2 ell / L) / (1 + (n pi ell / L)^2),
    phi_0 = 1/sqrt(L),  phi_n = sqrt(2/L) cos(n pi x / L).

The 2D eigenpairs are products.  The expansion is truncated at ``n_kl`` modes
retaining the largest variance, and the coefficients -- not only the realised
fields -- are stored in the scenario archive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from physics.fv2d import Grid

__all__ = [
    "KLExpansion",
    "Scenario",
    "ScenarioGenerator",
    "save_archive",
    "load_archive",
    "retained_variance",
]


def _eig_1d(L: float, ell: float, nx: int):
    """Discrete 1D Neumann eigenpairs of the exponential covariance.

    The integral operator is discretised with midpoint quadrature,

        (T f)_i = sum_j exp(-|x_i-x_j|/ell) f_j dx,

    and the symmetric matrix ``A = C dx`` is diagonalised.  This avoids the
    finite-interval correction to the infinite-domain formula
    ``2 ell / (1 + (n pi ell / L)^2)`` and guarantees that the retained
    variance is measured against the actual discretised field.
    """
    x = (np.arange(nx) + 0.5) * L / nx
    dx = L / nx
    C = np.exp(-np.abs(x[:, None] - x[None, :]) / ell)
    A = C * dx
    nu, V = np.linalg.eigh(A)
    order = np.argsort(nu)[::-1]
    nu = np.clip(nu[order], 0.0, None)
    # phi = v / sqrt(dx) so that sum_i phi_i^2 dx = 1
    phi = V[:, order].T / np.sqrt(dx)
    return nu, phi


@dataclass
class KLExpansion:
    grid: Grid
    sigma: float = 0.5
    ell_x: float = 20.0
    ell_y: float = 20.0
    n_kl: int = 0                 # 0 => auto-select for target_variance
    k0: float = 10.0
    target_variance: float = 0.95
    k_min: Optional[float] = None
    k_max: Optional[float] = None

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("sigma must be nonnegative")
        nu_x, phi_x = _eig_1d(self.grid.lx, self.ell_x, self.grid.nx)
        nu_y, phi_y = _eig_1d(self.grid.ly, self.ell_y, self.grid.ny)
        lam = np.outer(nu_x, nu_y)
        total = float(lam.sum())
        flat = np.sort(lam.ravel())[::-1]
        cum = np.cumsum(flat) / total
        if self.n_kl is None or self.n_kl <= 0:
            self.n_kl = int(np.searchsorted(cum, self.target_variance) + 1)
        self.n_kl = int(min(self.n_kl, flat.size))
        order = np.argsort(lam.ravel())[::-1][: self.n_kl]
        ix, iy = np.unravel_index(order, lam.shape)
        self.nu_x, self.phi_x = nu_x, phi_x
        self.nu_y, self.phi_y = nu_y, phi_y
        self.ix = ix
        self.iy = iy
        self.weights = lam[ix, iy]
        self.retained = float(self.weights.sum() / total)
        self.scale = self.sigma * np.sqrt(self.weights)
        # (n_kl, nx) and (n_kl, ny) design matrices
        self._px = phi_x[ix, :]
        self._py = phi_y[iy, :]
        if self.k_min is None:
            self.k_min = self.k0 * np.exp(-3.5 * self.sigma)
        if self.k_max is None:
            self.k_max = self.k0 * np.exp(3.5 * self.sigma)

    # -- sampling ---------------------------------------------------------
    def sample_coefficients(self, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(self.n_kl)

    def log_field(self, coeffs: np.ndarray) -> np.ndarray:
        c = self.scale * np.asarray(coeffs, dtype=float).ravel()
        if c.size != self.n_kl:
            raise ValueError("wrong number of KL coefficients")
        # G[j, i] = sum_k c_k py_k[j] px_k[i]
        return (self._py * c[:, None]).T @ self._px

    def conductivity(self, coeffs: np.ndarray) -> tuple[np.ndarray, float]:
        G = self.log_field(coeffs)
        K = self.k0 * np.exp(G)
        clamped = np.clip(K, self.k_min, self.k_max)
        frac = float(np.mean(clamped != K))
        return clamped, frac

    def to_dict(self) -> dict:
        return {
            "sigma": float(self.sigma),
            "ell_x": float(self.ell_x),
            "ell_y": float(self.ell_y),
            "n_kl": int(self.n_kl),
            "target_variance": float(self.target_variance),
            "k0": float(self.k0),
            "k_min": float(self.k_min),
            "k_max": float(self.k_max),
            "retained_variance": float(self.retained),
        }


@dataclass
class Scenario:
    """One random realisation (coefficients are the source of truth)."""

    seed: int
    coeffs: np.ndarray
    K: np.ndarray
    clamp_fraction: float = 0.0
    source_scale: float = 1.0
    gradient_scale: float = 1.0

    def to_dict(self) -> dict:
        return {
            "seed": int(self.seed),
            "clamp_fraction": float(self.clamp_fraction),
            "source_scale": float(self.source_scale),
            "gradient_scale": float(self.gradient_scale),
        }


@dataclass
class ScenarioGenerator:
    """Draw scenarios.  Only ``K`` is random in the baseline; the optional
    source-strength and regional-gradient multipliers are one-at-a-time
    switches for the sensitivity study."""

    grid: Grid
    kl: KLExpansion
    random_source: bool = False
    random_gradient: bool = False
    source_sigma: float = 0.2
    gradient_sigma: float = 0.1

    def sample(self, seed: int) -> Scenario:
        rng = np.random.default_rng(seed)
        coeffs = self.kl.sample_coefficients(rng)
        K, frac = self.kl.conductivity(coeffs)
        ss = 1.0
        gs = 1.0
        if self.random_source:
            ss = float(np.exp(rng.normal(0.0, self.source_sigma)))
        if self.random_gradient:
            gs = float(np.exp(rng.normal(0.0, self.gradient_sigma)))
        return Scenario(seed=seed, coeffs=coeffs, K=K, clamp_fraction=frac,
                        source_scale=ss, gradient_scale=gs)


def retained_variance(kl: KLExpansion) -> float:
    return kl.retained


# ---------------------------------------------------------------------------
# Archive I/O
# ---------------------------------------------------------------------------
def save_archive(
    path: str | Path,
    scenarios: list[Scenario],
    generator: ScenarioGenerator,
    extra: Optional[dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path.with_suffix(".npz"),
        coeffs=np.vstack([s.coeffs for s in scenarios]),
        seeds=np.array([s.seed for s in scenarios], dtype=np.int64),
        source_scale=np.array([s.source_scale for s in scenarios]),
        gradient_scale=np.array([s.gradient_scale for s in scenarios]),
    )
    meta = {
        "kl": generator.kl.to_dict(),
        "random_source": bool(generator.random_source),
        "random_gradient": bool(generator.random_gradient),
        "source_sigma": float(generator.source_sigma),
        "gradient_sigma": float(generator.gradient_sigma),
        "n_scenarios": len(scenarios),
        "clamp_fraction_max": float(max(s.clamp_fraction for s in scenarios)),
        "extra": extra or {},
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load_archive(path: str | Path) -> tuple[list[Scenario], dict]:
    path = Path(path)
    npz = np.load(path.with_suffix(".npz"))
    meta = json.loads(path.with_suffix(".json").read_text())
    coeffs = npz["coeffs"]
    seeds = npz["seeds"]
    ss = npz["source_scale"]
    gs = npz["gradient_scale"]
    kl_meta = meta["kl"]
    # K is rebuilt from the coefficients by the caller (grid is not stored).
    scenarios = [
        Scenario(
            seed=int(seeds[m]),
            coeffs=coeffs[m],
            K=np.empty((0, 0)),  # filled by the caller via kl.conductivity
            source_scale=float(ss[m]),
            gradient_scale=float(gs[m]),
        )
        for m in range(coeffs.shape[0])
    ]
    return scenarios, meta
