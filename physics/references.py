"""Analytic reference solutions used by the forward/optimization gates."""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

__all__ = ["thiem_drawdown", "ogata_banks"]


def thiem_drawdown(
    q: float,
    T: float,
    r: np.ndarray | float,
    r_ref: float,
    h_ref: float = 0.0,
) -> np.ndarray:
    """Thiem steady drawdown for a fully penetrating well.

    ``h(r) = h_ref - q / (2 pi T) * ln(r_ref / r)`` for ``r <= r_ref``.
    ``q > 0`` is extraction, ``h`` is the head.
    """
    r = np.maximum(np.asarray(r, dtype=float), 1e-12)
    return h_ref - (q / (2.0 * np.pi * T)) * np.log(r_ref / r)


def ogata_banks(x, t, v, D, C0=1.0):
    """1D Ogata--Banks solution of the advection--dispersion equation."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    t = np.maximum(t, 1e-12)
    a = (x - v * t) / (2.0 * np.sqrt(D * t))
    b = (x + v * t) / (2.0 * np.sqrt(D * t))
    return 0.5 * C0 * (erfc(a) + np.exp(v * x / D) * erfc(b))
