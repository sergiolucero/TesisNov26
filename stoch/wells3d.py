"""Well placement for the 3D model: plume mask and farthest-point layout.

The 3D analogue of :mod:`stoch.wells`.  A hexagonal lattice is awkward on a
3D plume, so the baseline is a deterministic farthest-point sampling of the
plume cell centres: the first well sits on the plume maximum and every next
one is the plume cell farthest from those already chosen.  This spreads a
requested number of wells over the whole plume, including its vertical extent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from physics.fv3d import Grid3D

__all__ = [
    "plume_mask3d",
    "mask_points3d",
    "farthest_point_wells3d",
    "coverage3d",
    "WellLayout3D",
    "place_wells3d",
]


def plume_mask3d(grid: Grid3D, C: np.ndarray, threshold: float) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    if C.shape != grid.shape:
        raise ValueError(f"C shape {C.shape} does not match grid {grid.shape}")
    mask = C >= threshold
    if mask.sum() < 4:
        mask = np.zeros_like(mask, dtype=bool)
        j, i, k = np.unravel_index(np.argmax(C), C.shape)
        mask[max(j - 1, 0): j + 2, max(i - 1, 0): i + 2,
             max(k - 1, 0): k + 2] = True
    return mask


def mask_points3d(grid: Grid3D, mask: np.ndarray) -> np.ndarray:
    j, i, k = np.nonzero(np.asarray(mask, dtype=bool))
    x = grid.x0 + (i + 0.5) * grid.dx
    y = grid.y0 + (j + 0.5) * grid.dy
    z = grid.z0 + (k + 0.5) * grid.dz
    return np.column_stack([x, y, z])


def farthest_point_wells3d(
    pts: np.ndarray,
    n_wells: int,
    start: int | None = None,
) -> np.ndarray:
    """Farthest-point sampling; deterministic and always returns n_wells."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    if pts.shape[0] == 0:
        raise ValueError("empty point set")
    n = min(int(n_wells), pts.shape[0])
    if start is None:
        start = 0
    chosen = [int(np.clip(start, 0, pts.shape[0] - 1))]
    dmin = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    for _ in range(n - 1):
        k = int(np.argmax(dmin))
        chosen.append(k)
        dmin = np.minimum(dmin, np.linalg.norm(pts - pts[k], axis=1))
    return pts[np.asarray(chosen)]


def coverage3d(pts: np.ndarray, wells: np.ndarray) -> tuple[float, float]:
    """Max distance and fraction within one cell of the nearest well."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    wells = np.atleast_2d(np.asarray(wells, dtype=float))
    d, _ = cKDTree(wells).query(pts)
    return float(np.max(d)), float(np.mean(d <= 1.0))


@dataclass
class WellLayout3D:
    grid: Grid3D
    mask: np.ndarray
    wells: np.ndarray
    method: str
    covering_radius: float
    coverage: float

    @property
    def n_wells(self) -> int:
        return int(self.wells.shape[0])


def place_wells3d(
    grid: Grid3D,
    C: np.ndarray,
    n_wells: int = 12,
    threshold: float = 0.05,
    method: str = "farthest",
) -> WellLayout3D:
    """Place ``n_wells`` inside the design plume ``C >= threshold``."""
    mask = plume_mask3d(grid, C, threshold)
    pts = mask_points3d(grid, mask)
    if method != "farthest":
        raise ValueError("3D baseline supports method='farthest' only")
    # start at the plume maximum so the hottest cell is always screened
    j, i, k = np.unravel_index(np.argmax(C), C.shape)
    hot = np.array([grid.x0 + (i + 0.5) * grid.dx,
                    grid.y0 + (j + 0.5) * grid.dy,
                    grid.z0 + (k + 0.5) * grid.dz])
    start = int(np.argmin(np.linalg.norm(pts - hot, axis=1)))
    wells = farthest_point_wells3d(pts, n_wells, start=start)
    cr, cov = coverage3d(pts, wells)
    return WellLayout3D(grid, mask, wells, method, cr, cov)
