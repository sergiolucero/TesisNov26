"""Well layout: hexagonal lattice, centroidal Voronoi tessellation, diagnostics.

The wells are *here-and-now*: they are placed before the conductivity field is
observed and do not move during the optimization.  Only the rates are
decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from physics.fv2d import Grid

__all__ = [
    "plume_mask",
    "hex_lattice",
    "cvt",
    "place_wells",
    "covering_radius",
    "coverage_fraction",
    "WellLayout",
]


def plume_mask(
    grid: Grid,
    C: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Binary design-plume mask ``C >= threshold`` (shape ``(ny, nx)``)."""
    C = np.asarray(C, dtype=float)
    if C.shape != (grid.ny, grid.nx):
        raise ValueError("C shape does not match the grid")
    mask = C >= threshold
    if mask.sum() < 4:
        # fall back to the cell of the maximum so the code never crashes
        mask = np.zeros_like(mask, dtype=bool)
        j, i = np.unravel_index(np.argmax(C), C.shape)
        mask[max(j - 1, 0): j + 2, max(i - 1, 0): i + 2] = True
    return mask


def _mask_points(grid: Grid, mask: np.ndarray) -> np.ndarray:
    j, i = np.nonzero(mask)
    x = grid.x0 + (i + 0.5) * grid.dx
    y = grid.y0 + (j + 0.5) * grid.dy
    return np.column_stack([x, y])


def hex_lattice(
    grid: Grid,
    mask: np.ndarray,
    n_wells: int,
    centroid: Optional[np.ndarray] = None,
    spacing: Optional[float] = None,
) -> np.ndarray:
    """Place ``n_wells`` on a hexagonal lattice clipped to the design plume.

    Uses ``ell_w = sqrt(2 A_p / (sqrt(3) N_w))`` (Eq. ``hexspacing``) and
    retains the points closest to the plume centroid.  If the lattice yields
    too few interior points, the remaining wells are filled by farthest-point
    sampling of the plume cells, so the result always has ``n_wells`` points.
    """
    mask = np.asarray(mask, dtype=bool)
    pts = _mask_points(grid, mask)
    if pts.shape[0] == 0:
        raise ValueError("empty design plume")
    if centroid is None:
        centroid = pts.mean(axis=0)
    centroid = np.asarray(centroid, dtype=float)

    area = mask.sum() * grid.dx * grid.dy
    if spacing is None:
        spacing = np.sqrt(2.0 * area / (np.sqrt(3.0) * max(n_wells, 1)))

    # lattice indices covering the plume bounding box
    lo = pts.min(axis=0) - 2 * spacing
    hi = pts.max(axis=0) + 2 * spacing
    i_range = np.arange(np.floor((lo[0] - centroid[0]) / spacing) - 1,
                        np.ceil((hi[0] - centroid[0]) / spacing) + 2)
    j_range = np.arange(np.floor((lo[1] - centroid[1]) / (spacing * np.sqrt(3) / 2)) - 1,
                        np.ceil((hi[1] - centroid[1]) / (spacing * np.sqrt(3) / 2)) + 2)
    I, J = np.meshgrid(i_range, j_range)
    cand = np.column_stack([
        centroid[0] + spacing * (I.ravel() + J.ravel() / 2.0),
        centroid[1] + spacing * (np.sqrt(3) / 2.0) * J.ravel(),
    ])

    # keep lattice points inside the plume (tol = half a cell)
    tol = 0.5 * min(grid.dx, grid.dy)
    tree = cKDTree(pts)
    d, _ = tree.query(cand)
    cand = cand[d <= tol]
    if cand.shape[0] >= n_wells:
        order = np.argsort(np.linalg.norm(cand - centroid, axis=1))
        return cand[order[:n_wells]]

    chosen = list(cand)
    # farthest-point fill from the plume cell centres
    if chosen:
        dmin = np.min(np.linalg.norm(pts[:, None, :] - np.asarray(chosen)[None, :, :], axis=2), axis=1)
    else:
        dmin = np.full(pts.shape[0], np.inf)
    for _ in range(n_wells - len(chosen)):
        k = int(np.argmax(dmin))
        chosen.append(pts[k])
        dmin = np.minimum(dmin, np.linalg.norm(pts - pts[k], axis=1))
    return np.asarray(chosen)


def cvt(
    grid: Grid,
    mask: np.ndarray,
    n_wells: int,
    init: Optional[np.ndarray] = None,
    iters: int = 60,
    tol: float = 1e-4,
    seed: int = 0,
) -> np.ndarray:
    """Centroidal Voronoi tessellation by Lloyd's algorithm (discrete)."""
    pts = _mask_points(grid, mask)
    if init is None:
        # low-discrepancy-ish initialisation: random cells
        rng = np.random.default_rng(seed)
        init = pts[rng.choice(pts.shape[0], size=n_wells, replace=False)]
    w = np.asarray(init, dtype=float).copy()

    for _ in range(iters):
        tree = cKDTree(w)
        _, lab = tree.query(pts)
        new = np.empty_like(w)
        for k in range(n_wells):
            sel = pts[lab == k]
            if sel.shape[0] == 0:
                # re-seed an empty cell at the farthest plume point
                d, _ = cKDTree(w).query(pts)
                new[k] = pts[int(np.argmax(d))]
            else:
                new[k] = sel.mean(axis=0)
        move = np.max(np.linalg.norm(new - w, axis=1))
        w = new
        if move < tol:
            break
    return w


def covering_radius(grid: Grid, mask: np.ndarray, wells: np.ndarray) -> float:
    """Max distance from a plume cell centre to the nearest well."""
    pts = _mask_points(grid, mask)
    d, _ = cKDTree(np.atleast_2d(wells)).query(pts)
    return float(np.max(d))


def coverage_fraction(
    grid: Grid,
    mask: np.ndarray,
    wells: np.ndarray,
    radius: float,
) -> float:
    """Fraction of plume cells within ``radius`` of a well."""
    pts = _mask_points(grid, mask)
    d, _ = cKDTree(np.atleast_2d(wells)).query(pts)
    return float(np.mean(d <= radius))


@dataclass
class WellLayout:
    grid: Grid
    mask: np.ndarray
    wells: np.ndarray
    method: str
    spacing: float
    covering_radius: float
    coverage: float

    @property
    def n_wells(self) -> int:
        return int(self.wells.shape[0])


def place_wells(
    grid: Grid,
    mask: np.ndarray,
    n_wells: int,
    method: str = "hex",
    centroid: Optional[np.ndarray] = None,
    coverage_radius: Optional[float] = None,
    seed: int = 0,
) -> WellLayout:
    """Factory for the baseline layouts."""
    pts = _mask_points(grid, mask)
    if centroid is None:
        centroid = pts.mean(axis=0)
    area = mask.sum() * grid.dx * grid.dy
    spacing = np.sqrt(2.0 * area / (np.sqrt(3.0) * max(n_wells, 1)))
    if method == "hex":
        wells = hex_lattice(grid, mask, n_wells, centroid=centroid, spacing=spacing)
    elif method == "cvt":
        wells = cvt(grid, mask, n_wells, iters=80, seed=seed)
    else:
        raise ValueError("method must be 'hex' or 'cvt'")
    cr = covering_radius(grid, mask, wells)
    if coverage_radius is None:
        coverage_radius = spacing / np.sqrt(3.0)
    cov = coverage_fraction(grid, mask, wells, coverage_radius)
    return WellLayout(grid, mask, wells, method, spacing, cr, cov)
