"""Well-layout checks (gate O-0)."""

import unittest

import numpy as np

from physics.fv2d import Grid
from stoch.wells import (
    covering_radius,
    coverage_fraction,
    cvt,
    hex_lattice,
    place_wells,
    plume_mask,
)


class TestWells(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(40, 20, 200.0, 100.0)
        X, Y = self.grid.centres()
        C = np.exp(-(((X - 100) ** 2 + (Y - 50) ** 2) / (2 * 25 ** 2)))
        self.mask = plume_mask(self.grid, C, 0.25)
        self.area = self.mask.sum() * self.grid.dx * self.grid.dy

    def test_hex_count_and_spacing(self):
        n = 9
        wells = hex_lattice(self.grid, self.mask, n)
        self.assertEqual(wells.shape, (n, 2))
        spacing = np.sqrt(2 * self.area / (np.sqrt(3) * n))
        # no two wells closer than a small fraction of the spacing
        d = np.linalg.norm(wells[:, None, :] - wells[None, :, :], axis=2)
        d[d == 0] = np.inf
        self.assertGreater(d.min(), 0.4 * spacing)

    def test_points_inside_plume(self):
        wells = hex_lattice(self.grid, self.mask, 7)
        for x, y in wells:
            i = int(np.clip(x / self.grid.dx, 0, self.grid.nx - 1))
            j = int(np.clip(y / self.grid.dy, 0, self.grid.ny - 1))
            # near the boundary, allow one cell
            patch = self.mask[max(j - 1, 0):j + 2, max(i - 1, 0):i + 2]
            self.assertTrue(patch.any(), msg=f"well ({x},{y}) outside plume")

    def test_covering_radius(self):
        n = 9
        wells = hex_lattice(self.grid, self.mask, n)
        spacing = np.sqrt(2 * self.area / (np.sqrt(3) * n))
        cr = covering_radius(self.grid, self.mask, wells)
        self.assertLess(cr, 1.5 * spacing / np.sqrt(3))
        cov = coverage_fraction(self.grid, self.mask, wells, spacing / np.sqrt(3))
        self.assertGreater(cov, 0.8)

    def test_cvt_reduces_quantization(self):
        rng = np.random.default_rng(0)
        pts = np.column_stack([
            (np.arange(self.mask.sum()) % self.grid.nx) * self.grid.dx,
            (np.arange(self.mask.sum()) // self.grid.nx) * self.grid.dy,
        ])
        init = pts[rng.choice(len(pts), 6, replace=False)]
        cvt_pts = cvt(self.grid, self.mask, 6, init=init, iters=50)
        self.assertEqual(cvt_pts.shape, (6, 2))

        def quant(points):
            j, i = np.nonzero(self.mask)
            xy = np.column_stack([(i + 0.5) * self.grid.dx, (j + 0.5) * self.grid.dy])
            d = np.min(np.linalg.norm(xy[:, None, :] - points[None, :, :], axis=2), axis=1)
            return float(np.mean(d ** 2))

        self.assertLessEqual(quant(cvt_pts), quant(init) * 1.05)

    def test_place_wells(self):
        for method in ("hex", "cvt"):
            layout = place_wells(self.grid, self.mask, 5, method=method, seed=0)
            self.assertEqual(layout.n_wells, 5)
            self.assertGreater(layout.covering_radius, 0.0)
            self.assertTrue(0.0 <= layout.coverage <= 1.0)


if __name__ == "__main__":
    unittest.main()
