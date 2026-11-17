"""Tests for the q*-annotated plume plotting in stoch/report.py.

The figure is a deliverable, and the defect it fixes was silent: wells were
drawn as identical dots regardless of pumping rate, so a solution that puts the
whole budget on one well looked indistinguishable from an arbitrary layout.
These tests pin the two things that matter -- the annotation actually fires for
active wells, and the no-pump / no-q paths still work.
"""

import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physics.fv2d import Grid
from stoch.report import plot_plume


def _grid():
    return Grid(nx=20, ny=10, lx=200.0, ly=100.0)


def _field(grid, peak=1.0):
    C = np.zeros((grid.ny, grid.nx))
    C[grid.ny // 2, grid.nx // 2] = peak
    return C


class TestPlotPlume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.grid = _grid()
        self.wells = np.array([[50.0, 50.0], [100.0, 50.0], [150.0, 50.0]])
        self.comp = ((180.0, 50.0),)
        self.C = _field(self.grid)

    def tearDown(self):
        plt.close("all")

    def _render(self, **kw):
        p = Path(self.tmp) / f"p_{abs(hash(tuple(sorted(kw))))}.png"
        plot_plume(self.grid, self.C, self.wells, self.comp, p, **kw)
        self.assertTrue(p.is_file(), "figure was not written")
        return p

    def test_writes_without_q(self):
        p = self._render(creg=0.5, title="no q")
        self.assertGreater(p.stat().st_size, 1000)

    def test_writes_with_q(self):
        q = np.array([14.8, 0.0, 0.0])
        p = self._render(creg=0.5, q=q, title="with q")
        self.assertGreater(p.stat().st_size, 1000)

    def test_all_zero_q(self):
        # The no-pump comparison passes an all-zero q; must not divide by zero
        # or draw a zero-size marker.
        p = self._render(creg=0.5, q=np.zeros(3), title="off")
        self.assertGreater(p.stat().st_size, 1000)

    def test_q_length_mismatch_is_tolerated(self):
        # A short q must not raise: extra wells are treated as idle.
        p = self._render(creg=0.5, q=np.array([5.0]), title="short q")
        self.assertGreater(p.stat().st_size, 1000)

    def test_active_well_annotation(self):
        """One active well: its rate should appear in the title summary."""
        import matplotlib.axes

        q = np.array([0.0, 14.8, 0.0])
        captured = {}
        orig = matplotlib.axes.Axes.set_title

        def spy(self, label, *a, **kw):
            captured["title"] = label
            return orig(self, label, *a, **kw)

        matplotlib.axes.Axes.set_title = spy
        try:
            self._render(creg=0.5, q=q, title="t")
        finally:
            matplotlib.axes.Axes.set_title = orig
        self.assertIn("1/3 wells active", captured["title"])
        self.assertIn("14.8", captured["title"])

    def test_title_reports_total_rate(self):
        import matplotlib.axes

        q = np.array([10.0, 0.0, 5.0])
        captured = {}
        orig = matplotlib.axes.Axes.set_title

        def spy(self, label, *a, **kw):
            captured["title"] = label
            return orig(self, label, *a, **kw)

        matplotlib.axes.Axes.set_title = spy
        try:
            self._render(creg=0.5, q=q, title="t")
        finally:
            matplotlib.axes.Axes.set_title = orig
        self.assertIn("2/3 wells active", captured["title"])
        self.assertIn("15.0", captured["title"])


if __name__ == "__main__":
    unittest.main()
