"""Tests for the plume figures in stoch/report.py.

The figure is a deliverable, and the original defects were silent: wells were
drawn as identical dots regardless of pumping rate, so a solution that puts the
whole budget on one well looked indistinguishable from an arbitrary layout; and
one figure drew only the remediated field, so nothing showed what the policy
was buying.

These tests pin what matters -- the annotation fires for active wells, the
no-pump / no-q paths work, and the merged untreated-vs-remediated figure returns
honest summary numbers with each field panel on its own colour scale.
"""

import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physics.fv2d import Grid
from stoch.report import plot_plume, plot_remediation


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


class TestPlotRemediation(unittest.TestCase):
    """The merged figure: untreated field, remediated field, breakthrough."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.grid = _grid()
        self.wells = np.array([[50.0, 50.0], [100.0, 50.0], [150.0, 50.0]])
        self.q = np.array([0.0, 12.5, 0.0])
        # Untreated field is 8x the treated one: the two panels must not share
        # a colour scale or the remediated panel would look empty.
        self.C_off = _field(self.grid, peak=8.0)
        self.C_on = _field(self.grid, peak=1.0)

    def tearDown(self):
        plt.close("all")

    def _render(self, **kw):
        p = Path(self.tmp) / "contamination.png"
        out = plot_remediation(self.grid, self.C_off, self.C_on, self.wells, p,
                               creg=1.0, q=self.q, **kw)
        self.assertTrue(p.is_file(), "figure was not written")
        self.assertGreater(p.stat().st_size, 1000)
        return out

    def test_summary_numbers(self):
        out = self._render()
        self.assertAlmostEqual(out["peak_off"], 8.0)
        self.assertAlmostEqual(out["peak_on"], 1.0)
        self.assertAlmostEqual(out["peak_reduction_pct"], 87.5)
        self.assertAlmostEqual(out["creg"], 1.0)

    def test_with_breakthrough_curves(self):
        t = np.linspace(0.0, 365.0, 11)
        out = self._render(times=t,
                           C_comp_off=np.linspace(0.0, 4.0, 10),
                           C_comp_on=np.linspace(0.0, 0.5, 10),
                           window_start=0.8)
        self.assertAlmostEqual(out["compliance_max_off"], 4.0)
        self.assertAlmostEqual(out["compliance_max_on"], 0.5)

    def test_curve_length_mismatch_is_tolerated(self):
        # C_record has one row per step while .times has nsteps+1 entries.
        t = np.linspace(0.0, 365.0, 11)
        out = self._render(times=t, C_comp_off=np.zeros(10),
                           C_comp_on=None, window_start=0.8)
        self.assertAlmostEqual(out["compliance_max_off"], 0.0)
        self.assertNotIn("compliance_max_on", out)

    def test_all_zero_q_and_no_q(self):
        p = Path(self.tmp) / "zero.png"
        plot_remediation(self.grid, self.C_off, self.C_off, self.wells, p,
                         creg=1.0, q=np.zeros(3))
        self.assertGreater(p.stat().st_size, 1000)
        p2 = Path(self.tmp) / "none.png"
        plot_remediation(self.grid, self.C_off, self.C_off, self.wells, p2,
                         creg=1.0)
        self.assertGreater(p2.stat().st_size, 1000)

    def test_independent_colour_scales(self):
        """Each field panel is normalized to its own peak, not a shared one.

        A shared scale would render the remediated panel as an almost black
        rectangle next to the untreated peak, hiding the comparison the figure
        exists to make.
        """
        import matplotlib.axes

        vmax = []
        orig = matplotlib.axes.Axes.imshow

        def spy(self, X, *a, **kw):
            vmax.append(kw.get("vmax"))
            return orig(self, X, *a, **kw)

        matplotlib.axes.Axes.imshow = spy
        try:
            self._render()
        finally:
            matplotlib.axes.Axes.imshow = orig
        self.assertEqual(len(vmax), 2)
        self.assertAlmostEqual(vmax[0], 8.0)
        self.assertAlmostEqual(vmax[1], 1.0)


if __name__ == "__main__":
    unittest.main()
