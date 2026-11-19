"""Tests for the plume movie renderer.

The movie is a deliverable, and the failure modes are quiet: a default
per-frame colour scale makes the plume look stationary, and a stride that
drops the final frame hides the cleaned-up state.  These tests pin the GIF
format, the fixed global scale, and the "always keep the last step" rule.
"""

import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physics.fv2d import Grid
from stoch.report import animate_plume


def _grid():
    return Grid(nx=12, ny=8, lx=120.0, ly=80.0)


def _fields(grid, nframes=6, peak=4.0):
    """A travelling Gaussian blob, flattened like transport_run's C_fields."""
    X, Y = grid.centres()
    fields = np.zeros((nframes, grid.n))
    for k in range(nframes):
        x0 = 20.0 + 10.0 * k
        C = peak * np.exp(-((X - x0) ** 2 + (Y - 40.0) ** 2) / 100.0)
        fields[k] = C.ravel()
    return fields


class TestAnimatePlume(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.grid = _grid()
        self.wells = np.array([[30.0, 40.0], [80.0, 40.0]])
        self.comp = ((110.0, 40.0),)

    def tearDown(self):
        plt.close("all")

    def test_writes_gif_and_reports_scale(self):
        fields = _fields(self.grid)
        times = np.arange(fields.shape[0]) * 10.0
        out = self.tmp / "plume.gif"
        summary = animate_plume(
            self.grid, fields, times, self.wells, self.comp, out,
            creg=1.0, q=np.array([3.0, 0.0]), fps=8, dpi=60,
        )
        self.assertTrue(out.is_file())
        self.assertGreater(out.stat().st_size, 1000)
        self.assertEqual(out.read_bytes()[:3], b"GIF")
        self.assertEqual(summary["n_frames"], fields.shape[0])
        self.assertAlmostEqual(summary["vmax"], float(fields.max()))
        self.assertEqual(summary["path"], str(out))

    def test_stride_always_keeps_last_frame(self):
        fields = _fields(self.grid, nframes=10)
        times = np.arange(10) * 10.0
        out = self.tmp / "stride.gif"
        summary = animate_plume(
            self.grid, fields, times, self.wells, self.comp, out,
            stride=4, fps=8, dpi=50,
        )
        # Frames 0, 4, 8 would be the naive stride; the final frame 9 must
        # also be present so the movie ends on the stored final state.
        self.assertEqual(summary["n_frames"], 4)
        self.assertEqual(summary["n_steps"], 10)

    def test_bad_field_shape_raises(self):
        out = self.tmp / "bad.gif"
        with self.assertRaises(ValueError):
            animate_plume(
                self.grid, np.zeros((3, self.grid.n + 1)),
                np.arange(3.0), self.wells, self.comp, out,
            )

    def test_bad_suffix_raises(self):
        fields = _fields(self.grid)
        with self.assertRaises(ValueError):
            animate_plume(
                self.grid, fields, np.arange(fields.shape[0]) * 10.0,
                self.wells, self.comp, self.tmp / "movie.avi",
            )


if __name__ == "__main__":
    unittest.main()
