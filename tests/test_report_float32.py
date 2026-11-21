"""Reporting must survive float32 objective values.

A float32 GPU run reached the plotting stage with `ev.G`/`ev.tau` in float32;
`np.histogram` then refused to build bins over the tiny representable range and
raised, aborting the run *after* the optimization had succeeded and before
metrics.json was written. The fix casts to float64 inside the plot helper.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np

matplotlib.use("Agg")

from stoch.report import plot_distributions  # noqa: E402


class _Eval:
    def __init__(self, dtype) -> None:
        # A range this narrow is what float32 objectives actually look like when
        # every scenario is compliant (G = -1 exactly, up to rounding).
        self.G = np.array([-1.0, -1.0 + 1e-7, -1.0 - 1e-7, -0.9999999], dtype=dtype)
        self.tau = np.zeros(4, dtype=dtype)


class TestFloat32Plotting(unittest.TestCase):
    def test_float32_and_float64_both_plot(self):
        with TemporaryDirectory() as tmp:
            for dtype in (np.float32, np.float64):
                path = Path(tmp) / f"dist_{np.dtype(dtype).name}.png"
                plot_distributions(_Eval(dtype), 365.0, path)
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
