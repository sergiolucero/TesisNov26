#!/usr/bin/env python3
"""Tests for the 3D certificate tool (SCRIPTS/check_tail3d.py).

The statistics are the part that can be silently wrong -- a Wilson interval on
the violation count instead of the compliant count puts the bounds on the wrong
side of the gate and turns a failing policy into a passing one.  So the anchors
below are the four published reference values from DOCS/Results.md, plus the
verdict direction (PASS needs the LOWER bound above the gate).
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "SCRIPTS"
for p in (str(ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod          # dataclasses/annotations need this
    spec.loader.exec_module(mod)
    return mod


ct = _load("check_tail3d")

GATE = 0.94


class TestWilson(unittest.TestCase):
    def test_published_anchors(self):
        """Reference values quoted in DOCS/Results.md -- must not drift."""
        cases = {
            (39, 40): (0.871, 0.996),
            (97, 100): (0.915, 0.990),
            (40, 40): (0.912, 1.000),
            (1, 1): (0.207, 1.000),
        }
        for (k, n), (lo_ex, hi_ex) in cases.items():
            lo, hi = ct.wilson_interval(k, n)
            self.assertAlmostEqual(lo, lo_ex, places=3, msg=f"{k}/{n} lower")
            self.assertAlmostEqual(hi, hi_ex, places=3, msg=f"{k}/{n} upper")

    def test_interval_brackets_the_estimate_and_stays_in_the_unit_interval(self):
        for k, n in [(0, 10), (5, 10), (10, 10), (7, 400), (0, 1)]:
            lo, hi = ct.wilson_interval(k, n)
            self.assertLessEqual(lo, k / n + 1e-12)
            self.assertGreaterEqual(hi, k / n - 1e-12)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)

    def test_narrows_with_sample_size(self):
        widths = []
        for n in (16, 100, 400):
            lo, hi = ct.wilson_interval(n, n)      # zero violations
            widths.append(hi - lo)
        self.assertGreater(widths[0], widths[1])
        self.assertGreater(widths[1], widths[2])

    def test_empty_sample_is_uninformative(self):
        self.assertEqual(ct.wilson_interval(0, 0), (0.0, 1.0))


class TestVerdict(unittest.TestCase):
    def test_small_perfect_sample_is_inconclusive_not_a_pass(self):
        """The 2D lesson: 40/40 has a lower bound of 0.912, below the gate."""
        self.assertEqual(ct.verdict(0, 40, GATE), "INCONCLUSIVE")

    def test_large_clean_sample_passes(self):
        self.assertEqual(ct.verdict(1, 400, GATE), "PASS")
        self.assertEqual(ct.verdict(0, 400, GATE), "PASS")

    def test_clear_failure_fails(self):
        # 34/400 violations: reliability 0.915, upper bound below 0.94
        self.assertEqual(ct.verdict(34, 400, GATE), "FAIL")

    def test_verdict_uses_the_compliant_count(self):
        """A policy that violates EVERY scenario must not be able to pass."""
        self.assertEqual(ct.verdict(40, 40, GATE), "FAIL")
        self.assertEqual(ct.verdict(400, 400, GATE), "FAIL")


class TestLoadRun(unittest.TestCase):
    def _write(self, d: Path, cfg: dict, pol: dict | None, met: dict | None):
        (d / "run_config.json").write_text(json.dumps(cfg))
        if pol is not None:
            (d / "policy.json").write_text(json.dumps(pol))
        if met is not None:
            (d / "metrics.json").write_text(json.dumps(met))

    def test_reads_policy_json(self):
        cfg = {"nx": 12, "ny": 6, "nz": 3, "m_train": 4, "m_test": 8,
               "seed_train": 1000, "seed_test": 500000}
        pol = {"q": [1.0, 0.0], "wells": [[10.0, 10.0, 3.0, 1.0],
                                          [20.0, 10.0, 3.0, 1.0]]}
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write(d, cfg, pol, {"opt": {"q_star": [9.0, 9.0]}})
            got_cfg, got_pol, got_met = ct.load_run(d)
        self.assertEqual(got_cfg["nx"], 12)
        self.assertEqual(got_pol["q"], [1.0, 0.0])     # policy.json wins
        self.assertIsNotNone(got_met)

    def test_falls_back_to_metrics_q_star(self):
        cfg = {"nx": 12, "ny": 6, "nz": 3, "m_train": 4, "m_test": 8}
        met = {"opt": {"q_star": [3.0, 0.0]}, "wells": [[10.0, 10.0, 3.0, 1.0],
                                                        [20.0, 10.0, 3.0, 1.0]]}
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write(d, cfg, None, met)
            _cfg, got_pol, _met = ct.load_run(d)
        self.assertEqual(got_pol["q"], [3.0, 0.0])
        self.assertEqual(len(got_pol["wells"]), 2)

    def test_missing_run_config_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                ct.load_run(Path(tmp))


class TestRebuildUsesTheRunsOwnConstructors(unittest.TestCase):
    def test_rebuild_shapes_and_scalars(self):
        cfg = {"nx": 10, "ny": 6, "nz": 3, "lx": 100.0, "ly": 60.0, "lz": 12.0,
               "k0": 10.0, "kv_ratio": 0.1, "phi": 0.3, "R": 1.0,
               "alpha_l": 5.0, "alpha_th": 0.5, "alpha_tv": 0.05, "decay": 0.0,
               "sigma": 0.5, "ell_x": 20.0, "ell_y": 20.0, "ell_z": 6.0,
               "n_kl": 8, "target_variance": 0.95,
               "grad": 1e-3, "source_xyz": [30.0, 30.0, 6.0], "source_q": 10.0,
               "source_c": 100.0, "source_days": 365.0,
               "compliance_xyz": [[90.0, 30.0, 6.0]], "obs_xyz": [[50.0, 30.0, 6.0]],
               "eps_well": 1.5, "dt": 36.5, "nsteps": 4,
               "c_e": 1.0, "c_w": 1.0, "c_m": 1.0, "c_tau": 1.0, "c_fail": 10.0,
               "creg": 1.0, "window_start": 0.8, "h_min": -10.0, "eps_time": 0.05,
               "alpha": 0.05, "alpha_h": 0.01}
        grid, aq, kl, fwd, cost = ct.rebuild(cfg, [[10.0, 10.0, 3.0, 1.0]])
        self.assertEqual((grid.nx, grid.ny, grid.nz), (10, 6, 3))
        self.assertEqual(grid.n, 180)
        self.assertEqual(aq.K.shape, (3, 6, 10))
        self.assertEqual(kl.n_kl, 8)
        self.assertEqual(fwd.nsteps, 4)
        self.assertEqual(cost.creg, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
