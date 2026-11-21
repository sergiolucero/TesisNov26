"""Gates for `SCRIPTS/check_tail.py`'s statistics.

The script's job is to turn "the run says reliability 0.99" into a claim someone
else can check, so the two numbers it leans on have to be right:

* `wilson_interval` -- a binomial interval, not a normal one, because 39/40 and
  391/400 are both "0.975" and only the second is evidence;
* `tail_stats` -- how many draws the CVaR average, and how uncertain that average
  is. This is the statistic that showed a run's reported tail (+0.005 on its own
  sample) is not a property of the aquifer (+0.25..+0.62 on three fresh draws).
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SCRIPTS"))

from check_tail import tail_stats, wilson_interval  # noqa: E402


class TestWilson(unittest.TestCase):
    def test_wide_when_the_sample_is_small(self):
        lo, hi = wilson_interval(16, 16)
        self.assertAlmostEqual(lo, 0.806, places=3)
        self.assertEqual(hi, 1.0)

    def test_narrows_with_n_at_the_same_rate(self):
        lo40, _ = wilson_interval(39, 40)
        lo400, _ = wilson_interval(391, 400)
        self.assertAlmostEqual(lo40, 0.871, places=3)
        self.assertAlmostEqual(lo400, 0.958, places=3)
        # 0.975 at M=40 is consistent with the 0.94 gate being missed; at M=400
        # the same rate clears it. That difference is the whole point.
        self.assertLess(lo40, 0.94)
        self.assertGreater(lo400, 0.94)

    def test_bounds_stay_inside_the_unit_interval(self):
        for k, n in ((0, 16), (1, 400), (400, 400), (0, 1)):
            lo, hi = wilson_interval(k, n)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)
            self.assertLessEqual(lo, hi)

    def test_empty_sample_is_nan_not_an_exception(self):
        lo, hi = wilson_interval(0, 0)
        self.assertTrue(math.isnan(lo) and math.isnan(hi))


class TestTailStats(unittest.TestCase):
    def test_counts_the_draws_the_average_uses(self):
        # 100 values, alpha=0.05 -> the average is over the worst 5.
        G = np.arange(100, dtype=float)
        ts = tail_stats(G, 0.05)
        self.assertEqual(ts["n_tail"], 5)
        self.assertAlmostEqual(ts["tail_mean"], 97.0)       # mean(95..99)
        self.assertAlmostEqual(ts["tail_min"], 95.0)
        self.assertAlmostEqual(ts["tail_max"], 99.0)

    def test_standard_error_is_the_tail_spread_not_the_whole_spread(self):
        rng = np.random.default_rng(0)
        G = np.concatenate([rng.normal(-1.0, 0.05, 380),  # the bulk: tight
                            rng.normal(1.0, 1.50, 20)])   # the tail: wide
        ts = tail_stats(G, 0.05)
        self.assertEqual(ts["n_tail"], 20)
        self.assertGreater(ts["tail_sd"], 1.0)
        self.assertAlmostEqual(ts["tail_se"], ts["tail_sd"] / math.sqrt(20), places=6)
        # A tail of 20 wide draws cannot pin its own mean to better than ~0.3,
        # which is the order of the draw-to-draw spread we measured (0.25..0.62).
        self.assertGreater(ts["tail_se"], 0.2)

    def test_no_spread_gives_no_uncertainty(self):
        ts = tail_stats(np.full(50, 3.0), 0.05)
        # alpha*M = 2.5, so the average is two whole draws plus a half-weighted
        # third one -- the fractional draw is why n_tail is 3 and not 2.
        self.assertEqual(ts["n_tail"], 3)
        self.assertAlmostEqual(ts["tail_sd"], 0.0)

    def test_tiny_sample_reduces_to_the_worst_value(self):
        # alpha*M < 1: CVaR is the max, so one draw and no spread estimate.
        ts = tail_stats(np.array([-2.0, -1.0, 0.0, 5.0]), 0.05)
        self.assertEqual(ts["n_tail"], 1)
        self.assertAlmostEqual(ts["tail_mean"], 5.0)
        self.assertTrue(math.isnan(ts["tail_se"]))

    def test_all_good_scenarios_still_produce_a_tail(self):
        # A policy with no failures has CVaR < 0 but the *tail* is still the
        # worst 5% -- "safe" does not mean "no tail to average".
        ts = tail_stats(np.linspace(-1.0, -0.5, 200), 0.05)
        self.assertEqual(ts["n_tail"], 10)
        self.assertLess(ts["tail_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
