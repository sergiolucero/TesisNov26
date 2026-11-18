"""Tests for SCRIPTS/seed_sweep.py (the multi-seed harness).

These guard the two things that make the harness trustworthy: seed-spec
parsing and the median/IQR arithmetic. Both are easy to get subtly wrong and
hard to notice, because a wrong statistic still *looks* like a number.
"""

import importlib.util
import statistics as st
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "seed_sweep", ROOT / "SCRIPTS" / "seed_sweep.py")
seed_sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(seed_sweep)


class TestParseSeeds(unittest.TestCase):
    def test_range(self):
        self.assertEqual(seed_sweep.parse_seeds("0-4"), [0, 1, 2, 3, 4])

    def test_list(self):
        self.assertEqual(seed_sweep.parse_seeds("0,2,5"), [0, 2, 5])

    def test_mixed(self):
        self.assertEqual(seed_sweep.parse_seeds("0-2,7"), [0, 1, 2, 7])

    def test_single(self):
        self.assertEqual(seed_sweep.parse_seeds("3-3"), [3])

    def test_whitespace(self):
        self.assertEqual(seed_sweep.parse_seeds(" 0 , 2 "), [0, 2])

    def test_empty_raises(self):
        with self.assertRaises(SystemExit):
            seed_sweep.parse_seeds("")

    def test_reversed_range_raises(self):
        with self.assertRaises(SystemExit):
            seed_sweep.parse_seeds("3-1")

    def test_bad_token_raises(self):
        for bad in ("-", "a-b", "x", "1-", "-2"):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                seed_sweep.parse_seeds(bad)


class TestDig(unittest.TestCase):
    def test_nested(self):
        obj = {"metrics": {"regions": {"full": 1.5}}}
        self.assertEqual(seed_sweep.dig(obj, "metrics.regions.full"), 1.5)

    def test_missing_key_raises(self):
        with self.assertRaises(KeyError):
            seed_sweep.dig({"a": {}}, "a.b")

    def test_non_dict_raises(self):
        with self.assertRaises(KeyError):
            seed_sweep.dig({"a": 1}, "a.b")


class TestMedianIQR(unittest.TestCase):
    def test_matches_statistics_median(self):
        for vals in ([1, 2, 3, 4, 5],
                     [8.14e-2, 1.17e-2, 1.61e-2, 3.59e-2, 1.42e-2]):
            with self.subTest(vals=vals):
                got = seed_sweep.median_iqr(vals)["median"]
                self.assertAlmostEqual(got, st.median(vals), places=15)

    def test_quartiles(self):
        s = seed_sweep.median_iqr([1, 2, 3, 4, 5])
        self.assertAlmostEqual(s["q1"], 2.0)
        self.assertAlmostEqual(s["q3"], 4.0)
        self.assertAlmostEqual(s["iqr"], 2.0)

    def test_single_value(self):
        s = seed_sweep.median_iqr([5.0])
        self.assertEqual(s["median"], 5.0)
        self.assertEqual(s["iqr"], 0.0)

    def test_empty(self):
        self.assertEqual(seed_sweep.median_iqr([]), {})

    def test_order_independent(self):
        a = seed_sweep.median_iqr([3.0, 1.0, 2.0])
        b = seed_sweep.median_iqr([1.0, 3.0, 2.0])
        self.assertEqual(a["median"], b["median"])
        self.assertEqual(a["iqr"], b["iqr"])

    def test_spread_reported(self):
        # The Phase 3 baseline spread was ~7x; min/max must surface that.
        s = seed_sweep.median_iqr([1.17e-2, 8.14e-2])
        self.assertEqual(s["min"], 1.17e-2)
        self.assertEqual(s["max"], 8.14e-2)


if __name__ == "__main__":
    unittest.main()
