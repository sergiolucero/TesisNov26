"""Unit tests for the ensemble-average seed-reduction utility.

Only the pure core is exercised here (``ensemble_rel_l2`` and ``median_iqr``),
so the tests are fast and do not need model training. The contract being
protected: averaging identical members must not change the error, averaging
symmetric independent perturbations must reduce it, and the per-member list is
returned unmodified.
"""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "ensemble_eval", ROOT / "SCRIPTS" / "ensemble_eval.py")
ensemble_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ensemble_eval)

RNG = np.random.default_rng(0)


class TestEnsembleRelL2(unittest.TestCase):
    def setUp(self):
        xi = np.linspace(0, 1, 64)
        tau = np.linspace(0, 0.5, 32)
        Xi, Tau = np.meshgrid(xi, tau, indexing="xy")
        self.u_ref = 0.5 * (1 + np.sin(2 * np.pi * Xi)) * (1 - np.exp(-Tau / 0.1))

    def test_identical_members_match_single(self):
        preds = [self.u_ref.copy() for _ in range(3)]
        per_member, ensemble = ensemble_eval.ensemble_rel_l2(preds, self.u_ref)
        self.assertEqual(len(per_member), 3)
        for e in per_member:
            self.assertAlmostEqual(e, 0.0, places=12)
        self.assertAlmostEqual(ensemble, 0.0, places=12)

    def test_symmetric_noise_averaging_reduces_error(self):
        noise = RNG.normal(0.0, 0.05, size=self.u_ref.shape)
        preds = [self.u_ref + noise, self.u_ref - noise]
        per_member, ensemble = ensemble_eval.ensemble_rel_l2(preds, self.u_ref)
        self.assertAlmostEqual(per_member[0], per_member[1], places=12)
        self.assertLess(ensemble, per_member[0])

    def test_ensemble_never_worse_than_worst_member(self):
        noise = RNG.normal(0.0, 0.05, size=self.u_ref.shape)
        preds = [self.u_ref + noise, self.u_ref, self.u_ref - noise]
        per_member, ensemble = ensemble_eval.ensemble_rel_l2(preds, self.u_ref)
        self.assertLessEqual(ensemble, max(per_member))

    def test_median_iqr_empty(self):
        self.assertEqual(ensemble_eval.median_iqr([]), {})


if __name__ == "__main__":
    unittest.main()
