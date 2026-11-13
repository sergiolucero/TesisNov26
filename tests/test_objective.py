"""Cost, cleanup time, CVaR and mass-balance checks (gate O-2)."""

import unittest

import numpy as np

from stoch.objective import (
    CostConfig,
    cleanup_time,
    cvar,
    cvar_and_weights,
    scenario_objective,
    sigmoid,
    soft_last_violation,
)
from tests.helpers import tiny_archive


class TestCleanupTime(unittest.TestCase):
    def test_last_violation(self):
        dt = 10.0
        tau, fail = cleanup_time(np.array([1.0, 1.0, -1.0, -1.0]), dt, 40.0)
        self.assertAlmostEqual(tau, 20.0)
        self.assertFalse(fail)
        tau, fail = cleanup_time(np.array([-1.0, -1.0]), dt, 20.0)
        self.assertEqual(tau, 0.0)
        self.assertFalse(fail)
        tau, fail = cleanup_time(np.array([1.0, 0.5, 1.0]), dt, 30.0)
        self.assertAlmostEqual(tau, 30.0)
        self.assertTrue(fail)

    def test_soft_limit(self):
        v = np.array([0.8, 0.4, -0.1, -0.5, -0.8])
        hard, _ = cleanup_time(v, 1.0, 5.0)
        soft, _ = soft_last_violation(v, 1.0, 1e-4)
        self.assertLess(abs(soft - hard), 1e-6)

    def test_soft_derivative(self):
        v = np.array([-0.5, 0.2, 0.8, -0.1, -0.4])
        dt, eps = 2.0, 0.15
        _, d = soft_last_violation(v, dt, eps)
        h = 1e-6
        fd = np.zeros_like(v)
        for i in range(v.size):
            vp = v.copy(); vp[i] += h
            vm = v.copy(); vm[i] -= h
            fd[i] = (soft_last_violation(vp, dt, eps)[0]
                     - soft_last_violation(vm, dt, eps)[0]) / (2 * h)
        self.assertLess(np.max(np.abs(d - fd)) / max(np.max(np.abs(fd)), 1e-12), 1e-6)


class TestCVaR(unittest.TestCase):
    def test_constant(self):
        x = np.full(10, 3.0)
        val, w = cvar_and_weights(x, 0.2)
        self.assertAlmostEqual(val, 3.0)
        self.assertAlmostEqual(w.sum(), 1.0)

    def test_above_mean(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=500)
        self.assertGreater(cvar(x, 0.1), x.mean())

    def test_weights_direction(self):
        x = np.array([5.0, 1.0, -2.0, 0.5, -1.0])
        _, w = cvar_and_weights(x, 0.2)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertEqual(int(np.argmax(w)), int(np.argmax(x)))


class TestObjective(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive = tiny_archive()
        cls.cost = CostConfig()

    def test_cost_decomposition(self):
        q = np.array([2.0, 3.0])
        ev = self.archive.evaluate(q, self.cost)
        for d in ev.details:
            total = d["J_pump"] + d["J_treat"] + d["J_res"] + d["J_time"]
            self.assertAlmostEqual(total, d["J"], places=8)
            self.assertLess(abs(d["mass_balance_error"]), 1e-12)

    def test_monotone_pumping(self):
        ev0 = self.archive.evaluate([0.0, 0.0], self.cost)
        ev1 = self.archive.evaluate([2.0, 2.0], self.cost)
        # more pumping reduces the violation and the residual mass
        self.assertGreater(np.mean(ev0.G), np.mean(ev1.G))
        self.assertGreater(ev0.details[0]["J_res"], ev1.details[0]["J_res"])
        self.assertGreater(ev1.details[0]["J_treat"], ev0.details[0]["J_treat"])

    def test_sigmoid_limits(self):
        z = np.linspace(-3, 3, 11)
        self.assertTrue(np.all(sigmoid(z) + sigmoid(-z) - 1.0 < 1e-12))
        self.assertLess(sigmoid(-50.0), 1e-15)
        self.assertGreater(sigmoid(50.0), 1.0 - 1e-15)


if __name__ == "__main__":
    unittest.main()
