"""Discrete-adjoint gradient checks (gate O-3)."""

import unittest

import numpy as np

from stoch.adjoint import check_gradient, scenario_G_gradient
from stoch.objective import CostConfig
from tests.helpers import tiny_archive


class TestAdjoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive = tiny_archive()
        cls.cost = CostConfig()

    def test_cost_gradient(self):
        q = np.array([2.0, 3.0])
        direction = np.array([0.3, 0.7])
        res = check_gradient(self.archive, q, self.cost, scenario_index=0,
                             delta=1e-5, direction=direction)
        self.assertLess(res["rel_error"], 1e-6,
                        msg=f"adjoint={res['adjoint']:.8g} fd={res['finite_difference']:.8g}")

    def test_g_gradient(self):
        q = np.array([2.0, 3.0])
        direction = np.array([0.3, 0.7])
        ev = self.archive.evaluate(q, self.cost, keep=True)
        d = ev.details[0]
        prob = self.archive.problems[0]
        grad = scenario_G_gradient(self.archive.grid, prob, q, d["res"], d, self.cost)
        adj = float(np.dot(grad, direction))
        h = 1e-5
        Gp = self.archive.evaluate(q + h * direction, self.cost).G[0]
        Gm = self.archive.evaluate(q - h * direction, self.cost).G[0]
        fd = (Gp - Gm) / (2 * h)
        self.assertLess(abs(adj - fd) / max(abs(fd), 1e-12), 1e-4,
                        msg=f"adjoint={adj:.8g} fd={fd:.8g}")

    def test_two_wells(self):
        q = np.array([1.0, 4.0])
        res = check_gradient(self.archive, q, self.cost, scenario_index=1,
                             delta=1e-5, direction=np.array([1.0, 0.0]))
        self.assertLess(res["rel_error"], 1e-5)


if __name__ == "__main__":
    unittest.main()
