"""Chance-constraint reformulation checks (gate O-4)."""

import unittest

import numpy as np

from stoch.chance import (
    compare_reformulations,
    empirical_probability,
    sigmoid_probability,
)


class TestChance(unittest.TestCase):
    def test_empirical(self):
        G = np.array([-1.0, 0.0, 0.5, 2.0, -3.0])
        self.assertAlmostEqual(empirical_probability(G), 0.6)

    def test_sigmoid_limit(self):
        G = np.array([-2.0, -1.0, 0.5, 3.0])
        p_exact = empirical_probability(G)
        p_small = sigmoid_probability(G, 1e-6)
        self.assertAlmostEqual(p_exact, p_small, places=5)

    def test_compare_reformulations(self):
        # toy: G_m(q) = q - 1 + noise (10 scenarios), J = q^2, exact 90%
        M = 10
        noise = np.linspace(-0.4, 0.4, M)

        def evaluate(q):
            qq = float(q[0])
            return {"J": qq ** 2, "G": 1.0 - qq + noise}

        grid = [np.array([t]) for t in np.linspace(0.0, 2.0, 21)]
        v = compare_reformulations(evaluate, grid, alpha=0.1,
                                   eps_list=(0.1, 0.01))
        self.assertGreaterEqual(v.exact_reliability, 0.9)
        self.assertGreaterEqual(v.cvar_J, v.exact_J - 1e-12)
        self.assertTrue(v.sigmoid_exact_feasible)
        self.assertGreaterEqual(v.exact_q[0], 1.0 - 1e-9)


if __name__ == "__main__":
    unittest.main()
