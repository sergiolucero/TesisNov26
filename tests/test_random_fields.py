"""Karhunen--Lo\\`eve / log-normal conductivity checks (gate O-1)."""

import unittest

import numpy as np

from physics.fv2d import Grid
from stoch.random_fields import KLExpansion, ScenarioGenerator


class TestKL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = Grid(40, 20, 200.0, 100.0)
        cls.kl = KLExpansion(cls.grid, sigma=0.5, ell_x=20.0, ell_y=20.0,
                             n_kl=0, target_variance=0.95)

    def test_retained_variance(self):
        self.assertGreaterEqual(self.kl.retained, 0.95)
        self.assertGreater(self.kl.n_kl, 0)
        self.assertLessEqual(self.kl.n_kl, self.grid.nx * self.grid.ny)

    def test_reproducible(self):
        gen = ScenarioGenerator(self.grid, self.kl)
        a = gen.sample(123)
        b = gen.sample(123)
        self.assertTrue(np.array_equal(a.coeffs, b.coeffs))
        self.assertTrue(np.array_equal(a.K, b.K))

    def test_marginal_statistics(self):
        rng = np.random.default_rng(7)
        n = 400
        fields = np.stack([
            self.kl.log_field(self.kl.sample_coefficients(rng)) for _ in range(n)
        ])
        mean = fields.mean()
        var = fields.var(ddof=1)
        se = var * np.sqrt(2.0 / (n - 1))
        target = self.kl.sigma ** 2 * self.kl.retained
        self.assertLess(abs(mean), 5 * self.kl.sigma / np.sqrt(n))
        self.assertLess(abs(var - target), 4 * se)

    def test_covariance_shape(self):
        rng = np.random.default_rng(11)
        n = 300
        fields = np.stack([
            self.kl.log_field(self.kl.sample_coefficients(rng)) for _ in range(n)
        ])
        G = fields - fields.mean(axis=0)
        # pointwise variance and covariance at an x-lag, averaged over y
        c0 = float(np.mean(G * G))
        lag = 20.0
        di = int(round(lag / self.grid.dx))
        clag = float(np.mean(G[:, :, 0] * G[:, :, di]))
        target = self.kl.sigma ** 2 * self.kl.retained * np.exp(-lag / self.kl.ell_x)
        self.assertLess(abs(c0 - self.kl.sigma ** 2 * self.kl.retained),
                        0.15 * self.kl.sigma ** 2)
        self.assertLess(abs(clag - target), 0.30 * target)

    def test_clamp_fraction(self):
        gen = ScenarioGenerator(self.grid, self.kl)
        fracs = [gen.sample(s).clamp_fraction for s in range(20)]
        self.assertLess(max(fracs), 0.01)

    def test_k_bounds(self):
        s = ScenarioGenerator(self.grid, self.kl).sample(3)
        self.assertTrue(np.all(s.K >= self.kl.k_min))
        self.assertTrue(np.all(s.K <= self.kl.k_max))


if __name__ == "__main__":
    unittest.main()
