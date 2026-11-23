"""Parity tests for the batched 3D torch backend inlined in
``full3D_colab_run.py``.

The reference is the NumPy/splu path in the same file; these tests assert the
torch path reproduces it to Krylov tolerance on a tiny grid, on CPU, in
float64 -- the same contract ``tests/test_fv2d_torch.py`` enforces for the 2D
backend.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

import full3D_colab_run as colab


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestFV3DTorchParity(unittest.TestCase):
    def setUp(self):
        self.grid = colab.Grid3D(6, 4, 2, 120.0, 80.0, 20.0)
        self.aquifer = colab.Aquifer3D(
            K=np.full((2, 4, 6), 10.0), kv_ratio=0.1, phi=0.30, R=1.0,
            alpha_l=5.0, alpha_th=0.5, alpha_tv=0.05, decay=0.0,
        )
        self.kl = colab.KL3D(self.grid, 0.5, 20.0, 20.0, 6.0, n_kl=0, k0=10.0,
                             kv_ratio=0.1, target_variance=0.95)
        self.fwd = colab.ForwardConfig3D(
            dt=36.5, nsteps=8, grad=1e-3,
            source_xyz=(30.0, 40.0, 10.0), source_q=10.0, source_c=100.0,
            source_days=365.0, compliance_xyz=[(90.0, 40.0, 10.0)],
            obs_xyz=[(60.0, 40.0, 10.0)], eps_well=1.5,
        )
        self.cost = colab.CostConfig3D()
        self.scen = colab.sample_scenarios(self.grid, self.kl, [1000, 1001])
        z_mid = 10.0
        half = 0.5 * 0.3 * 20.0
        self.well_specs = colab.place_wells_3d(
            self.grid, np.array([[60.0, 40.0]]), 3, z_mid + half, z_mid - half,
            seed=0,
        )

    def test_single_candidate_parity(self):
        arch_np = colab.Archive3D(self.grid, self.aquifer, self.well_specs,
                                  self.scen, self.fwd)
        arch_t = colab.TorchArchive3D(self.grid, self.aquifer, self.well_specs,
                                      self.scen, self.fwd, device="cpu",
                                      dtype="float64", rtol=1e-11, maxiter=4000)
        q = np.array([12.5, 0.0, 8.0])
        ev_np = arch_np.evaluate(q, self.cost)
        ev_t = arch_t.evaluate_many(q, self.cost)

        Jn = ev_np.J
        Jt = ev_t.J[0].detach().cpu().numpy()
        np.testing.assert_allclose(Jt, Jn, rtol=1e-8, atol=1e-6)
        np.testing.assert_allclose(ev_t.G[0].detach().cpu().numpy(), ev_np.G,
                                   rtol=1e-8, atol=1e-6)
        np.testing.assert_allclose(ev_t.H[0].detach().cpu().numpy(), ev_np.H,
                                   rtol=1e-8, atol=1e-7)
        np.testing.assert_allclose(ev_t.tau[0].detach().cpu().numpy(), ev_np.tau,
                                   rtol=1e-8, atol=1e-9)

        s_np = ev_np.summary()
        s_t = ev_t.summary()
        self.assertLessEqual(s_t["max_mass_balance_error"], 1e-8)
        np.testing.assert_allclose(s_t["mean_J"], s_np["mean_J"], rtol=1e-8)

        Cfn = ev_np.details[0]["C_final"]
        Cft = ev_t.details["C_final"][0, 0].detach().cpu().numpy()
        np.testing.assert_allclose(Cft, Cfn, rtol=1e-5, atol=1e-7)

    def test_batched_candidates_match_numpy(self):
        arch_np = colab.Archive3D(self.grid, self.aquifer, self.well_specs,
                                  self.scen, self.fwd)
        arch_t = colab.TorchArchive3D(self.grid, self.aquifer, self.well_specs,
                                      self.scen, self.fwd, device="cpu",
                                      dtype="float64", rtol=1e-11, maxiter=4000)
        Q = np.array([[12.5, 0.0, 8.0], [5.0, 10.0, 2.0], [0.0, 0.0, 0.0]])
        ev_t = arch_t.evaluate_many(Q, self.cost)
        Jt = ev_t.J.detach().cpu().numpy()
        for i in range(Q.shape[0]):
            Jn = arch_np.evaluate(Q[i], self.cost).J
            np.testing.assert_allclose(Jt[i], Jn, rtol=1e-8, atol=1e-6)


class TestGridAxesAndWellPlacement(unittest.TestCase):
    """A well must be where it was asked to be, on a grid that cannot hide an
    axis swap.

    Fields are ``(nz, ny, nx)``.  ``Grid3D.centres()`` once returned the
    coordinates as ``(nx, ny, nz)`` because ``meshgrid(x, y, z, indexing="ij")``
    does that, and ``gaussian_well_weights_3d`` ravels those arrays -- so every
    well kernel was written into transposed cells.  The flow solve was then
    self-consistent but *not* the documented geometry: wells that were supposed
    to sit on the plume (and be screened over 30 % of the thickness) sat
    somewhere else entirely.  Nothing in the run output could reveal it, because
    the reported coordinates are the requested ones.
    """

    def test_centres_are_in_field_order(self):
        nx, ny, nz = 8, 4, 2
        g = colab.Grid3D(nx, ny, nz, 8.0, 4.0, 2.0)
        X, Y, Z = g.centres()
        self.assertEqual(X.shape, (nz, ny, nx))
        self.assertEqual(X.shape, (g.nz, g.ny, g.nx))
        # x varies along the last axis, y along the middle, z along the first
        self.assertGreater(np.ptp(X[0, 0, :]), 0.0)
        self.assertLess(np.ptp(X[0, :, 0]), 1e-12)
        self.assertGreater(np.ptp(Y[0, :, 0]), 0.0)
        self.assertLess(np.ptp(Y[:, 0, 0]), 1e-12)
        self.assertGreater(np.ptp(Z[:, 0, 0]), 0.0)
        self.assertLess(np.ptp(Z[0, 0, :]), 1e-12)
        np.testing.assert_allclose(X[0, 0, :], (np.arange(nx) + 0.5) * g.dx)
        np.testing.assert_allclose(Y[0, :, 0], (np.arange(ny) + 0.5) * g.dy)
        np.testing.assert_allclose(Z[:, 0, 0], (np.arange(nz) + 0.5) * g.dz)

    def test_well_kernel_lands_on_the_requested_cell(self):
        g = colab.Grid3D(8, 4, 2, 8.0, 4.0, 2.0)      # dx = dy = dz = 1
        w = colab.gaussian_well_weights_3d(g, 6.5, 2.5, 1.9, 0.1, eps_cells=1.0)
        cx, cy, cz = g.cell_centre(int(np.argmax(w)))
        self.assertAlmostEqual(cx, 6.5, places=6)
        self.assertAlmostEqual(cy, 2.5, places=6)
        self.assertLessEqual(abs(cz - 1.0), 0.5 + 1e-9)

    def test_well_kernel_respects_the_screen_interval(self):
        g = colab.Grid3D(6, 4, 6, 6.0, 4.0, 6.0)      # 6 layers, dz = 1
        w = colab.gaussian_well_weights_3d(g, 3.5, 2.5, 2.4, 1.6,
                                          eps_cells=1.0).reshape(g.nz, g.ny, g.nx)
        column = w[:, 2, 3]
        self.assertGreater(column.sum(), 0.0)
        # layers centred at 1.5 and 2.5 sit inside [1.6, 2.4]?  No: 1.5 < 1.6 and
        # 2.5 > 2.4, so the screen must select neither and the code must fall
        # back to the single mid-screen cell rather than leaking outside it.
        outside = np.concatenate([w[:1], w[3:]])
        self.assertLess(float(np.abs(outside).sum()), 1e-12)

    def test_weights_are_normalised_and_finite(self):
        g = colab.Grid3D(8, 4, 2, 8.0, 4.0, 2.0)
        w = colab.gaussian_well_weights_3d(g, 6.5, 2.5, 1.9, 0.1, eps_cells=1.5)
        self.assertEqual(w.shape, (g.n,))
        self.assertTrue(np.all(np.isfinite(w)))
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
        self.assertGreaterEqual(float(w.min()), 0.0)


if __name__ == "__main__":
    unittest.main()
