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


if __name__ == "__main__":
    unittest.main()
