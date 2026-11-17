"""Parity gates for the batched SAA layer (`stoch.saa_torch`).

The contract: with the same scenarios, the same configuration and float64, the
GPU archive must produce the same objective, the same chance statistics and the
same optimizer answer as ``stoch.saa.Archive``.  Anything looser would make the
two backends incomparable, which is exactly what the repository's O-gates are
meant to prevent.
"""

import unittest

import numpy as np

from physics.fv2d import Aquifer, Grid
from stoch.objective import CostConfig
from stoch.objective_torch import cvar_batch, soft_last_violation_batch
from stoch.optimize import SAAProblem, multi_start, pattern_search_batched, project_capacity
from stoch.random_fields import KLExpansion, ScenarioGenerator
from stoch.saa import Archive, ForwardConfig
from stoch.saa_torch import TorchArchive

GRID = Grid(24, 12, 100.0, 50.0)
WELLS = np.array([[40.0, 30.0], [60.0, 20.0], [70.0, 40.0], [50.0, 10.0]])
M_SCEN = 4
NSTEPS = 12
Q0 = np.array([3.0, 5.0, 0.0, 2.0])
PROBLEM_KW = dict(q_max=50.0, Q_max=200.0, alpha=0.05, alpha_h=0.01,
                  chance="cvar", rho_g=1e6, rho_h=1e6)


def build():
    """Reference and batched archives over the same scenario set."""
    aq = Aquifer(K=np.full((GRID.ny, GRID.nx), 10.0), phi=0.30, b=10.0,
                 alpha_l=5.0, alpha_t=0.5)
    kl = KLExpansion(GRID, sigma=0.5, ell_x=20.0, ell_y=20.0, n_kl=6, k0=10.0)
    gen = ScenarioGenerator(GRID, kl)
    scenarios = [gen.sample(1000 + i) for i in range(M_SCEN)]
    fwd = ForwardConfig(
        dt=36.5, nsteps=NSTEPS, grad=1e-3,
        source_xy=(20.0, 25.0), source_q=10.0, source_c=100.0, source_days=365.0,
        compliance_xy=((90.0, 25.0),), obs_xy=((50.0, 25.0),), eps_well=1.5,
    )
    ref = Archive(GRID, aq, WELLS, scenarios, fwd)
    batched = TorchArchive(GRID, aq, WELLS, scenarios, fwd,
                           device="cpu", dtype="float64")
    return ref, batched, fwd


class TestObjectiveParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref, cls.batched, cls.fwd = build()
        cls.cost = CostConfig()
        cls.ev_ref = cls.ref.evaluate(Q0, cls.cost, keep=True)
        cls.ev_t = cls.batched.evaluate(Q0, cls.cost, keep=True)

    def test_cost_and_chance_aggregates(self):
        for name in ("J", "G", "H"):
            a = getattr(self.ev_ref, name)
            b = getattr(self.ev_t, name)
            scale = max(np.abs(a).max(), 1e-30)
            self.assertLess(np.abs(a - b).max() / scale, 1e-7, msg=name)
        self.assertLess(np.abs(self.ev_ref.tau - self.ev_t.tau).max(), 1e-9)
        self.assertTrue(np.array_equal(self.ev_ref.failed, self.ev_t.failed))
        self.assertTrue(np.array_equal(self.ev_ref.compliant, self.ev_t.compliant))

    def test_summary_matches(self):
        a = self.ev_ref.summary()
        b = self.ev_t.summary()
        # Statistics like se_J are differences of large numbers, so they are
        # compared against the objective scale rather than their own.
        atol = 1e-6 * max(abs(a["mean_J"]), 1.0)
        for key, value in a.items():
            if not isinstance(value, float):
                continue
            self.assertLess(abs(value - b[key]), atol, msg=key)

    def test_cost_terms_match(self):
        for m in range(M_SCEN):
            for key in ("J", "J_pump", "J_treat", "J_res", "J_time", "G", "H"):
                self.assertLess(
                    abs(self.ev_ref.details[m][key] - self.ev_t.details[m][key])
                    / max(abs(self.ev_ref.details[m][key]), 1e-12),
                    1e-6,
                    msg=f"{key} scenario {m}",
                )

    def test_mass_balance_within_gate(self):
        tol = 1e-3  # the O2 gate is 1e-2; the reference is ~1e-15
        for d in self.ev_t.details:
            self.assertLess(abs(d["mass_balance_error"]), tol)

    def test_final_field_and_heads(self):
        for m in range(M_SCEN):
            C_ref = self.ev_ref.details[m]["C_final"]
            C_t = self.ev_t.details[m]["C_final"]
            self.assertLess(np.abs(C_ref - C_t).max() / np.abs(C_ref).max(), 1e-8)
            h_ref = self.ev_ref.details[m]["h"]
            h_t = self.ev_t.details[m]["h"]
            self.assertLess(np.abs(h_ref - h_t).max() / max(np.abs(h_ref).max(), 1e-30),
                            1e-9)

    def test_numpy_cross_check_on_subset(self):
        """The GPU archive can reproduce itself through the SciPy path."""
        ev = self.batched.evaluate_numpy(Q0, self.cost)
        # Only the first scenario of the sub-archive is compared.
        self.assertEqual(ev.M, 1)
        self.assertLess(abs(ev.J[0] - self.ev_ref.J[0]) / abs(self.ev_ref.J[0]), 1e-8)

    def test_batch_of_one_equals_single(self):
        ev = self.batched.evaluate_many(np.atleast_2d(Q0), self.cost)
        self.assertLess(np.abs(ev.J[0].numpy() - self.ev_t.J).max(), 1e-6)


class TestBatchedHelpers(unittest.TestCase):
    def test_cvar_matches_reference(self):
        from stoch.objective import cvar

        rng = np.random.default_rng(0)
        for alpha in (0.01, 0.05, 0.25, 0.5, 0.9):
            for _ in range(10):
                x = rng.standard_normal(M_SCEN)
                ref = cvar(x, alpha)
                got = float(cvar_batch(torch_tensor(x).reshape(1, -1), alpha)[0])
                self.assertLess(abs(got - ref), 1e-10, msg=f"alpha={alpha}")

    def test_cvar_degenerate_cases(self):
        # alpha*M < 1 -> worst value; alpha >= 1 -> mean
        import torch

        x = torch.tensor([[3.0, 1.0, 2.0]])
        self.assertAlmostEqual(float(cvar_batch(x, 0.05)[0]), 3.0, places=12)
        self.assertAlmostEqual(float(cvar_batch(x, 1.0)[0]), 2.0, places=12)

    def test_soft_last_violation_matches_reference(self):
        import torch

        from stoch.objective import soft_last_violation

        rng = np.random.default_rng(1)
        v = rng.standard_normal((NSTEPS, 1, 3)) * 0.5
        tau, d = soft_last_violation_batch(
            torch.as_tensor(v, dtype=torch.float64), 36.5, 0.05
        )
        for m in range(3):
            tau_ref, d_ref = soft_last_violation(v[:, 0, m], 36.5, 0.05)
            self.assertLess(abs(tau_ref - float(tau[0, m])), 1e-9)
            self.assertLess(np.abs(d_ref - d[:, 0, m].numpy()).max(), 1e-9)


def torch_tensor(x):
    import torch

    return torch.as_tensor(np.asarray(x, dtype=float), dtype=torch.float64)


class TestOptimizerParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref, cls.batched, cls.fwd = build()
        cls.cost = CostConfig()
        cls.p_ref = SAAProblem(cls.ref, cls.cost, **PROBLEM_KW)
        cls.p_t = SAAProblem(cls.batched, cls.cost, **PROBLEM_KW)

    def test_penalised_objective_matches(self):
        f_ref = self.p_ref.objective(Q0)
        f_t = self.p_t.objective(Q0)
        self.assertLess(abs(f_ref - f_t) / max(abs(f_ref), 1e-12), 1e-7)

    def test_batch_objective_matches_single(self):
        Q = np.stack([Q0, project_capacity(0.5 * Q0, 50.0, 200.0)])
        f_batch, ev = self.p_t.batch_objective(Q)
        for k, q in enumerate(Q):
            self.assertLess(abs(f_batch[k] - self.p_ref.objective(q))
                            / max(abs(f_batch[k]), 1e-12), 1e-7)
        self.assertEqual(tuple(ev.J.shape), (2, M_SCEN))

    def test_batched_search_reaches_the_reference_optimum(self):
        r_ref = multi_start(self.p_ref, n_starts=1, method="pattern", seed=0,
                            max_iter=25, max_evals=120, batched=False)
        r_t = multi_start(self.p_t, n_starts=1, method="pattern", seed=0,
                          max_iter=25, max_evals=120, batched=True)
        self.assertEqual(r_t.method, "pattern-batched")
        # Same neighbourhood: the batched sweep accepts the best of the compass
        # set instead of the first improvement, so the trajectory differs but
        # the objective must agree to solver tolerance.
        self.assertLess(abs(r_ref.fun - r_t.fun) / max(abs(r_ref.fun), 1e-12), 1e-3)
        self.assertLess(np.abs(r_ref.q - r_t.q).max(), 0.25 * PROBLEM_KW["q_max"])

    def test_batched_search_requires_a_batch_capable_archive(self):
        with self.assertRaises(TypeError):
            pattern_search_batched(self.p_ref, Q0, max_iter=1)


class TestBackendSelection(unittest.TestCase):
    def test_make_archive_backends(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "SCRIPTS"))
        import run_optimization as ro

        self.assertIsInstance(ro.make_archive(GRID, Aquifer(K=np.full((GRID.ny, GRID.nx), 10.0)),
                                              WELLS, [], ForwardConfig(), backend="numpy"),
                              Archive)
        aq = Aquifer(K=np.full((GRID.ny, GRID.nx), 10.0))
        with self.assertRaises(ValueError):
            ro.make_archive(GRID, aq, WELLS, [], ForwardConfig(), backend="quantum")

    def test_auto_backend_without_accelerator_is_numpy(self):
        import sys
        from pathlib import Path

        import torch

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "SCRIPTS"))
        import run_optimization as ro

        if not torch.cuda.is_available():
            self.assertEqual(ro.resolve_backend("auto", "auto"), "numpy")
        self.assertEqual(ro.resolve_backend("torch", "cpu"), "torch")


if __name__ == "__main__":
    unittest.main()
