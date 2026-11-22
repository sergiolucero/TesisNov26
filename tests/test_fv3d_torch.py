#!/usr/bin/env python3
"""Parity and hardening tests for the 3D torch backend (full3D_colab_run.py).

The 3D batch path (`TorchArchive3D` / `transport_batch3d` / `cg3d` /
`bicgstab3d`) drives every GPU run, so it needs the same regression gate the 2D
backend has: parity against the NumPy reference on flow heads and the objective,
batch-axis independence, and the numeric hardenings that were each added in
response to a real failure (float32 reduction floor, `max(0.0, nan)`, a stall
that burns maxiter, a nan objective that quietly becomes `best=nan`).

Everything runs on CPU in float64 unless a test says otherwise; the grid is
small enough that the whole file costs seconds.
"""

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

f3d = importlib.util.spec_from_file_location("f3d_under_test",
                                             ROOT / "full3D_colab_run.py")
f3d_mod = importlib.util.module_from_spec(f3d)
assert f3d.loader is not None
sys.modules["f3d_under_test"] = f3d_mod
f3d.loader.exec_module(f3d_mod)

try:
    import torch
except ImportError:                                  # pragma: no cover
    torch = None


def _problem(nx=12, ny=8, nz=4, M=2, nsteps=6, n_kl=6):
    """A small but genuinely heterogeneous 3D problem, both backends share it."""
    grid = f3d_mod.Grid3D(nx, ny, nz, 120.0, 80.0, 12.0)
    aq = f3d_mod.Aquifer3D(K=np.full((nz, ny, nx), 10.0), kv_ratio=0.1, phi=0.3,
                           R=1.0, alpha_l=4.0, alpha_th=0.4, alpha_tv=0.04,
                           decay=0.0)
    kl = f3d_mod.KL3D(grid, 0.5, 20.0, 20.0, 6.0, n_kl=n_kl, k0=10.0,
                      kv_ratio=0.1, target_variance=0.95)
    fwd = f3d_mod.ForwardConfig3D(
        dt=36.5, nsteps=nsteps, grad=1e-3, source_xyz=(20.0, 40.0, 6.0),
        source_q=10.0, source_c=100.0, source_days=365.0,
        compliance_xyz=[(110.0, 40.0, 6.0)], obs_xyz=[(60.0, 40.0, 6.0)],
        eps_well=1.5)
    scen = f3d_mod.sample_scenarios(grid, kl, [500000 + i for i in range(M)])
    wells = np.array([[60.0, 40.0, 7.0, 5.0], [80.0, 50.0, 7.0, 5.0]])
    cost = f3d_mod.CostConfig3D()
    return dict(grid=grid, aq=aq, wells=wells, scen=scen, fwd=fwd, cost=cost,
                q=np.array([12.5, 3.0]))


def rel(a, b) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    scale = max(np.abs(a).max(), np.abs(b).max(), 1e-30)
    return float(np.abs(a - b).max() / scale)


@unittest.skipIf(torch is None, "torch not installed")
class TestBackendParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = _problem()
        cls.p = p
        cls.ref = f3d_mod.Archive3D(p["grid"], p["aq"], p["wells"], p["scen"],
                                    p["fwd"])
        cls.ev_ref = cls.ref.evaluate(p["q"], p["cost"])
        cls.arch = f3d_mod.TorchArchive3D(p["grid"], p["aq"], p["wells"],
                                          p["scen"], p["fwd"], device="cpu",
                                          dtype="float64")
        cls.ev_t = cls.arch.evaluate(p["q"], p["cost"])

    def test_objective_parity(self):
        self.assertLess(rel(self.ev_ref.J, self.ev_t.J), 1e-8)
        self.assertLess(rel(np.asarray(self.ev_ref.G), np.asarray(self.ev_t.G)), 1e-8)
        self.assertLess(rel(self.ev_ref.H, self.ev_t.H), 1e-8)

    def test_cleanup_and_failure_flags_identical(self):
        np.testing.assert_allclose(np.asarray(self.ev_t.tau),
                                   np.asarray(self.ev_ref.tau), rtol=0, atol=0)
        np.testing.assert_array_equal(np.asarray(self.ev_t.failed),
                                      np.asarray(self.ev_ref.failed))

    def test_cost_components_and_field_parity(self):
        for key in ("J_pump", "J_treat", "J_res", "J_time"):
            a = [d[key] for d in self.ev_ref.details]
            b = [d[key] for d in self.ev_t.details]
            self.assertLess(rel(a, b), 1e-7, key)
        for m in range(len(self.ev_ref.details)):
            self.assertLess(rel(self.ev_ref.details[m]["C_final"],
                                self.ev_t.details[m]["C_final"]), 1e-8, f"scen {m}")

    def test_heads_match_superposition(self):
        """h(q) = h0 + q @ hw on both backends, and the numpy A matches."""
        nw = self.p["wells"].shape[0]
        q = np.asarray(self.p["q"], dtype=float)
        h_t = self.arch.resp.head(torch.as_tensor(q, dtype=torch.float64)).numpy()
        h0 = np.stack([pr.resp.h0 for pr in self.ref.problems])
        hw = np.stack([pr.resp.hw for pr in self.ref.problems])
        h_np = h0 + np.einsum("si,min->smn", q[None, :], hw.reshape(len(self.ref.problems), nw, -1))
        self.assertLess(rel(h_np, h_t.reshape(h_np.shape)), 1e-8)

    def test_mass_ledger_is_at_noise_level(self):
        """Absolute, not relative: the NumPy error is ~1e-15 and torch ~1e-11."""
        err = max(abs(d["mass_balance_error"]) for d in self.ev_t.details)
        self.assertLess(err, 1e-6)

    def test_float32_tracks_float64(self):
        a32 = f3d_mod.TorchArchive3D(self.p["grid"], self.p["aq"], self.p["wells"],
                                     self.p["scen"], self.p["fwd"], device="cpu",
                                     dtype="float32")
        ev = a32.evaluate(self.p["q"], self.p["cost"])
        self.assertLess(rel(self.ev_ref.J, ev.J), 1e-4)
        self.assertTrue(np.all(np.isfinite(np.asarray(ev.J))))


@unittest.skipIf(torch is None, "torch not installed")
class TestBatchAxis(unittest.TestCase):
    def test_row_zero_of_a_batch_equals_the_single_evaluation(self):
        p = _problem(M=2, nsteps=3)
        arch = f3d_mod.TorchArchive3D(p["grid"], p["aq"], p["wells"], p["scen"],
                                      p["fwd"], device="cpu", dtype="float64")
        q = np.asarray(p["q"], dtype=float)
        Q = np.stack([q, q * 0.5, np.zeros_like(q)])
        batch = arch.evaluate_many(Q, p["cost"])
        single = arch.evaluate(q, p["cost"])
        # Batching changes the memory layout, hence the reduction order: expect
        # float-level agreement (~1e-10 relative), not bit equality.
        np.testing.assert_allclose(batch.J[0].numpy(), np.asarray(single.J),
                                   rtol=1e-9, atol=1e-6)
        np.testing.assert_allclose(batch.J[2].numpy(),
                                   arch.evaluate(Q[2], p["cost"]).J,
                                   rtol=1e-9, atol=1e-6)

    def test_zero_pumping_is_not_an_error(self):
        """A degenerate candidate (all wells shut) must still evaluate."""
        p = _problem(M=1, nsteps=3)
        arch = f3d_mod.TorchArchive3D(p["grid"], p["aq"], p["wells"], p["scen"],
                                      p["fwd"], device="cpu", dtype="float64")
        ev = arch.evaluate(np.zeros(p["wells"].shape[0]), p["cost"])
        self.assertTrue(np.all(np.isfinite(np.asarray(ev.J))))
        self.assertLess(rel(ev.J, self.ref_J(p, np.zeros(p["wells"].shape[0]))), 1e-8)

    @staticmethod
    def ref_J(p, q):
        ref = f3d_mod.Archive3D(p["grid"], p["aq"], p["wells"], p["scen"], p["fwd"])
        return ref.evaluate(q, p["cost"]).J


@unittest.skipIf(torch is None, "torch not installed")
class TestKrylovHardening(unittest.TestCase):
    def test_float32_scalars_are_computed_wide(self):
        """A float32 dot over ~1e5 similar terms loses ~1e-7 relative; the
        float64 promotion keeps the result within float32 rounding of truth."""
        x = torch.full((1, 20000), 1e3, dtype=torch.float32)
        got = float(f3d_mod._t_dot(x, x).item())
        self.assertAlmostEqual(got / (20000 * 1e6), 1.0, places=5)
        self.assertEqual(f3d_mod._t_dot(x, x).dtype, torch.float32)
        self.assertEqual(f3d_mod._t_norm(x).dtype, torch.float32)

    def test_nan_residual_is_not_swallowed(self):
        """max(0.0, nan) == 0.0 would report a converged solve with residual 0."""
        rep = f3d_mod.TorchKrylovReport()
        rep.add(f3d_mod.TorchKrylovReport(iterations=5, solves=1, worst_iterations=5,
                                          worst_residual=float("nan"),
                                          converged=False))
        self.assertTrue(math.isnan(rep.worst_residual))
        self.assertFalse(rep.converged)

    def test_stall_guard_fires_only_without_a_new_best(self):
        """STALL_WINDOW consecutive iterations with no new best, no more."""
        g = f3d_mod._Stall3D()
        calls = 1                                  # the first call sets the best
        while not g.check(1e-3):
            calls += 1
            self.assertLess(calls, 5 * f3d_mod.STALL_WINDOW, "never stalled")
        self.assertEqual(calls, f3d_mod.STALL_WINDOW + 1)

    def test_stall_guard_does_not_fire_while_improving(self):
        """Slow, steady convergence must never be mistaken for a stall."""
        g = f3d_mod._Stall3D()
        r = 1.0
        for _ in range(10 * f3d_mod.STALL_WINDOW):
            r *= 0.99
            self.assertFalse(g.check(r))

    def test_non_finite_residual_stalls_immediately(self):
        self.assertTrue(f3d_mod._Stall3D().check(float("nan")))

    def test_report_exposes_the_stalled_flag(self):
        d = f3d_mod.TorchKrylovReport(stalled=True).as_dict()
        self.assertTrue(d["stalled"])
        self.assertFalse(f3d_mod.TorchKrylovReport().as_dict()["stalled"])

    def test_float32_default_tolerance_is_reachable(self):
        self.assertEqual(f3d_mod.torch_default_rtol(torch.float64), 1e-11)
        self.assertEqual(f3d_mod.torch_default_rtol(torch.float32), 1e-5)

    def test_stalled_solve_is_bounded(self):
        """A deliberately unconvergeable system must stop near the window, not
        at maxiter (the failure that cost a GPU session 4000 iterations/step)."""
        n = 40
        A = torch.eye(n, dtype=torch.float64)
        A[0, 0] = 0.0                              # singular -> no convergence
        x_star = torch.randn(n, 1, dtype=torch.float64)
        b = (A @ x_star).reshape(1, n)

        def matvec(v):
            return (A @ v.reshape(n, 1)).reshape(-1, n)

        _x, rep = f3d_mod.cg3d(matvec, b, rtol=1e-14, maxiter=3000)
        self.assertLess(rep.iterations, 3 * f3d_mod.STALL_WINDOW)


@unittest.skipIf(torch is None, "torch not installed")
class TestFiniteGuard(unittest.TestCase):
    def test_finite_values_pass(self):
        f3d_mod.finite_or_raise3d([1.0, 2.0], "objective")

    def test_nan_raises_with_actionable_text(self):
        with self.assertRaises(RuntimeError) as ctx:
            f3d_mod.finite_or_raise3d([1.0, float("nan")], "objective", "hint")
        msg = str(ctx.exception)
        self.assertIn("not finite", msg)
        self.assertIn("candidate(s) [1]", msg)
        self.assertIn("hint", msg)
        self.assertIn("krylov", msg.lower())

    def test_inf_raises(self):
        with self.assertRaises(RuntimeError):
            f3d_mod.finite_or_raise3d([float("inf")], "objective")

    def test_the_problem_actually_calls_the_guard(self):
        p = _problem(M=1, nsteps=2)
        arch = f3d_mod.TorchArchive3D(p["grid"], p["aq"], p["wells"], p["scen"],
                                      p["fwd"], device="cpu", dtype="float64")
        prob = f3d_mod.SAAProblemTorch3D(arch, p["cost"], q_max=50.0, Q_max=200.0)

        def poisoned(Q, cost, keep=False):
            m = arch.M
            nan = torch.full((np.atleast_2d(Q).shape[0], m), float("nan"),
                             dtype=torch.float64)
            return SimpleNamespace(mean_J=nan[:, 0], G=nan, H=-nan,
                                   tau=nan, failed=nan)

        arch.evaluate_many = poisoned
        with self.assertRaises(RuntimeError):
            prob.objective(p["q"])

    def test_transport_health_is_tracked(self):
        p = _problem(M=1, nsteps=2)
        arch = f3d_mod.TorchArchive3D(p["grid"], p["aq"], p["wells"], p["scen"],
                                      p["fwd"], device="cpu", dtype="float64")
        arch.evaluate(p["q"], p["cost"])
        self.assertEqual(arch.krylov.solves, arch.fwd.nsteps)
        self.assertTrue(arch.krylov.converged)
        self.assertFalse(arch.krylov.stalled)
        self.assertIn("transport", arch.krylov_hint())


class TestRunsWithoutTorch(unittest.TestCase):
    """The numpy path must not need torch at all (the 2D lesson)."""

    def test_numpy_archive_is_independent_of_the_torch_classes(self):
        p = _problem(M=1, nsteps=2)
        ref = f3d_mod.Archive3D(p["grid"], p["aq"], p["wells"], p["scen"], p["fwd"])
        ev = ref.evaluate(p["q"], p["cost"])
        self.assertTrue(np.all(np.isfinite(np.asarray(ev.J))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
