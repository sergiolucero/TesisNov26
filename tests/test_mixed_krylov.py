"""Mixed-precision BiCGSTAB: fp64 recurrence, float32 operator.

Motivation (measured on a T4, 160x96, M=100): the pure-float32 batched transport
lost a candidate to a NaN residual after 28 iterations. Promoting only the
Krylov *scalars* fixed M=40 but not M=100, so the recurrence itself now runs in
float64 while the stencil matvec -- the cost centre -- stays float32.
"""
from __future__ import annotations

import unittest

import torch

import physics.fv2d_torch as fvt


def _hard_system(n: int = 300, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """A moderately ill-conditioned SPD system plus a known exact solution."""
    gen = torch.Generator().manual_seed(seed)
    u = torch.randn(n, n, generator=gen, dtype=torch.float64)
    s = torch.logspace(0.0, -7.0, n, dtype=torch.float64)
    a = (u * s) @ u.T + 1e-2 * torch.eye(n, dtype=torch.float64)
    a = 0.5 * (a + a.T)
    x_true = torch.randn(3, n, generator=gen, dtype=torch.float64)
    return a, x_true @ a.T, x_true


class TestMixedPrecision(unittest.TestCase):
    def setUp(self):
        self._saved = fvt.MIXED_KRYLOV

    def tearDown(self):
        fvt.MIXED_KRYLOV = self._saved

    def test_float64_path_is_untouched(self):
        # Smaller system: cond ~1e7 at n=300 cannot reach 1e-11 in 800 iterations
        # at all (it stalls near 7e-10), which is a property of the matrix, not of
        # the mixed-precision switch we are testing.
        a, b, x_true = _hard_system(n=80)
        x, rep = fvt.bicgstab_solve(lambda q: q @ a.T, b, rtol=1e-10, maxiter=2000,
                                    mixed_precision=True)
        self.assertEqual(x.dtype, torch.float64)   # no cast, no promotion
        # Solution error is bounded by cond(A) ~ 1e7 times the residual tolerance
        # (1e-11), so ~1e-4; measured 2.3e-6.
        self.assertLess(float((x - x_true).norm() / x_true.norm()), 1e-4)
        self.assertTrue(rep.converged)
        self.assertFalse(rep.stalled)

    def test_mixed_mode_returns_caller_dtype(self):
        a, b, _ = _hard_system()
        x, _rep = fvt.bicgstab_solve(lambda q: q @ a.to(torch.float32).T,
                                     b.to(torch.float32), rtol=1e-6, maxiter=500,
                                     mixed_precision=True)
        self.assertEqual(x.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(x).all()))

    def test_mixed_beats_plain_float32_recurrence(self):
        """On a hard system the fp64 recurrence must reach a lower residual."""
        a, b, _ = _hard_system()
        a32, b32 = a.to(torch.float32), b.to(torch.float32)
        _x_plain, plain = fvt.bicgstab_solve(lambda q: q @ a32.T, b32, rtol=1e-8,
                                            maxiter=800, mixed_precision=False)
        _x_mixed, mixed = fvt.bicgstab_solve(lambda q: q @ a32.T, b32, rtol=1e-8,
                                             maxiter=800, mixed_precision=True)
        # Measured locally: plain 1.0e-04, mixed 1.8e-05 (the float32 *operator*
        # noise floor stops both short of rtol; what matters is that the fp64
        # recurrence gets several times further).
        self.assertLess(mixed.worst_residual, 0.5 * plain.worst_residual)

    def test_switch_defaults_to_module_constant(self):
        a, b, _ = _hard_system(n=80)
        fvt.MIXED_KRYLOV = True
        x, _ = fvt.bicgstab_solve(lambda q: q @ a.to(torch.float32).T,
                                  b.to(torch.float32), rtol=1e-6, maxiter=200)
        self.assertTrue(bool(torch.isfinite(x).all()))
        fvt.MIXED_KRYLOV = False
        x, _ = fvt.bicgstab_solve(lambda q: q @ a.to(torch.float32).T,
                                  b.to(torch.float32), rtol=1e-6, maxiter=200)
        self.assertTrue(bool(torch.isfinite(x).all()))


if __name__ == "__main__":
    unittest.main()
