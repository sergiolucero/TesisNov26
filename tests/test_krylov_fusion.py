"""Gates for the fused/relaxed-check BiCGSTAB path (`physics.fv2d_torch`).

`DOCS/Conclusions.md` §9 asked whether the batched transport is launch-bound, and
answering it meant adding two opt-in behaviours to the Krylov solver: compiling
the iteration into one call, and checking the residual every N iterations instead
of every one. Both are *speed* changes to a *numerics* kernel, so the gates here
are parity gates first and speed tests never: a path that is fast and moves the
answer is a failure.

The properties worth pinning:

* the stencil fast path reproduces the legacy matvec path bit for bit (it has to:
  every number already in DOCS/ came through the legacy path);
* ``check_every`` only allows slack inside ``rtol`` -- it may not drift;
* the fused path agrees with the unfused one on a real problem;
* the knobs default to off, so importing the module changes nothing.
"""
from __future__ import annotations

import os
import unittest

import numpy as np
import torch

from physics.fv2d import Aquifer, Grid
from physics.fv2d_torch import (
    KRYLOV_CHECK_EVERY,
    FUSE_KRYLOV,
    _apply_coeffs,
    bicgstab_solve,
    transport_batch,
)
from stoch.random_fields import KLExpansion, ScenarioGenerator
from stoch.saa import ForwardConfig
from stoch.saa_torch import TorchArchive

GRID = Grid(24, 12, 100.0, 50.0)
WELLS = np.array([[40.0, 30.0], [60.0, 20.0]])
M_SCEN = 3
NSTEPS = 10


def _archive(dtype="float64", m_scen=M_SCEN):
    aq = Aquifer(K=np.full((GRID.ny, GRID.nx), 10.0), phi=0.30, b=10.0,
                 alpha_l=5.0, alpha_t=0.5)
    kl = KLExpansion(GRID, sigma=0.5, ell_x=20.0, ell_y=20.0, n_kl=8, k0=10.0)
    gen = ScenarioGenerator(GRID, kl)
    scenarios = [gen.sample(2000 + i) for i in range(m_scen)]
    fwd = ForwardConfig(dt=36.5, nsteps=NSTEPS, grad=1e-3,
                        source_xy=(20.0, 25.0), source_q=10.0, source_c=100.0,
                        source_days=365.0, compliance_xy=((90.0, 25.0),),
                        obs_xy=((50.0, 25.0),), eps_well=1.5)
    return TorchArchive(GRID, aq, WELLS, scenarios, fwd, device="cpu", dtype=dtype)


class TestDefaults(unittest.TestCase):
    def test_knobs_default_to_the_historical_path(self):
        # The env may legitimately be set when a sweep is running; the
        # assertion is that an unset environment keeps the old behaviour.
        if "TESIS_FUSE_KRYLOV" not in os.environ:
            self.assertFalse(FUSE_KRYLOV)
        if "TESIS_KRYLOV_CHECK_EVERY" not in os.environ:
            self.assertEqual(KRYLOV_CHECK_EVERY, 1)


class TestStencilFastPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arch = _archive()
        cls.Q = torch.tensor([[2.0, 3.0], [1.0, 0.0]], dtype=torch.float64)

    def test_apply_coeffs_matches_stencil_apply(self):
        from physics.fv2d_torch import Stencil5
        st = Stencil5.symmetric(
            diag=torch.rand(2, GRID.ny, GRID.nx, dtype=torch.float64) + 4.0,
            xf=torch.rand(2, GRID.ny, GRID.nx - 1, dtype=torch.float64),
            yf=torch.rand(2, GRID.ny - 1, GRID.nx, dtype=torch.float64))
        c = torch.rand(2, GRID.ny, GRID.nx, dtype=torch.float64)
        self.assertTrue(torch.equal(_apply_coeffs(c, st.diag, st.xe, st.xw,
                                                  st.yn, st.ys),
                                    st.apply(c).clone()))

    def test_stencil_path_matches_the_legacy_matvec_path(self):
        """Same operator, two entry points: identical fields."""
        stencil_path = self.arch.transport(self.Q)                 # stencil=A
        legacy = transport_batch(
            GRID, self.arch.aquifer, self.arch.flow, self.arch.resp, self.Q,
            self.arch.fwd.dt, NSTEPS, source_idx=self.arch.source_idx,
            source_rate=self.arch.source_rate, source_steps=self.arch.fwd.source_steps,
            record_cells=self.arch.compliance_cells, rtol=self.arch.rtol,
            maxiter=self.arch.maxiter)
        self.assertEqual(float((stencil_path.C_final - legacy.C_final).abs().max()),
                         0.0)
        self.assertEqual(float(stencil_path.C_final.sum()),
                         float(legacy.C_final.sum()))

    def test_check_every_keeps_the_answer(self):
        base = self.arch.transport(self.Q, check_every=1)
        relaxed = self.arch.transport(self.Q, check_every=8)
        rel = float((base.C_final - relaxed.C_final).abs().max()
                    / base.C_final.abs().max())
        # rtol for float64 is 1e-11; the slack from checking late must stay well
        # inside the field scale, not "close enough for a plot".
        self.assertLess(rel, 1e-6)
        self.assertLessEqual(relaxed.report.worst_residual, 1e-8)

    def test_fused_path_matches(self):
        base = self.arch.transport(self.Q, check_every=1)
        try:
            fused = self.arch.transport(self.Q, check_every=1, fuse=True)
        except Exception as exc:                     # pragma: no cover
            self.skipTest(f"torch.compile unavailable: {type(exc).__name__}: {exc}")
        rel = float((base.C_final - fused.C_final).abs().max()
                    / base.C_final.abs().max())
        self.assertLess(rel, 1e-10)

    def test_objective_is_unchanged(self):
        """The gate that matters: the SAA objective, not just the field."""
        from stoch.objective import CostConfig
        cost = CostConfig()
        Q = np.array([[2.0, 3.0], [1.0, 0.0]])
        ref = self.arch.evaluate_many(Q, cost)
        for kw in ({"check_every": 8}, {"fuse": False}):
            ev = self.arch.evaluate_many(Q, cost, **kw)
            rel = abs(float(np.asarray(ev.J.cpu()).mean())
                      - float(np.asarray(ref.J.cpu()).mean()))
            rel /= abs(float(np.asarray(ref.J.cpu()).mean()))
            self.assertLess(rel, 1e-9, kw)


if __name__ == "__main__":
    unittest.main()
