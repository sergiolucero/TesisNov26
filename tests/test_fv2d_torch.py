"""Parity gates for the batched PyTorch backend against the NumPy reference.

These tests are the contract that lets ``SCRIPTS/run_optimization.py`` swap
backends without changing any physics: the batched solver must reproduce
``physics.fv2d`` on flow, velocity, transport and the mass diagnostics to
solver tolerance, and its stencil must be the exact adjoint of the matrix the
SciPy path assembles.

They run on CPU in float64, so they need no GPU and are fast enough for the
regular suite (a handful of cells, a dozen time steps).
"""

import unittest

import numpy as np
import torch

from physics.fv2d import (
    Aquifer,
    FlowModel,
    Grid,
    flow_responses,
    transport_run,
)
from physics.fv2d_torch import (
    FlowModelTorch,
    advection_stencil,
    default_rtol,
    flow_responses_torch,
    resolve_device,
    torch_dtype,
    transport_batch,
)

GRID = Grid(20, 10, 80.0, 40.0)
WELLS = [(25.0, 20.0), (55.0, 26.0)]
DT, NSTEPS = 5.0, 8
H_LEFT = 0.2


def _fields(n=2, seed=7):
    rng = np.random.default_rng(seed)
    return np.stack([10.0 * np.exp(0.35 * rng.standard_normal((GRID.ny, GRID.nx)))
                     for _ in range(n)])


def _aquifer(K):
    return Aquifer(K=K, phi=0.30, b=10.0, alpha_l=5.0, alpha_t=0.5)


class TestDevicePlumbing(unittest.TestCase):
    def test_resolve_device(self):
        self.assertEqual(resolve_device("cpu").type, "cpu")
        if not torch.cuda.is_available():
            # 'auto' must degrade to CPU rather than raise on a GPU-less box.
            self.assertEqual(resolve_device("auto").type, "cpu")
            with self.assertRaises(RuntimeError):
                resolve_device("cuda")

    def test_dtype_and_tolerance(self):
        self.assertEqual(torch_dtype("float32"), torch.float32)
        self.assertLess(default_rtol(torch.float64), default_rtol(torch.float32))
        with self.assertRaises(ValueError):
            torch_dtype("float16")


class TestFlowParity(unittest.TestCase):
    def setUp(self):
        self.K = _fields(2)
        self.torch_flow = FlowModelTorch(GRID, self.K, phi=0.30, b=10.0,
                                         h_left=H_LEFT, h_right=0.0,
                                         dtype=torch.float64, device="cpu")
        self.torch_resp = flow_responses_torch(self.torch_flow, WELLS, 1.5)
        self.numpy = []
        for K in self.K:
            aq = _aquifer(K)
            flow = FlowModel(GRID, aq, h_left=H_LEFT, h_right=0.0)
            self.numpy.append((aq, flow, flow_responses(flow, WELLS, 1.5)))

    def test_operator_matches_sparse_matrix(self):
        """The stencil must equal the assembled sparse matrix, entry by entry."""
        _, flow, _ = self.numpy[0]
        rng = np.random.default_rng(0)
        h = rng.standard_normal((GRID.ny, GRID.nx))
        ref = flow.A @ h.ravel()
        got = self.torch_flow.stencil().apply(
            torch.as_tensor(h, dtype=torch.float64).unsqueeze(0)
        )[0].numpy().reshape(-1)
        self.assertLess(np.abs(ref - got).max() / np.abs(ref).max(), 1e-12)
        self.assertLess(
            np.abs(flow.A.diagonal() - self.torch_flow.diag[0].numpy().reshape(-1)).max(),
            1e-12,
        )

    def test_background_solve_and_unit_responses(self):
        self.assertTrue(self.torch_resp.report.converged)
        for m, (_, _, resp) in enumerate(self.numpy):
            h0 = self.torch_resp.h0[m].numpy()
            self.assertLess(np.abs(h0 - resp.h0).max() / np.abs(resp.h0).max(), 1e-9)
            hw = self.torch_resp.hw[m].numpy()
            self.assertLess(np.abs(hw - resp.hw).max() / np.abs(resp.hw).max(), 1e-9)

    def test_velocity(self):
        aq, flow, resp = self.numpy[0]
        q = np.array([4.0, 7.0])
        vx_ref, vy_ref = flow.velocity(resp.head(q))
        h = self.torch_resp.head(torch.as_tensor(q, dtype=torch.float64))
        vx, vy = self.torch_flow.velocity(h.reshape(2, GRID.ny, GRID.nx))
        self.assertLess(np.abs(vx[0].numpy() - vx_ref).max() / np.abs(vx_ref).max(), 1e-9)
        self.assertLess(np.abs(vy[0].numpy() - vy_ref).max() /
                        max(np.abs(vy_ref).max(), 1e-30), 1e-9)

    def test_superposition_is_linear(self):
        q = torch.tensor([3.0, -1.0], dtype=torch.float64)
        h = self.torch_resp.head(q)
        manual = (self.torch_resp.h0
                  + 3.0 * self.torch_resp.hw[:, 0]
                  - 1.0 * self.torch_resp.hw[:, 1])
        self.assertLess(float((h - manual).abs().max()), 1e-12)


class TestTransportParity(unittest.TestCase):
    def setUp(self):
        self.K = _fields(2)
        self.torch_flow = FlowModelTorch(GRID, self.K, phi=0.30, b=10.0,
                                         h_left=H_LEFT, h_right=0.0,
                                         dtype=torch.float64, device="cpu")
        self.torch_resp = flow_responses_torch(self.torch_flow, WELLS, 1.5)
        self.q = np.array([4.0, 7.0])
        self.src = GRID.index(10.0, 20.0)
        self.rec = [GRID.index(70.0, 20.0)]
        self.batch = transport_batch(
            GRID, _aquifer(self.K[0]), self.torch_flow, self.torch_resp, self.q,
            DT, NSTEPS, source_idx=self.src, source_rate=1000.0, source_steps=3,
            record_cells=self.rec, store_fields=True,
        )

    def test_matches_reference_transport(self):
        for m, K in enumerate(self.K):
            aq = _aquifer(K)
            flow = FlowModel(GRID, aq, h_left=H_LEFT, h_right=0.0)
            resp = flow_responses(flow, WELLS, 1.5)
            res = transport_run(GRID, aq, flow, resp, self.q, DT, NSTEPS,
                                source_idx=self.src, source_rate=1000.0,
                                source_steps=3, record_cells=self.rec,
                                store_fields=True, return_operator=True)
            C = self.batch.C_fields[:, 0, m, :].numpy()
            self.assertLess(np.abs(C - res.C_fields).max() /
                            np.abs(res.C_fields).max(), 1e-8)
            self.assertLess(np.abs(self.batch.C_well[:, 0, m, :].numpy()
                                   - res.C_well).max(), 1e-8)
            self.assertLess(np.abs(self.batch.C_record[:, 0, m, :].numpy()
                                   - res.C_record).max(), 1e-8)
            for key in ("total", "source", "well", "decay"):
                ref = res.mass[key]
                got = self.batch.mass[key][:, 0, m].numpy()
                self.assertLess(np.abs(got - ref).max() /
                                max(np.abs(ref).max(), 1e-30), 1e-8)
            # Outflow is ~1e-8 of the source mass here, so it is compared
            # against the total injected mass instead of its own scale.
            ref = res.mass["outflow"]
            got = self.batch.mass["outflow"][:, 0, m].numpy()
            self.assertLess(np.abs(got - ref).max() /
                            max(abs(res.mass["source"][-1]), 1e-30), 1e-8)

    def test_global_mass_balance(self):
        """The identity the O2 gate checks, at batched-solver tolerance."""
        mass = self.batch.mass
        bal = (mass["total"][-1] - mass["total"][0]
               - (mass["source"][-1] - mass["well"][-1]
                  - mass["outflow"][-1] - mass["decay"][-1]))
        denom = np.maximum(np.abs(mass["source"][-1].numpy()),
                           np.abs(mass["total"][0].numpy()))
        self.assertLess(float(np.abs(bal.numpy() / denom).max()), 1e-8)

    def test_krylov_report(self):
        report = self.batch.report
        self.assertTrue(report.converged)
        self.assertEqual(report.solves, NSTEPS)
        self.assertLess(report.worst_residual, 1e-9)

    def test_batch_axis_is_independent(self):
        """A candidate in a batch must equal the same candidate run alone."""
        Q = np.stack([self.q, 0.5 * self.q, np.zeros(2)])
        multi = transport_batch(GRID, _aquifer(self.K[0]), self.torch_flow,
                                self.torch_resp, Q, DT, NSTEPS,
                                source_idx=self.src, source_rate=1000.0,
                                source_steps=3, record_cells=self.rec)
        self.assertEqual(tuple(multi.C_well.shape), (NSTEPS, 3, 2, 2))
        self.assertLess(float((multi.C_well[:, 0] - self.batch.C_well[:, 0]).abs().max()),
                        1e-10)
        single = transport_batch(GRID, _aquifer(self.K[0]), self.torch_flow,
                                 self.torch_resp, 0.5 * self.q, DT, NSTEPS,
                                 source_idx=self.src, source_rate=1000.0,
                                 source_steps=3, record_cells=self.rec)
        # Both solves stop at the same residual, so they agree to solver
        # tolerance rather than bit-for-bit.
        self.assertLess(float((multi.C_well[:, 1, 0] - single.C_well[:, 0, 0]).abs().max()),
                        1e-8)


class TestAdvectionStencil(unittest.TestCase):
    """The stencil must be the same operator the SciPy path assembles."""

    def test_upwind_operator_matches_reference(self):
        from physics.fv2d import advection_operator

        rng = np.random.default_rng(3)
        phi_t = np.full((GRID.ny, GRID.nx), 3.0)
        vx = rng.standard_normal((GRID.ny, GRID.nx + 1)) * 0.05
        vy = np.zeros((GRID.ny + 1, GRID.nx))
        c = rng.standard_normal((1, GRID.ny, GRID.nx))

        ref = advection_operator(GRID, vx, vy, phi_t)
        st = advection_stencil(
            GRID,
            torch.as_tensor(phi_t, dtype=torch.float64).unsqueeze(0),
            torch.as_tensor(vx, dtype=torch.float64).unsqueeze(0),
            torch.as_tensor(vy, dtype=torch.float64).unsqueeze(0),
        )
        got = st.apply(torch.as_tensor(c, dtype=torch.float64))[0].numpy()
        expected = (ref @ c[0].ravel()).reshape(GRID.ny, GRID.nx)
        self.assertLess(np.abs(got - expected).max() / np.abs(expected).max(), 1e-12)

        # transpose of the same operator, checked against the sparse transpose
        psi = rng.standard_normal(GRID.ny * GRID.nx)
        got_t = st.apply_transpose(
            torch.as_tensor(psi.reshape(1, GRID.ny, GRID.nx), dtype=torch.float64)
        )[0].numpy().ravel()
        expected_t = ref.T @ psi
        self.assertLess(np.abs(got_t - expected_t).max() / np.abs(expected_t).max(), 1e-12)

    def test_fixed_signs_are_linear_in_the_flux(self):
        """Fixing the upwind pattern makes A(v) linear -- the adjoint premise."""
        rng = np.random.default_rng(5)
        phi_t = torch.full((1, GRID.ny, GRID.nx), 3.0, dtype=torch.float64)
        vx1 = torch.as_tensor(rng.standard_normal((GRID.ny, GRID.nx + 1)),
                              dtype=torch.float64).unsqueeze(0) * 0.05
        vx2 = torch.as_tensor(rng.standard_normal((GRID.ny, GRID.nx + 1)),
                              dtype=torch.float64).unsqueeze(0) * 0.05
        vy = torch.zeros(1, GRID.ny + 1, GRID.nx, dtype=torch.float64)
        sign = torch.sign(vx1)
        c = torch.as_tensor(rng.standard_normal((1, GRID.ny, GRID.nx)),
                            dtype=torch.float64)

        st1 = advection_stencil(GRID, phi_t, vx1, vy, vx_sign=sign)
        st2 = advection_stencil(GRID, phi_t, vx2, vy, vx_sign=sign)
        st12 = advection_stencil(GRID, phi_t, vx1 + vx2, vy, vx_sign=sign)
        lhs = st12.apply(c)
        rhs = st1.apply(c) + st2.apply(c)
        self.assertLess(float((lhs - rhs).abs().max()),
                        1e-12 * max(1.0, float(lhs.abs().max())))


if __name__ == "__main__":
    unittest.main()
