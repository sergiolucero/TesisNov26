"""External checks on the 3D finite-volume solver (``DOCS/3D_plan.md`` 3D-4).

The 3D stack's own evidence is agreement between its NumPy and batched-torch
paths, which catches divergence but not a *shared* modelling error.  These tests
supply the outside reference:

* **Flow, against Thiem.**  A fully penetrating well in an 8 m slab must draw
  down like a 2D well with transmissivity ``K * lz``; the shape error is compared
  the same way the 2D O0 gate does (``<= 1e-2``).
* **Transport, against the trusted 2D kernel.**  The column case is extruded
  through both solvers and must agree to round-off.  This is the check
  ``3D_plan.md`` asks for ("checked against the 2D kernels by extrusion") and it
  is the one that would have caught the transposed-grid bug in the verifiers.
* **Transport, against analytics.**  Front position against ``v = K grad / phi``
  and the breakthrough spread against ``2 D t`` with ``D = alpha_l v``.

What is *not* claimed here.  The column does not reproduce the Ogata--Banks
*shape* as tightly as the moments: measured through the breakthrough curve, the
effective dispersion of the shared 2D/3D discretization is 0.24, 0.29, 0.31 of
the analytic ``D`` at 1000/2000/3000 d, i.e. the discrete front is steeper than
the analytic one early and converges upward.  Since the 3D solution is bit-equal
to the 2D one, that gap is a property of the shared discretization (first-order
upwind plus implicit Euler), not of the 3D implementation -- so it is documented
rather than gated. A kernel-level improvement (higher-order upwind or Crank--
Nicolson) would have to move both solvers at once.
"""
from __future__ import annotations

import unittest

import numpy as np

from physics import fv2d, fv3d
from physics.references import thiem_drawdown

K0, GRAD, PHI, ALPHA_L = 10.0, 1e-3, 0.3, 10.0
V = K0 * GRAD / PHI                      # pore velocity, m/d
D = ALPHA_L * V                          # longitudinal dispersion, m2/d


def _column_3d(nx: int = 300, alpha_l: float = ALPHA_L, ny: int = 1, nz: int = 1):
    """Uniform flow along x through a zero-transverse-extent column."""
    grid = fv3d.Grid3D(nx, ny, nz, float(nx), float(max(ny, 1)),
                       float(max(nz, 1)))
    aq = fv3d.Aquifer3D(K=np.full(grid.shape, K0), phi=PHI, alpha_l=alpha_l,
                        alpha_th=alpha_l, alpha_tv=alpha_l)
    flow = fv3d.FlowModel3D(grid, aq, h_left=GRAD * grid.lx, h_right=0.0)
    resp = fv3d.flow_responses3d(flow, [(nx - 10.0, 0.5 * grid.ly,
                                         0.5 * grid.lz)], eps_cells=0.5)
    return grid, aq, flow, resp


def _feed_3d(nx: int = 300, nsteps: int = 100, dt: float = 10.0,
             alpha_l: float = ALPHA_L):
    """Continuous injection at the inlet, held at C = 1 by construction.

    ``source_rate = v phi dy dz`` is exactly the advective mass flux that leaves
    the first cell, so the inlet sits at C = 1 without hard-coding a Dirichlet
    row -- the same trick the 2D column uses.
    """
    grid, aq, flow, resp = _column_3d(nx, alpha_l=alpha_l)
    vx, vy, vz = flow.velocity(resp.h0)
    src = V * PHI * grid.dy * grid.dz
    out = fv3d.transport_run3d(grid, aq, flow, resp, [0.0], dt=dt, nsteps=nsteps,
                               source_idx=0, source_rate=src,
                               disp_ref=(vx, vy, vz))
    return grid, out, out.C_final.reshape(grid.ny, grid.nx, grid.nz).mean(axis=(0, 2))


class TestFlowAgainstThiem(unittest.TestCase):
    def test_fully_penetrating_well_reproduces_the_thiem_shape(self):
        grid = fv3d.Grid3D(80, 80, 4, 800.0, 800.0, 8.0)
        aq = fv3d.Aquifer3D(K=np.full(grid.shape, K0), phi=PHI)
        flow = fv3d.FlowModel3D(grid, aq, h_left=0.0, h_right=0.0,
                                bc_y="dirichlet")
        resp = fv3d.flow_responses3d(flow, [(400.0, 400.0, 0.5 * grid.lz)],
                                     eps_cells=0.3, eps_z_cells=8.0)
        q0 = 20.0
        head = resp.head([q0]).reshape(grid.shape).mean(axis=2)
        X, Y, _ = grid.centres()
        r = np.sqrt((X[:, :, 0] - 400.0) ** 2 + (Y[:, :, 0] - 400.0) ** 2)
        sel = (r >= 25.0) & (r <= 100.0)
        order = np.argsort(r.ravel())
        r_ref = 30.0
        head_at_r = float(np.interp(r_ref, r.ravel()[order], head.ravel()[order]))

        transmissivity = K0 * grid.lz
        num = head[sel] - head_at_r
        ana = (thiem_drawdown(q0, transmissivity, r[sel], r_ref=1.0)
               - thiem_drawdown(q0, transmissivity, np.array([r_ref]),
                                r_ref=1.0)[0])
        err = float(np.max(np.abs(num - ana)) / np.max(np.abs(ana)))
        self.assertLess(err, 1e-2, f"Thiem shape error {err:.3g}")


class TestExtrudedColumn(unittest.TestCase):
    def test_seepage_velocity_is_exact(self):
        _grid, _aq, flow, resp = _column_3d(nx=40)
        vx, vy, vz = flow.velocity(resp.h0)
        self.assertAlmostEqual(float(np.mean(vx[:, 1:-1, :])) / V, 1.0, places=12)
        self.assertLess(float(np.abs(vy).max()), 1e-12)
        self.assertLess(float(np.abs(vz).max()), 1e-12)

    def test_head_has_no_transverse_structure(self):
        grid, _aq, flow, resp = _column_3d(nx=40, ny=3, nz=4)
        field = resp.h0.reshape(grid.shape)
        self.assertLess(float(np.ptp(field, axis=(0, 2)).max()), 1e-12)

    def test_matches_the_2d_kernel_by_extrusion(self):
        """The 3D port must reproduce the trusted 2D solve bit for bit."""
        nx = 200
        grid3, out3, prof3 = _feed_3d(nx=nx, nsteps=100)
        _grid, aq3, flow3, resp3 = _column_3d(nx)
        vx3, vy3, vz3 = flow3.velocity(resp3.h0)

        grid2 = fv2d.Grid(nx, 2, float(nx), 2.0)
        aq2 = fv2d.Aquifer(K=np.full((grid2.ny, grid2.nx), K0), b=1.0, phi=PHI,
                           alpha_l=ALPHA_L, alpha_t=ALPHA_L)
        flow2 = fv2d.FlowModel(grid2, aq2, h_left=GRAD * grid2.lx, h_right=0.0)
        resp2 = fv2d.flow_responses(flow2, [(nx - 10.0, 0.5 * grid2.ly)],
                                    eps_cells=0.5)
        vx2, vy2 = flow2.velocity(resp2.h0)
        out2 = fv2d.transport_run(grid2, aq2, flow2, resp2, [0.0], dt=10.0,
                                  nsteps=100, source_idx=0,
                                  source_rate=V * PHI * grid2.dy * 2.0,
                                  disp_ref=(vx2, vy2))
        prof2 = out2.C_final.reshape(grid2.ny, nx).mean(axis=0)
        rel = float(np.linalg.norm(prof2 - prof3) / np.linalg.norm(prof3))
        self.assertLess(rel, 1e-9, f"2D and 3D disagree by {rel:.3g}")

    def test_front_travels_at_the_pore_velocity(self):
        """Pure advection: the 50% crossing must move at ``v``, not near it."""
        t = 2000.0
        grid, _out, prof = _feed_3d(nx=300, nsteps=int(t / 10.0), alpha_l=1e-9)
        x = (np.arange(grid.nx) + 0.5) * grid.dx
        below = np.where(prof < 0.5)[0]
        self.assertGreater(below.size, 1, "the front never crossed half the inlet")
        i = int(below[0])
        x50 = float(np.interp(0.5, [prof[i], prof[i - 1]], [x[i], x[i - 1]]))
        expect = 0.5 * grid.dx + V * t
        # a fixed sub-cell lag (~0.8 dx) is expected from first-order upwind;
        # a wrong porosity or face-area scaling shows up as a proportional error.
        self.assertLess(abs(x50 - expect) / expect, 0.02,
                        f"front at {x50:.2f} m, expected {expect:.2f} m")

    def test_breakthrough_spread_scales_with_dispersion(self):
        """``var(x) = 2 D t`` sets the width; a 2x error in D fails this."""
        t = 2000.0
        grid, _out, prof = _feed_3d(nx=300, nsteps=int(t / 10.0))
        x = (np.arange(grid.nx) + 0.5) * grid.dx
        pdf = np.clip(-np.gradient(prof, grid.dx), 0.0, None)   # breakthrough
        mass = float(pdf.sum() * grid.dx)
        mean = float((pdf * x).sum() * grid.dx / mass)
        var = float((pdf * (x - mean) ** 2).sum() * grid.dx / mass)
        d_eff = var / (2.0 * t)
        self.assertLess(abs(d_eff - D) / D, 0.20,
                        f"effective dispersion {d_eff:.3f} vs {D:.3f} m2/d")

    def test_mass_ledger_closes(self):
        _grid, out, _prof = _feed_3d(nx=200, nsteps=200)
        m = out.mass
        residual = (m["total"][-1] - m["total"][0] + m["well"][-1]
                    + m["outflow"][-1] + m["decay"][-1] - m["source"][-1])
        self.assertLess(abs(float(residual)) / max(abs(float(m["source"][-1])), 1e-12),
                        1e-10, "source mass is not accounted for")


if __name__ == "__main__":
    unittest.main()
