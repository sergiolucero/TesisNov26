r"""External checks on the 3D finite-volume solver (``DOCS/3D_plan.md`` 3D-4).

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

* **Transport, against Ogata--Banks, with the right inlet.**  A mass-flux inlet is
  not a Dirichlet inlet, and the difference is large when ``v x / D`` is large:
  the classical formula carries a reflected term (:math:`e^{vx/D}\operatorname{erfc} b`)
  that a pure advective inlet has no analogue for, so the correct reference is the
  swept front :math:`\tfrac12 C_0 \operatorname{erfc}((x - vt)/2\sqrt{Dt})`.
  The measured interior relative L2 against that reference is 1.1e-2 at
  ``dx = 1 m`` and 4.0e-3 at ``dx = 0.5 m``, first order in ``dx``, and it drops
  to 3.4e-3 once the known upwind dispersion ``v dx / 2`` is added to ``D`` --
  which is what turns it second order and tells us the discretisation is the
  only error left.  Earlier versions of this file compared against the Dirichlet
  formula and reported a 0.41 "shape error" that did not converge; that was the
  wrong reference, not a bug, and no amount of amplitude fitting could fix it
  because the difference is a shift of ``D/v`` (5 m here), not a scale factor.

What is *not* claimed here.  Spatial *moments* are exact to the level of the
upwind correction (a pulse's variance grows as ``2Dt`` with
``D_eff = alpha_l v + v dx/2``, and the transverse moments grow at
``alpha_t v``), the flux-inlet front matches the swept front at the resolutions
above, and the 3D solution is bit-equal to the trusted 2D one.  A kernel-level
improvement (higher-order upwind or Crank--Nicolson) would still have to move
both solvers at once.
"""
from __future__ import annotations

import unittest

import numpy as np

from physics import fv2d, fv3d
from physics.references import ogata_banks, thiem_drawdown
from scipy.special import erfc

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


def _swept_front(x, t, v, d_eff, c0=1.0):
    """Analytic solution for a mass-flux (Cauchy) inlet: the swept front."""
    s = 2.0 * np.sqrt(max(d_eff * t, 1e-30))
    return 0.5 * c0 * erfc((x - v * t) / s)


def _flux_inlet_prof(nx, alpha_l=ALPHA_L, ny=1, nz=1, cfl=0.4):
    """Plane mass-flux inlet held at C = 1 by construction, ``dx = 1 m``."""
    grid = fv3d.Grid3D(nx, ny, nz, float(nx), float(max(ny, 1)),
                       float(max(nz, 1)))
    aq = fv3d.Aquifer3D(K=np.full(grid.shape, K0), phi=PHI, alpha_l=alpha_l,
                        alpha_th=alpha_l, alpha_tv=alpha_l)
    flow = fv3d.FlowModel3D(grid, aq, h_left=GRAD * grid.lx, h_right=0.0)
    resp = fv3d.flow_responses3d(flow, [(grid.lx - 10.0, 0.5 * grid.ly,
                                         0.5 * grid.lz)], eps_cells=0.5)
    flow.release_factorization()
    idx = np.arange(grid.n).reshape(grid.ny, grid.nx, grid.nz)
    inlet = idx[:, 0, :].ravel()
    dt = cfl * grid.dx / V
    nsteps = int(round((0.5 * grid.lx / V) / dt))
    res = fv3d.transport_run3d(
        grid, aq, flow, resp, [0.0], dt=dt, nsteps=nsteps,
        source_idx=inlet,
        source_rate=V * PHI * grid.dy * grid.dz * inlet.size,
        store_fields=True,
    )
    prof = res.C_final.reshape(grid.ny, grid.nx, grid.nz).mean(axis=(0, 2))
    return (np.arange(nx) + 0.5) * grid.dx, prof, nsteps * dt, res


class TestOgataBanksFluxInlet(unittest.TestCase):
    """G3.1: the discrete front against the analytic front, and its order.

    The reference is the swept front, not the classical Ogata--Banks formula:
    this solver injects a mass *flux* at the inlet, which has no upstream
    dispersive leak, so the reflected ``exp(vx/D) erfc(b)`` term of the
    Dirichlet problem does not belong in the comparison.  At ``lx = 240,
    v = K grad / phi, D = 10 v`` the two differ by a shift of ``D/v = 10`` cells,
    which is why the previously reported "0.41 shape error" never converged.
    """

    def _errors(self, nx):
        x, prof, t, _res = _flux_inlet_prof(nx)
        core = (x > 2.0) & (x < nx - 2.0)
        norm = float(np.linalg.norm(prof[core]))
        e_swept = float(np.linalg.norm(prof[core] - _swept_front(x, t, V, D)[core])
                        / norm)
        e_ob = float(np.linalg.norm(
            prof[core] - ogata_banks(x, t, V, D, C0=1.0)[core]) / norm)
        return e_swept, e_ob, x, prof, t

    def test_front_matches_the_analytic_swept_front(self):
        e_swept, _e_ob, _x, _p, _t = self._errors(480)
        self.assertLess(e_swept, 1.5e-2,
                        f"interior rel-L2 {e_swept:.3g} is outside the G3.1 gate")

    def test_error_is_first_order_in_dx(self):
        e_coarse, _o, _x, _p, _t = self._errors(240)
        e_fine, _o2, _x2, _p2, _t2 = self._errors(480)
        self.assertLess(e_fine, 0.65 * e_coarse,
                        f"halving dx moved the error {e_coarse:.3g} -> {e_fine:.3g}")

    def test_upwind_dispersion_explains_the_residual(self):
        """Adding the known ``v dx / 2`` to D must not make the fit worse."""
        x, prof, t, _res = _flux_inlet_prof(480)
        core = (x > 2.0) & (x < x[-1] - 2.0)
        norm = float(np.linalg.norm(prof[core]))
        e_phys = float(np.linalg.norm(prof[core] - _swept_front(x, t, V, D)[core])
                       / norm)
        e_num = float(np.linalg.norm(
            prof[core] - _swept_front(x, t, V, D + 0.5 * V * (x[1] - x[0]))[core])
            / norm)
        self.assertLessEqual(e_num, 1.05 * e_phys,
                             "the residual is not explained by upwind dispersion")

    def test_the_dirichlet_formula_is_the_wrong_reference(self):
        """Guard the reason, not just the number: OB must be visibly worse.

        If someone "fixes" this test by switching to the classical formula, the
        inlet type has been misread -- the flux inlet sits ``D/v`` ahead of the
        Dirichlet front, and that shift does not shrink with ``dx``.
        """
        e_swept, e_ob, _x, _p, _t = self._errors(480)
        self.assertGreater(e_ob, 3.0 * e_swept,
                           "the Dirichlet formula should not fit a flux inlet")

    def test_flux_inlet_mass_ledger_closes(self):
        _x, _p, _t, res = _flux_inlet_prof(240)
        m = res.mass
        src = float(m["source"][-1])
        residual = (float(m["total"][-1]) - float(m["total"][0])
                    + float(m["well"][-1]) + float(m["outflow"][-1])
                    + float(m["decay"][-1]) - src)
        self.assertLess(abs(residual) / src, 1e-10,
                        f"mass ledger leaves {residual:.3e} of {src:.3e}")


class TestDispersionMoments(unittest.TestCase):
    """G3.2 (partial): a pulse's second moments grow as ``2 D t`` per axis.

    This measures the dispersion tensor the operator is supposed to assemble,
    independently of any inlet or outflow choice: put a pulse in the middle of
    the domain and read the variance growth.  Longitudinal growth carries the
    first-order upwind term ``v dx / 2`` on top of ``alpha_l v``.

    Two windows matter.  The longitudinal moment is read at the END, where the
    plume has spread ~9 m in a 120 m domain.  The transverse moments are read
    EARLY: with ``alpha_t v t`` spreading, the y variance reaches the no-flow
    walls (``ly^2/12 = 5.3``) at about 1000 d and stops growing, which is a
    property of the box and not of the operator -- reading it late reports 0.6
    of the true coefficient for that reason alone.
    """

    K = 1.0                      # slow enough that the pulse stays inside
    ALPHA_T = 0.5                # = ALPHA_L / 20, deliberately != alpha_l

    def setUp(self):
        self.v = self.K * GRAD / PHI
        self.dl = ALPHA_L * self.v
        self.dt = self.v
        self.nx, self.ny, self.nz = 120, 8, 6
        self.grid = fv3d.Grid3D(self.nx, self.ny, self.nz, 120.0, 8.0, 6.0)
        self.aq = fv3d.Aquifer3D(K=np.full(self.grid.shape, self.K), phi=PHI,
                                 alpha_l=ALPHA_L, alpha_th=self.ALPHA_T,
                                 alpha_tv=self.ALPHA_T)
        self.flow = fv3d.FlowModel3D(self.grid, self.aq, h_left=GRAD * 120.0,
                                     h_right=0.0)
        self.resp = fv3d.flow_responses3d(
            self.flow, [(60.0, 4.0, 3.0)], eps_cells=1.0)
        self.flow.release_factorization()

    def _moments(self, C):
        g = self.grid
        m = (PHI * g.cell_volume * C).reshape(g.ny, g.nx, g.nz)
        tot = float(m.sum())
        xs, ys, zs = g.centres()
        out = []
        for coord in (xs, ys, zs):
            mu = float((m * coord).sum() / tot)
            out.append((mu, float((m * (coord - mu) ** 2).sum() / tot)))
        return tot, out

    def _run(self, dt=20.0, ns=60):
        g = self.grid
        c0 = np.zeros(g.n)
        c0[g.index(60.0, 4.0, 3.0)] = 1.0 / (PHI * g.cell_volume)
        return fv3d.transport_run3d(g, self.aq, self.flow, self.resp, [0.0],
                                    dt=dt, nsteps=ns, c_init=c0,
                                    store_fields=True)

    def test_longitudinal_moment_includes_upwind_dispersion(self):
        res = self._run()
        _t0, m0 = self._moments(res.C_fields[0])
        _t1, m1 = self._moments(res.C_fields[-1])
        t = 60 * 20.0
        d_long = (m1[0][1] - m0[0][1]) / (2 * t)
        expect = self.dl + 0.5 * self.v * self.grid.dx
        self.assertLess(abs(d_long / expect - 1.0), 0.10,
                        f"longitudinal dispersion {d_long:.5f}, expected "
                        f"{expect:.5f} = alpha_l v + v dx/2")

    def test_transverse_moments_grow_at_alpha_t_v(self):
        res = self._run()
        _t0, m0 = self._moments(res.C_fields[0])
        _t1, m1 = self._moments(res.C_fields[10])     # t = 200 d, walls far
        t = 10 * 20.0
        d_y = (m1[1][1] - m0[1][1]) / (2 * t)
        d_z = (m1[2][1] - m0[2][1]) / (2 * t)
        expect = self.ALPHA_T * self.v
        self.assertLess(abs(d_y / expect - 1.0), 0.25,
                        f"transverse y dispersion {d_y:.6f} vs {expect:.6f}")
        self.assertLess(abs(d_z / expect - 1.0), 0.25,
                        f"transverse z dispersion {d_z:.6f} vs {expect:.6f}")

    def test_centroid_drifts_at_the_pore_velocity(self):
        res = self._run()
        _t0, m0 = self._moments(res.C_fields[0])
        _t1, m1 = self._moments(res.C_fields[-1])
        t = 60 * 20.0
        self.assertAlmostEqual((m1[0][0] - m0[0][0]) / (self.v * t), 1.0,
                               places=2, msg="centroid is not at v t")
        # ... and not at the Darcy velocity, which is phi times smaller
        self.assertGreater((m1[0][0] - m0[0][0]) / t / (self.K * GRAD), 2.5)

    def test_pulse_mass_is_conserved(self):
        res = self._run()
        tot0, _m0 = self._moments(res.C_fields[0])
        tot1, _m1 = self._moments(res.C_fields[-1])
        self.assertAlmostEqual(tot1 / tot0, 1.0, places=8, msg="mass was lost")


class TestGeometryConventions(unittest.TestCase):
    """Cheap guards for the axes and the source API, which are easy to get wrong."""

    def test_centres_returns_x_y_z_in_field_order(self):
        nx, ny, nz = 5, 3, 2
        grid = fv3d.Grid3D(nx, ny, nz, 5.0, 3.0, 2.0)
        xs, ys, zs = grid.centres()
        self.assertEqual(xs.shape, (ny, nx, nz))
        self.assertEqual(xs.shape, grid.shape)
        # x must vary along axis 1, y along axis 0, z along axis 2
        self.assertGreater(np.ptp(xs[0, :, 0]), 0.0)
        self.assertLess(np.ptp(xs[:, 0, 0]), 1e-12)
        self.assertGreater(np.ptp(ys[:, 0, 0]), 0.0)
        self.assertLess(np.ptp(ys[0, :, 0]), 1e-12)
        self.assertGreater(np.ptp(zs[0, 0, :]), 0.0)
        self.assertLess(np.ptp(zs[0, :, 0]), 1e-12)
        # and the values must be the cell centres in metres
        np.testing.assert_allclose(xs[0, :, 0], (np.arange(nx) + 0.5) * grid.dx)
        np.testing.assert_allclose(ys[:, 0, 0], (np.arange(ny) + 0.5) * grid.dy)
        np.testing.assert_allclose(zs[0, 0, :], (np.arange(nz) + 0.5) * grid.dz)

    def test_source_can_be_a_set_of_cells(self):
        grid = fv3d.Grid3D(20, 4, 3, 20.0, 4.0, 3.0)
        aq = fv3d.Aquifer3D(K=np.full(grid.shape, K0), phi=PHI, alpha_l=ALPHA_L)
        flow = fv3d.FlowModel3D(grid, aq, h_left=GRAD * 20.0, h_right=0.0)
        resp = fv3d.flow_responses3d(flow, [(10.0, 2.0, 1.5)], eps_cells=1.0)
        idx = np.arange(grid.n).reshape(grid.ny, grid.nx, grid.nz)
        inlet = idx[:, 0, :].ravel()
        total = 4.0
        res = fv3d.transport_run3d(grid, aq, flow, resp, [0.0], dt=1.0,
                                   nsteps=3, source_idx=inlet,
                                   source_rate=total)
        self.assertAlmostEqual(float(res.mass["source"][-1]), total * 3.0,
                               places=10, msg="the total rate must be split")
        per_cell = np.full(inlet.size, total / inlet.size)
        res2 = fv3d.transport_run3d(grid, aq, flow, resp, [0.0], dt=1.0,
                                    nsteps=3, source_idx=inlet,
                                    source_rate=per_cell)
        np.testing.assert_allclose(res2.C_final, res.C_final, rtol=0, atol=1e-12)
        with self.assertRaises(ValueError):
            fv3d.transport_run3d(grid, aq, flow, resp, [0.0], dt=1.0, nsteps=1,
                                 source_idx=inlet, source_rate=np.ones(3))


if __name__ == "__main__":
    unittest.main()
