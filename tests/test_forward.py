"""Forward-model gates: velocity, Thiem shape, mass balance."""

import unittest

import numpy as np

from physics.fv2d import (
    Aquifer,
    FlowModel,
    Grid,
    flow_responses,
    transport_run,
)
from physics.references import thiem_drawdown


class TestFlow(unittest.TestCase):
    def test_uniform_velocity(self):
        g = Grid(30, 12, 120.0, 48.0)
        aq = Aquifer(K=np.full((g.ny, g.nx), 10.0))
        flow = FlowModel(g, aq, h_left=1e-3 * g.lx, h_right=0.0)
        resp = flow_responses(flow, [(60.0, 24.0)], eps_cells=1.0)
        vx, vy = flow.velocity(resp.h0)
        v_exact = 10.0 * 1e-3 / 0.3
        self.assertLess(np.max(np.abs(vx[:, 1:-1] - v_exact)) / v_exact, 1e-10)
        self.assertLess(np.max(np.abs(vy)), 1e-10)

    def test_thiem_shape(self):
        q, K, b = 20.0, 10.0, 10.0
        g = Grid(100, 100, 1000.0, 1000.0)
        aq = Aquifer(K=np.full((g.ny, g.nx), K), b=b)
        flow = FlowModel(g, aq, h_left=0.0, h_right=0.0, bc_y="dirichlet")
        resp = flow_responses(flow, [(500.0, 500.0)], eps_cells=0.3)
        h = resp.head([q]).reshape(g.ny, g.nx)
        X, Y = g.centres()
        r = np.sqrt((X - 500.0) ** 2 + (Y - 500.0) ** 2)
        sel = (r >= 25.0) & (r <= 100.0)
        order = np.argsort(r.ravel())
        h1 = np.interp(30.0, r.ravel()[order], h.ravel()[order])
        th = thiem_drawdown(q, K * b, r[sel], r_ref=1.0)
        th1 = thiem_drawdown(q, K * b, 30.0, r_ref=1.0)
        err = np.max(np.abs((h[sel] - h1) - (th - th1))) / np.max(np.abs(th - th1))
        self.assertLess(err, 1e-2)

    def test_well_superposition(self):
        g = Grid(24, 12, 96.0, 48.0)
        aq = Aquifer(K=np.full((g.ny, g.nx), 8.0))
        flow = FlowModel(g, aq, h_left=0.15, h_right=0.0)
        resp = flow_responses(flow, [(30.0, 24.0), (70.0, 24.0)], 1.2)
        q = np.array([4.0, 7.0])
        h = resp.head(q)
        h_check = resp.h0 + 4.0 * resp.hw[0] + 7.0 * resp.hw[1]
        self.assertLess(np.max(np.abs(h - h_check)), 1e-12)


class TestTransport(unittest.TestCase):
    def setUp(self):
        self.g = Grid(24, 12, 96.0, 48.0)
        self.aq = Aquifer(K=np.full((self.g.ny, self.g.nx), 10.0))
        self.flow = FlowModel(self.g, self.aq, h_left=0.0, h_right=0.0)
        self.resp = flow_responses(self.flow, [(48.0, 24.0)], 1.2)

    def test_no_flow_mass_conservation(self):
        g, aq = self.g, self.aq
        X, Y = g.centres()
        C0 = 8.0 * np.exp(-(((X - 48) ** 2 + (Y - 24) ** 2) / (2 * 6 ** 2))).ravel()
        res = transport_run(g, aq, self.flow, self.resp, [0.0], 5.0, 30, c_init=C0)
        m = res.mass
        rel = abs(m["total"][-1] - m["total"][0]) / m["total"][0]
        self.assertLess(rel, 1e-10)

    def test_source_and_well_mass_balance(self):
        g, aq = self.g, self.aq
        flow = FlowModel(g, aq, h_left=0.2, h_right=0.0)
        resp = flow_responses(flow, [(70.0, 24.0)], 1.2)
        res = transport_run(
            g, aq, flow, resp, [8.0], 5.0, 40,
            source_idx=g.index(20.0, 24.0), source_rate=500.0,
            source_steps=10,
        )
        m = res.mass
        bal = (m["total"][-1] - m["total"][0]
               - (m["source"][-1] - m["well"][-1] - m["outflow"][-1] - m["decay"][-1]))
        self.assertLess(abs(bal) / m["source"][-1], 1e-12)
        self.assertGreater(m["well"][-1], 0.0)

    def test_well_sink_is_distributed(self):
        g, aq = self.g, self.aq
        flow = FlowModel(g, aq, h_left=0.2, h_right=0.0)
        resp = flow_responses(flow, [(60.0, 24.0)], 1.5)
        self.assertAlmostEqual(float(resp.weights.sum()), 1.0, places=12)
        # the pumped well must lower the ambient head
        h = resp.head([5.0])
        self.assertLess(h[resp.well_cells[0]], resp.h0[resp.well_cells[0]])


if __name__ == "__main__":
    unittest.main()
