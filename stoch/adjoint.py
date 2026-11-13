"""Discrete adjoint gradient for the high-fidelity transport model.

The transport step is linear in the concentration,

    A(q) C^n = st * C^{n-1} + s,

with ``A(q) = A_adv(v(q)) + A_disp + diag(st + decay + sum_i q_i w_i)``.
Because the flow problem is linear in ``q``, ``v(q) = v0 + sum_i q_i v_i``
and the advection operator is *linear* in ``q``.  The dispersion tensor is
frozen at the background velocity, so

    dA/dq_i = A_adv(v_i) + diag(w_i).

The discrete adjoint is then exact (to solver tolerance) and its cost is one
extra transport solve, independent of the number of wells.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import splu

from physics.fv2d import Grid, advection_operator

__all__ = [
    "scenario_cost_gradient",
    "scenario_G_gradient",
    "scenario_H_gradient",
    "scenario_dJdC",
    "check_gradient",
]


def scenario_dJdC(
    grid: Grid,
    problem,
    res,
    q: np.ndarray,
    d: dict,
    cost,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble ``dJ/dC`` (nt, n) and the explicit ``dJ/dq`` for one scenario."""
    nt = res.C_well.shape[0]
    n = grid.n
    dJdC = np.zeros((nt, n))
    q = np.asarray(q, dtype=float).ravel()

    # treatment: c_w q_i dt w_i . C  (implicit part)
    wsum = q @ problem.resp.weights        # (n,)
    dJdC += (cost.c_w * actual_dt(res) * wsum)[None, :]

    # residual mass at the final time
    phi = problem.flow.aquifer.phi_field((grid.ny, grid.nx)).ravel()
    dJdC[-1] += cost.c_m * problem.flow.aquifer.b * problem.flow.aquifer.R * phi * grid.cell_volume

    # smooth cleanup-time term
    if d.get("time_deriv") is not None:
        arg = np.asarray(d["arg_cells"], dtype=int)
        comp = np.asarray(problem_compliance_cells(problem), dtype=int)
        t = np.arange(nt)
        dJdC[t, comp[arg]] += d["time_deriv"]

    # explicit q dependence: pumping + explicit treatment
    dJdq = np.full(q.size, cost.c_e * grid_time(res))
    dJdq += cost.c_w * actual_dt(res) * np.sum(res.C_well, axis=0)
    return dJdC, dJdq


def actual_dt(res) -> float:
    """Time step used by the transport run."""
    return float(res.times[1] - res.times[0])


def grid_time(res) -> float:
    return float(res.times[-1])


def problem_compliance_cells(problem):
    return problem.compliance_cells


def _adjoint_solve(
    grid: Grid,
    problem,
    res,
    dJdC: np.ndarray,
    dJdq: np.ndarray,
) -> np.ndarray:
    """Backward adjoint recursion and gradient assembly."""
    ops = res.operators
    if not ops:
        raise ValueError("transport result must be run with return_operator=True")
    A = ops["A"]
    st = ops["st"]
    phi_t = ops["phi_t"]
    luT = splu(A.T.tocsc())
    nw = problem.resp.hw.shape[0]
    nt = res.C_fields.shape[0] - 1
    C = res.C_fields

    # unit-response advection operators with the upwind pattern of the
    # total velocity (so A(v) is differentiated linearly)
    vx, vy = ops["vx"], ops["vy"]
    adv = []
    for i in range(nw):
        vx_i, vy_i = problem.flow.velocity(problem.resp.hw[i])
        adv.append(advection_operator(grid, vx_i, vy_i, phi_t,
                                       vx_sign=vx, vy_sign=vy))

    grad = np.asarray(dJdq, dtype=float).copy()
    psi_next = np.zeros(grid.n)
    for n in range(nt - 1, -1, -1):
        rhs = dJdC[n] + st * psi_next
        psi = luT.solve(rhs)
        Cn = C[n + 1]
        for i in range(nw):
            grad[i] -= psi @ (adv[i] @ Cn) + float(np.dot(problem.resp.weights[i], psi * Cn))
        psi_next = psi
    return grad


def scenario_cost_gradient(
    grid: Grid,
    problem,
    q,
    res,
    d: dict,
    cost,
) -> np.ndarray:
    """Gradient of the smooth part of ``J`` for one scenario."""
    dJdC, dJdq = scenario_dJdC(grid, problem, res, q, d, cost)
    return _adjoint_solve(grid, problem, res, dJdC, dJdq)


def scenario_G_gradient(
    grid: Grid,
    problem,
    q,
    res,
    d: dict,
    cost,
) -> np.ndarray:
    """Gradient of the scalar compliance violation ``G`` for one scenario."""
    nt = res.C_well.shape[0]
    i0 = int(np.floor(cost.window_start * nt))
    v = d["v_t"]
    k = i0 + int(np.argmax(v[i0:]))
    cell = int(problem.compliance_cells[int(d["arg_cells"][k])])
    dJdC = np.zeros((nt, grid.n))
    dJdC[k, cell] = 1.0
    return _adjoint_solve(grid, problem, res, dJdC, np.zeros(q.size))


def scenario_H_gradient(problem, q, h: np.ndarray, cost) -> np.ndarray:
    """Gradient of the drawdown margin ``H`` (flow is linear in ``q``)."""
    vals = h[problem.obs_cells] - cost.h_min
    k = int(np.argmin(vals))
    return problem.resp.hw[:, problem.obs_cells[k]].copy()


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------
def check_gradient(
    archive,
    q,
    cost,
    scenario_index: int = 0,
    delta: float = 1e-4,
    direction: np.ndarray | None = None,
) -> dict:
    """Compare the discrete adjoint to central finite differences.

    The objective used here is the smooth part (``time_mode='soft'``); this
    is exactly the function the gradient-based optimizer minimises.
    """
    q = np.asarray(q, dtype=float).ravel()
    nw = q.size
    if direction is None:
        direction = np.ones(nw) / nw
    direction = np.asarray(direction, dtype=float).ravel()

    prob = archive.problems[scenario_index]
    fwd = archive.fwd
    from physics.fv2d import transport_run
    from stoch.objective import scenario_objective

    def run_one(qq):
        h = prob.resp.head(qq)
        res = transport_run(
            archive.grid, prob.flow.aquifer, prob.flow, prob.resp, qq,
            fwd.dt, fwd.nsteps,
            source_idx=prob.source_idx, source_rate=prob.source_rate,
            source_steps=fwd.source_steps,
            record_cells=archive.compliance_cells,
            store_fields=True, return_operator=True,
        )
        d = scenario_objective(
            archive.grid, prob.flow.aquifer, res, qq, h[archive.obs_cells],
            fwd.dt, fwd.T, cost,
        )
        return res, d

    res, d = run_one(q)
    grad = scenario_cost_gradient(archive.grid, prob, q, res, d, cost)
    Jp = run_one(q + delta * direction)[1]["J"]
    Jm = run_one(q - delta * direction)[1]["J"]
    fd = (Jp - Jm) / (2 * delta)
    adj = float(np.dot(grad, direction))
    rel = abs(adj - fd) / max(abs(fd), 1e-12)
    return {"adjoint": adj, "finite_difference": fd, "rel_error": rel,
            "gradient": grad, "J": d["J"]}
