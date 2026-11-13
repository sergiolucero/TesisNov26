"""Cost functional, compliance violation, drawdown violation and cleanup time.

Implements Eqs. (eq:cost)--(eq:cleanuptime) of ``optimization.tex`` with the
baseline linear pumping cost.  The only non-smooth ingredient used by the
gradient-based optimizer is the cleanup time; it is replaced by a smooth
sigmoid surrogate (``time_mode="soft"``).  The exact hard cleanup time is
always reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from physics.fv2d import Aquifer, Grid, TransportResult

__all__ = [
    "CostConfig",
    "sigmoid",
    "dsigmoid",
    "compliance_trace",
    "cleanup_time",
    "scenario_objective",
    "cvar",
    "cvar_weights",
]


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def dsigmoid(z):
    s = sigmoid(z)
    return s * (1.0 - s)


@dataclass
class CostConfig:
    c_e: float = 1.0
    c_w: float = 1.0
    c_m: float = 1.0
    c_tau: float = 1.0
    c_fail: float = 10.0
    creg: float = 1.0
    window_start: float = 0.8
    h_min: float = -10.0
    time_mode: str = "soft"          # "soft" (differentiable) or "hard"
    eps_time: float = 0.05           # relative to creg
    rho_g: float = 1.0               # chance-constraint penalty weight
    rho_h: float = 1.0
    alpha: float = 0.05
    alpha_h: float = 0.01


def compliance_trace(
    C_record: np.ndarray,
    creg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-time maximum violation and the originating compliance cell."""
    v = C_record - creg
    cells = np.argmax(v, axis=1)
    return v[np.arange(v.shape[0]), cells], cells


def cleanup_time(v: np.ndarray, dt: float, T: float) -> tuple[float, bool]:
    """Last time the compliance violation is positive.

    This is the first time after which the site is compliant for the rest of
    the horizon.  ``tau = 0`` means compliant from the start; ``fail`` means
    the last violation occurs at the final time.
    """
    v = np.asarray(v, dtype=float)
    idx = np.nonzero(v > 0.0)[0]
    if idx.size == 0:
        return 0.0, False
    last = int(idx[-1])
    return float((last + 1) * dt), bool(last == v.size - 1)


def soft_last_violation(v: np.ndarray, dt: float, eps: float):
    """Differentiable surrogate of :func:`cleanup_time`.

    Returns ``(tau, dtau/dv)``.  The last violation index is smoothed with
    ``s_n = sigmoid(v_n/eps)`` and ``tau = sum_n t_n s_n prod_{k>n}(1-s_k)``.
    """
    v = np.asarray(v, dtype=float).ravel()
    nt = v.size
    s = sigmoid(v / eps)
    ds = dsigmoid(v / eps) / eps
    t = (np.arange(nt) + 1) * dt
    R = np.ones(nt + 1)
    for n in range(nt - 1, -1, -1):
        R[n] = R[n + 1] * (1.0 - s[n])
    tau = float(np.sum(t * s * R[1:]))
    dtau = np.zeros(nt)
    A = 0.0
    for m in range(nt):
        denom = 1.0 - s[m]
        ratio = A / denom if denom > 1e-9 else 0.0
        dtau[m] = t[m] * ds[m] * R[m + 1] - ds[m] * ratio
        A += t[m] * s[m] * R[m + 1]
    return tau, dtau


def scenario_objective(
    grid: Grid,
    aquifer: Aquifer,
    res: TransportResult,
    q: np.ndarray,
    h_obs: np.ndarray,
    dt: float,
    T: float,
    cfg: CostConfig,
) -> dict:
    """Evaluate ``J``, ``G``, ``H`` and ``tau_c`` for one scenario.

    ``res.C_record`` must contain the compliance-cell time series in a fixed
    order; ``h_obs`` are the heads at the drawdown observation points.
    """
    q = np.asarray(q, dtype=float).ravel()
    C_comp = res.C_record
    if C_comp is None:
        raise ValueError("transport result has no compliance record")
    nt = C_comp.shape[0]
    times = (np.arange(nt) + 1) * dt

    v_t, arg_cells = compliance_trace(C_comp, cfg.creg)
    i0 = int(np.floor(cfg.window_start * nt))
    G = float(np.max(v_t[i0:]))

    H = float(np.min(h_obs - cfg.h_min))

    # exact cleanup time (reported even in soft mode)
    tau_c, failed = cleanup_time(v_t, dt, T)

    # pumping: linear baseline c_e * sum_i q_i T
    J_pump = cfg.c_e * float(np.sum(q)) * T

    # treatment: c_w * sum_i q_i * integral C_well dt
    J_treat = cfg.c_w * float(np.sum(res.C_well * q[None, :]) * dt)

    # residual mass at T
    phi = aquifer.phi_field((grid.ny, grid.nx)).ravel()
    C_final = res.C_final
    J_res = cfg.c_m * float(np.sum(aquifer.b * aquifer.R * phi * C_final) * grid.cell_volume)

    eps = max(cfg.eps_time * cfg.creg, 1e-12)
    if cfg.time_mode == "soft":
        tau_soft, dtau = soft_last_violation(v_t, dt, eps)
        zmax = float(np.max(v_t) / eps)
        J_fail = cfg.c_fail * float(sigmoid(zmax))
        J_time = cfg.c_tau * tau_soft + J_fail
        dJdv = cfg.c_tau * dtau
        k = int(np.argmax(v_t))
        dJdv[k] += cfg.c_fail * dsigmoid(zmax) / eps
        time_deriv = dJdv
    elif cfg.time_mode == "hard":
        J_time = cfg.c_tau * min(tau_c, T) + (cfg.c_fail if failed else 0.0)
        time_deriv = None
    else:
        raise ValueError("time_mode must be 'soft' or 'hard'")

    J = J_pump + J_treat + J_res + J_time
    mass = res.mass
    bal = (mass["total"][-1] - mass["total"][0]
           - (mass["source"][-1] - mass["well"][-1]
              - mass["outflow"][-1] - mass["decay"][-1]))
    denom = max(abs(mass["source"][-1]), abs(mass["total"][0]), 1e-30)
    mass_err = float(bal / denom)
    return {
        "J": J,
        "J_pump": J_pump,
        "J_treat": J_treat,
        "J_res": J_res,
        "J_time": J_time,
        "G": G,
        "H": H,
        "tau_c": tau_c,
        "failed": bool(failed),
        "v_t": v_t,
        "arg_cells": arg_cells,
        "time_deriv": time_deriv,
        "compliance_prob": float(np.all(v_t[i0:] <= 0.0)),
        "C_comp": C_comp,
        "C_well": res.C_well,
        "C_final": C_final,
        "mass_balance_error": mass_err,
    }


# ---------------------------------------------------------------------------
# CVaR helpers (shared by the SAA and the chance constraints)
# ---------------------------------------------------------------------------
def cvar_and_weights(values, alpha: float):
    """Empirical CVaR at level ``1-alpha`` and its gradient weights.

    Returns ``(cvar, w)`` with ``d CVaR / d values = w``.  Handles the
    small-sample case ``alpha * M < 1`` where CVaR reduces to the worst value.
    """
    x = np.asarray(values, dtype=float).ravel()
    M = x.size
    order = np.argsort(x)[::-1]
    xs = x[order]
    aM = alpha * M
    k = int(np.floor(aM))
    if k <= 0:
        w = np.zeros(M)
        w[order[0]] = 1.0
        return float(xs[0]), w
    if k >= M:
        return float(xs.mean()), np.full(M, 1.0 / M)
    eta = xs[k]
    ws = np.zeros(M)
    ws[:k] = 1.0 / (alpha * M)
    ws[k] = (alpha * M - k) / (alpha * M)
    w = np.empty(M)
    w[order] = ws
    val = (xs[:k].sum() + (alpha * M - k) * eta) / (alpha * M)
    return float(val), w


def cvar(values, alpha: float) -> float:
    """Empirical CVaR at level ``1-alpha`` (Rockafellar--Uryasev)."""
    return cvar_and_weights(values, alpha)[0]


def cvar_weights(values, alpha: float, eta: float | None = None) -> np.ndarray:
    """Weights ``w_m`` such that ``d CVaR / d values = w``."""
    return cvar_and_weights(values, alpha)[1]
