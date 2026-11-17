"""Batched, device-resident cost functional for the stochastic optimization.

This is :mod:`stoch.objective` written so that ``S`` candidate rate vectors
evaluated on ``M`` scenarios at once reduce to a handful of tensor kernels
instead of ``S*M`` Python loops.  The formulas are transcribed from
:func:`stoch.objective.scenario_objective` (same terms, same sigmoid
surrogate, same mass-balance identity), so the numbers agree term by term --
``tests/test_saa_torch.py`` checks that against the NumPy reference.

All results carry leading axes ``(S, M)`` (or ``(nt, S, M)`` for time series).
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from physics.fv2d_torch import TransportBatch
from stoch.objective import CostConfig

__all__ = [
    "cvar_batch",
    "sigmoid_t",
    "dsigmoid_t",
    "cleanup_time_batch",
    "soft_last_violation_batch",
    "scenario_objective_batch",
]


def cvar_batch(values: torch.Tensor, alpha: float) -> torch.Tensor:
    """Batched empirical CVaR at level ``1 - alpha`` (Rockafellar--Uryasev).

    ``values`` shaped ``(S, M)``; returns ``(S,)``.  Matches
    :func:`stoch.objective.cvar`, including the small-sample case
    ``alpha * M < 1`` where CVaR degenerates to the worst value.
    """
    M = values.shape[1]
    sorted_values = torch.sort(values, dim=1, descending=True).values
    aM = alpha * M
    k = int(math.floor(aM))
    if k <= 0:
        return sorted_values[:, 0]
    if k >= M:
        return sorted_values.mean(dim=1)
    eta = sorted_values[:, k]
    head = sorted_values[:, :k].sum(dim=1)
    return (head + (aM - k) * eta) / (aM)


def sigmoid_t(z: torch.Tensor) -> torch.Tensor:
    """``0.5 * (1 + tanh(z/2))`` -- the same sigmoid as ``stoch.objective``."""
    return 0.5 * (1.0 + torch.tanh(0.5 * z))


def dsigmoid_t(z: torch.Tensor) -> torch.Tensor:
    s = sigmoid_t(z)
    return s * (1.0 - s)


def cleanup_time_batch(v: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact cleanup time per candidate/scenario.

    ``v`` is the compliance violation trace shaped ``(nt, S, M)``.  Returns
    ``(tau, failed)`` with ``tau = 0`` when the site is compliant throughout
    and ``failed`` when the last violation is at the final time -- the same
    convention as :func:`stoch.objective.cleanup_time`.
    """
    nt = v.shape[0]
    idx = torch.arange(nt, dtype=v.dtype, device=v.device).reshape(nt, *([1] * (v.dim() - 1)))
    positive = v > 0.0
    last = torch.where(positive, idx, torch.zeros_like(idx)).max(dim=0).values
    any_positive = positive.any(dim=0)
    tau = torch.where(any_positive, (last + 1.0) * dt, torch.zeros_like(last))
    failed = any_positive & (last == nt - 1)
    return tau, failed


def soft_last_violation_batch(
    v: torch.Tensor, dt: float, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable surrogate of the cleanup time, batched.

    Mirrors :func:`stoch.objective.soft_last_violation`::

        tau = sum_n t_n s_n prod_{k>n} (1 - s_k),      s_n = sigmoid(v_n / eps)

    Returns ``(tau, dtau/dv)`` with the derivative shaped like ``v``.
    """
    nt = v.shape[0]
    s = sigmoid_t(v / eps)
    ds = dsigmoid_t(v / eps) / eps
    t = (torch.arange(nt, dtype=v.dtype, device=v.device) + 1.0).reshape(
        nt, *([1] * (v.dim() - 1))
    ) * dt

    # R[n] = prod_{k>=n} (1 - s_k); R_shift[n] = R[n+1], with R_shift[nt-1] = 1.
    ones = torch.ones_like(s)
    tail = torch.flip(torch.cumprod(torch.flip(1.0 - s, dims=[0]), dim=0), dims=[0])
    r_shift = torch.cat([tail[1:], ones[:1]], dim=0)

    contrib = t * s * r_shift
    tau = contrib.sum(dim=0)

    # A[m] = sum_{n<m} t_n s_n R[n+1]; exclusive cumulative sum along time.
    a_excl = torch.cat([torch.zeros_like(contrib[:1]), torch.cumsum(contrib, dim=0)[:-1]],
                       dim=0)
    denom = 1.0 - s
    ratio = torch.where(denom > 1e-9, a_excl / torch.clamp(denom, min=1e-9),
                        torch.zeros_like(denom))
    dtau = t * ds * r_shift - ds * ratio
    return tau, dtau


def scenario_objective_batch(
    res: TransportBatch,
    Q: torch.Tensor,
    h_obs: torch.Tensor,
    dt: float,
    T: float,
    cost: CostConfig,
    mass_weight: torch.Tensor,
) -> dict:
    """Batched ``J``, ``G``, ``H``, ``tau_c`` for ``S`` candidates x ``M`` scenarios.

    Parameters
    ----------
    res
        Output of :func:`physics.fv2d_torch.transport_batch`.
    Q
        Rate vectors, shape ``(S, nw)``.
    h_obs
        Heads at the drawdown observation points, shape ``(S, M, n_obs)``.
    mass_weight
        ``b * R * phi * cell_volume`` on the grid, shaped ``(n,)`` or
        ``(M, n)``; multiplies the final concentration to give the
        residual-mass term.
    """
    if res.C_record is None:
        raise ValueError("transport result has no compliance record")
    Q = Q.to(dtype=res.C_final.dtype, device=res.C_final.device)
    C_comp = res.C_record                        # (nt, S, M, n_rec)
    nt = C_comp.shape[0]

    # -- compliance trace, violation G and drawdown H --------------------
    v_t, arg_cells = (C_comp - cost.creg).max(dim=-1)     # (nt, S, M)
    i0 = int(nt * cost.window_start)
    G = v_t[i0:].max(dim=0).values
    H = (h_obs - cost.h_min).min(dim=-1).values

    tau_c, failed = cleanup_time_batch(v_t, dt)

    # -- cost components -------------------------------------------------
    J_pump = cost.c_e * Q.sum(dim=-1).reshape(-1, 1).expand_as(G) * T
    J_treat = cost.c_w * dt * torch.einsum("nsmi,si->sm", res.C_well, Q)
    J_res = cost.c_m * (res.C_final * mass_weight).sum(dim=-1)

    eps = max(cost.eps_time * cost.creg, 1e-12)
    if cost.time_mode == "soft":
        tau_soft, dtau = soft_last_violation_batch(v_t, dt, eps)
        zmax = v_t.max(dim=0).values / eps
        J_fail = cost.c_fail * sigmoid_t(zmax)
        J_time = cost.c_tau * tau_soft + J_fail
        dJdv = cost.c_tau * dtau
        k = v_t.argmax(dim=0)
        bump = cost.c_fail * dsigmoid_t(zmax) / eps
        one_hot = torch.zeros_like(dJdv).scatter(0, k.unsqueeze(0), 1.0)
        dJdv = dJdv + one_hot * bump.unsqueeze(0)
        time_deriv: Optional[torch.Tensor] = dJdv
    elif cost.time_mode == "hard":
        J_time = cost.c_tau * tau_c.clamp(max=T) + torch.where(
            failed, torch.full_like(tau_c, cost.c_fail), torch.zeros_like(tau_c)
        )
        time_deriv = None
    else:
        raise ValueError("time_mode must be 'soft' or 'hard'")

    J = J_pump + J_treat + J_res + J_time

    # -- mass balance ----------------------------------------------------
    mass = res.mass
    bal = (mass["total"][-1] - mass["total"][0]
           - (mass["source"][-1] - mass["well"][-1]
              - mass["outflow"][-1] - mass["decay"][-1]))
    denom = torch.maximum(torch.maximum(mass["source"][-1].abs(),
                                        mass["total"][0].abs()),
                          torch.full_like(mass["total"][0], 1e-30))
    mass_err = bal / denom

    compliant = (v_t[i0:] <= 0.0).all(dim=0)

    return {
        "J": J,
        "J_pump": J_pump,
        "J_treat": J_treat,
        "J_res": J_res,
        "J_time": J_time,
        "G": G,
        "H": H,
        "tau_c": tau_c,
        "failed": failed,
        "v_t": v_t,
        "arg_cells": arg_cells,
        "time_deriv": time_deriv,
        "compliance_prob": compliant,
        "C_comp": C_comp,
        "C_well": res.C_well,
        "C_final": res.C_final,
        "mass_balance_error": mass_err,
    }
