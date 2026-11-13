"""Outer optimization: pattern search, SLSQP with adjoint gradients, multi-start.

The chance constraints are handled by an exact penalty on the CVaR
reformulation (or on the empirical probability if requested).  The penalty
weight should be increased in continuation if the final policy is infeasible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from stoch.adjoint import (
    scenario_H_gradient,
    scenario_G_gradient,
    scenario_cost_gradient,
)
from stoch.objective import CostConfig, cvar_and_weights

__all__ = [
    "SAAProblem",
    "OptResult",
    "project_capacity",
    "pattern_search",
    "slsqp_adjoint",
    "multi_start",
    "solve_saa",
]


def project_capacity(
    q: np.ndarray,
    q_max: float = 50.0,
    Q_max: float = 200.0,
) -> np.ndarray:
    """Clip to the box and scale down if the plant capacity is exceeded."""
    q = np.clip(np.asarray(q, dtype=float), 0.0, q_max)
    s = q.sum()
    if s > Q_max:
        q = q * (Q_max / s)
    return q


@dataclass
class OptResult:
    q: np.ndarray
    fun: float
    nit: int
    method: str
    history: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"q": self.q.tolist(), "fun": self.fun, "nit": self.nit,
                "method": self.method, "metrics": self.metrics}


class SAAProblem:
    """Penalised SAA objective with an optional discrete-adjoint gradient.

    ``f(q) = mean_m J_m(q) + rho_g [CVaR(G)]_+^2 + rho_h [CVaR(-H)]_+^2``.
    """

    def __init__(
        self,
        archive,
        cost: CostConfig,
        q_max: float = 50.0,
        Q_max: float = 200.0,
        alpha: float = 0.05,
        alpha_h: float = 0.01,
        chance: str = "cvar",
        eps_sigmoid: float = 0.05,
        rho_g: float = 1.0,
        rho_h: float = 1.0,
    ) -> None:
        self.archive = archive
        self.cost = cost
        self.q_max = q_max
        self.Q_max = Q_max
        self.alpha = alpha
        self.alpha_h = alpha_h
        self.chance = chance
        self.eps_sigmoid = eps_sigmoid
        self.rho_g = rho_g
        self.rho_h = rho_h
        self.n_wells = archive.well_xy.shape[0]
        self._cache_q: Optional[np.ndarray] = None
        self._cache_ev = None
        self.n_eval = 0

    # -- evaluation -------------------------------------------------------
    def evaluate(self, q, keep: bool = True):
        q = project_capacity(q, self.q_max, self.Q_max)
        if self._cache_q is not None and np.allclose(q, self._cache_q):
            return self._cache_ev
        ev = self.archive.evaluate(q, self.cost, keep=keep)
        self.n_eval += 1
        self._cache_q = q.copy()
        self._cache_ev = ev
        return ev

    def chance_values(self, ev) -> tuple[float, float]:
        if self.chance == "exact":
            pg = float(np.mean(ev.G <= 0.0))
            ph = float(np.mean(-ev.H <= 0.0))
            return -(pg - (1.0 - self.alpha)), -(ph - (1.0 - self.alpha_h))
        cg, _ = cvar_and_weights(ev.G, self.alpha)
        ch, _ = cvar_and_weights(-ev.H, self.alpha_h)
        return cg, ch

    def objective(self, q, keep: bool = False) -> float:
        ev = self.evaluate(q, keep=keep)
        cg, ch = self.chance_values(ev)
        return float(ev.mean_J + self.rho_g * max(cg, 0.0) ** 2
                     + self.rho_h * max(ch, 0.0) ** 2)

    def penalty(self, ev) -> tuple[float, float]:
        cg, ch = self.chance_values(ev)
        return self.rho_g * max(cg, 0.0) ** 2, self.rho_h * max(ch, 0.0) ** 2

    # -- gradient ---------------------------------------------------------
    def gradient(self, q) -> np.ndarray:
        ev = self.evaluate(q, keep=True)
        if not ev.keep:
            ev = self.evaluate(q, keep=True)
        grid = self.archive.grid
        M = ev.M
        grad = np.zeros(self.n_wells)
        cg, ch = self.chance_values(ev)
        if self.chance == "exact":
            return np.zeros_like(grad)  # nonsmooth: use pattern search

        _, wg = cvar_and_weights(ev.G, self.alpha)
        _, wh = cvar_and_weights(-ev.H, self.alpha_h)

        for m, prob in enumerate(self.archive.problems):
            d = ev.details[m]
            res = d["res"]
            grad += scenario_cost_gradient(grid, prob, q, res, d, self.cost) / M
            if cg > 0.0 and wg[m] > 0.0:
                gG = scenario_G_gradient(grid, prob, q, res, d, self.cost)
                grad += 2.0 * self.rho_g * cg * wg[m] * gG
            if ch > 0.0 and wh[m] > 0.0:
                gH = scenario_H_gradient(prob, q, d["h"], self.cost)
                # constraint is -H <= 0
                grad -= 2.0 * self.rho_h * ch * wh[m] * gH
        return grad

    def fun_and_grad(self, q):
        q = project_capacity(q, self.q_max, self.Q_max)
        return self.objective(q, keep=True), self.gradient(q)


def pattern_search(
    problem: SAAProblem,
    q0: np.ndarray,
    step0: Optional[float] = None,
    tol: float = 1e-3,
    max_iter: int = 80,
    max_evals: int = 2000,
) -> OptResult:
    """Compass/coordinate pattern search with capacity projection."""
    q0 = project_capacity(q0, problem.q_max, problem.Q_max)
    q = q0.copy()
    best = problem.objective(q)
    step = step0 if step0 is not None else 0.25 * problem.q_max
    hist = [{"eval": 0, "f": best, "q": q.tolist()}]
    nit = 0
    evals = 1
    while step > tol and nit < max_iter and evals < max_evals:
        improved = False
        for i in range(problem.n_wells):
            for sgn in (1.0, -1.0):
                cand = q.copy()
                cand[i] += sgn * step
                cand = project_capacity(cand, problem.q_max, problem.Q_max)
                if np.allclose(cand, q):
                    continue
                fc = problem.objective(cand)
                evals += 1
                if fc < best - 1e-12:
                    q, best = cand, fc
                    improved = True
                    hist.append({"eval": evals, "f": best, "q": q.tolist()})
                    break
            if evals >= max_evals:
                break
        nit += 1
        if not improved:
            step *= 0.5
    ev = problem.evaluate(q, keep=False)
    return OptResult(q=q, fun=best, nit=nit, method="pattern", history=hist,
                     metrics=ev.summary())


def slsqp_adjoint(
    problem: SAAProblem,
    q0: np.ndarray,
    max_iter: int = 100,
    ftol: float = 1e-8,
) -> OptResult:
    """SLSQP with the discrete-adjoint gradient and the capacity constraint."""
    q0 = project_capacity(q0, problem.q_max, problem.Q_max)
    hist: list[dict] = []

    def fun(q):
        f = problem.objective(q, keep=True)
        hist.append({"eval": len(hist), "f": float(f), "q": np.asarray(q).tolist()})
        return f

    def jac(q):
        return problem.gradient(q)

    cons = [{
        "type": "ineq",
        "fun": lambda q: problem.Q_max - np.sum(q),
        "jac": lambda q: -np.ones_like(q),
    }]
    bounds = [(0.0, problem.q_max)] * problem.n_wells
    res = minimize(
        fun, q0, jac=jac, method="SLSQP", bounds=bounds, constraints=cons,
        options={"maxiter": max_iter, "ftol": ftol, "disp": False},
    )
    q = project_capacity(res.x, problem.q_max, problem.Q_max)
    ev = problem.evaluate(q, keep=False)
    return OptResult(q=q, fun=float(res.fun), nit=int(res.nit), method="slsqp",
                     history=hist, metrics=ev.summary())


def multi_start(
    problem: SAAProblem,
    n_starts: int = 5,
    method: str = "pattern",
    seed: int = 0,
    **kwargs,
) -> OptResult:
    """Random feasible starts, keep the best penalised objective."""
    rng = np.random.default_rng(seed)
    best: Optional[OptResult] = None
    results = []
    for k in range(n_starts):
        if k == 0:
            q0 = np.full(problem.n_wells, 0.5 * problem.q_max)
        else:
            w = rng.random(problem.n_wells) + 0.1
            q0 = w / w.sum() * rng.uniform(0.2, 1.0) * problem.Q_max
        q0 = project_capacity(q0, problem.q_max, problem.Q_max)
        if method == "pattern":
            kw = {k: v for k, v in kwargs.items()
                  if k in ("step0", "tol", "max_iter", "max_evals")}
            r = pattern_search(problem, q0, **kw)
        elif method == "slsqp":
            kw = {k: v for k, v in kwargs.items() if k in ("max_iter", "ftol")}
            r = slsqp_adjoint(problem, q0, **kw)
        else:
            raise ValueError("method must be 'pattern' or 'slsqp'")
        results.append(r)
        if best is None or r.fun < best.fun:
            best = r
    assert best is not None
    best.metrics["multi_start_spread"] = float(np.std([r.fun for r in results]))
    best.metrics["multi_start_best"] = float(np.min([r.fun for r in results]))
    return best


def solve_saa(
    problem: SAAProblem,
    method: str = "pattern",
    n_starts: int = 3,
    **kwargs,
) -> OptResult:
    return multi_start(problem, n_starts=n_starts, method=method, **kwargs)
