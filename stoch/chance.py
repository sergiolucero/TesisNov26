"""Chance-constraint reformulations: exact empirical, CVaR, and sigmoid.

The exact empirical chance constraint is

    (1/M) sum_m 1{G_m <= 0} >= 1 - alpha.

It is nonsmooth.  Two tractable surrogates are provided:

* CVaR (conservative): ``CVaR_{1-alpha}(G) <= 0``;
* sigmoid: ``(1/M) sum_m sigma(-G_m / eps) >= 1 - alpha``.

``compare_reformulations`` evaluates all three on a dense grid for a small
toy problem, which is the O-4 validation gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from stoch.objective import cvar, cvar_and_weights, sigmoid

__all__ = [
    "empirical_probability",
    "sigmoid_probability",
    "cvar",
    "chance_metrics",
    "continuation_schedule",
    "compare_reformulations",
    "ChanceVerdict",
]


def empirical_probability(G, tol: float = 0.0) -> float:
    return float(np.mean(np.asarray(G, dtype=float) <= tol))


def sigmoid_probability(G, eps: float, tol: float = 0.0) -> float:
    G = np.asarray(G, dtype=float)
    return float(np.mean(sigmoid(-(G - tol) / eps)))


def chance_metrics(G, H, alpha: float, alpha_h: float) -> dict:
    """CVaR and empirical reliability for both chance constraints."""
    G = np.asarray(G, dtype=float)
    H = np.asarray(H, dtype=float)
    cvar_g, wg = cvar_and_weights(G, alpha)
    cvar_h, wh = cvar_and_weights(-H, alpha_h)
    return {
        "cvar_G": cvar_g,
        "cvar_negH": cvar_h,
        "weights_G": wg,
        "weights_negH": wh,
        "prob_G": empirical_probability(G),
        "prob_H": empirical_probability(-H),
        "feasible_G": empirical_probability(G) >= 1.0 - alpha,
        "feasible_H": empirical_probability(-H) >= 1.0 - alpha_h,
    }


def continuation_schedule(eps0: float = 0.1, factor: float = 0.1, n: int = 3) -> list[float]:
    return [eps0 * factor ** k for k in range(n)]


@dataclass
class ChanceVerdict:
    exact_J: float
    exact_q: np.ndarray
    exact_reliability: float
    cvar_J: float
    cvar_q: np.ndarray
    cvar_reliability: float
    sigmoid_J: float
    sigmoid_q: np.ndarray
    sigmoid_reliability: float
    sigmoid_eps: float
    n_exact_feasible: int
    n_cvar_feasible: int
    n_sigmoid_feasible: int
    conservatism_cvar: float
    sigmoid_exact_feasible: bool

    def to_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.__dict__.items()}


def compare_reformulations(
    evaluate: Callable[[np.ndarray], dict],
    q_grid: Sequence[np.ndarray],
    alpha: float = 0.05,
    eps_list: Sequence[float] = (0.1, 0.01, 0.001),
) -> ChanceVerdict:
    """Evaluate exact/CVaR/sigmoid optima on a grid of candidate policies.

    ``evaluate(q)`` must return ``{"J": float, "G": np.ndarray}`` for the
    same scenario archive (common random numbers).
    """
    qs = [np.asarray(q, dtype=float) for q in q_grid]
    J = np.array([evaluate(q)["J"] for q in qs])
    G = [np.asarray(evaluate(q)["G"], dtype=float) for q in qs]
    M = G[0].size

    prob = np.array([np.mean(g <= 0.0) for g in G])
    cvar_g = np.array([cvar(g, alpha) for g in G])

    exact_mask = prob >= 1.0 - alpha
    if not np.any(exact_mask):
        raise RuntimeError("no grid point satisfies the exact chance constraint; "
                           "widen the grid or increase the pumping bound")
    ie = int(np.argmin(np.where(exact_mask, J, np.inf)))

    cvar_mask = cvar_g <= 0.0
    ic = int(np.argmin(np.where(cvar_mask, J, np.inf))) if np.any(cvar_mask) else None

    # sigmoid: smallest eps with a feasible grid point
    isig = None
    eps_used = float(eps_list[-1])
    for eps in eps_list:
        mask = np.array([sigmoid_probability(g, eps) >= 1.0 - alpha for g in G])
        if np.any(mask):
            isig = int(np.argmin(np.where(mask, J, np.inf)))
            eps_used = float(eps)
            break
    if isig is None:
        isig = int(np.argmin(J))
        eps_used = float(eps_list[-1])

    qe, qc, qs = qs[ie], qs[ic if ic is not None else ie], qs[isig]
    Ge = G[ie]
    return ChanceVerdict(
        exact_J=float(J[ie]),
        exact_q=qe,
        exact_reliability=float(prob[ie]),
        cvar_J=float(J[ic]) if ic is not None else float("nan"),
        cvar_q=qc,
        cvar_reliability=float(prob[ic]) if ic is not None else float("nan"),
        sigmoid_J=float(J[isig]),
        sigmoid_q=qs,
        sigmoid_reliability=float(prob[isig]),
        sigmoid_eps=eps_used,
        n_exact_feasible=int(exact_mask.sum()),
        n_cvar_feasible=int(cvar_mask.sum()),
        n_sigmoid_feasible=int(sum(
            sigmoid_probability(g, eps_used) >= 1.0 - alpha for g in G
        )),
        conservatism_cvar=float((J[ic] - J[ie]) / max(abs(J[ie]), 1e-12))
        if ic is not None else float("nan"),
        sigmoid_exact_feasible=bool(np.mean(G[isig] <= 0.0) >= 1.0 - alpha),
    )
