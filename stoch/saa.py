"""Scenario archive, sample-average approximation, and out-of-sample checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from physics.fv2d import (
    Aquifer,
    FlowModel,
    FlowResponses,
    Grid,
    flow_responses,
    transport_run,
)
from stoch.objective import CostConfig, cvar, scenario_objective
from stoch.random_fields import Scenario, ScenarioGenerator

__all__ = [
    "ForwardConfig",
    "ScenarioProblem",
    "Archive",
    "SAAEvaluation",
    "evaluate_archive",
]


@dataclass
class ForwardConfig:
    dt: float = 36.5
    nsteps: int = 100
    grad: float = 1e-3
    source_xy: tuple[float, float] = (30.0, 50.0)
    source_q: float = 10.0
    source_c: float = 100.0
    source_days: float = 365.0
    compliance_xy: Sequence[tuple[float, float]] = ((185.0, 50.0),)
    obs_xy: Sequence[tuple[float, float]] = ((100.0, 50.0),)
    eps_well: float = 1.5
    mu: float = 1.0  # depth-averaged thickness factor for K

    @property
    def T(self) -> float:
        return self.dt * self.nsteps

    @property
    def source_steps(self) -> int:
        return max(int(round(self.source_days / self.dt)), 1)


@dataclass
class ScenarioProblem:
    scenario: Scenario
    flow: FlowModel
    resp: FlowResponses
    source_idx: int
    source_rate: float
    compliance_cells: Optional[np.ndarray] = None

    @property
    def K(self) -> np.ndarray:
        return self.scenario.K


class Archive:
    """A fixed set of scenarios with cached unit-rate flow responses.

    The expensive flow factorisation is done once per scenario; every
    evaluation of a candidate ``q`` only re-solves the transport equation.
    """

    def __init__(
        self,
        grid: Grid,
        aquifer: Aquifer,
        well_xy: np.ndarray,
        scenarios: list[Scenario],
        fwd: ForwardConfig,
    ) -> None:
        self.grid = grid
        self.aquifer = aquifer
        self.well_xy = np.atleast_2d(np.asarray(well_xy, dtype=float))
        self.fwd = fwd
        self.scenarios = list(scenarios)
        self.problems: list[ScenarioProblem] = []
        self.compliance_cells = grid.indices(
            [p[0] for p in fwd.compliance_xy], [p[1] for p in fwd.compliance_xy]
        )
        self.obs_cells = grid.indices(
            [p[0] for p in fwd.obs_xy], [p[1] for p in fwd.obs_xy]
        )
        self.source_idx = grid.index(*fwd.source_xy)
        self._build()

    def _build(self) -> None:
        for s in self.scenarios:
            aq = Aquifer(
                K=s.K,
                phi=self.aquifer.phi,
                R=self.aquifer.R,
                b=self.aquifer.b,
                alpha_l=self.aquifer.alpha_l,
                alpha_t=self.aquifer.alpha_t,
                decay=self.aquifer.decay,
            )
            h_left = self.fwd.grad * self.grid.lx * s.gradient_scale
            flow = FlowModel(self.grid, aq, h_left=h_left, h_right=0.0)
            resp = flow_responses(flow, self.well_xy, self.fwd.eps_well)
            flow.release_factorization()
            rate = self.fwd.source_q * self.fwd.source_c * s.source_scale
            self.problems.append(
                ScenarioProblem(s, flow, resp, self.source_idx, rate,
                                compliance_cells=self.compliance_cells)
            )

    # -- evaluation -------------------------------------------------------
    def evaluate(
        self,
        q: Sequence[float],
        cost: CostConfig,
        keep: bool = False,
        progress: bool = False,
    ) -> "SAAEvaluation":
        return evaluate_archive(self, q, cost, keep=keep, progress=progress)


@dataclass
class SAAEvaluation:
    q: np.ndarray
    J: np.ndarray
    G: np.ndarray
    H: np.ndarray
    tau: np.ndarray
    failed: np.ndarray
    compliant: np.ndarray
    details: list[dict]
    keep: bool = False

    # -- summary statistics ----------------------------------------------
    @property
    def M(self) -> int:
        return self.J.size

    @property
    def mean_J(self) -> float:
        return float(np.mean(self.J))

    @property
    def se_J(self) -> float:
        return float(np.std(self.J, ddof=1) / np.sqrt(self.M)) if self.M > 1 else 0.0

    @property
    def reliability(self) -> float:
        return float(np.mean(self.G <= 0.0))

    @property
    def cvar_G(self) -> float:
        return cvar(self.G, 0.05)

    @property
    def cvar_H(self) -> float:
        return cvar(self.H, 0.01)

    @property
    def mean_tau(self) -> float:
        return float(np.mean(self.tau))

    def summary(self) -> dict:
        return {
            "M": int(self.M),
            "mean_J": self.mean_J,
            "se_J": self.se_J,
            "rel_se_J": self.se_J / max(abs(self.mean_J), 1e-12),
            "mean_G": float(np.mean(self.G)),
            "cvar_G": self.cvar_G,
            "reliability": self.reliability,
            "violation_prob": 1.0 - self.reliability,
            "mean_H": float(np.mean(self.H)),
            "cvar_H": self.cvar_H,
            "mean_tau": self.mean_tau,
            "median_tau": float(np.median(self.tau)),
            "p90_tau": float(np.quantile(self.tau, 0.9)),
            "failure_fraction": float(np.mean(self.failed)),
            "J_pump": float(np.mean([d["J_pump"] for d in self.details])),
            "J_treat": float(np.mean([d["J_treat"] for d in self.details])),
            "J_res": float(np.mean([d["J_res"] for d in self.details])),
            "J_time": float(np.mean([d["J_time"] for d in self.details])),
        }


def evaluate_archive(
    archive: Archive,
    q: Sequence[float],
    cost: CostConfig,
    keep: bool = False,
    progress: bool = False,
) -> SAAEvaluation:
    q = np.asarray(q, dtype=float).ravel()
    nw = archive.well_xy.shape[0]
    if q.size != nw:
        raise ValueError(f"q has length {q.size}, expected {nw}")
    fwd = archive.fwd
    dt, nt = fwd.dt, fwd.nsteps
    J = np.zeros(len(archive.problems))
    G = np.zeros_like(J)
    H = np.zeros_like(J)
    tau = np.zeros_like(J)
    failed = np.zeros_like(J, dtype=bool)
    compliant = np.zeros_like(J, dtype=bool)
    details: list[dict] = []

    for m, prob in enumerate(archive.problems):
        h = prob.resp.head(q)
        res = transport_run(
            archive.grid,
            prob.flow.aquifer,
            prob.flow,
            prob.resp,
            q,
            dt,
            nt,
            source_idx=prob.source_idx,
            source_rate=prob.source_rate,
            source_steps=fwd.source_steps,
            record_cells=archive.compliance_cells,
            store_fields=keep,
            return_operator=keep,
        )
        d = scenario_objective(
            archive.grid,
            prob.flow.aquifer,
            res,
            q,
            h[archive.obs_cells],
            dt,
            fwd.T,
            cost,
        )
        J[m] = d["J"]
        G[m] = d["G"]
        H[m] = d["H"]
        tau[m] = d["tau_c"]
        failed[m] = d["failed"]
        compliant[m] = d["compliance_prob"]
        if keep:
            d["res"] = res
            d["h"] = h
        details.append(d)
        if progress:
            print(f"  scenario {m + 1}/{len(archive.problems)}: J={J[m]:.4g} G={G[m]:.4g}")

    return SAAEvaluation(q, J, G, H, tau, failed, compliant, details, keep=keep)


def suggest_wells_from_archive(
    archive: Archive,
    cost: CostConfig,
    threshold: float,
    n_wells: int,
    method: str = "hex",
    seed: int = 0,
):
    """Design plume from the mean no-pumping plume of the archive."""
    from stoch.wells import place_wells, plume_mask

    zeros = np.zeros(archive.well_xy.shape[0])
    ev = archive.evaluate(zeros, cost, keep=True)
    Cmean = np.mean([d["C_final"] for d in ev.details], axis=0).reshape(
        archive.grid.ny, archive.grid.nx
    )
    mask = plume_mask(archive.grid, Cmean, threshold)
    return place_wells(archive.grid, mask, n_wells, method=method, seed=seed), Cmean
