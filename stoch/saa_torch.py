"""GPU-backed scenario archive for the sample-average approximation.

:class:`TorchArchive` is the batched counterpart of :class:`stoch.saa.Archive`.
It keeps the random-field ensemble on the accelerator and evaluates

* all ``M`` scenarios of one candidate, or
* ``S`` candidate rate vectors on all ``M`` scenarios (one batched call)

with the same cost functional as the NumPy path, so the existing optimizer in
:mod:`stoch.optimize` can be pointed at it without changing its logic:

* :meth:`TorchArchive.evaluate` returns a :class:`stoch.saa.SAAEvaluation`, so
  every reporting/plotting helper keeps working unchanged;
* :meth:`TorchArchive.evaluate_many` returns the batched
  :class:`SAAEvaluationBatch`, which :func:`stoch.optimize.pattern_search_batched`
  uses to evaluate a whole compass set in one kernel sweep.

The NumPy archive stays the reference for the verification gates (O1--O4):
those check solver-level properties (adjoint vs. central differences at 1e-8,
mass balance at 1e-15) and are meant to run against ``splu``/``float64``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from physics.fv2d import Aquifer, FlowModel, FlowResponses, Grid, flow_responses
from physics.fv2d_torch import (
    FlowModelTorch,
    FlowResponsesTorch,
    TransportBatch,
    flow_responses_torch,
    resolve_device,
    torch_dtype,
    transport_batch,
)
from stoch.objective import CostConfig
from stoch.objective_torch import scenario_objective_batch
from stoch.random_fields import Scenario
from stoch.saa import ForwardConfig, SAAEvaluation, ScenarioProblem

__all__ = ["TorchArchive", "SAAEvaluationBatch"]


# ---------------------------------------------------------------------------
# Batched evaluation container
# ---------------------------------------------------------------------------
@dataclass
class SAAEvaluationBatch:
    """``S`` candidates evaluated on ``M`` scenarios, entirely on the device.

    ``J``, ``G``, ``H``, ``tau`` and ``failed`` are shaped ``(S, M)``;
    ``details`` holds the full tensor bundle from
    :func:`stoch.objective_torch.scenario_objective_batch`.
    """

    Q: torch.Tensor
    J: torch.Tensor
    G: torch.Tensor
    H: torch.Tensor
    tau: torch.Tensor
    failed: torch.Tensor
    compliant: torch.Tensor
    details: dict
    keep: bool = False

    @property
    def S(self) -> int:
        return self.J.shape[0]

    @property
    def M(self) -> int:
        return self.J.shape[1]

    @property
    def mean_J(self) -> torch.Tensor:
        return self.J.mean(dim=1)

    @property
    def se_J(self) -> torch.Tensor:
        if self.M < 2:
            return torch.zeros(self.S, dtype=self.J.dtype, device=self.J.device)
        return self.J.std(dim=1, unbiased=True) / (self.M ** 0.5)

    @property
    def reliability(self) -> torch.Tensor:
        return (self.G <= 0.0).to(self.J.dtype).mean(dim=1)

    @property
    def mean_tau(self) -> torch.Tensor:
        return self.tau.mean(dim=1)

    def numpy(self) -> dict:
        """Host copies of the primary per-candidate arrays."""
        t = lambda x: x.detach().cpu().numpy()  # noqa: E731
        return {
            "Q": t(self.Q),
            "J": t(self.J),
            "G": t(self.G),
            "H": t(self.H),
            "tau": t(self.tau),
            "failed": t(self.failed),
            "compliant": t(self.compliant),
        }

    def summary(self, index: int = 0) -> dict:
        """Summary statistics of one candidate, shaped like ``SAAEvaluation``."""
        t = lambda x: float(x.detach().cpu())  # noqa: E731
        J = self.J[index]
        se = self.se_J[index]
        mean_J = t(J.mean())
        details = self.details
        return {
            "M": int(self.M),
            "mean_J": mean_J,
            "se_J": t(se),
            "rel_se_J": t(se) / max(abs(mean_J), 1e-12),
            "mean_G": t(self.G[index].mean()),
            "cvar_G": t(self._cvar(self.G[index], 0.05)),
            "reliability": t(self.reliability[index]),
            "violation_prob": 1.0 - t(self.reliability[index]),
            "mean_H": t(self.H[index].mean()),
            "mean_tau": t(self.tau[index].mean()),
            "median_tau": t(self.tau[index].median()),
            "failure_fraction": t(self.failed[index].to(self.J.dtype).mean()),
            "J_pump": t(details["J_pump"][index].mean()),
            "J_treat": t(details["J_treat"][index].mean()),
            "J_res": t(details["J_res"][index].mean()),
            "J_time": t(details["J_time"][index].mean()),
        }

    @staticmethod
    def _cvar(values: torch.Tensor, alpha: float) -> torch.Tensor:
        from stoch.objective_torch import cvar_batch

        return cvar_batch(values.unsqueeze(0), alpha)[0]


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------
class TorchArchive:
    """Batched random-field archive (GPU or CPU, ``float32`` or ``float64``).

    Construction mirrors :class:`stoch.saa.Archive`: one steady-flow
    factorisation (here: one batched Krylov solve) per scenario gives the
    unit-rate responses, and every candidate rate vector then only needs the
    transport solve, which is where the batched backend wins.
    """

    supports_batching = True

    def __init__(
        self,
        grid: Grid,
        aquifer: Aquifer,
        well_xy: np.ndarray,
        scenarios: Sequence[Scenario],
        fwd: ForwardConfig,
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "float64",
        rtol: Optional[float] = None,
        maxiter: int = 4000,
        label: str = "archive",
    ) -> None:
        self.grid = grid
        self.aquifer = aquifer
        self.well_xy = np.atleast_2d(np.asarray(well_xy, dtype=float))
        self.scenarios = list(scenarios)
        self.fwd = fwd
        self.device = resolve_device(device)
        self.dtype = torch_dtype(dtype)
        self.rtol = rtol
        self.maxiter = int(maxiter)
        self.label = label

        self.compliance_cells = grid.indices(
            [p[0] for p in fwd.compliance_xy], [p[1] for p in fwd.compliance_xy]
        )
        self.obs_cells = grid.indices(
            [p[0] for p in fwd.obs_xy], [p[1] for p in fwd.obs_xy]
        )
        self.source_idx = grid.index(*fwd.source_xy)

        K = np.stack([np.asarray(s.K, dtype=float) for s in self.scenarios])
        h_left = np.array([fwd.grad * grid.lx * s.gradient_scale
                           for s in self.scenarios], dtype=float)
        self.flow = FlowModelTorch(
            grid, K, phi=aquifer.phi, b=aquifer.b,
            h_left=h_left, h_right=0.0,
            dtype=self.dtype, device=self.device,
            rtol=rtol, maxiter=self.maxiter,
        )
        self.resp: FlowResponsesTorch = flow_responses_torch(
            self.flow, self.well_xy, fwd.eps_well
        )
        self.source_rate = torch.as_tensor(
            np.array([fwd.source_q * fwd.source_c * s.source_scale
                      for s in self.scenarios], dtype=float),
            dtype=self.dtype, device=self.device,
        )
        # b * R * phi * cell volume: multiplies C_final for the residual mass.
        self.mass_weight = (
            self.flow.b * float(aquifer.R) * self.flow.phi_flat * grid.cell_volume
        ).reshape(self.M, grid.n)

    # -- basic properties -------------------------------------------------
    def __len__(self) -> int:
        return len(self.scenarios)

    @property
    def M(self) -> int:
        return len(self.scenarios)

    @property
    def n_wells(self) -> int:
        return self.resp.n_wells

    def describe(self) -> dict:
        props = {}
        if self.device.type == "cuda":
            props = {
                "gpu": torch.cuda.get_device_name(self.device),
                "total_memory_GB": round(
                    torch.cuda.get_device_properties(self.device).total_memory / 1e9, 2
                ),
            }
        return {
            "backend": "torch",
            "label": self.label,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "scenarios": self.M,
            "wells": self.n_wells,
            "grid": [self.grid.nx, self.grid.ny],
            "flow_krylov": self.resp.report.as_dict(),
            **props,
        }

    # -- evaluation -------------------------------------------------------
    def _as_Q(self, q) -> torch.Tensor:
        if isinstance(q, torch.Tensor):
            Q = q.to(dtype=self.dtype, device=self.device)
        else:
            Q = torch.as_tensor(np.atleast_2d(np.asarray(q, dtype=float)),
                                dtype=self.dtype, device=self.device)
        if Q.dim() == 1:
            Q = Q.unsqueeze(0)
        if Q.shape[1] != self.n_wells:
            raise ValueError(
                f"q has {Q.shape[1]} entries, expected {self.n_wells} wells"
            )
        return Q

    def transport(self, Q: torch.Tensor, store_fields: bool = False) -> TransportBatch:
        """Batched transport solve for rate vectors ``Q`` (shape ``(S, nw)``)."""
        fwd = self.fwd
        return transport_batch(
            self.grid,
            self.aquifer,
            self.flow,
            self.resp,
            Q,
            fwd.dt,
            fwd.nsteps,
            source_idx=self.source_idx,
            source_rate=self.source_rate,
            source_steps=fwd.source_steps,
            record_cells=self.compliance_cells,
            store_fields=store_fields,
            rtol=self.rtol,
            maxiter=self.maxiter,
        )

    def evaluate_many(
        self,
        Q,
        cost: CostConfig,
        keep: bool = False,
    ) -> SAAEvaluationBatch:
        """Evaluate ``S`` candidates on all ``M`` scenarios in one batch."""
        Q = self._as_Q(Q)
        res = self.transport(Q, store_fields=keep)
        h = self.resp.head(Q)                                  # (S, M, n)
        h_obs = h[:, :, self.obs_cells]                        # (S, M, n_obs)
        out = scenario_objective_batch(
            res, Q, h_obs, self.fwd.dt, self.fwd.T, cost, self.mass_weight
        )
        return SAAEvaluationBatch(
            Q=Q, J=out["J"], G=out["G"], H=out["H"], tau=out["tau_c"],
            failed=out["failed"], compliant=out["compliance_prob"],
            details=out, keep=keep,
        )

    def evaluate(
        self,
        q,
        cost: CostConfig,
        keep: bool = False,
        progress: bool = False,
    ) -> SAAEvaluation:
        """Drop-in replacement for :meth:`stoch.saa.Archive.evaluate`.

        Returns a plain :class:`stoch.saa.SAAEvaluation` with NumPy arrays, so
        ``SAAEvaluation.summary()``, the reporting helpers and the plotting
        code work unchanged.
        """
        batch = self.evaluate_many(np.atleast_2d(np.asarray(q, dtype=float)), cost,
                                   keep=keep)
        Q = batch.Q[0].detach().cpu().numpy()
        npv = lambda t: t[0].detach().cpu().numpy()  # noqa: E731
        J = npv(batch.J)
        G = npv(batch.G)
        H = npv(batch.H)
        tau = npv(batch.tau)
        failed = npv(batch.failed)
        compliant = npv(batch.compliant)

        h_all = self.resp.head(batch.Q[0]).detach().cpu().numpy()   # (M, n)
        details: list[dict] = []
        for m in range(self.M):
            d = {
                "J": float(J[m]),
                "J_pump": float(batch.details["J_pump"][0, m]),
                "J_treat": float(batch.details["J_treat"][0, m]),
                "J_res": float(batch.details["J_res"][0, m]),
                "J_time": float(batch.details["J_time"][0, m]),
                "G": float(G[m]),
                "H": float(H[m]),
                "tau_c": float(tau[m]),
                "failed": bool(failed[m]),
                "compliance_prob": bool(compliant[m]),
                "mass_balance_error": float(batch.details["mass_balance_error"][0, m]),
                "v_t": npv(batch.details["v_t"])[:, m],
                "arg_cells": npv(batch.details["arg_cells"])[:, m],
                "C_comp": npv(batch.details["C_comp"])[:, m, :],
            }
            d["h"] = h_all[m]
            if keep:
                d["C_final"] = npv(batch.details["C_final"])[m]
                d["C_well"] = npv(batch.details["C_well"])[:, m, :]
            details.append(d)

        if progress:
            for m in range(self.M):
                print(f"  scenario {m + 1}/{self.M}: J={J[m]:.4g} G={G[m]:.4g}")

        return SAAEvaluation(Q, J, G, H, tau, failed, compliant, details, keep=keep)

    @property
    def problems(self):
        """Not available: the discrete adjoint is a NumPy-backend feature.

        ``SAAProblem.gradient`` (and therefore SLSQP and the O3 gate) reads
        ``archive.problems`` and the SciPy operators inside
        ``TransportResult.operators``.  Rather than return something that would
        silently produce a wrong gradient, this raises.
        """
        raise NotImplementedError(
            "the discrete adjoint / SLSQP path is implemented for the NumPy "
            "archive only; run with --backend numpy (the pattern search does "
            "not need gradients). Use get_problem(i) for per-scenario NumPy "
            "problems."
        )

    # -- helpers used by the NumPy-side reporting / figures ---------------
    def get_problem(self, index: int = 0) -> ScenarioProblem:
        """Build a NumPy reference problem for one scenario (figures, checks).

        This is the escape hatch that keeps the plotting code -- which wants a
        :class:`physics.fv2d.FlowModel`/``TransportResult`` -- working with a
        GPU archive without maintaining two figure code paths.
        """
        s = self.scenarios[index]
        aq = Aquifer(K=s.K, phi=self.aquifer.phi, R=self.aquifer.R,
                     b=self.aquifer.b, alpha_l=self.aquifer.alpha_l,
                     alpha_t=self.aquifer.alpha_t, decay=self.aquifer.decay)
        h_left = self.fwd.grad * self.grid.lx * s.gradient_scale
        flow = FlowModel(self.grid, aq, h_left=h_left, h_right=0.0)
        resp: FlowResponses = flow_responses(flow, self.well_xy, self.fwd.eps_well)
        flow.release_factorization()
        rate = self.fwd.source_q * self.fwd.source_c * s.source_scale
        return ScenarioProblem(s, flow, resp, self.source_idx, rate,
                               compliance_cells=self.compliance_cells)

    def evaluate_numpy(self, q, cost: CostConfig,
                       n_scenarios: int = 1) -> SAAEvaluation:
        """Evaluate one candidate with the NumPy reference on a subset.

        The escape hatch for "is the GPU telling me the same story?" -- build
        the SciPy problems for the first ``n_scenarios`` scenarios and solve
        them with ``splu``.  Every scenario costs one factorisation, so the
        default is a single scenario.
        """
        from stoch.saa import evaluate_archive

        q = np.atleast_1d(np.asarray(q, dtype=float))
        n = max(1, min(int(n_scenarios), self.M))
        sub = _ReferenceArchive(self, n)
        return evaluate_archive(sub, q, cost)


class _ReferenceArchive:
    """Minimal NumPy archive shim over the first ``n`` scenarios of a torch one."""

    def __init__(self, torch_archive: "TorchArchive", n: int) -> None:
        self.grid = torch_archive.grid
        self.aquifer = torch_archive.aquifer
        self.well_xy = torch_archive.well_xy
        self.fwd = torch_archive.fwd
        self.scenarios = torch_archive.scenarios[:n]
        self.problems = [torch_archive.get_problem(m) for m in range(n)]
        self.compliance_cells = torch_archive.compliance_cells
        self.obs_cells = torch_archive.obs_cells
        self.source_idx = torch_archive.source_idx

    def evaluate(self, q, cost, keep=False, progress=False):
        from stoch.saa import evaluate_archive

        return evaluate_archive(self, q, cost, keep=keep, progress=progress)
