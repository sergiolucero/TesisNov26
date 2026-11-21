"""Batched PyTorch (GPU) 2-D finite-volume flow and transport.

Why this module exists
----------------------
:mod:`physics.fv2d` is the exact, well-tested NumPy/SciPy reference.  Its
bottleneck is structural: every scenario owns one
``scipy.sparse.linalg.splu`` of the ``(nx*ny)`` transport matrix, and the
stochastic optimizer needs that factorisation once per scenario per candidate
rate vector per time step.  SciPy's sparse LU is serial and CPU-only, so on a
Colab GPU the accelerator sits idle while the CPU grinds through the
optimization stage.

This module keeps the *same discretisation* and changes the execution model:

* **Batched over scenarios and over candidate strategies.**  Tensors of shape
  ``(S, M, ny, nx)`` hold ``S`` candidate pumping vectors for each of ``M``
  random-field scenarios, so every kernel launch advances ``S*M`` PDE solves.
  That is the parallelism a GPU is built for.
* **Matrix-free stencils.**  The five-point operator is stored as face
  coefficients and applied with shifted multiplies, so assembly is a handful
  of elementwise kernels instead of Python-level sparse insertion.
* **Batched Krylov solves.**  Darcy flow is symmetric positive definite ->
  batched conjugate gradients.  Implicit-Euler transport is nonsymmetric
  (upwind advection) -> batched BiCGSTAB.  Both use Jacobi preconditioning and
  the transport loop is warm-started from the previous time level, so there is
  no sparse LU and no host round-trip per step.
* **Identical algebra.**  Coefficients, boundary conditions, well kernels,
  frozen dispersion and mass diagnostics are transcribed from
  :mod:`physics.fv2d`, so results match the NumPy reference to solver
  tolerance (``tests/test_fv2d_torch.py`` checks this).

Precision
---------
The solvers are iterative, so accuracy is set by ``rtol``.  ``float64`` (the
default) uses ``rtol=1e-11``, which reproduces ``splu`` well inside the 1e-8
the reference tests demand.  ``float32`` is supported because it is ~2x faster
on GPUs, but it is a screening mode: the verification gates in
``SCRIPTS/run_optimization.py`` stay on the NumPy ``float64`` path.

Everything here is device-agnostic: it runs on CPU (the test suite needs no
GPU) and on CUDA when available.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from physics.fv2d import Aquifer, Grid, gaussian_well_weights

__all__ = [
    "DTYPES",
    "resolve_device",
    "torch_dtype",
    "default_rtol",
    "KrylovReport",
    "Stencil5",
    "FlowModelTorch",
    "FlowResponsesTorch",
    "TransportBatch",
    "well_weights_torch",
    "flow_responses_torch",
    "advection_stencil",
    "dispersion_stencil",
    "face_dispersion_torch",
    "transport_batch",
    "cg_solve",
    "bicgstab_solve",
]


DTYPES = {"float32": torch.float32, "float64": torch.float64}


# ---------------------------------------------------------------------------
# Device / dtype plumbing
# ---------------------------------------------------------------------------
def resolve_device(spec: str | torch.device | None = "auto") -> torch.device:
    """Resolve ``auto`` / ``cuda`` / ``cuda:1`` / ``cpu`` / ``mps``."""
    if isinstance(spec, torch.device):
        return spec
    s = (spec or "auto").strip().lower()
    if s in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = torch.device(s)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"requested device {s!r} but torch.cuda.is_available() is False "
            f"(torch {torch.__version__}, CUDA build {torch.version.cuda})"
        )
    return dev


def torch_dtype(spec: str | torch.dtype = "float64") -> torch.dtype:
    if isinstance(spec, torch.dtype):
        return spec
    try:
        return DTYPES[str(spec).lower()]
    except KeyError:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {spec!r}") from None


def default_rtol(dtype: torch.dtype) -> float:
    """Relative Krylov tolerance that the given dtype can actually reach.

    float64 has ~2.2e-16 of head-room, so 1e-11 reproduces ``splu`` far inside
    the 1e-8 parity band.

    float32 has ~1.2e-7 and loses several digits to cancellation in a batched
    sparse solve: measured on the 80x40 benchmark the attainable relative
    residual sits at 9.8e-7 .. 1.0e-6, i.e. *exactly* at 1e-6.  Asking for 1e-6
    makes convergence a coin flip decided by reduction order, and the losing
    side pays the full ``maxiter`` on every one of the 100 time steps.  A
    production A100 run did that: 55 of 75 compass sweeps took 2.1 s, 20 took
    272 s median (508 s worst), 98% of the wall clock in that minority.  1e-5 is
    five decades above the noise floor, so it converges deterministically.
    """
    return 1e-11 if dtype == torch.float64 else 1e-5


# ---------------------------------------------------------------------------
# Batched Krylov solvers
# ---------------------------------------------------------------------------
@dataclass
class KrylovReport:
    """Aggregated convergence diagnostics (accumulated over a time loop)."""

    iterations: int = 0          # total Krylov iterations
    solves: int = 0              # number of batched solves
    worst_iterations: int = 0    # iterations of the slowest single solve
    worst_residual: float = 0.0  # final relative residual, worst case
    converged: bool = True
    stalled: bool = False        # gave up on the stagnation guard, not on maxiter

    def add(self, other: "KrylovReport") -> None:
        self.iterations += other.iterations
        self.solves += other.solves
        self.worst_iterations = max(self.worst_iterations, other.worst_iterations)
        # ``max(0.0, nan)`` is 0.0 in Python, which once hid a NaN solve behind
        # a reassuring 'worst residual 0.00e+00' in a production run's log.
        if not math.isfinite(other.worst_residual):
            self.worst_residual = float("nan")
        elif math.isfinite(self.worst_residual):
            self.worst_residual = max(self.worst_residual, other.worst_residual)
        self.converged = self.converged and other.converged
        self.stalled = self.stalled or other.stalled

    @property
    def iterations_per_solve(self) -> float:
        return self.iterations / self.solves if self.solves else 0.0

    def as_dict(self) -> dict:
        return {
            "iterations": int(self.iterations),
            "solves": int(self.solves),
            "iterations_per_solve": round(float(self.iterations_per_solve), 3),
            "worst_iterations": int(self.worst_iterations),
            "worst_residual": float(self.worst_residual),
            "converged": bool(self.converged),
            "stalled": bool(self.stalled),
        }


def _flat_norm(t: torch.Tensor) -> torch.Tensor:
    """Per-batch-element Euclidean norm for ``(B, ...)`` tensors.

    float32 is scaled first (LAPACK ``nrm2`` style).  Squaring a field whose
    largest entry is ~1e20 overflows the float32 exponent range and returns
    ``inf``, and ``inf/inf`` in the convergence test is ``nan`` -- which is how
    a *correct* solve can be reported as broken.  The scaled form bounds every
    squared term by 1, so the reduction cannot overflow.
    """
    flat = t.reshape(t.shape[0], -1)
    if t.dtype != torch.float32:
        return torch.linalg.vector_norm(flat, dim=1)
    if SCALAR_FP64:
        return torch.linalg.vector_norm(flat.double(), dim=1).to(torch.float32)
    amax = flat.abs().amax(dim=1)
    safe = torch.where(amax > 0, amax, torch.ones_like(amax))
    return amax * torch.linalg.vector_norm(flat / safe.unsqueeze(1), dim=1)


def _scalar(t: torch.Tensor) -> float:
    """Host-side float of a tensor on any device.

    ``float(t)`` works on CPU but raises on a CUDA tensor ("can't convert
    cuda:0 device type tensor to numpy"). These calls sit in the Krylov
    convergence tests, so a real GPU run failed immediately in cg_solve while
    every CPU test passed. ``.item()`` is device-agnostic.
    """
    return float(t.detach().item())


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-batch-element inner product for ``(B, ...)`` tensors.

    Same overflow argument as :func:`_flat_norm`: in float32 the product of two
    large work vectors can exceed the exponent range, so each operand is scaled
    by its own max and the scale is folded back afterwards.
    """
    fa = a.reshape(a.shape[0], -1)
    fb = b.reshape(b.shape[0], -1)
    if a.dtype != torch.float32:
        return (fa * fb).sum(dim=1)
    if SCALAR_FP64:
        return (fa.double() * fb.double()).sum(dim=1).to(torch.float32)
    sa = fa.abs().amax(dim=1)
    sb = fb.abs().amax(dim=1)
    sa = torch.where(sa > 0, sa, torch.ones_like(sa))
    sb = torch.where(sb > 0, sb, torch.ones_like(sb))
    return (sa * sb) * (fa / sa.unsqueeze(1) * (fb / sb.unsqueeze(1))).sum(dim=1)


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(num.dtype).tiny
    return num / torch.where(den.abs() < tiny, torch.full_like(den, tiny), den)


def _scale(coeff: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Reshape a per-batch scalar so it broadcasts against ``like``."""
    return coeff.reshape(-1, *([1] * (like.dim() - 1)))


#: A solve is declared stalled when it fails to set a new *best* residual for
#: this many consecutive iterations.  The test is deliberately weak ("no
#: improvement at all", not "no big improvement"): a well-conditioned CG that
#: needs 900 iterations still improves the best residual almost every step, so
#: it never trips the guard, while a float32 solve sitting on its ~1e-6 noise
#: floor bounces around without ever beating its best and is caught in
#: ``STALL_WINDOW`` steps instead of burning ``maxiter`` (4000) per time step.
STALL_WINDOW = 60
SCALAR_FP64 = os.environ.get("TESIS_SCALAR_FP64", "1") not in {"0", "false", "False"}
"""Accumulate Krylov scalars in float64 even when the vectors are float32.

A float32 Krylov loop loses accuracy where it multiplies *accumulated* work
vectors: ``<rhat, r>``, ``||r||`` and the beta/omega ratios sum ~1e4-1e5 terms of
similar magnitude, so the reduction error is what decides whether the residual
can reach 1e-6 at all. Measured on a T4 at 160x96 (M40) the batched BiCGSTAB
stalled with a NaN residual on 4 of 19 compass candidates; the fix that needs no
extra memory or code in the solver is to promote only the *scalars* to float64
(the vectors stay float32, so the stencil matvec -- the cost centre -- is
unchanged). Set ``TESIS_SCALAR_FP64=0`` to get the plain float32 reductions back
for an A/B measurement.
"""


MIXED_KRYLOV = os.environ.get("TESIS_MIXED_KRYLOV", "1") not in {"0", "false", "False"}
"""Run the whole BiCGSTAB recurrence in float64, with a float32 matvec.

Promoting only the *scalars* (``SCALAR_FP64``) fixed the float32 transport at
M=40 but not at M=100: with a batch of 1900 systems the float32 *recurrence*
itself -- ``p = r + beta*(p - omega*v)`` and the ``alpha``/``omega`` updates,
where errors accumulate rather than cancel -- lost one candidate to a NaN
residual after 28 iterations.

``MIXED_KRYLOV`` keeps the work vectors and the recurrence scalars in float64
and casts only the input of each matvec down to float32 (the operator holds
float32 face coefficients), casting the result back up. The stencil apply, which
is the cost centre, stays float32; the extra cost is fp64 elementwise traffic on
the ~8 work vectors. Set ``TESIS_MIXED_KRYLOV=0`` for the pure-float32 loop.
"""


class _StallGuard:
    """Track the best residual so far and detect a plateau.

    Bounding a stalled solve matters because the solvers are called once per
    implicit-Euler step: at 4000 iterations a single unreachable tolerance turns
    an 8 s compass sweep into a 500 s sweep (measured on a T4, 20 sweeps of 75).
    """

    __slots__ = ("best", "best_it", "stalled")

    def __init__(self) -> None:
        self.best = float("inf")
        self.best_it = 0
        self.stalled = False

    def check(self, iteration: int, residual: float) -> bool:
        """Record ``residual``; return True when the solve has stalled."""
        if not math.isfinite(residual):
            self.stalled = True
            return True
        if residual < self.best:
            self.best = residual
            self.best_it = iteration
            return False
        if iteration - self.best_it >= STALL_WINDOW:
            self.stalled = True
            return True
        return False


def cg_solve(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    diag: Optional[torch.Tensor] = None,
    rtol: float = 1e-11,
    maxiter: int = 2000,
) -> tuple[torch.Tensor, KrylovReport]:
    """Batched preconditioned conjugate gradients for SPD systems.

    Convergence is declared when *every* system in the batch satisfies
    ``||r|| <= rtol * ||b||``; ``maxiter`` bounds the number of iterations.
    """
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    bnorm = _flat_norm(b)
    scale = torch.where(bnorm > 0, bnorm, torch.ones_like(bnorm))

    z = r if diag is None else r / diag
    p = z.clone()
    rz = _dot(r, z)

    guard = _StallGuard()
    it = 0
    for it in range(1, maxiter + 1):
        cur = _scalar((_flat_norm(r) / scale).max())
        if cur <= rtol:
            break
        if guard.check(it, cur):
            break
        ap = matvec(p)
        alpha = _safe_div(rz, _dot(p, ap))
        x = x + _scale(alpha, p) * p
        r = r - _scale(alpha, r) * ap
        z = r if diag is None else r / diag
        rz_new = _dot(r, z)
        beta = _safe_div(rz_new, rz)
        p = z + _scale(beta, p) * p
        rz = rz_new

    residual = _scalar((_flat_norm(r) / scale).max())
    report = KrylovReport(iterations=it, solves=1, worst_iterations=it,
                          worst_residual=residual,
                          converged=residual <= max(rtol, 10.0 * rtol),
                          stalled=guard.stalled)
    return x, report


def bicgstab_solve(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    diag: Optional[torch.Tensor] = None,
    rtol: float = 1e-11,
    maxiter: int = 4000,
    mixed_precision: Optional[bool] = None,
) -> tuple[torch.Tensor, KrylovReport]:
    """Batched preconditioned BiCGSTAB for nonsingular systems.

    Used for the implicit-Euler transport operator (nonsymmetric upwind
    advection) and for its transpose in the discrete adjoint.

    ``mixed_precision`` (default ``MIXED_KRYLOV``) runs the recurrence in
    float64 for float32 inputs; it is a no-op for float64 inputs.
    """
    if mixed_precision is None:
        mixed_precision = MIXED_KRYLOV
    # Mixed-precision mode: the recurrence runs in float64 while the operator
    # stays float32 (see MIXED_KRYLOV above).  All work vectors therefore live
    # in float64; only the matvec sees float32.
    b0_dtype = b.dtype
    mixed = mixed_precision and b.dtype == torch.float32
    if mixed:
        _mv = matvec

        def matvec(q: torch.Tensor) -> torch.Tensor:  # noqa: F811 - local override
            return _mv(q.to(torch.float32)).to(torch.float64)

        work = torch.float64
    else:
        work = b.dtype
    b = b.to(work)
    if x0 is not None:
        x0 = x0.to(work)
    if diag is not None:
        diag = diag.to(work)

    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    bnorm = _flat_norm(b)
    scale = torch.where(bnorm > 0, bnorm, torch.ones_like(bnorm))

    rhat = r.clone()
    v = torch.zeros_like(b)
    p = torch.zeros_like(b)
    rho_old = torch.ones(b.shape[0], dtype=b.dtype, device=b.device)
    alpha = torch.ones_like(rho_old)
    omega = torch.ones_like(rho_old)

    guard = _StallGuard()
    it = 0
    for it in range(1, maxiter + 1):
        cur = _scalar((_flat_norm(r) / scale).max())
        if cur <= rtol:
            break
        if guard.check(it, cur):
            break
        rho = _dot(rhat, r)
        # BiCGSTAB breakdown guard.  <rhat, r> can vanish (or turn non-finite
        # after an alpha/omega blow-up), and the next beta = rho/rho_old *
        # alpha/omega then explodes; in float32 that ends in inf/NaN, which
        # poisoned 4 of 19 candidates in a real A100 run.  The textbook remedy
        # is to restart the affected elements with rhat = r.
        if not bool(torch.isfinite(rho).all()):
            okay = torch.isfinite(rho) & (rho.abs() > 0)
            mask = (~okay).reshape(-1, *([1] * (r.dim() - 1)))
            rhat = torch.where(mask, r, rhat)
            p = torch.where(mask, torch.zeros_like(p), p)
            v = torch.where(mask, torch.zeros_like(v), v)
            rho_old = torch.where(okay, rho_old, torch.ones_like(rho_old))
            alpha = torch.where(okay, alpha, torch.ones_like(alpha))
            omega = torch.where(okay, omega, torch.ones_like(omega))
            rho = _dot(rhat, r)
        beta = _safe_div(rho, rho_old) * _safe_div(alpha, omega)
        p = r + _scale(beta, r) * (p - _scale(omega, p) * v)
        phat = p if diag is None else p / diag
        v = matvec(phat)
        alpha = _safe_div(rho, _dot(rhat, v))
        s = r - _scale(alpha, r) * v
        if _scalar((_flat_norm(s) / scale).max()) <= rtol:
            x = x + _scale(alpha, x) * phat
            r = s
            break
        shat = s if diag is None else s / diag
        t = matvec(shat)
        omega = _safe_div(_dot(t, s), _dot(t, t))
        x = x + _scale(alpha, x) * phat + _scale(omega, x) * shat
        r = s - _scale(omega, r) * t
        rho_old = rho

    residual = _scalar((_flat_norm(r) / scale).max())
    if mixed:
        x = x.to(b0_dtype)
    report = KrylovReport(iterations=it, solves=1, worst_iterations=it,
                          worst_residual=residual,
                          converged=residual <= max(rtol, 10.0 * rtol),
                          stalled=guard.stalled)
    return x, report


# ---------------------------------------------------------------------------
# Five-point stencil
# ---------------------------------------------------------------------------
@dataclass
class Stencil5:
    """Face-coefficient representation of a cell-centred five-point operator.

    Face coefficients live on the interior faces, where the couplings are:

    * ``xe[b, j, i]`` -- coefficient of ``C[j, i+1]`` in row ``(j, i)``
      (east neighbour), for ``i = 0 .. nx-2``; shape ``(B, ny, nx-1)``.
    * ``xw[b, j, i]`` -- coefficient of ``C[j, i]`` in row ``(j, i+1)``
      (west neighbour), for ``i = 0 .. nx-2``.
    * ``yn[b, j, i]`` -- coefficient of ``C[j+1, i]`` in row ``(j, i)``
      (north neighbour); shape ``(B, ny-1, nx)``.
    * ``ys[b, j, i]`` -- coefficient of ``C[j, i]`` in row ``(j+1, i)``
      (south neighbour).

    Symmetric operators (dispersion, Darcy flow) have ``xe == xw``; upwind
    advection does not.  Keeping the two directions separate is what lets the
    transpose (the discrete adjoint) be exact: it is the same stencil with
    ``xe``/``xw`` and ``yn``/``ys`` swapped.
    """

    diag: torch.Tensor   # (B, ny, nx)
    xe: torch.Tensor     # (B, ny, nx-1)
    xw: torch.Tensor     # (B, ny, nx-1)
    yn: torch.Tensor     # (B, ny-1, nx)
    ys: torch.Tensor     # (B, ny-1, nx)

    # -- construction ----------------------------------------------------
    @staticmethod
    def zeros(diag: torch.Tensor) -> "Stencil5":
        b, ny, nx = diag.shape

        def z(*shape):
            return torch.zeros(shape, dtype=diag.dtype, device=diag.device)

        return Stencil5(diag, z(b, ny, nx - 1), z(b, ny, nx - 1),
                        z(b, ny - 1, nx), z(b, ny - 1, nx))

    @staticmethod
    def symmetric(diag: torch.Tensor, xf: torch.Tensor, yf: torch.Tensor) -> "Stencil5":
        """Stencil whose face couplings are symmetric (``xe == xw``)."""
        return Stencil5(diag, xf, xf, yf, yf)

    def __add__(self, other: "Stencil5") -> "Stencil5":
        return Stencil5(self.diag + other.diag, self.xe + other.xe,
                        self.xw + other.xw, self.yn + other.yn,
                        self.ys + other.ys)

    def scaled(self, factor: float) -> "Stencil5":
        return Stencil5(self.diag * factor, self.xe * factor, self.xw * factor,
                        self.yn * factor, self.ys * factor)

    def with_diagonal(self, diag: torch.Tensor) -> "Stencil5":
        return Stencil5(diag, self.xe, self.xw, self.yn, self.ys)

    # -- application -----------------------------------------------------
    def apply(self, c: torch.Tensor) -> torch.Tensor:
        """``A c`` for ``c`` shaped ``(B, ny, nx)``."""
        out = self.diag * c
        out[:, :, :-1] = out[:, :, :-1] + self.xe * c[:, :, 1:]
        out[:, :, 1:] = out[:, :, 1:] + self.xw * c[:, :, :-1]
        out[:, :-1, :] = out[:, :-1, :] + self.yn * c[:, 1:, :]
        out[:, 1:, :] = out[:, 1:, :] + self.ys * c[:, :-1, :]
        return out

    def apply_transpose(self, c: torch.Tensor) -> torch.Tensor:
        """``A^T c``: the same couplings with the two directions swapped."""
        out = self.diag * c
        out[:, :, :-1] = out[:, :, :-1] + self.xw * c[:, :, 1:]
        out[:, :, 1:] = out[:, :, 1:] + self.xe * c[:, :, :-1]
        out[:, :-1, :] = out[:, :-1, :] + self.ys * c[:, 1:, :]
        out[:, 1:, :] = out[:, 1:, :] + self.yn * c[:, :-1, :]
        return out

    def transpose(self) -> "Stencil5":
        return Stencil5(self.diag, self.xw, self.xe, self.ys, self.yn)


# ---------------------------------------------------------------------------
# Well kernels
# ---------------------------------------------------------------------------
def well_weights_torch(
    grid: Grid,
    well_xy: Sequence[Sequence[float]],
    eps_cells: float = 1.5,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Normalised Gaussian well kernels, shape ``(nw, ny, nx)``.

    Host-side delegation to :func:`physics.fv2d.gaussian_well_weights` keeps
    the well footprint bit-identical to the NumPy reference; the kernels are
    ``nw * nx * ny`` doubles, so porting them would buy nothing.
    """
    xy = np.atleast_2d(np.asarray(well_xy, dtype=float))
    w = np.vstack([gaussian_well_weights(grid, x, y, eps_cells) for x, y in xy])
    return torch.as_tensor(w, dtype=dtype, device=device).reshape(
        w.shape[0], grid.ny, grid.nx
    )


def _harmonic(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 2.0 * a * b / (a + b)


# ---------------------------------------------------------------------------
# Steady flow
# ---------------------------------------------------------------------------
class FlowModelTorch:
    """Batched steady Darcy flow ``A h = b`` for ``M`` conductivity fields.

    ``K`` has shape ``(M, ny, nx)``.  The operator is assembled once (face
    conductances plus a diagonal) and reused for the background solve and for
    every unit-rate well response -- the same trick as the single ``splu`` in
    :class:`physics.fv2d.FlowModel`, but shared across the whole batch.
    """

    def __init__(
        self,
        grid: Grid,
        K: torch.Tensor | np.ndarray,
        phi: float | torch.Tensor | np.ndarray = 0.30,
        b: float = 10.0,
        h_left: float | torch.Tensor | np.ndarray = 0.0,
        h_right: float | torch.Tensor | np.ndarray = 0.0,
        bc_y: str = "noflow",
        h_bottom: float | torch.Tensor | np.ndarray = 0.0,
        h_top: float | torch.Tensor | np.ndarray = 0.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        rtol: Optional[float] = None,
        maxiter: int = 2000,
    ) -> None:
        if bc_y not in {"noflow", "dirichlet"}:
            raise ValueError("bc_y must be 'noflow' or 'dirichlet'")
        self.grid = grid
        self.bc_y = bc_y
        self.b = float(b)
        self.dtype = dtype
        self.device = torch.device(device)
        self.rtol = default_rtol(dtype) if rtol is None else float(rtol)
        self.maxiter = int(maxiter)

        K = self._tensor(K)
        if K.dim() == 2:
            K = K.unsqueeze(0)
        if K.shape[-2:] != (grid.ny, grid.nx):
            raise ValueError("K shape does not match the grid")
        self.M = K.shape[0]
        self.K = K
        self.phi = self._to_field(phi, K)
        self._h_left = self._to_scalar(h_left)
        self._h_right = self._to_scalar(h_right)
        self._h_bottom = self._to_scalar(h_bottom)
        self._h_top = self._to_scalar(h_top)

        g = grid
        T = K * self.b
        self.T = T
        self.cx = _harmonic(T[:, :, :-1], T[:, :, 1:]) * g.dy / g.dx   # (M,ny,nx-1)
        self.cy = _harmonic(T[:, :-1, :], T[:, 1:, :]) * g.dx / g.dy   # (M,ny-1,nx)
        self.cxL = 2.0 * T[:, :, 0] * g.dy / g.dx                      # (M,ny)
        self.cxR = 2.0 * T[:, :, -1] * g.dy / g.dx
        if bc_y == "dirichlet":
            self.cyB = 2.0 * T[:, 0, :] * g.dx / g.dy
            self.cyT = 2.0 * T[:, -1, :] * g.dx / g.dy
        else:
            self.cyB = torch.zeros_like(T[:, 0, :])
            self.cyT = torch.zeros_like(T[:, -1, :])

        self.diag, self.rhs_const = self._assemble()
        # Face coefficients as matrix entries (A[i, i+1] = -cx in fv2d).
        self._stencil = Stencil5.symmetric(self.diag, -self.cx, -self.cy)

    # -- construction helpers -------------------------------------------
    def _tensor(self, value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=self.dtype, device=self.device)
        return torch.as_tensor(np.asarray(value, dtype=float),
                               dtype=self.dtype, device=self.device)

    def _to_field(self, value, like: torch.Tensor) -> torch.Tensor:
        """Expand scalar / grid field / per-scenario field to ``like``'s shape."""
        t = self._tensor(value)
        if t.dim() == 0:
            return t.expand_as(like).contiguous()
        if t.shape == like.shape[-2:]:
            return t.expand_as(like).contiguous()
        return torch.broadcast_tensors(t, like)[0].contiguous()

    def _to_scalar(self, value) -> torch.Tensor:
        """Per-scenario scalar shaped ``(M, 1)`` so it broadcasts with ``(M, ny)``."""
        t = self._tensor(np.atleast_1d(value).astype(float)).reshape(-1, 1)
        if t.shape[0] == 1:
            t = t.expand(self.M, 1)
        return t

    def _assemble(self) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.grid
        diag = torch.zeros(self.M, g.ny, g.nx, dtype=self.dtype, device=self.device)
        diag[:, :, :-1] += self.cx
        diag[:, :, 1:] += self.cx
        diag[:, :-1, :] += self.cy
        diag[:, 1:, :] += self.cy
        diag[:, :, 0] += self.cxL
        diag[:, :, -1] += self.cxR
        if self.bc_y == "dirichlet":
            diag[:, 0, :] += self.cyB
            diag[:, -1, :] += self.cyT

        rhs = torch.zeros_like(diag)
        rhs[:, :, 0] += self.cxL * self._h_left
        rhs[:, :, -1] += self.cxR * self._h_right
        if self.bc_y == "dirichlet":
            rhs[:, 0, :] += self.cyB * self._h_bottom
            rhs[:, -1, :] += self.cyT * self._h_top
        return diag, rhs

    # -- operator --------------------------------------------------------
    def stencil(self, batch: Optional[int] = None) -> Stencil5:
        """Flow operator as a :class:`Stencil5` (optionally tiled ``batch``x)."""
        st = self._stencil
        return st if batch in (None, 1) else _tile_stencil(st, batch)

    def apply(self, h: torch.Tensor) -> torch.Tensor:
        """``A h`` for ``h`` shaped ``(M, ny, nx)``."""
        return self._stencil.apply(h)

    def solve(self, rhs_extra: Optional[torch.Tensor] = None
              ) -> tuple[torch.Tensor, KrylovReport]:
        """Solve ``A h = rhs_const + rhs_extra`` (batched CG)."""
        rhs = self.rhs_const if rhs_extra is None else self.rhs_const + rhs_extra
        return cg_solve(self.apply, rhs, diag=self.diag, rtol=self.rtol,
                        maxiter=self.maxiter)

    # -- post-processing -------------------------------------------------
    def velocity(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Face-centred seepage velocity ``v = -K grad(h) / phi``.

        ``h`` shaped ``(..., ny, nx)`` may carry leading batch dimensions; the
        conductivity, porosity and boundary heads are broadcast against it.
        Returns ``vx`` shaped ``(..., ny, nx+1)`` and ``vy`` ``(..., ny+1, nx)``.
        """
        g = self.grid
        ny, nx = g.ny, g.nx
        if h.dim() < 2:
            raise ValueError("h must have at least two (grid) dimensions")
        if h.dim() >= 3 and h.shape[-3] not in (self.M,):
            raise ValueError(
                f"velocity expects the scenario axis to match the {self.M} "
                f"conductivity fields, got leading shape {tuple(h.shape[:-2])}"
            )
        K, phi = self.K, self.phi
        h_left, h_right = self._h_left, self._h_right
        h_bottom, h_top = self._h_bottom, self._h_top
        extra = h.dim() - K.dim()
        if extra > 0:
            lead = (1,) * extra
            K = K.view(*lead, *K.shape)
            phi = phi.view(*lead, *phi.shape)
            h_left = h_left.view(*lead, *h_left.shape)
            h_right = h_right.view(*lead, *h_right.shape)
            h_bottom = h_bottom.view(*lead, *h_bottom.shape)
            h_top = h_top.view(*lead, *h_top.shape)

        lead = h.shape[:-2]
        vx = torch.zeros(*lead, ny, nx + 1, dtype=h.dtype, device=h.device)
        vy = torch.zeros(*lead, ny + 1, nx, dtype=h.dtype, device=h.device)

        Kx = _harmonic(K[..., :-1], K[..., 1:])
        vx[..., 1:nx] = -Kx * (h[..., 1:] - h[..., :-1]) / g.dx / phi[..., 1:]
        vx[..., 0] = -K[..., 0] * (h[..., 0] - h_left) / (0.5 * g.dx) / phi[..., 0]
        vx[..., nx] = -K[..., -1] * (h_right - h[..., -1]) / (0.5 * g.dx) / phi[..., -1]

        # Arithmetic on the broadcast tensors keeps the leading batch dims, so
        # the same code covers a single scenario field and a (S, M) stack.
        Ky = _harmonic(K[..., :-1, :], K[..., 1:, :])
        vy[..., 1:ny, :] = -Ky * (h[..., 1:, :] - h[..., :-1, :]) / g.dy / phi[..., 1:, :]
        if self.bc_y != "noflow":
            # Dirichlet y-boundaries at half-cell distance (fv2d leaves these
            # at zero for no-flow, which is the default).
            vy[..., 0, :] = (-K[..., 0, :] * (h[..., 0, :] - h_bottom)
                             / (0.5 * g.dy) / phi[..., 0, :])
            vy[..., ny, :] = (-K[..., -1, :] * (h_top - h[..., -1, :])
                              / (0.5 * g.dy) / phi[..., -1, :])
        return vx, vy

    @property
    def phi_flat(self) -> torch.Tensor:
        """Porosity on the grid, shape ``(M, ny, nx)``."""
        return self.phi


def _tile_stencil(st: Stencil5, batch: int) -> Stencil5:
    """Repeat every coupled operator ``batch`` times along the batch axis."""
    tile = lambda t: (t.unsqueeze(1).expand(t.shape[0], batch, *t.shape[1:])
                      .reshape(t.shape[0] * batch, *t.shape[1:]))  # noqa: E731
    return Stencil5(tile(st.diag), tile(st.xe), tile(st.xw), tile(st.yn), tile(st.ys))


def _flat_face(t: torch.Tensor, S: int) -> torch.Tensor:
    """Lift ``(M, ...)`` face coefficients to ``(S*M, ...)`` for ``S`` candidates."""
    return (t.unsqueeze(0).expand(S, *t.shape)
            .reshape(S * t.shape[0], *t.shape[1:]).contiguous())


# ---------------------------------------------------------------------------
# Flow responses (linear superposition in q)
# ---------------------------------------------------------------------------
@dataclass
class FlowResponsesTorch:
    """Unit-rate flow responses for ``M`` scenarios.

    ``head(q) = h0 + q . hw`` with ``h0`` shaped ``(M, n)`` and ``hw`` shaped
    ``(M, nw, n)``, built exactly like :class:`physics.fv2d.FlowResponses`, so
    the superposition used by the optimizer is exact.
    """

    h0: torch.Tensor              # (M, n)
    hw: torch.Tensor              # (M, nw, n)
    weights: torch.Tensor         # (nw, n)
    well_cells: np.ndarray        # (nw,)
    well_xy: np.ndarray           # (nw, 2)
    report: KrylovReport = field(default_factory=KrylovReport)

    @property
    def n_wells(self) -> int:
        return self.hw.shape[1]

    def head(self, q: torch.Tensor) -> torch.Tensor:
        """Heads for a batch of rate vectors.

        ``q`` shaped ``(S, nw)`` -> ``(S, M, ny*nx)``; ``(nw,)`` -> ``(M, ny*nx)``.
        """
        q = q.to(dtype=self.h0.dtype, device=self.h0.device)
        if q.dim() == 1:
            return self.h0 + torch.einsum("i,min->mn", q, self.hw)
        return self.h0.unsqueeze(0) + torch.einsum("si,min->smn", q, self.hw)


def flow_responses_torch(
    flow: FlowModelTorch,
    well_xy: Sequence[Sequence[float]],
    eps_cells: float = 1.5,
) -> FlowResponsesTorch:
    """Background solve plus all unit-rate well responses in one Krylov batch.

    ``nw + 1`` right-hand sides share the same matrix per scenario, so they are
    tiled into a single batched CG call: one solve instead of ``nw + 1``
    sequential factorisations.
    """
    grid = flow.grid
    weights = well_weights_torch(grid, well_xy, eps_cells,
                                 dtype=flow.dtype, device=flow.device)
    nw = weights.shape[0]
    k = nw + 1

    rhs = torch.stack(
        [flow.rhs_const]
        + [-weights[i].expand_as(flow.rhs_const) for i in range(nw)],
        dim=1,
    )                                                          # (M, k, ny, nx)
    b = rhs.reshape(flow.M * k, grid.ny, grid.nx)
    st = flow.stencil(batch=k)
    solution, report = cg_solve(st.apply, b, diag=st.diag, rtol=flow.rtol,
                                maxiter=flow.maxiter)

    sol = solution.reshape(flow.M, k, grid.n)
    well_cells = np.argmax(weights.reshape(nw, -1).detach().cpu().numpy(), axis=1)
    return FlowResponsesTorch(
        h0=sol[:, 0, :],
        hw=sol[:, 1:, :],
        weights=weights.reshape(nw, grid.n),
        well_cells=well_cells,
        well_xy=np.atleast_2d(np.asarray(well_xy, dtype=float)),
        report=report,
    )


# ---------------------------------------------------------------------------
# Transport stencil assembly
# ---------------------------------------------------------------------------
def face_dispersion_torch(
    grid: Grid,
    aquifer: Aquifer,
    vx: torch.Tensor,
    vy: torch.Tensor,
    phi: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Face dispersion conductances, transcribed from ``fv2d.face_dispersion``.

    The tensor is frozen at the velocity passed in (the background, no-pumping
    velocity), which is what makes the discrete adjoint exact.
    """
    ny, nx = grid.ny, grid.nx
    if phi is None:
        phi = torch.as_tensor(np.asarray(aquifer.phi_field((ny, nx))),
                              dtype=vx.dtype, device=vx.device)
    phi = phi.to(dtype=vx.dtype, device=vx.device)

    vxc = 0.5 * (vx[..., :-1] + vx[..., 1:])          # (..., ny, nx)
    vyc = 0.5 * (vy[..., :-1, :] + vy[..., 1:, :])    # (..., ny, nx)
    speed = torch.sqrt(vxc ** 2 + vyc ** 2)
    speed = torch.clamp(speed, min=1e-12)
    a_l, a_t = float(aquifer.alpha_l), float(aquifer.alpha_t)
    Dxx = a_t * speed + (a_l - a_t) * vxc ** 2 / speed
    Dyy = a_t * speed + (a_l - a_t) * vyc ** 2 / speed
    Dxx_f = 0.5 * (Dxx[..., :, :-1] + Dxx[..., :, 1:])
    Dyy_f = 0.5 * (Dyy[..., :-1, :] + Dyy[..., 1:, :])
    phix = 0.5 * (phi[..., :, :-1] + phi[..., :, 1:])
    phiy = 0.5 * (phi[..., :-1, :] + phi[..., 1:, :])
    b = float(aquifer.b)
    gx = b * phix * Dxx_f * grid.dy / grid.dx
    gy = b * phiy * Dyy_f * grid.dx / grid.dy
    return gx, gy


def dispersion_stencil(gx: torch.Tensor, gy: torch.Tensor) -> Stencil5:
    """Symmetric five-point operator from face conductances.

    ``gx`` has shape ``(B, ny, nx-1)`` and ``gy`` ``(B, ny-1, nx)``; the
    diagonal is the sum of the incident conductances.
    """
    b, ny, nx = gx.shape[0], gx.shape[1], gx.shape[2] + 1
    diag = torch.zeros(b, ny, nx, dtype=gx.dtype, device=gx.device)
    diag[:, :, :-1] += gx
    diag[:, :, 1:] += gx
    diag[:, :-1, :] += gy
    diag[:, 1:, :] += gy
    return Stencil5.symmetric(diag, -gx, -gy)


def advection_stencil(
    grid: Grid,
    phi_t: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    vx_sign: Optional[torch.Tensor] = None,
    vy_sign: Optional[torch.Tensor] = None,
) -> Stencil5:
    """First-order upwind advection operator, transcribed from ``fv2d``.

    ``phi_t`` is the depth-integrated porosity ``b * phi`` shaped
    ``(B, ny, nx)`` (or broadcastable to it).  ``vx_sign`` / ``vy_sign``
    optionally fix the upwind direction independently of the flux magnitude;
    the discrete adjoint uses that to keep ``A(v)`` linear in the rate vector.

    Boundary x-faces with inflow use ``C = 0`` (clean water); the y-boundaries
    are no-flow in the baseline configuration.
    """
    ny, nx = grid.ny, grid.nx
    if vx_sign is None:
        vx_sign = vx
    if vy_sign is None:
        vy_sign = vy

    b = vx.shape[0]
    diag = torch.zeros(b, ny, nx, dtype=vx.dtype, device=vx.device)

    # interior x faces: face i sits between cells i-1 and i
    v = vx[..., 1:nx]
    a = v * grid.dy
    sign = vx_sign[..., 1:nx]
    pl = phi_t[..., :-1]
    pr = phi_t[..., 1:]
    pos = sign >= 0.0
    zero = torch.zeros((), dtype=vx.dtype, device=vx.device)
    # positive flux: outflow from the west cell uses the west porosity
    diag[..., :-1] += torch.where(pos, pl * a, zero)
    diag[..., 1:] += torch.where(pos, zero, -pr * a)
    xe = torch.where(pos, zero, pr * a)      # row i picks up C[i+1] (backflow)
    xw = torch.where(pos, -pl * a, zero)     # row i+1 picks up C[i] (inflow)

    # interior y faces: face j sits between rows j-1 and j
    v = vy[..., 1:ny, :]
    a = v * grid.dx
    sign = vy_sign[..., 1:ny, :]
    pu = phi_t[..., :-1, :]
    pd = phi_t[..., 1:, :]
    pos = sign >= 0.0
    diag[..., :-1, :] += torch.where(pos, pu * a, zero)
    diag[..., 1:, :] += torch.where(pos, zero, -pd * a)
    yn = torch.where(pos, zero, pd * a)      # row j picks up C[j+1] (backflow)
    ys = torch.where(pos, -pu * a, zero)     # row j+1 picks up C[j] (inflow)

    # x boundary outflow (inflow carries C = 0)
    v0 = vx[..., 0]
    m = vx_sign[..., 0] < 0
    diag[..., 0] += torch.where(m, -phi_t[..., 0] * v0 * grid.dy, zero)
    v1 = vx[..., nx]
    m = vx_sign[..., nx] > 0
    diag[..., -1] += torch.where(m, phi_t[..., -1] * v1 * grid.dy, zero)

    # y boundary outflow (no-flow in the baseline; kept for generality)
    m = vy_sign[..., 0, :] < 0
    diag[..., 0, :] += torch.where(m, -phi_t[..., 0, :] * vy[..., 0, :] * grid.dx, zero)
    m = vy_sign[..., ny, :] > 0
    diag[..., -1, :] += torch.where(m, phi_t[..., -1, :] * vy[..., ny, :] * grid.dx, zero)

    return Stencil5(diag, xe, xw, yn, ys)


# ---------------------------------------------------------------------------
# Batched transport
# ---------------------------------------------------------------------------
@dataclass
class TransportBatch:
    """Batched implicit-Euler transport result.

    Leading axes are ``(nsteps, S, M)`` for the time series and ``(S, M)`` for
    the summary quantities, where ``S`` is the number of candidate rate
    vectors and ``M`` the number of scenarios.
    """

    times: torch.Tensor                  # (nsteps+1,)
    C_well: torch.Tensor                 # (nsteps, S, M, nw)
    C_record: Optional[torch.Tensor]     # (nsteps, S, M, n_record)
    C_fields: Optional[torch.Tensor]     # (nsteps+1, S, M, n) if stored
    C_final: torch.Tensor                # (S, M, n)
    mass: dict[str, torch.Tensor]        # each (S, M)
    report: KrylovReport = field(default_factory=KrylovReport)
    # kept for the discrete adjoint
    stencil: Optional[Stencil5] = None
    st: Optional[torch.Tensor] = None
    decay: Optional[torch.Tensor] = None
    sink: Optional[torch.Tensor] = None
    phi_t: Optional[torch.Tensor] = None
    vx: Optional[torch.Tensor] = None
    vy: Optional[torch.Tensor] = None

    @property
    def n_steps(self) -> int:
        return self.C_well.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        return self.C_well.shape[1], self.C_well.shape[2]


def transport_batch(
    grid: Grid,
    aquifer: Aquifer,
    flow: FlowModelTorch,
    resp: FlowResponsesTorch,
    Q: torch.Tensor | np.ndarray,
    dt: float,
    nsteps: int,
    source_idx: Optional[int] = None,
    source_rate: float | torch.Tensor = 0.0,
    source_steps: Optional[int] = None,
    record_cells: Optional[Sequence[int]] = None,
    store_fields: bool = False,
    rtol: Optional[float] = None,
    maxiter: int = 4000,
    warm_start: bool = True,
) -> TransportBatch:
    """Batched implicit-Euler transport for ``S`` candidates x ``M`` scenarios.

    ``Q`` shaped ``(S, nw)`` (or ``(nw,)`` for a single candidate).  The
    operator depends on the candidate through the well sink and the advection
    velocities, but not on the time level, so it is assembled once and reused
    for all ``nsteps``; each step is a batched BiCGSTAB solve warm-started from
    the previous concentration field.
    """
    dev, dtype = flow.device, flow.dtype
    ny, nx, n = grid.ny, grid.nx, grid.n
    if not isinstance(Q, torch.Tensor):
        Q = torch.as_tensor(np.asarray(Q, dtype=float), dtype=dtype, device=dev)
    Q = Q.to(dtype=dtype, device=dev)
    single = Q.dim() == 1
    if single:
        Q = Q.unsqueeze(0)
    S, nw = Q.shape
    if nw != resp.n_wells:
        raise ValueError("Q length does not match the number of wells")
    M = flow.M
    B = S * M
    rtol = default_rtol(dtype) if rtol is None else float(rtol)

    # Depth-integrated porosity and the per-cell storage/decay terms.
    phi = flow.phi.reshape(M, n)                                    # (M, n)
    phi_t_m = (flow.b * phi)                                        # (M, n)
    V = grid.cell_volume
    st_m = (aquifer.R * phi_t_m * V / dt)                           # (M, n)
    decay_m = (aquifer.decay * aquifer.R * phi_t_m * V)             # (M, n)

    def flat_per_scenario(t: torch.Tensor) -> torch.Tensor:
        """(M, n) -> (B, n) for a batch of candidates x scenarios."""
        return t.unsqueeze(0).expand(S, M, n).reshape(B, n)

    st = flat_per_scenario(st_m)
    decay = flat_per_scenario(decay_m)
    phi_t = flat_per_scenario(phi_t_m)

    # Frozen background dispersion (no-pumping velocity), as in fv2d.
    h0_field = resp.h0.reshape(M, ny, nx)
    vx0, vy0 = flow.velocity(h0_field)
    gx, gy = face_dispersion_torch(grid, aquifer, vx0, vy0, phi=flow.phi)
    disp = dispersion_stencil(_flat_face(gx, S), _flat_face(gy, S))

    # Heads and velocities for every candidate and scenario.  The velocity
    # helper broadcasts the per-scenario conductivity against a leading
    # candidate axis, so the stack is kept as (S, M, ny, nx) here.
    h = resp.head(Q).reshape(S, M, ny, nx)
    vx, vy = flow.velocity(h)
    h = h.reshape(B, ny, nx)
    vx = vx.reshape(B, ny, nx + 1)
    vy = vy.reshape(B, ny + 1, nx)
    adv = advection_stencil(grid, phi_t.reshape(B, ny, nx), vx, vy)

    # Well sink: sum_i q_i w_i, per cell, per batch entry.
    sink = (Q @ resp.weights.reshape(nw, n)).reshape(S, n)          # (S, n)
    sink = sink.unsqueeze(1).expand(S, M, n).reshape(B, n)
    phi_2d = phi_t.reshape(B, ny, nx)
    sink_2d = sink.reshape(B, ny, nx)
    diag2d = (st + decay + sink).reshape(B, ny, nx)
    A = (adv + disp).with_diagonal((adv.diag + disp.diag) + diag2d)
    A_diag = A.diag

    C = torch.zeros(B, ny, nx, dtype=dtype, device=dev)
    src_flat = torch.zeros(B, n, dtype=dtype, device=dev)
    if source_idx is not None:
        # The rate may be a scalar or one value per scenario (random source
        # strength); it is broadcast over the (candidate, scenario) batch.
        # Not np.asarray(): source_rate may already be a CUDA tensor, and
        # np.asarray() goes through Tensor.__array__, which raises "can't
        # convert cuda:0 device type tensor to numpy". torch.as_tensor accepts
        # scalar / ndarray / tensor alike and keeps the device.
        if torch.is_tensor(source_rate):
            rate = source_rate.detach().to(device=dev, dtype=dtype).reshape(-1)
        else:
            rate = torch.as_tensor(np.asarray(source_rate, dtype=float),
                                   dtype=dtype, device=dev).reshape(-1)
        if rate.numel() == 1:
            rate_b = rate.expand(B)
        elif rate.numel() == M:
            rate_b = rate.reshape(1, M).expand(S, M).reshape(B)
        elif rate.numel() == B:
            rate_b = rate
        else:
            raise ValueError("source_rate must be scalar, one per scenario, "
                             f"or one per batch entry (got {rate.numel()})")
        if bool((rate != 0).any()):
            src_flat[:, int(source_idx)] = rate_b
    src = src_flat.reshape(B, ny, nx)

    if record_cells is not None:
        record_cells = np.atleast_1d(np.asarray(record_cells, dtype=int))
    nrec = 0 if record_cells is None else int(record_cells.size)
    C_well = torch.zeros(nsteps, S, M, nw, dtype=dtype, device=dev)
    C_rec = torch.zeros(nsteps, S, M, nrec, dtype=dtype, device=dev) if nrec else None
    C_fields = (torch.zeros(nsteps + 1, S, M, n, dtype=dtype, device=dev)
                if store_fields else None)
    if C_fields is not None:
        C_fields[0] = C.reshape(S, M, n)

    mass_total = torch.zeros(nsteps + 1, S, M, dtype=dtype, device=dev)
    mass_source = torch.zeros(nsteps + 1, S, M, dtype=dtype, device=dev)
    mass_well = torch.zeros(nsteps + 1, S, M, dtype=dtype, device=dev)
    mass_outflow = torch.zeros(nsteps + 1, S, M, dtype=dtype, device=dev)
    mass_decay = torch.zeros(nsteps + 1, S, M, dtype=dtype, device=dev)
    mass_total[0] = ((phi_2d * aquifer.R * C).sum(dim=(1, 2)) * V).reshape(S, M)

    vx_left, vx_right = vx[..., 0], vx[..., nx]
    weights_nw = resp.weights.reshape(nw, n)
    report = KrylovReport()

    for k in range(nsteps):
        active = src if (source_steps is None or k < source_steps) else torch.zeros_like(src)
        rhs = st.reshape(B, ny, nx) * C + active
        C, rep = bicgstab_solve(A.apply, rhs, x0=C if warm_start else None,
                                diag=A_diag, rtol=rtol, maxiter=maxiter)
        report.add(rep)
        if C_fields is not None:
            C_fields[k + 1] = C.reshape(S, M, n)

        Cf = C.reshape(S, M, n)
        C_well[k] = torch.einsum("in,smn->smi", weights_nw, Cf)
        if C_rec is not None:
            C_rec[k] = Cf[:, :, record_cells]

        C2 = C.reshape(B, ny, nx)
        mass_total[k + 1] = ((phi_2d * aquifer.R * C).sum(dim=(1, 2)) * V).reshape(S, M)
        mass_source[k + 1] = mass_source[k] + (active.sum(dim=(1, 2)) * dt).reshape(S, M)
        mass_well[k + 1] = mass_well[k] + ((C * sink_2d).sum(dim=(1, 2)) * dt).reshape(S, M)
        out = (torch.clamp(-vx_left, min=0.0) * phi_2d[..., 0] * C2[..., 0]).sum(dim=1) * grid.dy
        out = out + (torch.clamp(vx_right, min=0.0) * phi_2d[..., -1] * C2[..., -1]).sum(dim=1) * grid.dy
        mass_outflow[k + 1] = mass_outflow[k] + out.reshape(S, M) * dt
        mass_decay[k + 1] = mass_decay[k] + (
            (decay.reshape(B, ny, nx) * C).sum(dim=(1, 2)) * dt).reshape(S, M)

    mass = {
        "total": mass_total,
        "source": mass_source,
        "well": mass_well,
        "outflow": mass_outflow,
        "decay": mass_decay,
        "integrated_well_mass": mass_well[-1],
    }
    result = TransportBatch(
        times=torch.arange(nsteps + 1, dtype=dtype, device=dev) * dt,
        C_well=C_well,
        C_record=C_rec,
        C_fields=C_fields,
        C_final=C.reshape(S, M, n),
        mass=mass,
        report=report,
        stencil=A,
        st=st,
        decay=decay,
        sink=sink,
        phi_t=phi_t,
        vx=vx,
        vy=vy,
    )
    return result
