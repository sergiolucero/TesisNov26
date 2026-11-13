"""2D depth-averaged finite-volume flow and transport with extraction wells.

Discretisation
--------------
* cell-centred, five-point stencil, harmonic-mean face transmissivity;
* steady Darcy flow ``-div(T grad h) = b W - sum_i q_i delta_eps`` with
  Dirichlet heads on the x-boundaries and no-flow (or Dirichlet) on y;
* regularised wells: a normalised Gaussian kernel of width ``eps`` grid cells
  (Eq. 3.x of ``optimization.tex``);
* implicit-Euler transport with first-order upwind advection, a frozen
  (background) dispersion tensor and an explicit well sink ``q_i C(x_i)``;
* global mass-balance diagnostics for every run.

Sign conventions
----------------
``q_i >= 0`` is extraction.  Heads are in metres, ``q`` in m^3/d, ``K`` in
m/d, concentrations in mg/L.  The 2D transmissivity is ``T = K b``.

The frozen-dispersion choice is deliberate: it makes the discrete adjoint
exact and cheap (the advection operator is linear in ``q``), while capture is
dominated by advection.  Set ``alpha_l = alpha_t = 0`` for a pure-advection
check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

__all__ = [
    "Grid",
    "Aquifer",
    "FlowModel",
    "FlowResponses",
    "TransportResult",
    "gaussian_well_weights",
    "flow_responses",
    "advection_operator",
    "dispersion_operator",
    "face_dispersion",
    "transport_run",
]


# ---------------------------------------------------------------------------
# Geometry and aquifer
# ---------------------------------------------------------------------------
@dataclass
class Grid:
    """Rectangular cell-centred grid."""

    nx: int
    ny: int
    lx: float
    ly: float
    x0: float = 0.0
    y0: float = 0.0

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 2:
            raise ValueError("grid needs at least 2x2 cells")
        if self.lx <= 0 or self.ly <= 0:
            raise ValueError("domain lengths must be positive")

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def n(self) -> int:
        return self.nx * self.ny

    @property
    def area(self) -> float:
        return self.lx * self.ly

    @property
    def cell_volume(self) -> float:
        return self.dx * self.dy

    def index(self, x, y) -> int:
        """Flattened index of the cell containing ``(x, y)`` (clipped)."""
        i = int(np.floor((float(x) - self.x0) / self.dx))
        j = int(np.floor((float(y) - self.y0) / self.dy))
        i = min(max(i, 0), self.nx - 1)
        j = min(max(j, 0), self.ny - 1)
        return j * self.nx + i

    def indices(self, x, y) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        i = np.floor((x - self.x0) / self.dx).astype(int)
        j = np.floor((y - self.y0) / self.dy).astype(int)
        i = np.clip(i, 0, self.nx - 1)
        j = np.clip(j, 0, self.ny - 1)
        return j * self.nx + i

    def cell_centre(self, idx: int) -> tuple[float, float]:
        j, i = divmod(int(idx), self.nx)
        return self.x0 + (i + 0.5) * self.dx, self.y0 + (j + 0.5) * self.dy

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.x0 + (np.arange(self.nx) + 0.5) * self.dx
        y = self.y0 + (np.arange(self.ny) + 0.5) * self.dy
        return np.meshgrid(x, y)


@dataclass
class Aquifer:
    """Depth-averaged aquifer properties.

    ``K`` has shape ``(ny, nx)``.  ``phi`` may be a scalar or a field.
    """

    K: np.ndarray
    phi: float | np.ndarray = 0.30
    R: float = 1.0
    b: float = 10.0
    alpha_l: float = 5.0
    alpha_t: float = 0.5
    decay: float = 0.0

    def phi_field(self, shape: tuple[int, int]) -> np.ndarray:
        phi = np.asarray(self.phi, dtype=float)
        if phi.ndim == 0:
            return np.full(shape, float(phi))
        if phi.shape != shape:
            raise ValueError(f"phi has shape {phi.shape}, expected {shape}")
        return phi


# ---------------------------------------------------------------------------
# Wells
# ---------------------------------------------------------------------------
def gaussian_well_weights(
    grid: Grid,
    x: float,
    y: float,
    eps_cells: float = 1.5,
    truncate: float = 4.0,
) -> np.ndarray:
    """Normalised Gaussian well kernel on the grid (Eq. ``delta_eps``).

    Returns a vector of length ``grid.n`` summing to one.
    """
    X, Y = grid.centres()
    sx = max(eps_cells, 0.25) * grid.dx
    sy = max(eps_cells, 0.25) * grid.dy
    # truncate to keep the stencil small
    w = np.exp(-(((X - x) ** 2) / (2 * sx ** 2) + ((Y - y) ** 2) / (2 * sy ** 2)))
    w[w < np.exp(-0.5 * truncate ** 2)] = 0.0
    s = w.sum()
    if s <= 0:  # well exactly between cells at an extreme corner
        w = np.zeros(grid.n)
        w[grid.index(x, y)] = 1.0
        return w
    return (w / s).ravel()


# ---------------------------------------------------------------------------
# Steady flow
# ---------------------------------------------------------------------------
class FlowModel:
    """Steady Darcy flow with fixed wells and linear superposition.

    One sparse factorisation is reused for the background solve and all unit
    well responses (``N_w + 1`` right-hand sides).
    """

    def __init__(
        self,
        grid: Grid,
        aquifer: Aquifer,
        h_left: float = 0.0,
        h_right: float = 0.0,
        bc_y: str = "noflow",
        h_bottom: float = 0.0,
        h_top: float = 0.0,
    ) -> None:
        if aquifer.K.shape != (grid.ny, grid.nx):
            raise ValueError("K shape does not match the grid")
        self.grid = grid
        self.aquifer = aquifer
        self.h_left = float(h_left)
        self.h_right = float(h_right)
        self.bc_y = bc_y
        self.bc_x = "dirichlet"
        self.K = np.asarray(aquifer.K, dtype=float)
        self.T = self.K * aquifer.b
        self.phi = aquifer.phi_field((grid.ny, grid.nx))
        self.A, self.rhs_const = self._build_matrix(h_bottom, h_top)
        self._lu = splu(self.A.tocsc())

    # -- assembly ---------------------------------------------------------
    def _build_matrix(self, h_bottom: float, h_top: float):
        g = self.grid
        ny, nx, n = g.ny, g.nx, g.n
        idx = np.arange(n).reshape(ny, nx)
        diag = np.zeros(n)
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []

        # interior x faces
        cx = 2.0 * self.T[:, :-1] * self.T[:, 1:] / (self.T[:, :-1] + self.T[:, 1:])
        cx = cx * g.dy / g.dx
        left, right = idx[:, :-1].ravel(), idx[:, 1:].ravel()
        diag[left] += cx.ravel()
        diag[right] += cx.ravel()
        rows += [right, left]
        cols += [left, right]
        vals += [-cx.ravel(), -cx.ravel()]

        # interior y faces
        cy = 2.0 * self.T[:-1, :] * self.T[1:, :] / (self.T[:-1, :] + self.T[1:, :])
        cy = cy * g.dx / g.dy
        top, bottom = idx[:-1, :].ravel(), idx[1:, :].ravel()
        diag[top] += cy.ravel()
        diag[bottom] += cy.ravel()
        rows += [bottom, top]
        cols += [top, bottom]
        vals += [-cy.ravel(), -cy.ravel()]

        rhs = np.zeros(n)

        # Dirichlet x boundaries (half-cell distance)
        cxL = 2.0 * self.T[:, 0] * g.dy / g.dx
        cxR = 2.0 * self.T[:, -1] * g.dy / g.dx
        diag[idx[:, 0]] += cxL
        rhs[idx[:, 0]] += cxL * self.h_left
        diag[idx[:, -1]] += cxR
        rhs[idx[:, -1]] += cxR * self.h_right

        # y boundaries
        if self.bc_y == "dirichlet":
            cyB = 2.0 * self.T[0, :] * g.dx / g.dy
            cyT = 2.0 * self.T[-1, :] * g.dx / g.dy
            diag[idx[0, :]] += cyB
            rhs[idx[0, :]] += cyB * h_bottom
            diag[idx[-1, :]] += cyT
            rhs[idx[-1, :]] += cyT * h_top
        elif self.bc_y != "noflow":
            raise ValueError("bc_y must be 'noflow' or 'dirichlet'")

        rows.append(np.arange(n))
        cols.append(np.arange(n))
        vals.append(diag)
        A = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        ).tocsr()
        return A, rhs

    # -- solves -----------------------------------------------------------
    def solve(self, f: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve ``A h = rhs_const + f`` (``f`` is the integrated source)."""
        rhs = self.rhs_const.copy()
        if f is not None:
            rhs = rhs + np.asarray(f, dtype=float).ravel()
        return self._lu.solve(rhs)

    def release_factorization(self) -> None:
        """Free the LU factors once all unit responses have been computed."""
        self._lu = None
        self.A = None

    def solve_perturbation(self, f: np.ndarray) -> np.ndarray:
        """Solve ``A dh = f`` with zero boundary contribution.

        Used for the unit-rate well responses so that linear superposition
        ``h(q) = h0 + q @ hw`` is exact.
        """
        return self._lu.solve(np.asarray(f, dtype=float).ravel())

    def velocity(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Face-centred seepage velocity ``v = -K grad(h) / phi``.

        ``vx`` has shape ``(ny, nx+1)`` (x-faces), ``vy`` has shape
        ``(ny+1, nx)`` (y-faces).  Boundary y-faces are zero for no-flow.
        """
        g = self.grid
        ny, nx = g.ny, g.nx
        h2 = np.asarray(h, dtype=float).reshape(ny, nx)
        phi = self.phi
        vx = np.zeros((ny, nx + 1))
        vy = np.zeros((ny + 1, nx))

        Kx = 2.0 * self.K[:, :-1] * self.K[:, 1:] / (self.K[:, :-1] + self.K[:, 1:])
        vx[:, 1:nx] = -Kx * (h2[:, 1:] - h2[:, :-1]) / g.dx / phi[:, 1:]
        vx[:, 0] = -self.K[:, 0] * (h2[:, 0] - self.h_left) / (0.5 * g.dx) / phi[:, 0]
        vx[:, nx] = -self.K[:, -1] * (self.h_right - h2[:, -1]) / (0.5 * g.dx) / phi[:, -1]

        Ky = 2.0 * self.K[:-1, :] * self.K[1:, :] / (self.K[:-1, :] + self.K[1:, :])
        vy[1:ny, :] = -Ky * (h2[1:, :] - h2[:-1, :]) / g.dy / phi[1:, :]
        if self.bc_y == "noflow":
            vy[0, :] = 0.0
            vy[ny, :] = 0.0
        return vx, vy


@dataclass
class FlowResponses:
    """Unit-rate flow responses for one random field.

    ``head(q) = h0 + q @ hw`` and ``velocity(q)`` follows from linearity.
    """

    h0: np.ndarray
    hw: np.ndarray
    weights: np.ndarray
    well_cells: np.ndarray
    well_xy: np.ndarray

    def head(self, q: Sequence[float]) -> np.ndarray:
        q = np.asarray(q, dtype=float).ravel()
        if q.size != self.hw.shape[0]:
            raise ValueError("q has the wrong length")
        return self.h0 + q @ self.hw


def flow_responses(
    flow: FlowModel,
    well_xy: Sequence[Sequence[float]],
    eps_cells: float = 1.5,
) -> FlowResponses:
    """Solve the background and all unit-well flow problems."""
    well_xy = np.atleast_2d(np.asarray(well_xy, dtype=float))
    weights = np.vstack(
        [gaussian_well_weights(flow.grid, x, y, eps_cells) for x, y in well_xy]
    )
    h0 = flow.solve()
    hw = np.vstack([flow.solve_perturbation(-weights[i]) for i in range(weights.shape[0])])
    well_cells = np.argmax(weights, axis=1)
    return FlowResponses(h0, hw, weights, well_cells, well_xy)


# ---------------------------------------------------------------------------
# Transport operators
# ---------------------------------------------------------------------------
def advection_operator(
    grid: Grid,
    vx: np.ndarray,
    vy: np.ndarray,
    phi: float | np.ndarray,
    vx_sign: Optional[np.ndarray] = None,
    vy_sign: Optional[np.ndarray] = None,
) -> sp.csr_matrix:
    """First-order upwind advection operator ``div(phi v C)``.

    Boundary x-faces with inflow use ``C = 0`` (clean water).  The y
    boundaries are no-flow (or Dirichlet inflow with ``C = 0``).

    ``vx_sign``/``vy_sign`` optionally fix the upwind direction independently
    of the flux magnitude.  This is what makes the operator differentiable:
    ``A(v)`` is linear in ``v`` once the upwind pattern is fixed, so the
    adjoint can use ``A(v_i)`` built with the signs of the total velocity.
    """
    if vx_sign is None:
        vx_sign = vx
    if vy_sign is None:
        vy_sign = vy
    ny, nx, n = grid.ny, grid.nx, grid.n
    idx = np.arange(n).reshape(ny, nx)
    phi_f = np.full(n, float(phi)) if np.ndim(phi) == 0 else np.asarray(phi, float).ravel()
    p = phi_f[idx]

    diag = np.zeros(n)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    # interior x-faces: face i between cells (i-1) and i
    v = vx[:, 1:nx]
    a = (v * grid.dy).ravel()
    sgn = vx_sign[:, 1:nx].ravel()
    left = idx[:, :-1].ravel()
    right = idx[:, 1:].ravel()
    pl = p[:, :-1].ravel()
    pr = p[:, 1:].ravel()
    pos = sgn >= 0.0
    if np.any(pos):
        rl, rr, av, pv = left[pos], right[pos], a[pos], pl[pos]
        diag[rl] += pv * av
        rows.append(rr)
        cols.append(rl)
        vals.append(-pv * av)
    if np.any(~pos):
        rl, rr, av, pv = left[~pos], right[~pos], -a[~pos], pr[~pos]
        diag[rr] += pv * av
        rows.append(rl)
        cols.append(rr)
        vals.append(-pv * av)

    # interior y-faces
    v = vy[1:ny, :]
    a = (v * grid.dx).ravel()
    sgn = vy_sign[1:ny, :].ravel()
    up = idx[:-1, :].ravel()
    dn = idx[1:, :].ravel()
    pu = p[:-1, :].ravel()
    pd = p[1:, :].ravel()
    pos = sgn >= 0.0
    if np.any(pos):
        ru, rd, av, pv = up[pos], dn[pos], a[pos], pu[pos]
        diag[ru] += pv * av
        rows.append(rd)
        cols.append(ru)
        vals.append(-pv * av)
    if np.any(~pos):
        ru, rd, av, pv = up[~pos], dn[~pos], -a[~pos], pd[~pos]
        diag[rd] += pv * av
        rows.append(ru)
        cols.append(rd)
        vals.append(-pv * av)

    # x boundary outflow (inflow has C = 0)
    v = vx[:, 0]
    m = vx_sign[:, 0] < 0
    if np.any(m):
        c = idx[m, 0]
        diag[c] += -p[:, 0][m] * v[m] * grid.dy
    v = vx[:, -1]
    m = vx_sign[:, -1] > 0
    if np.any(m):
        c = idx[m, -1]
        diag[c] += p[:, -1][m] * v[m] * grid.dy

    # y boundary outflow (no-flow in the baseline; keep for generality)
    m = vy_sign[0, :] < 0
    if np.any(m):
        c = idx[0, :][m]
        diag[c] += -p[0, :][m] * vy[0, :][m] * grid.dx
    m = vy_sign[ny, :] > 0
    if np.any(m):
        c = idx[-1, :][m]
        diag[c] += p[-1, :][m] * vy[ny, :][m] * grid.dx

    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    return sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()


def face_dispersion(
    grid: Grid,
    aquifer: Aquifer,
    vx: np.ndarray,
    vy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Face dispersion conductances for the (diagonal) tensor.

    ``D_xx = a_T |v| + (a_L - a_T) v_x^2 / |v|`` and likewise ``D_yy``.
    The tensor is frozen at the velocity passed in.
    """
    ny, nx = grid.ny, grid.nx
    vxc = 0.5 * (vx[:, :-1] + vx[:, 1:])
    vyc = 0.5 * (vy[:-1, :] + vy[1:, :])
    speed = np.sqrt(vxc ** 2 + vyc ** 2)
    speed = np.maximum(speed, 1e-12)
    a_l, a_t = aquifer.alpha_l, aquifer.alpha_t
    Dxx = a_t * speed + (a_l - a_t) * vxc ** 2 / speed
    Dyy = a_t * speed + (a_l - a_t) * vyc ** 2 / speed
    Dxx_f = 0.5 * (Dxx[:, :-1] + Dxx[:, 1:])
    Dyy_f = 0.5 * (Dyy[:-1, :] + Dyy[1:, :])
    phi = aquifer.phi_field((ny, nx))
    phix = 0.5 * (phi[:, :-1] + phi[:, 1:])
    phiy = 0.5 * (phi[:-1, :] + phi[1:, :])
    # depth-integrated (multiply by aquifer thickness)
    gx = aquifer.b * phix * Dxx_f * grid.dy / grid.dx
    gy = aquifer.b * phiy * Dyy_f * grid.dx / grid.dy
    return gx, gy


def dispersion_operator(
    grid: Grid,
    gx: np.ndarray,
    gy: np.ndarray,
) -> sp.csr_matrix:
    """Symmetric dispersion operator from face conductances."""
    ny, nx, n = grid.ny, grid.nx, grid.n
    idx = np.arange(n).reshape(ny, nx)
    diag = np.zeros(n)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    left, right = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    g = gx.ravel()
    diag[left] += g
    diag[right] += g
    rows += [right, left]
    cols += [left, right]
    vals += [-g, -g]

    up, dn = idx[:-1, :].ravel(), idx[1:, :].ravel()
    g = gy.ravel()
    diag[up] += g
    diag[dn] += g
    rows += [dn, up]
    cols += [up, dn]
    vals += [-g, -g]

    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    return sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()


# ---------------------------------------------------------------------------
# Transport time stepping
# ---------------------------------------------------------------------------
@dataclass
class TransportResult:
    times: np.ndarray
    C_well: np.ndarray               # (nsteps, nw)
    C_record: Optional[np.ndarray]   # (nsteps, n_record)
    record_cells: Optional[np.ndarray]
    C_fields: Optional[np.ndarray]   # (nsteps+1, n) if stored
    mass: dict[str, np.ndarray]
    C_final_field: Optional[np.ndarray] = None
    operators: dict = field(default_factory=dict, repr=False)

    @property
    def C_final(self) -> np.ndarray:
        if self.C_final_field is not None:
            return self.C_final_field
        if self.C_fields is not None:
            return self.C_fields[-1]
        raise AttributeError("fields were not stored")


def _well_sink(q: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
    """Distributed well sink: ``sum_i q_i w_i`` on the cell diagonal.

    The flow source is the same Gaussian kernel, so the advective influx into
    the footprint equals ``q_i w_i`` cell by cell and the transport mass
    balance is exact.
    """
    return np.asarray(q, dtype=float) @ weights


def transport_run(
    grid: Grid,
    aquifer: Aquifer,
    flow: FlowModel,
    resp: FlowResponses,
    q: Sequence[float],
    dt: float,
    nsteps: int,
    source_idx: Optional[int] = None,
    source_rate: float = 0.0,
    source_steps: Optional[int] = None,
    c_init: Optional[np.ndarray] = None,
    record_cells: Optional[Sequence[int]] = None,
    store_fields: bool = False,
    disp_ref: Optional[tuple[np.ndarray, np.ndarray]] = None,
    return_operator: bool = False,
) -> TransportResult:
    """Implicit-Euler transport solve for a steady rate vector ``q``.

    Parameters
    ----------
    source_idx, source_rate
        Source cell and integrated mass rate (mass/time, not per volume).
        The source is active for the first ``nsteps`` steps; pass a tuple
        ``(idx, rate, active_steps)`` via ``source_rate`` if needed?  No --
        use :func:`transport_run_schedule` for a time-dependent source.
    disp_ref
        Background velocity used for the frozen dispersion tensor.  Defaults
        to the no-pumping velocity ``resp.h0``.
    """
    q = np.asarray(q, dtype=float).ravel()
    nw = q.size
    if resp.hw.shape[0] != nw:
        raise ValueError("q length does not match the number of wells")
    ny, nx, n = grid.ny, grid.nx, grid.n
    phi = aquifer.phi_field((ny, nx)).ravel()
    # depth-integrated porosity b*phi: storage, advection, dispersion and
    # decay are all per unit aquifer plan area; the well sink is q*C.
    phi_t = aquifer.b * phi
    V = grid.cell_volume
    st = aquifer.R * phi_t * V / dt
    decay = aquifer.decay * aquifer.R * phi_t * V

    if disp_ref is None:
        vx0, vy0 = flow.velocity(resp.h0)
    else:
        vx0, vy0 = disp_ref
    gx, gy = face_dispersion(grid, aquifer, vx0, vy0)
    Adisp = dispersion_operator(grid, gx, gy)

    h = resp.head(q)
    vx, vy = flow.velocity(h)
    Aadv = advection_operator(grid, vx, vy, phi_t)
    sink = _well_sink(q, resp.weights, n)
    A = (Aadv + Adisp + sp.diags(st + decay + sink)).tocsc()
    lu = splu(A)

    C = np.zeros(n) if c_init is None else np.asarray(c_init, dtype=float).ravel().copy()
    if C.size != n:
        raise ValueError("c_init has the wrong size")

    if record_cells is not None:
        record_cells = np.atleast_1d(np.asarray(record_cells, dtype=int))
    C_rec = np.zeros((nsteps, 0 if record_cells is None else record_cells.size))
    C_well = np.zeros((nsteps, nw))
    C_fields = np.zeros((nsteps + 1, n)) if store_fields else None
    if C_fields is not None:
        C_fields[0] = C

    src = np.zeros(n)
    if source_idx is not None and source_rate != 0.0:
        src[int(source_idx)] = float(source_rate)

    mass_total = np.zeros(nsteps + 1)
    mass_source = np.zeros(nsteps + 1)
    mass_well = np.zeros(nsteps + 1)
    mass_out = np.zeros(nsteps + 1)
    mass_decay = np.zeros(nsteps + 1)
    mass_total[0] = float(np.sum(phi_t * aquifer.R * C) * V)

    for k in range(nsteps):
        active = src if (source_steps is None or k < source_steps) else 0.0
        rhs = st * C + active
        C = lu.solve(rhs)
        if C_fields is not None:
            C_fields[k + 1] = C
        C_well[k] = resp.weights @ C
        if record_cells is not None:
            C_rec[k] = C[record_cells]

        Cc = C.reshape(ny, nx)
        mass_total[k + 1] = float(np.sum(phi_t * aquifer.R * C) * V)
        mass_source[k + 1] = mass_source[k] + float(np.sum(active)) * dt
        mass_well[k + 1] = mass_well[k] + float(np.dot(C, sink)) * dt
        out = 0.0
        phi2 = phi_t.reshape(ny, nx)
        v = vx[:, 0]
        out += float(np.sum(np.maximum(-v, 0.0) * phi2[:, 0] * Cc[:, 0])) * grid.dy
        v = vx[:, -1]
        out += float(np.sum(np.maximum(v, 0.0) * phi2[:, -1] * Cc[:, -1])) * grid.dy
        mass_out[k + 1] = mass_out[k] + out * dt
        mass_decay[k + 1] = mass_decay[k] + float(np.sum(decay * C)) * dt

    # cumulative integrals used by the objective
    integ_well_mass = float(np.sum(mass_well[-1]))  # already cumulative
    mass = {
        "total": mass_total,
        "source": mass_source,
        "well": mass_well,
        "outflow": mass_out,
        "decay": mass_decay,
        "integrated_well_mass": np.array(integ_well_mass),
    }
    ops = {"A": A, "lu": lu, "st": st, "vx": vx, "vy": vy, "phi_t": phi_t} if return_operator else {}
    return TransportResult(
        times=np.arange(nsteps + 1) * dt,
        C_well=C_well,
        C_record=C_rec if record_cells is not None else None,
        record_cells=record_cells,
        C_fields=C_fields,
        mass=mass,
        C_final_field=C.copy(),
        operators=ops,
    )
