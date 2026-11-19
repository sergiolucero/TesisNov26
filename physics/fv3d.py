"""3D finite-volume Darcy flow and advection--dispersion transport.

This is the three-dimensional companion to :mod:`physics.fv2d`.  It keeps the
same sign conventions and the same operator-splitting choices so a 3D result
can be checked against the trusted 2D kernels by extrusion:

* steady Darcy flow ``-div(K grad h) = -sum_i q_i delta_eps`` on a
  cell-centred, seven-point stencil with harmonic face conductivities;
* anisotropic conductivity ``diag(K_h, K_h, K_v)`` with ``K_v = kv_ratio*K_h``;
* Dirichlet heads on the x faces (regional gradient), no-flow y and z faces;
* regularised wells: a normalised 3D Gaussian kernel (screened interval can be
  emulated by a narrow ``eps_z``);
* implicit-Euler transport with first-order upwind advection and a frozen
  diagonal dispersion tensor with longitudinal, horizontal-transverse and
  vertical-transverse dispersivities.

Units are metres and days: ``K`` in m/d, ``q`` in m3/d, concentrations in
mg/L.  Unlike the depth-averaged 2D model there is no aquifer thickness factor
``b``; all operators are volume integrals over ``dx*dy*dz``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

__all__ = [
    "Grid3D",
    "Aquifer3D",
    "FlowModel3D",
    "FlowResponses3D",
    "TransportResult3D",
    "gaussian_well_weights3d",
    "flow_responses3d",
    "advection_operator3d",
    "face_dispersion3d",
    "dispersion_operator3d",
    "transport_run3d",
]


# ---------------------------------------------------------------------------
# Geometry and aquifer
# ---------------------------------------------------------------------------
@dataclass
class Grid3D:
    """Rectangular cell-centred 3D grid; ``K`` is stored as ``(ny, nx, nz)``."""

    nx: int
    ny: int
    nz: int
    lx: float
    ly: float
    lz: float
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1 or self.nz < 1:
            raise ValueError("grid needs at least 1 cell in each direction")
        if self.lx <= 0 or self.ly <= 0 or self.lz <= 0:
            raise ValueError("domain lengths must be positive")

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def dz(self) -> float:
        return self.lz / self.nz

    @property
    def n(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def volume(self) -> float:
        return self.lx * self.ly * self.lz

    @property
    def cell_volume(self) -> float:
        return self.dx * self.dy * self.dz

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.ny, self.nx, self.nz)

    def index(self, x: float, y: float, z: float) -> int:
        i = int(np.floor((float(x) - self.x0) / self.dx))
        j = int(np.floor((float(y) - self.y0) / self.dy))
        k = int(np.floor((float(z) - self.z0) / self.dz))
        i = min(max(i, 0), self.nx - 1)
        j = min(max(j, 0), self.ny - 1)
        k = min(max(k, 0), self.nz - 1)
        return (j * self.nx + i) * self.nz + k

    def indices(self, x, y, z) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        z = np.atleast_1d(np.asarray(z, dtype=float))
        i = np.clip(np.floor((x - self.x0) / self.dx).astype(int), 0, self.nx - 1)
        j = np.clip(np.floor((y - self.y0) / self.dy).astype(int), 0, self.ny - 1)
        k = np.clip(np.floor((z - self.z0) / self.dz).astype(int), 0, self.nz - 1)
        return (j * self.nx + i) * self.nz + k

    def cell_centre(self, idx: int) -> tuple[float, float, float]:
        k = idx % self.nz
        rem = idx // self.nz
        i = rem % self.nx
        j = rem // self.nx
        return (self.x0 + (i + 0.5) * self.dx,
                self.y0 + (j + 0.5) * self.dy,
                self.z0 + (k + 0.5) * self.dz)

    def centres(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.x0 + (np.arange(self.nx) + 0.5) * self.dx
        y = self.y0 + (np.arange(self.ny) + 0.5) * self.dy
        z = self.z0 + (np.arange(self.nz) + 0.5) * self.dz
        return np.meshgrid(x, y, z, indexing="ij")  # (nx, ny, nz)


@dataclass
class Aquifer3D:
    """3D aquifer properties.

    ``K`` is the horizontal conductivity field ``(ny, nx, nz)``.  The vertical
    conductivity is ``kv_ratio * K``; set ``kv_ratio`` below one for a layered
    or anisotropic site.
    """

    K: np.ndarray
    phi: float | np.ndarray = 0.30
    R: float = 1.0
    kv_ratio: float = 1.0
    alpha_l: float = 5.0
    alpha_th: float = 0.5
    alpha_tv: float = 0.5
    decay: float = 0.0

    def phi_field(self, shape: tuple[int, int, int]) -> np.ndarray:
        phi = np.asarray(self.phi, dtype=float)
        if phi.ndim == 0:
            return np.full(shape, float(phi))
        if phi.shape != shape:
            raise ValueError(f"phi has shape {phi.shape}, expected {shape}")
        return phi


# ---------------------------------------------------------------------------
# Wells
# ---------------------------------------------------------------------------
def gaussian_well_weights3d(
    grid: Grid3D,
    x: float,
    y: float,
    z: float,
    eps_cells: float = 1.5,
    eps_z_cells: Optional[float] = None,
    truncate: float = 4.0,
) -> np.ndarray:
    """Normalised 3D Gaussian well kernel; returns a vector of length ``n``.

    ``eps_z_cells`` controls the screened interval.  A small value gives a
    point-screened well; a large value approaches a fully penetrating well.
    """
    if eps_z_cells is None:
        eps_z_cells = eps_cells
    X, Y, Z = grid.centres()
    sx = max(eps_cells, 0.25) * grid.dx
    sy = max(eps_cells, 0.25) * grid.dy
    sz = max(eps_z_cells, 0.25) * grid.dz
    w = np.exp(
        -(((X - x) ** 2) / (2 * sx ** 2)
          + ((Y - y) ** 2) / (2 * sy ** 2)
          + ((Z - z) ** 2) / (2 * sz ** 2))
    )
    w[w < np.exp(-0.5 * truncate ** 2)] = 0.0
    s = w.sum()
    if s <= 0:
        w = np.zeros((grid.ny, grid.nx, grid.nz))
        w.ravel()[grid.index(x, y, z)] = 1.0
        return w.ravel()
    return (w / s).ravel()


# ---------------------------------------------------------------------------
# Steady flow
# ---------------------------------------------------------------------------
class FlowModel3D:
    """Steady Darcy flow with linear superposition of unit well responses."""

    def __init__(
        self,
        grid: Grid3D,
        aquifer: Aquifer3D,
        h_left: float = 0.0,
        h_right: float = 0.0,
        bc_y: str = "noflow",
        bc_z: str = "noflow",
    ) -> None:
        if aquifer.K.shape != grid.shape:
            raise ValueError(f"K shape {aquifer.K.shape} does not match grid "
                             f"{grid.shape}")
        if bc_y not in {"noflow", "dirichlet"} or bc_z not in {"noflow", "dirichlet"}:
            raise ValueError("boundary option must be 'noflow' or 'dirichlet'")
        self.grid = grid
        self.aquifer = aquifer
        self.h_left = float(h_left)
        self.h_right = float(h_right)
        self.bc_y = bc_y
        self.bc_z = bc_z
        self.K = np.asarray(aquifer.K, dtype=float)
        self.Kz = self.K * float(aquifer.kv_ratio)
        self.phi = aquifer.phi_field(grid.shape)
        self.A, self.rhs_const = self._build_matrix()
        self._lu = splu(self.A.tocsc())

    # -- assembly ---------------------------------------------------------
    def _build_matrix(self):
        g = self.grid
        ny, nx, nz, n = g.ny, g.nx, g.nz, g.n
        idx = np.arange(n).reshape(ny, nx, nz)
        dyz = g.dy * g.dz
        dxz = g.dx * g.dz
        dxy = g.dx * g.dy
        diag = np.zeros(n)
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []

        def add_face(a, b, gface):
            nonlocal diag
            gf = np.asarray(gface, dtype=float).ravel()
            a = a.ravel()
            b = b.ravel()
            diag[a] += gf
            diag[b] += gf
            rows.extend([b, a])
            cols.extend([a, b])
            vals.extend([-gf, -gf])

        # interior faces: harmonic mean of the two cell conductivities
        Kx = 2.0 * self.K[:, :-1, :] * self.K[:, 1:, :] / (
            self.K[:, :-1, :] + self.K[:, 1:, :])
        add_face(idx[:, :-1, :], idx[:, 1:, :], Kx * dyz / g.dx)

        Ky = 2.0 * self.K[:-1, :, :] * self.K[1:, :, :] / (
            self.K[:-1, :, :] + self.K[1:, :, :])
        add_face(idx[:-1, :, :], idx[1:, :, :], Ky * dxz / g.dy)

        Kz = 2.0 * self.Kz[:, :, :-1] * self.Kz[:, :, 1:] / (
            self.Kz[:, :, :-1] + self.Kz[:, :, 1:])
        add_face(idx[:, :, :-1], idx[:, :, 1:], Kz * dxy / g.dz)

        rhs = np.zeros(n)

        # Dirichlet x boundaries (half-cell distance)
        cxL = 2.0 * self.K[:, 0, :] * dyz / g.dx
        cxR = 2.0 * self.K[:, -1, :] * dyz / g.dx
        diag[idx[:, 0, :].ravel()] += cxL.ravel()
        rhs[idx[:, 0, :].ravel()] += (cxL * self.h_left).ravel()
        diag[idx[:, -1, :].ravel()] += cxR.ravel()
        rhs[idx[:, -1, :].ravel()] += (cxR * self.h_right).ravel()

        if self.bc_y == "dirichlet":
            cyB = 2.0 * self.K[0, :, :] * dxz / g.dy
            cyT = 2.0 * self.K[-1, :, :] * dxz / g.dy
            diag[idx[0, :, :].ravel()] += cyB.ravel()
            diag[idx[-1, :, :].ravel()] += cyT.ravel()
        if self.bc_z == "dirichlet":
            czB = 2.0 * self.Kz[:, :, 0] * dxy / g.dz
            czT = 2.0 * self.Kz[:, :, -1] * dxy / g.dz
            diag[idx[:, :, 0].ravel()] += czB.ravel()
            diag[idx[:, :, -1].ravel()] += czT.ravel()

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
        rhs = self.rhs_const.copy()
        if f is not None:
            rhs = rhs + np.asarray(f, dtype=float).ravel()
        return self._lu.solve(rhs)

    def solve_perturbation(self, f: np.ndarray) -> np.ndarray:
        return self._lu.solve(np.asarray(f, dtype=float).ravel())

    def release_factorization(self) -> None:
        self._lu = None
        self.A = None

    def velocity(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Face-centred seepage velocity ``v = -K grad(h) / phi``.

        ``vx`` is ``(ny, nx+1, nz)``, ``vy`` is ``(ny+1, nx, nz)`` and ``vz``
        is ``(ny, nx, nz+1)``.  No-flow boundary faces are zero.
        """
        g = self.grid
        ny, nx, nz = g.ny, g.nx, g.nz
        h3 = np.asarray(h, dtype=float).reshape(ny, nx, nz)
        phi = self.phi
        vx = np.zeros((ny, nx + 1, nz))
        vy = np.zeros((ny + 1, nx, nz))
        vz = np.zeros((ny, nx, nz + 1))

        Kx = 2.0 * self.K[:, :-1, :] * self.K[:, 1:, :] / (
            self.K[:, :-1, :] + self.K[:, 1:, :])
        vx[:, 1:nx, :] = (-Kx * (h3[:, 1:, :] - h3[:, :-1, :]) / g.dx
                          / phi[:, 1:, :])
        vx[:, 0, :] = (-self.K[:, 0, :] * (h3[:, 0, :] - self.h_left)
                       / (0.5 * g.dx) / phi[:, 0, :])
        vx[:, nx, :] = (-self.K[:, -1, :] * (self.h_right - h3[:, -1, :])
                        / (0.5 * g.dx) / phi[:, -1, :])

        Ky = 2.0 * self.K[:-1, :, :] * self.K[1:, :, :] / (
            self.K[:-1, :, :] + self.K[1:, :, :])
        vy[1:ny, :, :] = (-Ky * (h3[1:, :, :] - h3[:-1, :, :]) / g.dy
                          / phi[1:, :, :])
        if self.bc_y == "dirichlet":
            raise NotImplementedError("Dirichlet y boundaries are not implemented")

        Kz = 2.0 * self.Kz[:, :, :-1] * self.Kz[:, :, 1:] / (
            self.Kz[:, :, :-1] + self.Kz[:, :, 1:])
        vz[:, :, 1:nz] = (-Kz * (h3[:, :, 1:] - h3[:, :, :-1]) / g.dz
                          / phi[:, :, 1:])
        if self.bc_z == "dirichlet":
            raise NotImplementedError("Dirichlet z boundaries are not implemented")
        return vx, vy, vz


@dataclass
class FlowResponses3D:
    """Unit-rate flow responses: ``head(q) = h0 + q @ hw``."""

    h0: np.ndarray
    hw: np.ndarray
    weights: np.ndarray
    well_cells: np.ndarray
    well_xyz: np.ndarray

    def head(self, q: Sequence[float]) -> np.ndarray:
        q = np.asarray(q, dtype=float).ravel()
        if q.size != self.hw.shape[0]:
            raise ValueError("q has the wrong length")
        return self.h0 + q @ self.hw


def flow_responses3d(
    flow: FlowModel3D,
    well_xyz: Sequence[Sequence[float]],
    eps_cells: float = 1.5,
    eps_z_cells: Optional[float] = None,
) -> FlowResponses3D:
    well_xyz = np.atleast_2d(np.asarray(well_xyz, dtype=float))
    weights = np.vstack([
        gaussian_well_weights3d(flow.grid, x, y, z, eps_cells, eps_z_cells)
        for x, y, z in well_xyz
    ])
    h0 = flow.solve()
    hw = np.vstack([
        flow.solve_perturbation(-weights[i]) for i in range(weights.shape[0])
    ])
    well_cells = np.argmax(weights, axis=1)
    return FlowResponses3D(h0, hw, weights, well_cells, well_xyz)


# ---------------------------------------------------------------------------
# Transport operators
# ---------------------------------------------------------------------------
def advection_operator3d(
    grid: Grid3D,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    phi: float | np.ndarray,
    vx_sign: Optional[np.ndarray] = None,
    vy_sign: Optional[np.ndarray] = None,
    vz_sign: Optional[np.ndarray] = None,
) -> sp.csr_matrix:
    """First-order upwind advection ``div(phi v C)`` on the 3D grid."""
    if vx_sign is None:
        vx_sign = vx
    if vy_sign is None:
        vy_sign = vy
    if vz_sign is None:
        vz_sign = vz
    ny, nx, nz, n = grid.ny, grid.nx, grid.nz, grid.n
    idx = np.arange(n).reshape(ny, nx, nz)
    phi_f = (np.full(n, float(phi)) if np.ndim(phi) == 0
             else np.asarray(phi, float).ravel())
    p = phi_f.reshape(ny, nx, nz)

    diag = np.zeros(n)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    def add_faces(a, b, pa, pb, flux, sign):
        av = np.asarray(flux, float).ravel()
        sv = np.asarray(sign, float).ravel()
        a = a.ravel()
        b = b.ravel()
        pa = pa.ravel()
        pb = pb.ravel()
        pos = sv >= 0.0
        if np.any(pos):
            diag[a[pos]] += pa[pos] * av[pos]
            rows.append(b[pos])
            cols.append(a[pos])
            vals.append(-pa[pos] * av[pos])
        neg = ~pos
        if np.any(neg):
            diag[b[neg]] += pb[neg] * (-av[neg])
            rows.append(a[neg])
            cols.append(b[neg])
            vals.append(-pb[neg] * (-av[neg]))

    # interior x faces: area dy*dz
    add_faces(idx[:, :-1, :], idx[:, 1:, :], p[:, :-1, :], p[:, 1:, :],
              vx[:, 1:nx, :] * grid.dy * grid.dz, vx_sign[:, 1:nx, :])
    # interior y faces: area dx*dz
    add_faces(idx[:-1, :, :], idx[1:, :, :], p[:-1, :, :], p[1:, :, :],
              vy[1:ny, :, :] * grid.dx * grid.dz, vy_sign[1:ny, :, :])
    # interior z faces: area dx*dy
    add_faces(idx[:, :, :-1], idx[:, :, 1:], p[:, :, :-1], p[:, :, 1:],
              vz[:, :, 1:nz] * grid.dx * grid.dy, vz_sign[:, :, 1:nz])

    # x boundary outflow; inflow is clean water C=0
    area = grid.dy * grid.dz
    m = vx_sign[:, 0, :] < 0
    if np.any(m):
        diag[idx[:, 0, :][m]] += (-p[:, 0, :][m] * vx[:, 0, :][m] * area)
    m = vx_sign[:, -1, :] > 0
    if np.any(m):
        diag[idx[:, -1, :][m]] += (p[:, -1, :][m] * vx[:, -1, :][m] * area)

    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    return sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()


def face_dispersion3d(
    grid: Grid3D,
    aquifer: Aquifer3D,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Face conductances of the frozen diagonal dispersion tensor."""
    ny, nx, nz = grid.ny, grid.nx, grid.nz
    vxc = 0.5 * (vx[:, :-1, :] + vx[:, 1:, :])
    vyc = 0.5 * (vy[:-1, :, :] + vy[1:, :, :])
    vzc = 0.5 * (vz[:, :, :-1] + vz[:, :, 1:])
    speed = np.sqrt(vxc ** 2 + vyc ** 2 + vzc ** 2)
    speed = np.maximum(speed, 1e-12)
    a_l, a_th, a_tv = aquifer.alpha_l, aquifer.alpha_th, aquifer.alpha_tv
    Dxx = a_th * speed + (a_l - a_th) * vxc ** 2 / speed
    Dyy = a_th * speed + (a_l - a_th) * vyc ** 2 / speed
    Dzz = a_tv * speed + (a_l - a_tv) * vzc ** 2 / speed
    Dxx_f = 0.5 * (Dxx[:, :-1, :] + Dxx[:, 1:, :])
    Dyy_f = 0.5 * (Dyy[:-1, :, :] + Dyy[1:, :, :])
    Dzz_f = 0.5 * (Dzz[:, :, :-1] + Dzz[:, :, 1:])
    phi = aquifer.phi_field((ny, nx, nz))
    phix = 0.5 * (phi[:, :-1, :] + phi[:, 1:, :])
    phiy = 0.5 * (phi[:-1, :, :] + phi[1:, :, :])
    phiz = 0.5 * (phi[:, :, :-1] + phi[:, :, 1:])
    gx = phix * Dxx_f * (grid.dy * grid.dz) / grid.dx
    gy = phiy * Dyy_f * (grid.dx * grid.dz) / grid.dy
    gz = phiz * Dzz_f * (grid.dx * grid.dy) / grid.dz
    return gx, gy, gz


def dispersion_operator3d(
    grid: Grid3D,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
) -> sp.csr_matrix:
    ny, nx, nz, n = grid.ny, grid.nx, grid.nz, grid.n
    idx = np.arange(n).reshape(ny, nx, nz)
    diag = np.zeros(n)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    def add(a, b, g):
        a = a.ravel()
        b = b.ravel()
        g = np.asarray(g, float).ravel()
        diag[a] += g
        diag[b] += g
        rows.extend([b, a])
        cols.extend([a, b])
        vals.extend([-g, -g])

    add(idx[:, :-1, :], idx[:, 1:, :], gx)
    add(idx[:-1, :, :], idx[1:, :, :], gy)
    add(idx[:, :, :-1], idx[:, :, 1:], gz)

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
class TransportResult3D:
    times: np.ndarray
    C_well: np.ndarray
    C_record: Optional[np.ndarray]
    record_cells: Optional[np.ndarray]
    C_fields: Optional[np.ndarray]
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


def transport_run3d(
    grid: Grid3D,
    aquifer: Aquifer3D,
    flow: FlowModel3D,
    resp: FlowResponses3D,
    q: Sequence[float],
    dt: float,
    nsteps: int,
    source_idx: Optional[int] = None,
    source_rate: float = 0.0,
    source_steps: Optional[int] = None,
    c_init: Optional[np.ndarray] = None,
    record_cells: Optional[Sequence[int]] = None,
    store_fields: bool = False,
    disp_ref: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> TransportResult3D:
    """Implicit-Euler 3D transport for a steady extraction vector ``q``."""
    q = np.asarray(q, dtype=float).ravel()
    nw = q.size
    if resp.hw.shape[0] != nw:
        raise ValueError("q length does not match the number of wells")
    ny, nx, nz, n = grid.ny, grid.nx, grid.nz, grid.n
    phi = aquifer.phi_field((ny, nx, nz)).ravel()
    V = grid.cell_volume
    st = aquifer.R * phi * V / dt
    decay = aquifer.decay * aquifer.R * phi * V

    if disp_ref is None:
        vx0, vy0, vz0 = flow.velocity(resp.h0)
    else:
        vx0, vy0, vz0 = disp_ref
    gx, gy, gz = face_dispersion3d(grid, aquifer, vx0, vy0, vz0)
    Adisp = dispersion_operator3d(grid, gx, gy, gz)

    h = resp.head(q)
    vx, vy, vz = flow.velocity(h)
    Aadv = advection_operator3d(grid, vx, vy, vz, phi)
    sink = q @ resp.weights
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
    mass_total[0] = float(np.sum(phi * aquifer.R * C) * V)

    area = grid.dy * grid.dz
    for k in range(nsteps):
        active = src if (source_steps is None or k < source_steps) else 0.0
        C = lu.solve(st * C + active)
        if C_fields is not None:
            C_fields[k + 1] = C
        C_well[k] = resp.weights @ C
        if record_cells is not None:
            C_rec[k] = C[record_cells]

        C3 = C.reshape(ny, nx, nz)
        mass_total[k + 1] = float(np.sum(phi * aquifer.R * C) * V)
        mass_source[k + 1] = mass_source[k] + float(np.sum(active)) * dt
        mass_well[k + 1] = mass_well[k] + float(np.dot(C, sink)) * dt
        out = 0.0
        phi3 = phi.reshape(ny, nx, nz)
        v = vx[:, 0, :]
        out += float(np.sum(np.maximum(-v, 0.0) * phi3[:, 0, :]
                            * C3[:, 0, :])) * area
        v = vx[:, -1, :]
        out += float(np.sum(np.maximum(v, 0.0) * phi3[:, -1, :]
                            * C3[:, -1, :])) * area
        mass_out[k + 1] = mass_out[k] + out * dt
        mass_decay[k + 1] = mass_decay[k] + float(np.sum(decay * C)) * dt

    mass = {
        "total": mass_total,
        "source": mass_source,
        "well": mass_well,
        "outflow": mass_out,
        "decay": mass_decay,
        "integrated_well_mass": np.array(float(mass_well[-1])),
    }
    ops = {"A": A, "lu": lu, "st": st, "vx": vx, "vy": vy, "vz": vz,
           "phi": phi}
    return TransportResult3D(
        times=np.arange(nsteps + 1) * dt,
        C_well=C_well,
        C_record=C_rec if record_cells is not None else None,
        record_cells=record_cells,
        C_fields=C_fields,
        mass=mass,
        C_final_field=C.copy(),
        operators=ops,
    )
