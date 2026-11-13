# 2D Plan — From the 1D Sandbox to a Two-Dimensional Transport Model

This document is the **2D extension of `plan.md`**. It assumes the Phase 1
result is already trusted:

- 1D dimensionless ADE against Ogata–Banks (1961):
  **full-grid rel-L2 `1.35e-2`, corner-excluded interior `8.0e-3`**;
- the remaining full-grid error is the measure-zero inlet-corner singularity,
  not a network or loss bug (`Actions.md`, `HUMAN.md`).

The 2D plan deliberately repeats the Phase 1 discipline: **one new physical
ingredient per stage, one closed-form or manufactured reference per stage, and
an explicit pass/fail gate before advancing.**

---

## 0. What changes from 1D

| Ingredient | 1D sandbox | 2D extension | New risk |
|---|---|---|---|
| Space | `(ξ,τ)` | `(ξ,η,τ)` | more collocation points, 2D boundary layers |
| Velocity | constant `v` | uniform `v`, then Darcy `q = -K∇h` | divergence-free velocity, well singularities |
| Dispersion | scalar `D` | tensor `D_L`, `D_T`, anisotropy ratio | transverse plume spreading |
| Boundaries | two Dirichlet ends | inlet, outlet, no-flux walls, wells | corner singularities at inlet/wall junctions |
| Reference | Ogata–Banks | extruded OB, Green's functions, MMS, FV solver | no single closed form for heterogeneous `K` |
| Operator learning | 1D FNO | 2D FNO / autoregressive rollout | spectral bias, rollout stability |
| Cost | trivial | moderate | 10–100× more residual points |

The 2D code should be **dimension-agnostic where possible**. The 3D plan
(`3D_plan.md`) reuses the same `physics/` modules, changing only the grid,
the dispersion tensor, and the hard-boundary functions.

---

## 1. Governing equations

### 1.1 Flow (optional in Stage 2.1, required from Stage 2.3)

For a confined, constant-density aquifer,

```
-∇·(K ∇h) = W,          q = -K ∇h,          v = q / φ,
```

where `h` is hydraulic head, `K(x,y)` hydraulic conductivity, `q` specific
discharge, `φ` porosity, `v` seepage (pore) velocity, and `W` sources/sinks
(extraction negative, injection positive). On no-flow boundaries
`q·n = 0`; on Dirichlet boundaries `h = h_D`.

For the first 2D transport tests, keep the **decoupled** approach:

1. prescribe a uniform velocity `v = (U,0)`, or
2. solve the flow problem once (analytically or with a finite-volume solver),
3. feed the resulting velocity field to the transport PINN/FNO.

Do **not** start with a monolithic flow–transport PINN. Coupling introduces
second derivatives of `h` inside the transport residual and makes loss balancing
much harder. A monolithic variant is a later experiment (Stage 2.3b).

### 1.2 Transport

Conservative ADE with linear equilibrium sorption and first-order decay:

```
R ∂C/∂t + ∇·(v C) - ∇·(D ∇C) + λ R C = S,
```

with `R = 1 + ρ_b K_d/φ`, decay rate `λ ≥ 0`, source/sink `S`, and
`∇·v = 0` so that `∇·(vC) = v·∇C` for constant `φ`.

2D dispersion tensor:

```
D = α_T |v| I + (α_L - α_T) (v vᵀ)/|v| + D_m τ I.
```

In components, with `v = (v_x,v_y)` and `|v| = sqrt(v_x²+v_y²)`:

```
D_xx = α_L |v| v_x²/|v|² + α_T |v| v_y²/|v|² + D_m τ
D_yy = α_L |v| v_y²/|v|² + α_T |v| v_x²/|v|² + D_m τ
D_xy = (α_L - α_T) |v| v_x v_y/|v|²
```

For the first tests set `D_m τ` to zero or absorb it into `α_T`.

### 1.3 Dimensionless form

Use

```
ξ = x/L,   η = y/L,   τ = t U/L,   u = C/C_0,
Pe_L = U L / D_L,   Pe_T = U L / D_T,   r = Pe_T/Pe_L = D_L/D_T,
```

and, if flow is solved,

```
h̃ = h/H_0,   K̃ = K/K_0,   W̃ = W L²/(K_0 H_0).
```

The transport equation becomes

```
R u_τ + ṽ·∇u - (1/Pe_L) ∇·(D̃ ∇u) + λ̃ R u = S̃,
```

with `D̃ = D/D_L`. The flow equation becomes

```
-∇·(K̃ ∇h̃) = W̃.
```

All quantities are O(1); this is the same conditioning trick that made the
1D residual trainable.

---

## 2. Verification hierarchy (do not skip)

Every stage below has a **reference** and a **gate**. Use the gate number in
commit messages, run manifests, and `Actions.md`.

### Stage 2.1 — Extruded 1D verification

**Physics:** uniform flow `v=(U,0)`, no transverse source, initial and inlet
conditions independent of `η`, no-flux walls at `η=0,1`.

**Exact solution:** the 1D Ogata–Banks solution, extruded in `η`:

```
u_ref(ξ,η,τ) = u_OB(ξ,τ).
```

**Purpose:** prove the 2D machinery (grid, 2D autodiff, boundary handling,
sampling, metrics) has not broken the trusted 1D result.

**Gate G2.1:** corner-excluded interior rel-L2 `≤ 1.5e-2`; wall-normal
gradient `‖∂u/∂η‖_2 ≤ 1e-3`; full-grid error no worse than 1D by more than 20%.

### Stage 2.2 — Transverse dispersion and point/line sources

**Physics:** uniform flow, continuous point source in an infinite or large
domain, `D_T > 0`, anisotropic `α_L/α_T`.

**Reference (steady 2D point source):**

```
C_ss(x,y) = Q/(2π b √(D_L D_T))
            exp(U x/(2 D_L))
            K_0( (U/(2 D_L)) √(x² + (D_L/D_T) y²) ),
```

where `Q` is the mass release rate, `b` the aquifer thickness, and `K_0` the
modified Bessel function of the second kind. For a transient continuous source,
use the convolution of the instantaneous 2D Green's function:

```
C(x,y,t) = ∫_0^t  Ṁ/(4π φ b s √(D_L D_T))
           exp[-(x-Us)²/(4 D_L s) - y²/(4 D_T s)] ds.
```

For no-flux walls, add image sources; for a finite inflow/outflow domain,
choose `L` and `T` so the plume stays away from the boundaries, or use the
numerical reference solver.

**Tests:**

- `α_L/α_T = 1, 10, 100`; check plume elongation and transverse width;
- centerline peak and transverse profile against the reference;
- mass balance (Section 5);
- grid/point convergence at fixed `Pe_L`.

**Gate G2.2:** rel-L2 `≤ 5%` in the plume core `{u ≥ 10⁻³}`; centerline peak
error `≤ 5%`; mass-balance error `≤ 1%`.

### Stage 2.3 — Darcy flow, wells, and a divergence-free velocity field

**Physics:** steady flow with one extraction and one injection well;
homogeneous `K` first, then a single lens/layer.

**Reference:** superposition of Thiem solutions in a confined aquifer,

```
h(r) = h_0 ± Q_w/(2π T) ln(r/r_w),    T = K b,
```

with method-of-images corrections for no-flow boundaries. For the velocity
field, compare `v = -K∇h/φ` to the analytic gradient away from the well
singularity.

**Two implementations, in order:**

1. **Decoupled (recommended):** solve `h` with a finite-volume solver (or a
   separate flow PINN), evaluate `v` on the transport collocation points,
   then train the transport PINN with that frozen velocity.
2. **Monolithic (later):** shared trunk with two outputs `(h,C)`, losses for
   flow, transport, divergence, and BCs. Only attempt after (1) passes.

**Gate G2.3:** head rel-L2 `≤ 1e-3`; velocity rel-L2 `≤ 1e-2` away from wells;
`‖∇·v‖` small compared with `|v|/L`; transport gate as in G2.2.

### Stage 2.4 — Heterogeneity

**Physics:** layered `K`, then random log-normal `K(x,y)` with correlation
length `ℓ`, then (optionally) channelized fields.

**Reference:** a mass-conservative finite-volume (FV) solver on `128²`/`256²`
with a grid-convergence study; or a manufactured solution built with the same
heterogeneous `K`.

**Why MMS is mandatory here:** there is no simple closed form for arbitrary
`K`. Choose a smooth `u*(ξ,η,τ)`, substitute into the PDE, and define

```
S* = R u*_τ + v·∇u* - ∇·(D∇u*) + λ R u*.
```

Then `u*` is the exact solution of the forced problem. The PINN/FNO must
recover `u*` and the FV solver must show the expected order of accuracy.

**Gate G2.4:** FV grid convergence observed order `≥ 1.8`; PINN rel-L2
`≤ 2e-2` on the MMS problem; `≤ 5%` on held-out random `K` fields; mass
balance `≤ 2%`.

### Stage 2.5 — Remediation scenario and operator learning

**Scenario:** pump-and-treat with one or more extraction wells, an injection
well for flushing, a source zone, and a compliance monitoring well. Optional
permeable reactive barrier modeled as a zone with `λ > 0` or `R > 1`.

**Outputs that matter to a remediation engineer:**

- extracted mass and effluent concentration history;
- cleanup time to a regulatory threshold at the compliance well;
- capture zone / streamline pattern;
- sensitivity to `K`, source strength, and well rates.

**FNO:** learn the map

```
(K, source, well locations/rates, BCs)  ->  C(x,y,t)
```

from FV-generated scenarios, then use the surrogate for Monte Carlo and
optimization.

**Gate G2.5:** mass balance `≤ 2%`; cleanup-time error `≤ 10%`; FNO rollout
error `≤ 5%` at final time; surrogate `≥ 100×` faster than the FV reference
at comparable accuracy.

---

## 3. PINN design for 2D

### 3.1 Network and inputs

- Inputs `(ξ,η,τ)` scaled to `[-1,1]` (or `[0,1]`); optional Fourier features
  for high `Pe_L` (steep fronts).
- MLP: 4–6 hidden layers, 64–128 units, `tanh`; output `sigmoid` only for the
  soft-BC baseline. With hard constraints, use a linear output and let the
  ansatz enforce the bounds.
- For flow–transport: either two separate networks (decoupled) or a shared
  trunk with two heads (monolithic).

### 3.2 Hard constraints in a rectangle

Use the lift-plus-mask form

```
u_θ(ξ,η,τ) = P(ξ,η,τ) + B(ξ,η,τ) N_θ(ξ,η,τ),
```

where `P` satisfies the Dirichlet data and `B` vanishes on the Dirichlet
boundaries and at `τ=0`. For the box `(0,1)²` with inlet at `ξ=0`, outlet at
`ξ=1`, and no-flux walls at `η=0,1`, a practical choice is

```
s(τ) = 1 - exp(-τ/ε),
P(ξ,η,τ) = (1-ξ) g(η,τ) s(τ),
B(ξ,η,τ) = ξ(1-ξ) [1 + γ η²(1-η)²] s(τ).
```

Then:

- `u(ξ,η,0)=0` exactly (because `s(0)=0`);
- `u(0,η,τ)=g(η,τ)s(τ) -> g` exponentially fast;
- `u(1,η,τ)=0` (Dirichlet outlet; consistent with the semi-infinite reference);
- `∂B/∂η = 0` at `η=0,1`, so the no-flux walls are respected by the mask,
  while `u` itself is not forced to zero there.

For a general site geometry, replace the product mask by a signed-distance
function or an R-function; for no-flux boundaries, either use a mask with zero
normal derivative or enforce the flux softly.

### 3.3 Losses

```
L = λ_r L_res + λ_b L_bc + λ_0 L_ic + λ_d L_data
    + λ_div L_div + λ_flux L_flux.
```

- `L_res`: mean squared PDE residual on collocation points;
- `L_bc`: Dirichlet/Neumann/outflow mismatch;
- `L_ic`: initial condition;
- `L_data`: optional sparse measurements (monitoring wells);
- `L_div`: `‖∇·v‖²` for the flow network;
- `L_flux`: no-flux/outflow penalties.

Use **non-dimensional residuals**; report each term separately in the manifest.
Balance with gradient norms or NTK weighting (Phase 3 of `plan.md`), not with
hand-tuned constants alone.

### 3.4 Sampling

- Uniform/Latin-hypercube or Sobol points in `(ξ,η,τ)`;
- oversample near the inlet, the source, and the plume front;
- residual-based adaptive refinement (RAR) every `N` steps: add points where
  `|R|` is largest;
- for high `Pe_L`, use a **causal time weighting** or train in time windows;
- always keep a fixed evaluation grid independent of the training points.

### 3.5 Optimization and diagnostics

- Adam first (`2×10⁴–10⁵` steps, resampled collocation), then L-BFGS
  (`500–2000` iterations) on a fixed set;
- curriculum on `Pe_L`: start at `Pe_L≈5`, ramp to the target;
- log: total and per-term losses, PDE-residual percentiles, BC/IC errors,
  gradient norms, mass balance, and wall-clock per 1k steps;
- save `run_config.json` + `metrics.json` + `model.pt` in
  `out/2d/<run_id>/`, following `GROWTH.md`.

---

## 4. FNO design for 2D

### 4.1 Operator and data

Learn

```
G_θ : (K, S, well configuration) -> C(·,·,·).
```

For the uniform-flow stages, the “parameter” input can be a small vector
`(Pe_L, Pe_T, source location, source strength)` broadcast to a constant
channel. For heterogeneous stages, `K(x,y)` is a channel.

### 4.2 Architectures to compare

1. **Space–time FNO (3D spectral conv on `ξ,η,τ`)** — direct, but memory-hungry
   and less stable for long horizons.
2. **Autoregressive FNO2d** —
   `C^{n+1} = G_θ(C^n, K, S; Δτ)`, rolled out over `N_t` steps. Recommended
   for long-time remediation scenarios.
3. **U-NO / multi-resolution FNO** — if the plume has both sharp fronts and
   large-scale spreading.

Start with (2). Use (1) only for short-horizon experiments.

### 4.3 Training

- Normalize each channel (mean/std or min–max); keep a mask for solid/no-flow
  regions;
- loss = one-step rel-L2 + rollout loss with increasing horizon
  (pushforward/temporal bundling);
- sweep modes `k_max ∈ {8,12,16,24}` and width `∈ {32,64}`;
- test resolution invariance: train on `128²`, evaluate on `64²` and `256²`;
- optional physics-informed term: finite-difference or spectral residual on
  the output grid;
- report one-step, 10-step, full-rollout, spectral, and mass-balance errors.

### 4.4 Data generation

Build `fv2d.py` (or wrap an existing simulator) with:

- cell-centered FV, harmonic-mean face conductivities for Darcy;
- upwind or flux-limited advection, implicit diffusion, operator splitting;
- positivity-preserving limiting for high `Pe_L`;
- grid-convergence test (`64²`, `128²`, `256²`);
- random log-normal `K` via FFT with prescribed correlation length and
  anisotropy;
- storage in `.npz` for smoke tests, Zarr/HDF5 for production.

---

## 5. Metrics and acceptance gates

Report **all** of the following, not just one number:

1. **Field error**
   ```
   E_rel = ‖u_θ-u_ref‖₂ / ‖u_ref‖₂,
   E_rel^int = same over Ω_int = {ξ,η > δ} (boundary-layer excluded).
   ```
2. **Boundary-layer / corner error** — 2D has additional singularities where
   a Dirichlet inlet meets a no-flux wall. Always report the interior metric.
3. **Mass balance**
   ```
   M(t) = ∫_Ω φ R C dV,
   B(t) = |M(t)-M(0)-∫₀ᵗ[∫S dV - ∫_{Γ_out} φ v·n C dA
                           - ∫λRφC dV] dt| / max(M, M_ref).
   ```
4. **Breakthrough curves** at monitoring wells:
   rel-L2, peak concentration, arrival time, cleanup time.
5. **Plume moments**: total mass, centroid, covariance (spread).
6. **PDE/BC residuals** on a fixed evaluation grid (mean, 95th percentile,
   maximum).
7. **FNO-specific**: one-step error, rollout error vs horizon, spectral error,
   resolution-transfer error, conservation error.
8. **Cost**: wall-clock, GPU memory, number of residual evaluations.

### Gate summary

| Gate | Check | Threshold |
|---|---|---|
| G2.1 | extruded 1D, interior rel-L2 | `≤ 1.5e-2` |
| G2.1 | wall-normal gradient | `≤ 1e-3` |
| G2.2 | plume-core rel-L2 | `≤ 5%` |
| G2.2 | mass balance | `≤ 1%` |
| G2.3 | head / velocity rel-L2 | `≤ 1e-3` / `≤ 1e-2` |
| G2.4 | FV observed order | `≥ 1.8` |
| G2.4 | PINN MMS / held-out K | `≤ 2e-2` / `≤ 5%` |
| G2.5 | cleanup-time error | `≤ 10%` |
| G2.5 | FNO rollout error | `≤ 5%` |
| G2.5 | surrogate speedup | `≥ 100×` |

---

## 6. Work breakdown

| Phase | Work | Deliverables | Definition of done |
|---|---|---|---|
| **2D-0** | Refactor into `physics/` + `solve_2d_*.py`; port config/manifest/checkpoint helpers | `physics/transport.py`, `physics/references.py`, `physics/fv2d.py`, tests | unit tests pass; 1D code unchanged |
| **2D-1** | Extruded 1D verification | `solve_2d_pinn.py`, `out/2d/extruded/` | G2.1 |
| **2D-2** | Point/line source, transverse dispersion, anisotropy | reference integrals, `out/2d/source/` | G2.2 |
| **2D-3** | Darcy flow + wells; decoupled and monolithic | `physics/flow.py`, `out/2d/wells/` | G2.3 |
| **2D-4** | Layered and random `K`; MMS; FV reference | `physics/fv2d.py`, MMS suite | G2.4 |
| **2D-5** | Remediation scenario, FNO2d, rollout, surrogate | `solve_2d_fno.py`, `out/2d/fno/` | G2.5 |
| **2D-6** | Report, figures, update `Actions.md` and `HUMAN.md` | `2D_results.md`, PDF appendix | all gates green |

Suggested order: **2D-0 → 2D-1 → 2D-2 → 2D-3 → 2D-4 → 2D-5**. Do not start
2D-4 before G2.2 passes; do not start 2D-5 before G2.3 and the FV solver are
trusted.

---

## 7. Compute budget (order of magnitude)

| Task | Grid / points | Hardware | Wall-clock |
|---|---|---|---|
| 2D PINN, uniform flow | `N_r=2×10⁴–10⁵` | CPU or 8–16 GB GPU | hours–1 day |
| 2D PINN, wells | `N_r=5×10⁴–2×10⁵` | 16 GB GPU | 1–3 days |
| FV reference, one `K` field | `128²–256²`, implicit | CPU, parallel scenarios | minutes–hours |
| FNO2d, 500–1000 scenarios | `128²`, 4–6 layers | 16–24 GB GPU | 12–48 h |
| FNO2d, `256²` | `256²`, 32–64 channels | 24–40 GB GPU | 2–5 days |

Start with `128²`; only move to `256²` after the gates pass.

---

## 8. Risks and mitigations

| Risk | Why it happens | Mitigation |
|---|---|---|
| Inlet/wall corner singularity | Dirichlet inlet meets no-flux wall | report interior metric; round the corner or exclude a small band; local refinement |
| No-flux hard constraints are hard | flux BCs are not simple point values | use a zero-normal-derivative mask or soft penalty; verify `∂u/∂n` |
| Coupled flow–transport instability | `v` depends on `∇h`, second derivatives | decouple first; freeze velocity; only then try monolithic |
| Heterogeneous `K` has high-frequency content | FNO truncates modes | more modes, multi-resolution/U-NO, or local PINN refinement |
| Plume front is under-resolved | high `Pe_L` | Fourier features, causal weighting, time marching, finer sampling near front |
| FV solver is not mass-conservative | naive advection/dispersion | TPFA + upwind + implicit diffusion; test mass balance every run |
| Overfitting one “hydrology” | FNO sees one `K` field | randomize `K`, source, wells; held-out scenario splits |
| Metrics only look good on the full grid | corner/boundary layer dominates | always report interior and boundary-excluded errors |

---

## 9. Proposed file layout

```
TESIS/
├── physics/
│   ├── __init__.py
│   ├── transport.py      # ADE residual, 2D/3D dispersion tensor
│   ├── flow.py           # Darcy residual, wells, K-field generators
│   ├── references.py     # OB, Green's functions, MMS builders
│   ├── fv2d.py           # finite-volume reference solver (2D)
│   ├── grids.py          # grids, Sobol/LHS sampling, image sources
│   └── metrics.py        # rel-L2, mass balance, BTC, plume moments
├── solve_2d_pinn.py
├── solve_2d_fno.py
├── configs/2d/*.yaml
├── out/2d/<run_id>/      # config, metrics, model, figures
├── tests/
│   ├── test_transport_2d.py
│   ├── test_references_2d.py
│   ├── test_fv2d.py
│   └── test_metrics.py
└── 2D_results.md
```

Keep `hydrology.py` and `solve_1d_pinn.py` as the trusted regression baseline;
the 2D code should import shared helpers rather than fork them.

---

## 10. Immediate next actions

1. Create `physics/` and move the dimensionless residual, dispersion tensor,
   and metrics there; add a 2D MMS unit test.
2. Implement `solve_2d_pinn.py` with the extruded 1D case first; reproduce
   G2.1.
3. Implement the 2D Green's-function references and `fv2d.py`; run the
   `α_L/α_T` sweep.
4. Add the well flow and decoupled velocity path.
5. Generate the first random-`K` dataset and train FNO2d; compare rollout
   error to the FV reference.
6. Update `Actions.md` with gate results and `HUMAN.md` with the new commands.
