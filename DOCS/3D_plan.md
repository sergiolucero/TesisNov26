# 3D Plan — Three-Dimensional Groundwater Transport and Remediation

This document is the **3D extension of `plan.md`**, and it depends on the
2D plan (`2D_plan.md`). The rule is simple:

> **Do not start a full 3D model until the corresponding 2D gate is green, and
> do not use 3D unless a 2.5D model has been shown to be insufficient for the
> question being asked.**

Three dimensions are justified by vertical flow and vertical heterogeneity,
not by “3D looks more realistic.” The first 3D work should therefore be a
**2.5D axisymmetric or vertical-slice model**, which is cheap enough to debug
and is a genuine subset of the 3D equations.

---

## 0. Why 3D, and when not to use it

Use 3D only when at least one of these is essential:

1. **Partially penetrating wells** — flow has a strong vertical component near
   the screen; a depth-averaged 2D model cannot represent it.
2. **Layered / vertically heterogeneous `K`** — vertical connectivity controls
   plume migration and cleanup.
3. **Vertical anisotropy** `K_v/K_h ≪ 1` — confinement and leakage.
4. **Density-driven flow** (saltwater intrusion, dense DNAPL) — requires a
   coupled variable-density flow and transport model.
5. **True site geometry** — non-box domains, sloping layers, faults.

If none applies, a depth-averaged 2D model plus a vertical post-processing
check is cheaper and easier to verify. Record the justification in the run
manifest.

### 0.1 The 2.5D bridge

Before a full 3D model, implement:

- **Axisymmetric `(r,z,t)` model** for a single well in a layered aquifer.
  Use `r > r_w` to avoid the logarithmic well singularity; impose a flux or
  head condition at `r_w`.
- **Vertical slice `(x,z,t)` model** with uniform `y`, to test vertical
  coupling and no-flow top/bottom boundaries.

Both are much cheaper than 3D, have analytic/numerical references, and reuse
the same physics modules. A 3D result that disagrees with the axisymmetric
limit is a bug until proven otherwise.

---

## 1. Governing equations in 3D

### 1.1 Flow

```
-∇·(K ∇h) = W,          q = -K ∇h,          v = q/φ,
```

with the anisotropic conductivity tensor

```
K = diag(K_h, K_h, K_v)        (horizontal isotropy, vertical anisotropy),
```

or a full symmetric positive-definite tensor for a dipping-layer site.
Boundary conditions: `h = h_D` on Dirichlet boundaries, `q·n = 0` on no-flow
boundaries, and well terms

```
W_w = ± Q_w δ(x-x_w) δ(y-y_w) δ(z-z_w)   (screened interval).
```

For a partially penetrating well, distribute `Q_w` over the screened interval
rather than using a point source.

### 1.2 Transport

```
R ∂C/∂t + ∇·(v C) - ∇·(D ∇C) + λ R C = S,
```

with the 3D dispersion tensor written in the standard form with longitudinal
dispersivity `α_L`, horizontal transverse dispersivity `α_TH`, and vertical
transverse dispersivity `α_TV` (Bear, 1972). The 2D tensor in `2D_plan.md` is
the special case `α_TH = α_TV = α_T`.

For the first 3D tests use the same simplifications as in 1D/2D:

- constant `φ`, `R`, `λ`;
- no density coupling;
- uniform flow, then steady Darcy flow, then layered `K`, then random `K`.

Do **not** start with variable-density flow. It changes the flow equation
(`ρ(C)`), adds a buoyancy term, and requires a different set of references.

### 1.3 Dimensionless groups

```
ξ=x/L, η=y/L, ζ=z/H, τ=tU/L, u=C/C_0,
Pe_L=UL/D_L, Pe_TH=UL/D_TH, Pe_TV=UH/D_TV,
κ=K_v/K_h, ε=H/L.
```

The transport equation is the same as in 2D, with the 3D tensor. The flow
equation becomes

```
-∇·(K̃ ∇h̃) = W̃,
```

where `K̃` carries the anisotropy ratio `κ`. Report `Pe_L`, `Pe_TH`, `Pe_TV`,
`κ`, and `ε` with every 3D run.

---

## 2. Verification hierarchy

### Stage 3.1 — Extruded 1D/2D verification

**Physics:** uniform flow along `x`; initial, inlet, and wall conditions
independent of `y` and `z`; no-flux lateral and bottom/top boundaries.

**Reference:** `u_ref(ξ,η,ζ,τ) = u_OB(ξ,τ)`.

**Purpose:** prove the 3D grid, 3D autodiff, hard masks, and metrics recover
the trusted 1D result before adding vertical physics.

**Gate G3.1:** interior rel-L2 `≤ 1.5e-2`; `‖∂u/∂η‖` and `‖∂u/∂ζ‖`
`≤ 1e-3`; full-grid error no worse than 1D by more than 20%.

**Status of the finite-volume half (2026-11-23): PASSED**, with one caveat about
which analytic solution applies. A mass-flux inlet is not a Dirichlet inlet, so
the classical Ogata--Banks formula -- which carries the reflected
`exp(vx/D) erfc(b)` term -- is the wrong reference: at the parameters used it
differs by a *shift* of `D/v` (5--10 cells) that never shrinks with `dx`. Against
the swept front `0.5 C0 erfc((x - vt)/2 sqrt(Dt))` the interior relative L2 is
`1.1e-2` at `dx = 1 m` and `4.0e-3` at `dx = 0.5 m` (first order in `dx`; `3.4e-3`
once the upwind term `v dx/2` is added to `D`). The transverse requirements are
satisfied at machine precision on an extruded column, and the dispersion tensor
itself is verified by pulse moments (`alpha_L v + v dx/2` longitudinally,
`alpha_T v` transversely, centroid at the pore velocity). Tests:
`tests/test_fv3d_reference.py`. This is the reference solver
(`physics/fv3d.py`); the production 3D pipeline is `full3D_colab_run.py`, whose
own verification is mass balance plus NumPy/torch parity plus the same moment
test.

### Stage 3.2 — 3D point-source Green's function

**Physics:** uniform flow, continuous point source in an infinite domain,
anisotropic dispersion `D_L`, `D_T`, `D_V`.

**Reference (steady continuous point source):**

```
C_ss(x,y,z) = Q/(4π √(D_L D_T D_V))
              exp(Ux/(2D_L))
              exp(-m R)/R,
R = √(x²/D_L + y²/D_T + z²/D_V),
m = U/(2√D_L).
```

For a transient release, convolve the 3D instantaneous Green's function:

```
C(x,y,z,t) = ∫₀ᵗ  Ṁ/(φ (4π s)^{3/2} √(D_L D_T D_V))
             exp[-(x-Us)²/(4D_L s) - y²/(4D_T s) - z²/(4D_V s)] ds.
```

Add image sources for no-flux boundaries. Use a large enough domain that the
plume does not reach the boundaries during the simulation window.

**Gate G3.2:** plume-core rel-L2 `≤ 5%`; mass balance `≤ 1%`; correct
`D_L/D_T/D_V` scaling of the plume second moments.

### Stage 3.3 — 3D flow, partially penetrating wells, axisymmetric check

**Physics:** steady Darcy flow with one extraction and one injection well,
screened over a finite interval; layered `K`; `K_v/K_h < 1`.

**References:**

1. **Axisymmetric `(r,z)` finite-volume/PINN model** — the primary reference.
   Compare 3D and axisymmetric heads and fluxes away from the well.
2. **Thiem solution** for a fully penetrating well in a homogeneous aquifer,
   as a limiting case.
3. **Flow MMS** for layered `K`: choose `h*(x,y,z)`, compute
   `W* = -∇·(K∇h*)`, and impose `h*` on all boundaries.

**Gate G3.3:** 3D vs axisymmetric head rel-L2 `≤ 2%`; 3D vs axisymmetric
velocity rel-L2 `≤ 5%`; flow MMS observed order `≥ 1.8` for the FV reference;
discrete divergence `‖∇·v‖` small compared with `|v|/L`.

### Stage 3.4 — Layered and random 3D heterogeneity

**Physics:** three-layer `K` with a low-permeability confining unit, then a
random log-normal 3D field with anisotropic correlation lengths
(`ℓ_h`, `ℓ_v`).

**Reference:** 3D finite-volume solver on `64³`/`128³`, tested with MMS and
grid convergence.

**Gate G3.4:** FV observed order `≥ 1.8`; PINN MMS rel-L2 `≤ 2e-2`; FNO
held-out-field error `≤ 5–10%`; mass balance `≤ 2%`.

### Stage 3.5 — 3D FNO and uncertainty

**Operator:**

```
G_θ : (K(x,y,z), source, well screen/rates) -> C(x,y,z,t).
```

**Reference data:** 3D FV scenarios; storage and sampling plan in Section 6.

**Gate G3.5:** one-step error `≤ 2%`; 100-step rollout `≤ 10%`; stable
(no blow-up) over the full remediation horizon; resolution transfer
`64³ -> 128³` error `≤ 15%`; conservation error `≤ 2%`.

### Stage 3.6 — Application / surrogate

**Scenario:** pump-and-treat or permeable reactive barrier in a layered 3D
aquifer. Use the FNO surrogate for Monte Carlo over `K`, source strength, and
well rates; validate a subset with the FV solver.

**Gate G3.6:** surrogate vs FV on held-out scenarios `≤ 10%` in cleanup time
and extracted mass; `≥ 100×` speedup; uncertainty rankings stable under
surrogate error.

---

## 3. PINN design for 3D

### 3.1 Network and hard constraints

- Inputs `(ξ,η,ζ,τ)`; 4–8 hidden layers, 64–128 units, `tanh`; optional
  Fourier features for sharp fronts.
- Hard mask on a box `(0,1)³` with inlet at `ξ=0`, outlet at `ξ=1`, no-flux on
  the other four faces:

```
s(τ) = 1 - exp(-τ/ε),
P(ξ,η,ζ,τ) = (1-ξ) g(η,ζ,τ) s(τ),
B(ξ,η,ζ,τ) = ξ(1-ξ) [1+γη²(1-η)²] [1+γζ²(1-ζ)²] s(τ).
```

`B` vanishes at `ξ=0,1` and at `τ=0`, and has zero normal derivative at the
no-flux faces, so the mask respects the flux condition without forcing `u=0`
there. For layered media with discontinuous `K`, a single global network may
struggle; use domain decomposition (XPINN) across layers with interface
continuity of `h`, `C`, and normal flux.

- For axisymmetric 2.5D, use inputs `(r,z,τ)` and multiply the residual by `r`
  in the weak/weighted loss to avoid the coordinate singularity at `r=0`; or
  start at `r=r_w>0` with a well flux condition.

### 3.2 Losses

```
L = λ_r L_res + λ_b L_bc + λ_0 L_ic + λ_d L_data
    + λ_div L_div + λ_flux L_flux + λ_int L_interface.
```

Use non-dimensional residuals and log-scaled `K` (`log K`) as network input or
as a conditioning channel. Report each loss term and its gradient norm.

### 3.3 Sampling in 4D

- Sobol/Halton points in `(ξ,η,ζ,τ)`; `N_r = 10⁵–10⁶` per step is typical.
- Importance sampling near wells, the source, layer interfaces, and the plume
  front.
- RAR every `N` steps; add points where `|R|` is in the top percentile.
- Time marching / causal weighting: train early times first, then advance.
- Chunk the residual evaluation (as in `inspect_residual.py`) and use gradient
  accumulation; never build the full `N_x N_y N_z N_t` graph at once.

### 3.4 Optimization

- Adam `10⁵–10⁶` steps (resampled or RAR), then L-BFGS `10³` iterations on a
  fixed set.
- Curriculum: `Pe_L`, then `κ`, then heterogeneity.
- Mixed precision (`float32`) for speed; `float64` only for high-`Pe` or
  verification runs.
- Checkpoint every `N` steps and on SIGTERM; save `run_config.json`,
  `metrics.json`, `model.pt`.

---

## 4. FNO design for 3D

### 4.1 Spectral convolution and parameter count

A 3D Fourier layer keeps `k_x × k_y × k_z` modes and learns

```
R_l ∈ C^{k_x × k_y × k_z × c_in × c_out}.
```

The parameter count per layer is

```
N_spec = k_x k_y k_z c_in c_out.
```

Examples:

| modes | channels | parameters per layer |
|---|---|---|
| `8³` | 32 | `8³·32² = 524k` |
| `12³` | 32 | `12³·32² = 1.77M` |
| `16³` | 32 | `16³·32² = 4.19M` |
| `16³` | 64 | `16³·64² = 16.8M` |

A naive `16³`/64-channel FNO with several layers is large but not impossible;
`8³`–`12³` with 32 channels is the sensible starting point. Use **factorized
or low-rank spectral convolutions** if memory becomes the bottleneck.

### 4.2 Architectures

1. **Autoregressive FNO3d** (recommended first):
   `C^{n+1} = G_θ(C^n, K, S; Δτ)`, with a 3D spectral conv over `(x,y,z)` and
   time as a rollout dimension. Memory per step is manageable, and the rollout
   matches remediation time horizons.
2. **Space–time FNO4d** (direct `(x,y,z,t)` spectral conv): only for short
   horizons or small grids; parameter and memory cost grow quickly.
3. **Factorized FNO / tensorized FNO** for `128³` and larger.
4. **U-NO / multi-resolution** for layered media with sharp interfaces.

### 4.3 Training

- Normalize `K` in log space; normalize `C` per channel;
- rollout loss with increasing horizon and pushforward;
- add a physics-informed residual (finite differences on the output grid) if
  data are scarce;
- sweep modes `{8,12,16}` and width `{32,64}`;
- test resolution transfer `64³ -> 96³ -> 128³`;
- monitor mass balance and positivity; add a soft positivity penalty if needed.

---

## 5. Data, storage, and parallelization

### 5.1 3D finite-volume reference

- Cell-centered FV; TPFA for Darcy with harmonic face means;
- upwind/flux-limited advection, implicit diffusion, operator splitting;
- wells distributed over screened cells;
- grid convergence on `32³`, `64³`, `128³`;
- MMS for layered `K` and for transport with a prescribed velocity.

### 5.2 Storage estimate

One `float32` snapshot on a `128³` grid is `128³·4 B ≈ 8.4 MB`.
A scenario with 100 saved snapshots is `~840 MB`; 500 scenarios is `~420 GB`.
Plan accordingly:

- save every `k`-th time step (e.g. 20–40 snapshots per scenario);
- store in **Zarr** (chunked, compressed, parallel reads) or HDF5;
- keep `.npz` only for small smoke tests;
- version datasets by content: `fv3d_128_n500_k8_v1.zarr`.

### 5.3 Parallelization

- Scenario generation is embarrassingly parallel (multiprocessing/joblib or a
  cluster array job).
- FNO training: data-parallel across GPUs; spectral conv is already local in
  frequency space.
- PINN training: gradient accumulation across chunks; distributed data
  parallel if multiple GPUs are available.

---

## 6. Metrics and acceptance gates

Use the same metrics as in 2D, plus 3D-specific checks:

1. **Field error** `E_rel`, `E_rel^int` (exclude well neighborhoods and
   inlet/wall edges).
2. **Mass balance** over the 3D domain (Section 5 of `2D_plan.md`).
3. **Vertical profiles** of `C` and `h` at monitoring locations.
4. **Layer breakthrough curves** and vertical mass distribution.
5. **Well capture / extracted mass** and cleanup time.
6. **PDE/BC residuals** (mean, 95th percentile, maximum).
7. **FNO rollout**, spectral, resolution-transfer, and conservation errors.
8. **Cost**: wall-clock, GPU memory, storage per scenario.

### Gate summary

| Gate | Check | Threshold |
|---|---|---|
| G3.1 | extruded 1D/2D, interior rel-L2 | `≤ 1.5e-2` |
| G3.1 | `∂u/∂η`, `∂u/∂ζ` | `≤ 1e-3` |
| G3.1 | **FV status** | **PASS** (rel-L2 4.0e-3 at dx = 0.5 m; see Stage 3.1) |
| G3.2 | 3D Green's function plume core | `≤ 5%` |
| G3.2 | mass balance | `≤ 1%` |
| G3.3 | 3D vs axisymmetric head / velocity | `≤ 2%` / `≤ 5%` |
| G3.4 | FV observed order | `≥ 1.8` |
| G3.4 | PINN MMS / FNO held-out `K` | `≤ 2e-2` / `≤ 10%` |
| G3.5 | FNO one-step / 100-step rollout | `≤ 2%` / `≤ 10%` |
| G3.5 | resolution transfer / conservation | `≤ 15%` / `≤ 2%` |
| G3.6 | cleanup time / extracted mass | `≤ 10%` |
| G3.6 | surrogate speedup | `≥ 100×` |

---

## 7. Work breakdown

| Phase | Work | Deliverables | Definition of done |
|---|---|---|---|
| **3D-0** | 2.5D axisymmetric and vertical-slice models; shared 3D physics | `solve_rz_pinn.py`, `physics/transport3d.py` | axisymmetric vs Thiem/MMS |
| **3D-1** | Extruded 3D verification | `solve_3d_pinn.py`, `out/3d/extruded/` | G3.1 |
| **3D-2** | 3D Green's functions, transient/steady | `physics/references3d.py` | G3.2 |
| **3D-3** | Darcy flow, partially penetrating wells, axisymmetric check | `physics/flow3d.py`, `out/3d/wells/` | G3.3 |
| **3D-4** | Layered and random 3D `K`; FV reference; MMS | `physics/fv3d.py`, MMS suite | G3.4 |
| **3D-5** | FNO3d autoregressive, rollout, resolution transfer | `solve_3d_fno.py`, `out/3d/fno/` | G3.5 |
| **3D-6** | Surrogate-based uncertainty and remediation application | `3D_results.md` | G3.6 |

Suggested order: **3D-0 → 3D-1 → 3D-2 → 3D-3 → 3D-4 → 3D-5 → 3D-6**.
If 3D-0 already answers the site question, stop there; document why 3D was not
needed.

---

## 8. Compute and storage budget (order of magnitude)

| Task | Grid / points | Hardware | Wall-clock | Storage |
|---|---|---|---|---|
| 2.5D axisymmetric PINN | `N_r ~ 5×10⁴` | CPU / 8 GB GPU | hours | MB |
| 3D PINN, extruded | `N_r ~ 10⁵` | 16–24 GB GPU | 1–2 days | MB |
| 3D PINN, layered/random `K` | `N_r ~ 2×10⁵–10⁶` | 40–80 GB GPU | 3–10 days | MB |
| 3D FV, one scenario | `128³`, implicit | CPU, parallel | hours | ~0.2–0.8 GB |
| FNO3d, 200–500 scenarios | `64³`, 8–12 modes | 24–40 GB GPU | 2–7 days | 50–200 GB |
| FNO3d, `128³` | `128³`, factorized | 80 GB GPU / multi-GPU | weeks | 0.4–2 TB |

Use `64³` for architecture search; use `128³` only after G3.4 and G3.5 pass at
`64³`. Compress and subsample snapshots aggressively.

---

## 9. Risks and mitigations

| Risk | Why it happens | Mitigation |
|---|---|---|
| Curse of dimensionality | 4D input, 3D output | start 2.5D; Sobol + RAR; domain decomposition; GPU |
| Memory blow-up in autograd | second derivatives on `10⁶` points | chunked residual, gradient accumulation, checkpointing |
| FNO3d parameter explosion | `k³ c_in c_out` | 8–12 modes, 32 channels, factorized/low-rank spectral conv |
| Data storage explosion | `128³ × 100 × 500` | Zarr, save fewer snapshots, compression, `64³` first |
| Well singularity | `ln r` / `1/r` near wells | exclude well cells from error metrics; use analytical well model or local refinement |
| Layer interface errors | discontinuous `K` | XPINN/domain decomposition; interface flux loss; MMS |
| Unstable rollout | autoregressive error growth | pushforward, multi-step loss, spectral normalization, conservative corrections |
| Overfitting the training `K` fields | too few scenarios | randomize `K`, layers, wells; held-out scenario splits |
| False confidence from full-grid metrics | boundary/well singularities | always report interior/well-excluded errors and mass balance |
| 3D is unnecessary | 2.5D would answer the question | require a written justification before 3D-3 |

---

## 10. Proposed file layout

```
TESIS/
├── physics/
│   ├── transport3d.py    # 3D ADE residual and dispersion tensor
│   ├── flow3d.py         # 3D Darcy, layered K, wells
│   ├── references3d.py   # Green's functions, axisymmetric limits, MMS
│   ├── fv3d.py           # 3D finite-volume reference solver
│   ├── rz.py             # axisymmetric (r,z) operators
│   └── metrics3d.py      # vertical profiles, layer BTCs, mass balance
├── solve_rz_pinn.py
├── solve_3d_pinn.py
├── solve_3d_fno.py
├── configs/3d/*.yaml
├── out/3d/<run_id>/
├── data/3d/*.zarr
├── tests/
│   ├── test_transport_3d.py
│   ├── test_references_3d.py
│   ├── test_fv3d.py
│   └── test_axisymmetric.py
└── 3D_results.md
```

---

## 11. Immediate next actions

1. Implement the axisymmetric `(r,z,t)` transport/flow model and verify it
   against the Thiem solution and flow MMS.
2. Create `physics/transport3d.py` and the 3D MMS test; confirm the residual
   reduces to the 2D/1D residual in the appropriate limits.
3. Run the extruded 3D PINN verification (G3.1) before any layered `K`.
4. Implement the 3D Green's functions and verify plume moments (G3.2).
5. Build the 3D FV solver and run the `64³` grid-convergence study.
6. Only then start FNO3d; use `64³` and 8–12 modes for the architecture
   search.
7. Record the 3D justification, all gates, and the compute/storage numbers in
   `Actions.md` and `HUMAN.md`.
