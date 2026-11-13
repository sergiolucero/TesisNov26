# Optimization Plan — Stochastic Pump-and-Treat Remediation

This document is the **execution plan** for the stochastic optimization layer.
The mathematical formulation is in:

- `optimization.tex` / `optimization.pdf` — full problem statement, well
  layout, random-field model, chance constraints, SAA/CVaR, adjoint, and
  surrogate-based optimization.

It builds on the already-trusted forward chain:

- `formulation.tex` / `formulation.pdf` — 1D dimensionless ADE and the
  Ogata–Banks reference;
- `2D_plan.md` — 2D flow, transport, wells, heterogeneity, FV references,
  and gates G2.1–G2.5;
- `3D_plan.md` — 2.5D/3D extension and gates G3.1–G3.6;
- `Actions.md` / `HUMAN.md` — current results, environment, and gotchas.

The optimization layer is added **only after** the relevant forward gates are
green. The rule is the same as in the forward plans:

> One new ingredient per stage, one reference per stage, one explicit gate
> before advancing.

---

## 0. Problem in one page

| Element | Baseline choice | Extension |
|---|---|---|
| Random field | Log-normal \(K(\mathbf{x},\omega)=K_0 e^{G}\), \(G\) Gaussian, \(\sigma_G=0.5\), \(\ell=20\) m | Random porosity, source strength/location, boundary head, inlet concentration, dispersivities |
| Decision | Nonnegative steady extraction rates \(q_i\) at fixed wells | Piecewise-constant pumping schedule \(q_{ik}\) over \(N_p\) periods |
| Well layout | \(N_w=9\) or 16 wells on a hexagonal lattice over the design plume | Centroidal Voronoi tessellation for irregular plumes; optional well-placement MINLP |
| Objective | Expected pumping + treatment + residual mass + cleanup-time cost | Mean–CVaR cost; robust/DRO variants |
| Constraints | \(0\le q_i\le q_i^{\max}\), \(\sum_i q_i\le Q_{\max}\), drawdown limit | Total-volume budget, activation costs, time-dependent capacity |
| Reliability | \(\mathbb{P}(C(\mathbf{x}_c,t)\le C_{\mathrm{reg}}\ \forall \mathbf{x}_c,t)\ge 1-\alpha\), \(\alpha=0.05\) | Joint chance constraint over multiple monitoring wells and times; CVaR surrogate |
| Solution | SAA + CVaR/sigmoid chance constraint + adjoint or FNO surrogate | Stochastic gradient, derivative-free global search, active learning |
| Certificate | Out-of-sample cost, reliability, CVaR, cleanup-time distribution | Robustness under distribution shift, value of information |

The full stochastic program is Eq. (5.14) in `optimization.tex`:

\[
\begin{aligned}
\min_{\mathbf{q}\in\mathcal{Q}_{\mathrm{ad}}}\quad
& \mathbb{E}\bigl[J(\mathbf{q},\omega)\bigr]\\
\text{s.t.}\quad
& \mathbb{P}\bigl(G(\mathbf{q},\omega)\le 0\bigr)\ge 1-\alpha,\\
& \mathbb{P}\bigl(H(\mathbf{q},\omega)\ge 0\bigr)\ge 1-\alpha_h,\\
& (\mathbf{h},\mathbf{C})=\mathcal{S}(\mathbf{q},\omega)\quad\text{a.s.}
\end{aligned}
\]

where \(G\) is the compliance violation, \(H\) is the drawdown violation, and
\(\mathcal{S}\) is the flow–transport solution map.

---

## 1. What must be built

### 1.1 Forward modules (reuse, do not fork)

| Module | Role | Status |
|---|---|---|
| `physics/flow.py` | Darcy flow with wells, unit-rate responses \(h^{(i)}\), \(\mathbf{v}^{(i)}\) | reuse/extend 2D/3D flow |
| `physics/transport.py` | ADE residual, dispersion tensor, mass balance | reuse |
| `physics/fv2d.py`, `physics/fv3d.py` | Finite-volume reference, well cells | reuse |
| `physics/references.py` | Thiem, Ogata–Banks, MMS, Green's functions | reuse |

Add only:
- a well-sink hook in the flow solver (regularized \(\delta_\varepsilon\));
- a unit-response mode that returns \((h^{(i)},\mathbf{v}^{(i)})\) for each
  well;
- a well-concentration sampler \(C(\mathbf{x}_i,t)\) for the treatment cost.

### 1.2 New optimization modules

| Module | Contents |
|---|---|
| `stoch/random_fields.py` | KL expansion, log-normal \(K\), random source/BCs, scenario archive |
| `stoch/wells.py` | Hexagonal lattice, CVT, well regularization, coverage diagnostics |
| `stoch/objective.py` | \(J\), \(G\), \(H\), cleanup time \(\tau_c\), cost decomposition, mass balance |
| `stoch/saa.py` | Train/test splits, SAA objective, common random numbers, replications |
| `stoch/chance.py` | Exact MINLP, CVaR, sigmoid smoothing, continuation schedule |
| `stoch/adjoint.py` | Discrete adjoint, continuous adjoint, finite-difference gradient check |
| `stoch/optimize.py` | Pattern search, L-BFGS-B/trust region, projected gradient, multi-start |
| `stoch/surrogate.py` | FNO wrapper, training data generator, autodiff optimization, active learning |
| `stoch/report.py` | Metrics, confidence intervals, figures, `run_config.json` |

---

## 2. Design decisions to freeze before coding

Write these into `configs/opt/benchmark.yaml` and do not change them silently
between runs.  The corresponding template parameter table is Appendix D of
`optimization.tex`.

### 2.1 Domain and aquifer

- 2D depth-averaged baseline: \(200\times100\) m\(^2\), thickness \(b=10\) m.
- Porosity \(\phi_0=0.30\), retardation \(R=1\), decay \(\lambda=0\) for the
  first tracer tests; switch on \(R>1\), \(\lambda>0\) only after the tracer
  gates pass.
- \(K_0=10\) m/d, regional gradient \(10^{-3}\), \(U\approx0.033\) m/d.
- \(\alpha_L=5\) m, \(\alpha_T=0.5\) m (\(r=10\)).

### 2.2 Random field

- Log-normal \(K\), \(\sigma_G=0.5\), correlation length \(\ell=20\) m.
- KL truncation at 95–99% retained variance; start with \(N_{KL}=64\), verify
  convergence with \(N_{KL}=128\).
- Clamp \(K\) to \([K_{\min},K_{\max}]\) and report the clamp fraction.
- Optional random source: total mass \(M_s\), release duration \(T_s\), and
  centroid \(\mathbf{x}_s\); start with deterministic source, then switch on
  one random variable at a time.

### 2.3 Wells

- \(N_w=9\) for the first end-to-end run; \(N_w=16\) and 25 for the full study.
- Hexagonal lattice with spacing \(\ell_w=\sqrt{2A_p/(\sqrt{3}N_w)}\) over the
  design plume; CVT for irregular plumes.
- Design plume \(\Omega_p\): 95% probability contour of the reference plume
  (or the mean plume for the first run). Record which definition was used.
- Well regularization \(\varepsilon=1\)–2 grid cells; never smaller than
  \(r_w\).
- Rates \(q_i\in[0,50]\) m\(^3\)/d, plant capacity \(Q_{\max}=200\) m\(^3\)/d.

### 2.4 Cost and regulation

- \(C_{\mathrm{reg}}=1\) mg/L at 1–3 compliance wells.
- \(T=3650\) d; source release 365 d; compliance evaluated over the last 20%
  of the horizon (or from the cleanup target date).
- Start with dimensionless costs: \(c_e=1\), \(c_w=1\), \(c_m=1\),
  \(c_\tau=1\). Calibrate to site economics only after the optimization
  machinery is verified.
- Pumping-cost baseline: linear \(c_e\sum_i\int q_i\,dt\). Lift-dependent
  variant is a sensitivity, not the first run.

### 2.5 Chance constraint

- \(\alpha=0.05\) (95% reliability), \(\alpha_h=0.01\) for drawdown.
- Baseline reformulation: CVaR constraint, Eq. (6.5) in
  `optimization.tex`.
- Differentiable alternative: sigmoid approximation, Eqs. (6.6)–(6.7), with
  continuation \(\varepsilon_s=10^{-1},10^{-2},10^{-3}\).
- Exact MINLP, Eq. (6.3), only for a small toy problem used to validate the
  approximations.

### 2.6 SAA sizes

| Purpose | \(M\) | Notes |
|---|---:|---|
| Debug / derivative-free exploration | 50–100 | same random numbers across candidates |
| Coarse SAA optimization | 200–500 | train/test split |
| Final optimization | 2000–5000 | independent test archive |
| Probability certificate | 5000–20000 | DKW-based sample size if a tight probability estimate is required |

Always store the KL coefficients and random variables, not only the realized
fields.

---

## 3. Verification hierarchy

### Stage O-0 — Deterministic regression and well layout

**Work**
- Create `configs/opt/benchmark.yaml` and `stoch/wells.py`.
- Implement the hexagonal lattice and CVT; reproduce the well-layout figure of
  `optimization.tex`.
- Implement the regularized well source and the unit-response flow solves.
- Verify:
  - one-well steady flow against the Thiem solution away from the well;
  - the velocity field against the analytic gradient;
  - transport mass balance Eq. (2.10) for a conservative tracer;
  - the no-pumping limit against the existing 2D forward solution.

**Deliverables:** `stoch/wells.py`, `tests/test_wells.py`,
`out/opt/O0/`, figure of the well layout.

**Gate O0:** existing G2.1–G2.3/G3.1–G3.3 green; Thiem/velocity error
\(\le10^{-2}\) away from the well; mass-balance error \(\le1\%\).

---

### Stage O-1 — Random-field generator

**Work**
- Implement the KL expansion for the chosen covariance (exponential or
  Matérn), anisotropic correlation lengths, and log-normal \(K\).
- Implement optional random source, boundary head, inlet concentration, and
  dispersivities.
- Generate a scenario archive with metadata: seed, \(N_{KL}\), covariance
  parameters, clamp fraction.
- Statistical checks:
  - mean/variance of \(\log K\);
  - covariance at several separations;
  - connectivity/percolation statistics;
  - boundedness of \(K\) and \(\phi\);
  - reproducibility for the same seed.

**Deliverables:** `stoch/random_fields.py`,
`tests/test_random_fields.py`, `out/opt/O1/scenarios/`.

**Gate O1:** marginal and covariance checks within Monte Carlo confidence
bands; clamp fraction \(<1\%\); same seed reproduces the same archive.

---

### Stage O-2 — Objective, cleanup time, and SAA

**Work**
- Implement \(J_{\mathrm{pump}}\), \(J_{\mathrm{treat}}\), \(J_{\mathrm{res}}\),
  \(J_{\mathrm{time}}\), \(G\), \(H\), and \(\tau_c\).
- Implement the train/test split and common random numbers.
- Verify the cost decomposition on a fixed policy and a no-pumping policy.
- Check that \(J_M(\mathbf{q})\) and \(\hat p_M(\mathbf{q})\) converge as
  \(M\) increases.
- Implement the scenario replication loop for SAA gap estimates.

**Deliverables:** `stoch/objective.py`, `stoch/saa.py`,
`tests/test_objective.py`, `out/opt/O2/`.

**Gate O2:** relative standard error of \(\hat J\) \(\le1\%\); compliance
probability error \(\le1\%\) at fixed \(\mathbf{q}\); mass balance \(\le1\%\).

---

### Stage O-3 — Adjoint gradient and local optimization

**Work**
- Implement the discrete adjoint Eqs. (7.3)–(7.4) in `optimization.tex`.
- Implement the continuous adjoint as a cross-check (Appendix A).
- Gradient check against central finite differences for one well and two
  wells, several step sizes, several random directions.
- Run L-BFGS-B on a small SAA with \(N_w=1,2\) and compare to a brute-force
  grid search.
- Verify that the optimizer respects the bounds and the plant capacity.

**Deliverables:** `stoch/adjoint.py`, `stoch/optimize.py`,
`tests/test_adjoint.py`, `out/opt/O3/`.

**Gate O1:** relative gradient error \(\le10^{-4}\) in double precision;
one/two-well optimum within the brute-force tolerance; KKT residuals small.

---

### Stage O-4 — Chance-constraint reformulations

**Work**
- Implement the exact MINLP for a small toy problem (\(M\le30\), \(N_w=1\)).
- Implement the CVaR constraint and the sigmoid approximation.
- Compare the three formulations on the toy problem:
  - feasible set and optimal value;
  - conservatism of CVaR;
  - sensitivity to \(\varepsilon_s\);
  - runtime.
- Implement the continuation schedule for \(\varepsilon_s\).

**Deliverables:** `stoch/chance.py`, `tests/test_chance.py`,
`out/opt/O4/`.

**Gate O4:** at small \(M\), the CVaR/sigmoid solutions are feasible for the
exact empirical chance constraint; the CVaR objective is not lower than the
exact MINLP objective; the sigmoid solution converges as
\(\varepsilon_s\downarrow0\).

---

### Stage O-5 — FNO surrogate and surrogate optimization

**Work**
- Generate FNO training data: scenarios \(\times\) rate vectors, including
  \(\mathbf{q}=0\), the capacity boundary, and asymmetric patterns.
- Train the FNO with the field loss, well-concentration loss, and mass-balance
  penalty.
- Test:
  - held-out scenarios;
  - held-out rate vectors;
  - rollout error over the remediation horizon;
  - resolution transfer if applicable;
  - decision-relevant errors (cost, cleanup time, extracted mass,
    compliance probability).
- Implement the surrogate SAA optimization with autodiff and a trust region.
- Implement the active-learning loop: validate the candidate with FV, add the
  worst scenarios and the candidate neighborhood, retrain.

**Deliverables:** `stoch/surrogate.py`, `out/opt/O5/fno/`,
`out/opt/O5/surrogate_opt/`.

**Gate O3:** field relative \(L^2\) \(\le5\%\); cost error \(\le2\%\);
cleanup-time error \(\le5\%\); compliance-probability error \(\le1\%\);
surrogate speedup \(\ge100\times\).

---

### Stage O-6 — Full problem, certificate, and sensitivity

**Work**
- Run the full problem with \(N_w=9\) and 16, \(M=2000\)–5000, and the chosen
  chance-constraint reformulation.
- Run \(K\) independent SAA replications and evaluate all candidates on the
  independent test archive.
- Report:
  - expected cost and decomposition;
  - reliability and violation probability;
  - CVaR of cost and of violation;
  - cleanup-time distribution;
  - residual mass;
  - pumping distribution and active wells;
  - optimality gap;
  - sensitivity to \(K\), source, and cost coefficients;
  - value of information (perfect vs imperfect \(K\)).
- Stress test under distribution shift: \(\ell\), \(\sigma_G\), source
  strength, and boundary gradient.
- Write `optimization_results.md` and update `Actions.md` / `HUMAN.md`.

**Deliverables:** `out/opt/O6/`, `optimization_results.md`, figures, policy
archive.

**Gates O4–O7:**
- out-of-sample objective gap \(\le5\%\);
- out-of-sample reliability \(\ge1-\alpha-1\%\);
- robustness: reliability \(\ge1-\alpha-5\%\), cost degradation \(\le20\%\)
  under moderate distribution shift;
- surrogate speedup \(\ge100\times\);
- same config/seed/archive gives identical metrics.

---

## 4. Numerical methods: what to use when

### 4.1 Recommended path

1. **Explore** with pattern search or CMA-ES on a coarse SAA
   (\(M=50\)–100), using common random numbers. This finds a good basin and
   reveals whether the chance constraint is active.
2. **Refine** with the discrete adjoint and L-BFGS-B/trust region on a larger
   SAA (\(M=200\)–1000), using the CVaR or sigmoid constraint.
3. **Accelerate** with the FNO surrogate for large \(M\) and many optimizer
   iterations. Validate the candidate with the FV model and active learning.
4. **Certificate** on an independent test archive, with \(K\) SAA
   replications and a confidence interval.

Do not skip step 1: the chance-constrained problem is nonconvex, and a local
gradient method started from a naive uniform pumping policy often converges to
a poor local optimum.

### 4.2 When to use which method

| Situation | Method |
|---|---|
| \(N_w\le5\), nonsmooth chance constraint | Pattern search / CMA-ES on coarse SAA |
| \(N_w=10\)–50, smooth/CVaR constraint, FV available | Discrete adjoint + L-BFGS-B/trust region |
| Many scenarios, many optimizer iterations | FNO surrogate + autodiff |
| \(N_w\le10\), each FV run very expensive | Bayesian optimization |
| Rare-event compliance (\(\alpha\ll0.05\)) | Importance sampling + CVaR, or more scenarios |
| Time-dependent rates | Piecewise-constant parameterization + adjoint in time |

### 4.3 Surrogate guardrails

- **Trust region:** limit \(\|\mathbf{q}-\mathbf{q}_{\mathrm{train}}\|\) to the
  region covered by the training data.
- **Decision-relevant validation:** a small field error does not guarantee a
  small cost or reliability error. Validate cost, cleanup time, extracted
  mass, and compliance probability separately.
- **Active learning:** add the worst validation scenarios and the candidate
  neighborhood to the training set.
- **Control variate:** use the FNO prediction as a control variate for the FV
  estimate; the difference has much lower variance.
- **Versioning:** store the FNO weights, training config, and data archive
  hash with every optimization run.

---

## 5. Data pipeline

### 5.1 Scenario archive

```
out/opt/<run_id>/scenarios/
  train.npz      # KL coefficients, random variables, metadata
  test.npz       # independent seed
  manifest.json  # covariance, N_KL, clamp fraction, generation code SHA
```

Store the **coefficients**, not only the fields. A field archive cannot be
reused to generate a nearby scenario or to check the KL truncation.

### 5.2 FNO dataset

For each training scenario:
- draw \(N_q\) rate vectors from a space-filling design over
  \(\mathcal{Q}_{\mathrm{ad}}\);
- solve the FV model;
- store \(C(\mathbf{x},t)\), \(C(\mathbf{x}_i,t)\), mass-balance diagnostics,
  and the metadata.

Recommended starting point: 500–1000 scenarios \(\times\) 10–30 rate vectors
at \(128^2\). Increase the scenario count before increasing the rate-vector
count if the heterogeneity is the dominant source of error.

### 5.3 Storage

- `.npz` for small smoke tests;
- Zarr/HDF5 for production archives;
- version by content, e.g. `fv2d_128_n1000_q20_v1.zarr`;
- compress and subsample time snapshots; keep the well and compliance time
  series at full resolution.

---

## 6. Metrics and acceptance gates

### 6.1 Metrics to report

1. Expected cost \(\hat J\), standard error, and decomposition into pumping,
   treatment, residual, and time terms.
2. Reliability \(\hat p_{\mathrm{compliance}}\) and violation probability.
3. \(\widehat{\mathrm{CVaR}}_{1-\alpha}(J)\) and
   \(\widehat{\mathrm{CVaR}}_{1-\alpha}(G)\).
4. Cleanup time: mean, median, 90th percentile, failure fraction.
5. Residual mass at \(T\).
6. Pumping: total volume, active wells, per-well rates, spatial distribution.
7. Optimality gap from SAA replications or a relaxed lower bound.
8. Decision-relevant surrogate errors: cost, cleanup time, extracted mass,
   compliance probability.
9. Sensitivity: \(\partial J/\partial q_i\), shadow price of the chance
   constraint, value of information.
10. Well-layout diagnostics: covering radius, capture-zone coverage, distance
    to plume centroid.

### 6.2 Gate summary

| Gate | Check | Threshold |
|---|---|---|
| O0 | Deterministic forward regression (Thiem/MMS/FV) | G2.x/G3.x green; velocity error \(\le10^{-2}\); mass balance \(\le1\%\) |
| O1 | Adjoint gradient vs finite difference | relative error \(\le10^{-4}\) |
| O2 | SAA convergence at fixed \(\mathbf{q}\) | relative SE \(\le1\%\); probability error \(\le1\%\) |
| O3 | Surrogate vs FV on held-out scenarios/rates | field rel-\(L^2\) \(\le5\%\); cost \(\le2\%\); cleanup time \(\le5\%\) |
| O4 | Out-of-sample optimality | objective gap \(\le5\%\); reliability \(\ge1-\alpha-1\%\) |
| O5 | Robustness under distribution shift | reliability \(\ge1-\alpha-5\%\); cost degradation \(\le20\%\) |
| O6 | Surrogate speedup at comparable decision accuracy | \(\ge100\times\) |
| O7 | Reproducibility | identical metrics for same config/seed/archive |

A run is **solved** only when O0–O4 and O7 are green. O5–O6 are required for a
production or reporting claim.

---

## 7. Proposed file layout

```
TESIS/
├── physics/
│   ├── flow.py              # Darcy flow, wells, unit responses
│   ├── transport.py         # ADE, dispersion, mass balance
│   ├── fv2d.py, fv3d.py     # finite-volume references
│   └── references.py        # Thiem, Ogata–Banks, MMS, Green's functions
├── stoch/
│   ├── random_fields.py     # KL, log-normal K, random source/BCs
│   ├── wells.py             # hexagonal lattice, CVT, regularization
│   ├── objective.py         # J, G, H, cleanup time, cost decomposition
│   ├── saa.py               # train/test splits, SAA, replications
│   ├── chance.py            # MINLP, CVaR, sigmoid, continuation
│   ├── adjoint.py           # discrete/continuous adjoint, gradient check
│   ├── optimize.py          # pattern search, L-BFGS-B, trust region
│   ├── surrogate.py         # FNO wrapper, autodiff, active learning
│   └── report.py            # metrics, figures, manifest
├── configs/opt/
│   ├── benchmark.yaml
│   ├── explore.yaml
│   └── certificate.yaml
├── out/opt/<run_id>/
│   ├── run_config.json
│   ├── scenarios/{train,test}.npz
│   ├── surrogate.pt
│   ├── policy.json
│   ├── metrics.json
│   └── figures/
├── tests/
│   ├── test_random_fields.py
│   ├── test_wells.py
│   ├── test_objective.py
│   ├── test_adjoint.py
│   ├── test_chance.py
│   └── test_saa.py
├── optimization.tex / .pdf
├── optimization_plan.md
└── optimization_results.md   # written at stage O-6
```

Keep `hydrology.py`, `solve_1d_pinn.py`, and the 2D/3D forward scripts as the
trusted regression baseline. The optimization code imports the forward modules
rather than forking them.

---

## 8. Compute budget (order of magnitude)

| Task | Size | Hardware | Wall-clock |
|---|---|---|---|
| KL generation | \(M=10^3\), \(N_{KL}\sim10^2\) | CPU | minutes |
| One FV flow+transport scenario | \(128^2\), 100 steps | 1 core | 1–5 min |
| Flow unit responses | \(N_w+1\) solves per scenario | 1 core/scenario | \(N_w+1\) flow solves |
| \(M=10^3\) high-fidelity scenarios | \(128^2\) | 32 cores | 1–3 h |
| FNO training | \(10^3\) scenarios, \(128^2\) | 16–24 GB GPU | 12–48 h |
| Adjoint gradient | one scenario, \(128^2\) | 1 core | 2–5\(\times\) forward |
| SAA optimization (adjoint) | \(M=10^2\)–\(10^3\), \(10^2\) iterations | CPU/GPU | hours |
| Surrogate optimization | \(M=10^3\)–\(10^4\), \(10^3\) iterations | GPU | minutes–hours |
| Final certificate | \(M_{\mathrm{test}}=10^3\)–\(10^4\) | 32 cores | hours |

The FNO surrogate is the enabling technology for large \(M\) and many
optimizer iterations. Start with the CPU-sized \(128^2\) problem; move to
\(256^2\) or 3D only after the 2D gates pass.

---

## 9. Risks and mitigations

| Risk | Why it happens | Mitigation |
|---|---|---|
| Chance constraint is nonconvex/nonsmooth | Empirical indicator; joint space–time constraint | CVaR/sigmoid surrogate; multi-start; derivative-free exploration |
| SAA overfitting | Optimizing on a finite scenario archive | Independent test set; \(K\) SAA replications; increase \(M\); cross-validation |
| Surrogate bias | FNO trained away from the optimal \(\mathbf{q}\) | Trust region; active learning; FV validation; control variate |
| Curse of dimensionality in \(K\) | \(N_{KL}\) too large | Truncated KL at 95–99% variance; sensitivity analysis; dominant modes |
| Well singularity | Point source \(\ln r\)/\(1/r\) | Regularized wells; exclude well cells from error metrics; local refinement |
| Local optima | Nonlinear transport and chance constraints | Multi-start; continuation in \(\alpha\) and \(Q_{\max}\); global exploration |
| Rare-event compliance | \(\alpha\) small; plain MC inefficient | Importance sampling; CVaR; more scenarios near the tail |
| Cost-coefficient uncertainty | Economic parameters are estimates | Sensitivity analysis; Pareto frontier; robust cost scenarios |
| Plume-footprint uncertainty | Fixed wells may miss the realized plume | Design plume from a probability contour; robust layout; allow \(q_i=0\) |
| Time-dependent rates | \(N_wN_p\) grows quickly | Piecewise-constant parameterization; coarse-to-fine continuation; adjoint in time |
| Optimizer exploits a surrogate artifact | Surrogate is cheaper than FV | Trust region; out-of-sample FV certificate; never report a surrogate-only policy |

---

## 10. Immediate next actions

1. **Create the benchmark config and well layout.**
   `configs/opt/benchmark.yaml`, `stoch/wells.py`; reproduce the well figure
   and check the covering radius.
2. **Implement the random-field generator.**
   `stoch/random_fields.py`; pass the marginal/covariance/connectivity checks.
3. **Implement the objective and SAA.**
   `stoch/objective.py`, `stoch/saa.py`; verify cost decomposition and SAA
   convergence at fixed \(\mathbf{q}\).
4. **Implement and check the adjoint.**
   `stoch/adjoint.py`; finite-difference gradient check at one and two wells.
5. **Implement the chance-constraint reformulations.**
   `stoch/chance.py`; compare exact MINLP, CVaR, and sigmoid on a toy problem.
6. **Generate the FNO dataset and train the surrogate.**
   Include \(\mathbf{q}=0\), the capacity boundary, and asymmetric patterns.
7. **Run the surrogate optimization with active learning.**
   Validate the candidate with FV; add the worst scenarios; retrain.
8. **Run the full problem and write the certificate.**
   \(N_w=9\) then 16, \(M=2000\)–5000, independent test archive,
   `optimization_results.md`.
9. **Update `Actions.md` and `HUMAN.md`** with the new commands, gates, and
   gotchas.

### Definition of done for the first end-to-end solve

- O0–O4 and O7 green on the \(N_w=9\), 2D, log-normal-\(K\) baseline.
- A policy \(\mathbf{q}^\star\) with:
  - out-of-sample reliability \(\ge95\%\);
  - out-of-sample cost within \(5\%\) of the training SAA estimate;
  - expected cleanup time and residual mass reported with confidence
    intervals;
  - a reproducible manifest and scenario archive.
- A short results note (`optimization_results.md`) explaining what the policy
  does, where it pumps, why it is reliable, and what remains uncertain.

---

## 11. What not to do

- **Do not optimize on a single \(K\) field.** A deterministic optimum is not a
  stochastic policy.
- **Do not trust a surrogate-only policy.** Validate the recommended
  \(\mathbf{q}\) with the FV model on scenarios not used in training.
- **Do not report only the field relative \(L^2\) error.** Report cost,
  reliability, cleanup time, and extracted mass.
- **Do not start with \(N_w=50\) and 3D.** Start with \(N_w=9\), 2D, steady
  rates, and a small SAA; add one ingredient at a time.
- **Do not hide the chance-constraint approximation.** State whether the run
  used the exact empirical constraint, CVaR, or the sigmoid approximation,
  and report the resulting conservatism.
- **Do not forget mass balance.** A cheap policy that creates or destroys
  contaminant mass is a bug, not an optimum.
