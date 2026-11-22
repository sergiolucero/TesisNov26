# Results

Every number below comes from a run whose artifacts are in this repository, and
each certificate has been re-derived independently by `SCRIPTS/check_tail.py`
(NumPy reference, including scenarios no run trained on) rather than taken from
the run's own summary.
Section 5 lists what is *not* established.

---

## 1. Phase 1 — 1D advection–dispersion surrogate (PINN / FNO)

Reference: Ogata–Banks. Geometry: `Pe = vL/D = 20`, `τ_max = 0.5`, MLP 4×64 tanh,
Adam with resampled collocation + L-BFGS polish on a fixed grid.

| Surrogate | BC treatment | full-grid rel-L2 | interior rel-L2 | notes |
| :--- | :--- | ---: | ---: | :--- |
| PINN | soft penalty | 2.43e-2 | — | baseline |
| PINN | hard (`eps = 1e-3`) | **1.61e-2** | **1.35e-2** | residual error is the measure-zero inlet corner |
| FNO | — | see `DOCS/Conclusions.md` §5 | — | normalized coordinates were required |

The hard-constraint answer `u = (1-ξ)s(τ) + ξ(1-ξ)s(τ)·NN` encodes the inlet
discontinuity exactly, which is why the corner error disappears from the interior
metric. Full narrative and the FNO negative-then-fixed story: `DOCS/Conclusions.md`.

*Metric caveat, stated in the paper as well:* `solve_1d_pinn.region_errors` reports
a family of regions on the 256×256 space–time grid. The checkpoint archived in
`out/pinn/model.pt` evaluates to `full = 1.61e-2`, `tau,xi>0.005 = 1.35e-2`,
`tau,xi>0.05 = 1.42e-2`, `left_layer = 7.8e-3`. The often-quoted
"1.35e-2 full / 8.0e-3 interior" pair is therefore two *different* region
definitions, not the strict full grid — and 1.35e-2 is also the best of five
seeds, whose full-grid spread is 1.17e-2 … 8.14e-2 (factor 7, median 1.61e-2).
Quote the median and the interval, never the best seed alone.
`FIGURES/fig1_1d_verification.png` annotates exactly the two entries it plots so
the figure cannot drift from the table.

---

## 2. Phase 2 — 2D forward model and stochastic optimization

Benchmark (`configs/opt/hydrology_gpu.yaml`): 200×120 m at 160×96 cells
(15,360 unknowns), log-normal `K` with `σ = 0.65`, `ℓ = 15 m`, KL expansion at
95% variance (`n_kl = 1529`), 9 candidate wells on a hexagonal lattice, source
10 m³/d at 100 mg/L for 3650 d, compliance 1.0 mg/L at `x = 185 m` over the last
20% of the horizon. Chance constraint: `P(G ≤ 0) ≥ 1 − α`, `α = 0.05`, handled
as a CVaR-5% penalty; `M_train = 100` scenarios, `M_test = 40`, `M_holdout = 40`.

### 2.1 Certified policies

Two runs used the same search (pattern search over well rates with capacity
projection, batched compass sweep on a GPU) and differ only in floating-point
precision. Both are independent, fully certified optima:

| | float64 (`20261118-114831`) | float32 mixed (`20261120-111849`) |
| :--- | ---: | ---: |
| q\* [m³/d] | `[0, 9.72, 5.86, 0, 3.38, 0, 0, 0, 0]` | `[0, 9.33, 8.16, 0, 0, 0, 0, 0, 0]` |
| total pumping | 19.0 | 17.5 |
| active wells | 3 | 2 |
| J (train / test) | 433,945 / 434,008 | 428,592 / 428,579 |
| CVaR-5%(G) train / test | +0.018 / −0.007 | +0.005 / +0.009 |
| reliability train / test | 0.990 / 0.975 | 0.990 / 0.975 |
| holdout during search | 1.000 | 0.975 |
| max mass-balance error | 4.2e-11 | 2.3e-05 |
| wall clock (T4) | 4.14 h | **1.67 h** |
| gates | all pass | all pass |

Two facts matter more than the numbers. First, both optima sit **on** the
constraint (`CVaR ≈ 0` from either side), which is what the solution of a
chance-constrained program must do — these are the cheapest policies that are
still safe, not policies that happen to be safe. Second, the two policies differ
by only **1.2% in cost while using different well subsets**: near the optimum the
objective is flat along the direction "which two or three of the nine wells carry
the duty". Multi-start at this geometry therefore samples *near-equivalent
policies*, not cost.

**Accepted answer.** The float32 policy (2 wells, J = 428,592 train / 428,579
test) is the reported result for this geometry. The reasons not to keep searching
are the ones above: the two independent backends land 1.2% apart on *different*
well subsets, a fresh draw reproduces J to 0.02% (§2.2), and the compass search
had already passed the point where the step size, not the budget, decides the
outcome.

The obvious next experiment — replicas with offset scenario seeds — has since
been measured and does not work, for a reason worth stating plainly: **J barely
depends on the conductivity realization.**  The accepted policy scores 428,664.8
on scenarios 7000-7019 and 428,647.9 on 37000-37019, disjoint draws that differ by
0.004%; at M=400 the same comparison moves 0.02%.  Replicas seeded differently
therefore re-derive the same policy, and two such runs launched against this
config reproduced the accepted run's first three compass sweeps digit for digit
before being stopped.  So the flat direction is *not* sampling noise that more
seeds would average out — it is the cost model (pumping cost against cleanup
time), which admits several near-equal policies, and the 1.2% gap between the two
backends is decided by float32 noise in the fine compass steps, not by which
aquifers were drawn.  Resolving *that* would need a deliberate tie-break rule or a
finer step in float64, and it would not change the answer by more than the 1.2%
already measured.

### 2.2 Out-of-sample certificate with Wilson intervals

A run reports the tail of *its own* sample. The question this section answers is
what survives when the sample is replaced, and it is answered with
`SCRIPTS/check_tail.py`, which rebuilds scenarios through the run's own
constructors and re-solves the reported `q*` on the NumPy reference backend.

**The run's own continuation first.** Re-evaluating each policy on 400 scenarios
continuing from its train seed:

| policy | violations | reliability | 95% Wilson CI | gate 0.94 |
| :--- | ---: | ---: | :--- | :--- |
| float64 `q*` (3 wells) | 1/400 | 0.998 | **[0.986, 1.000]** | PASS |
| float32 `q*` (2 wells) | 7/400 | 0.983 | **[0.964, 0.991]** | PASS |

The first 100 of those 400 *are* the training fields, so this is closer to a
backend-agreement check than to new evidence. The interval is still the point: a
bare `0.975` at M = 40 is consistent with a true reliability anywhere in
~[0.87, 1.00]; `399/400` is not. Sample-size ladders are non-monotone (the same
violating scenario enters the tail at different rates), which is why a single M is
never quoted without the interval:

| M | 40 | 60 | 100 | 200 | 300 | 400 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| violations, float64 policy | 0 | 0 | 1 | 1 | 1 | 1 |
| reliability | 1.000 | 1.000 | 0.990 | 0.995 | 0.997 | 0.998 |
| Wilson low | 0.912 | 0.940 | 0.946 | 0.972 | 0.981 | 0.986 |
| violations, float32 policy | 0 | 0 | 1 | 3 | 6 | 7 |
| reliability | 1.000 | 1.000 | 0.990 | 0.985 | 0.980 | 0.983 |
| Wilson low | 0.912 | 0.940 | 0.946 | 0.957 | 0.957 | 0.964 |

**A fully independent draw.** With `--seed-base 900000` (nothing the optimizer
ever saw, float32 policy):

| M | violations | reliability | 95% Wilson CI | gate 0.94 |
| :--- | ---: | ---: | :--- | :--- |
| 100 | 4/100 | 0.960 | [0.902, 0.984] | **inconclusive** |
| 200 | 5/200 | 0.975 | [0.943, 0.989] | PASS |
| 400 | 9/400 | 0.978 | [0.958, 0.988] | PASS |

J = 428,509 (0.02 % from the reported 428,592) and CVaR-5 %(G) = +0.616 rather
than +0.005: the certificate holds on unseen scenarios, and the tail is worse than
the run claimed. The 100-scenario row is the useful one -- it is the sample size
the search itself used, and it is *not* enough to certify anything.

#### Five policies on one common fresh draw

Each policy's `q*` re-solved on the same 400 unseen scenarios
(`runs/tail_calibration/*.json`, seed base 900000). This is the calibration
experiment: policies deliberately spanning the safety range, from a policy
selected on only 16 scenarios to one that pumps all nine wells.

| policy | in-run train rel | J (fresh) | mean G | CVaR-5 %(G) ± s.e. | reliability [95 % CI] | gate |
| :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `20261118-041443` (M=16, 4 wells) | 1.000 | 423,041 | −0.697 | +1.865 ± 0.41 | 365/400 = 0.912 [0.881, 0.936] | **FAIL** |
| `20261121-215921` (M=40 screening, 2 wells) | 0.975 | 409,645 | −0.669 | +2.345 ± 0.65 | 366/400 = 0.915 [0.884, 0.939] | **FAIL** |
| `20261120-111849` (accepted, 2 wells) | 0.990 | 428,509 | −0.872 | +0.616 ± 0.47 | 391/400 = 0.978 [0.958, 0.988] | PASS |
| `20261118-114831` (float64, 3 wells) | 0.990 | 433,924 | −0.903 | +0.251 ± 0.32 | 392/400 = 0.980 [0.961, 0.990] | PASS |
| `20261118-052228` (6 wells) | 1.000 | 577,917 | −1.000 | −1.000 ± 0.00 | 400/400 = 1.000 [0.990, 1.000] | PASS |
| `20261118-091111` (9 wells) | 1.000 | 821,250 | −1.000 | −1.000 ± 0.00 | 400/400 = 1.000 [0.990, 1.000] | PASS |

Three readings:

1. **The audit discriminates.** The M = 16 policy reported reliability 1.000 on
   its own sixteen scenarios and fails on 400 unseen ones (0.912, interval
   entirely below the 0.94 gate). An out-of-sample certificate is not a
   formality that everything passes.
2. **Safety costs money, and the first 2 % costs the most.** Going from the
   accepted policy to the float64 one buys a smaller tail estimate for +1.3 % of
   J; going to "no failures at all" costs +35 % (6 wells) or +92 % (9 wells).
   The two safe policies also saturate the compliance metric at its floor
   (`mean G = −1.000`, standard deviation 0): once the plume never reaches
   `x = 185 m`, `G` cannot reward them further, so their `CVaR = −1.000` is a
   cap, not a measurement of safety margin.
3. **A cheaper policy is not a better one.** The M = 40 screening run is the
   cheapest policy in the table (409,645, 4.4 % below the accepted policy) and it
   fails on 34 of 400 unseen scenarios: 0.915 with an interval that is entirely
   below the gate. Cheaper-by-search and safer are opposite directions here, and
   only the fresh draw says which one you bought.
4. **Reliability separates policies; the tail does not.** 0.980 against 0.978 is
   not a difference at M = 400 (intervals overlap almost completely), and neither
   is 0.616 against 0.251 -- see below.

#### The tail does not transport; reliability does

The accepted policy re-solved on three disjoint 400-scenario draws:

| seed base | J | mean G | CVaR-5 %(G) | tail draws | s.e. | reliability [95 % CI] |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 900000 | 428,509 | −0.872 | **+0.616** | 20 | ±0.47 | 0.978 [0.958, 0.988] |
| 950000 | 428,605 | −0.901 | **+0.254** | 20 | ±0.21 | 0.975 [0.955, 0.986] |
| 990000 | 428,579 | −0.890 | **+0.334** | 20 | ±0.24 | 0.978 [0.958, 0.988] |

J reproduces to 0.02 % and reliability to 0.003. The CVaR number moves by a
factor of 2.4 -- and the three values are **statistically indistinguishable**:
their spread (0.36) is smaller than the standard error of any one of them. That
is arithmetic, not bad luck. `CVaR-5 %` averages the worst 5 % of draws, which at
`M = 400` is 20 scenarios; those 20 are the most variable ones in the set, and
`sd(tail)/sqrt(20)` is 0.2-0.5. A tail mean estimated from 20 extreme draws has
about a third of a unit of uncertainty in it, so it cannot resolve differences
smaller than that.

Hence the reporting rule this thesis adopts:

- **Certify with reliability and its Wilson interval.** It is a binomial
  proportion estimated from all 400 draws: stable, transportable, and comparable
  across policies and backends.
- **Quote CVaR with its standard error**, and only as a statement about where the
  policy sits relative to the constraint (`CVaR ≈ 0` on the search sample), never
  as a validated number. The run's `+0.005` and the fresh draw's `+0.62` are the
  same policy; they differ by which twenty scenarios were drawn.
- **Do not rank policies by tail value** at `M = 400`. Rank by reliability, and
  say the tail is unresolved.

`check_tail.py` prints the tail block (`n_tail`, tail mean/s.d./s.e., range) and
writes it to `--json`, so this uncertainty travels with the number instead of
being reconstructed later from memory.

#### Restarts do not rescue an under-sampled search

Two more searches were run from random starts to ask whether the certified optima
are one basin or several, and whether a smaller training set finds them faster.
Both replicas used `configs/opt/hydrology_gpu.yaml` at `160x96` with
`M_train = M_test = M_holdout = 40`, `n_starts = 2`, `max_evals = 400`,
float32 with the float64 Krylov recurrence, and differed only in `optim.seed`
(7 and 8) — so both optimise the *same* 40-scenario objective from different
starting points (`runs/colab_multi/20261121-215921`).

| | replica r0 (seed 7) | replica r1 (seed 8) |
| :--- | ---: | ---: |
| start 1 (deterministic `0.5 q_max`) | 409,863 at nit 28 | 409,863 at nit 28 |
| start 2 (random) | 421,639 | 415,576 |
| selected `q*` | `[2.59, 0, 9.72, 0…0]` | `[2.59, 0, 9.72, 0…0]` |
| pumping / active wells | 12.32 m³/d / 2 | 12.32 m³/d / 2 |
| train reliability (M=40) | 0.975 | 0.975 |
| test reliability (M=40) | 0.900 | 0.900 |
| holdout reliability | 0.900 | 0.925 |
| gates | reliability **False**, holdout **False** | reliability **False**, holdout **False** |
| wall clock / evaluations | 1.69 h / 802 | 1.67 h / 802 |

Three things follow, and none of them is the hoped-for one.

1. **The restarts found nothing new.** Both random starts, from different seeds,
   converged *worse* than the deterministic start, so both replicas returned the
   same `q*`. This landscape has a dominant attractor at M = 40, not a family of
   near-equivalent optima; the earlier float64/float32 pair (two well-subsets at
   nearly equal cost) was a pair of *accepted* policies, not evidence of many
   basins.
2. **The attractor is the fragile one.** Halving the training set lowers the J
   the search can see by 4.4 % and moves the certificate from 0.978 to 0.915. The
   search is not stuck; it is optimising the wrong objective, and more restarts
   optimise it more consistently.
3. **The mid-run guard caught it while it was still running.** The holdout check
   fired at iteration 16 on both replicas (`rel = 0.900`, `CVaR = +1.63`) and
   again at iteration 24 (`rel = 0.900` / `0.925`). The check costs ~7 s
   (1.2 % of the sweep) and turns a post-mortem into an abort decision.

The ladder for the screening policy grows *worse* with sample size — 0/16, 1/20,
2/40, 3/60, 5/100, 6/150, 13/200, 28/300, 34/400 — which is the opposite direction
from the two accepted policies, whose violation count stays flat while the tail
mean drifts. Under-sampling can hide violations or exaggerate them; the sign is
not predictable, so the sample size cannot be chosen by watching CVaR.

The failure was statistical, not numerical: 802 evaluations of float32 with the
float64 Krylov recurrence produced not one non-finite objective and no stalled
solve, so the dtype did its job and the sample size did not.

The practical rule this settles: **a screening run may use a smaller sample
(M = 40 here), but its output is a candidate, not a policy.** The candidate is accepted
only after `check_tail.py` on a fresh 400-scenario draw, and the run's own
`final_test_reliability` gate is best read as a smoke detector for exactly this
failure.

### 2.3 Effect of the policy on the plume

At `q*`, evaluated on 80 held-out scenarios (median with IQR): peak concentration
35.81 → 0.433 mg/L, concentration at the compliance point 17.23 → 0.0260 mg/L,
and scenarios meeting the 1.0 mg/L limit go from 1/80 to 78/80. The plume is
displaced rather than erased (its peak moves to `x ≈ 115 m`, just down-gradient
of the active well), which is the physically expected signature of pump-and-treat.

Baselines on the same scenario set: no pumping J = 322,341 (reliability 0.000,
mean cleanup 3650 d); uniform half-capacity J = 1,186,250 (reliability 1.000,
pumping cost 821,250).

---

### 2.4 Three dimensions: the first certified policy at 8,000 cells

The 3D runner has two geometries worth quoting. The **paper preset**
(`32x16x8` = 4,096 cells, `n_KL` 1419, 64/128 scenarios, 9 wells) gives
`q* = [12.5, 12.5, 12.5, 12.5, 0, 0, 0, 0, 0]` m3/d, train J 547,605.94 / test
547,665.31, test reliability 0.969 and mass balance 7e-15 — reproduced exactly on
a third machine with a different BLAS. But the paper preset resolves the 20 m
correlation length with only 2.4 cells, so a larger geometry
**`size_mid`** (`40x20x10` = 8,000 cells, dx=dy=5 m, dz=2 m, `n_KL` 2062) was
added: four cells per correlation length and three screened layers, so the
partially penetrating wells produce real vertical flow.

On that geometry two searches ran with the same wells, same seeds, same cost
model, differing only in how many scenarios the search could afford. Both
policies were then audited on **400 fresh fields** (seed base 900,000) that no
search saw:

| search | M_train | train J | in-run test rel. | 400-field reliability [95% Wilson] | verdict |
|---|---|---|---|---|---|
| NumPy, 90-min budget | 24 | 519,263 | 0.875 | 0.9075 [0.875, 0.932] | **FAIL** |
| torch on a T4 | 100 | 501,060 | 0.938 | **0.9675 [0.945, 0.981]** | **PASS** |
| | | | | fresh J: 519,433 vs 501,050 | |

- **The M_train=24 policy is rejected**, not merely unproven: its interval's
  upper bound (0.932) is below the 0.94 gate. `alpha*M_train = 1.2`, so the tail
  was priced from about one scenario, the search pushed CVaR(G) onto the
  constraint (+0.036) and bought 37 violations in 400 fields.
- **The M_train=100 policy is the first certified 3D policy** at this geometry:
  13/400 violations, lower bound 0.945 > 0.94. It is also 3.5% cheaper on the
  audit draw (501,050 vs 519,433).
- **The in-run test row is not a certificate either way.** The passing policy's
  own test row (0.9375 at M=48) fails the gate its audit passes; at M=48 the
  interval is [0.83, 0.98]. A 128-field draw still cannot separate the policies
  (0.961 [0.912, 0.983] vs 0.930 [0.872, 0.963], both INCONCLUSIVE) although the
  ordering is right. 400 fields decide it.
- **The two optima are different designs**: 42.19 m3/d over five wells versus
  37.27 m3/d from a single well, and the single well gives the tighter plume
  (final peak 3.7 vs 6.0 mg/L; 3.9% vs 6.3% of cells above 1 mg/L).

What made M_train=100 affordable is the cost lever, not the hardware name: the
batched torch backend evaluates 100 scenarios in **6.2 s** where NumPy needs
~78 s (~11x), because one batched Krylov solve replaces 100 sparse
factorisations. CPU torch is already 2.4x faster than NumPy at this geometry
(7.75 s vs 18.5 s per 8-scenario evaluation), so the honest statement is that
the *backend* bought the scenario count and the GPU bought the wall clock
(1,057 s total, search 1,003 s, converged at nit=23). Krylov health is recorded
in `metrics.json`: flow CG 217 iterations to 8.6e-12, transport 100 solves x 100
steps converged, `stalled: false`.

Artifacts: `out3d_colab/plume3d_size_mid_gpu/{metrics.json,policy.json,
movies/,cert/fresh_400.json}` and `out3d_colab/plume3d_size_mid/cert/`. Movie
statistics come from `SCRIPTS/plume_movie3d.stack_stats` on `movies/fields.npz`.

## 3. Compute: what the GPU actually buys

Measured, not estimated (`DOCS/GPU.md` has the full audit):

| setting | geometry | result |
| :--- | :--- | :--- |
| batched compass sweep, float32, T4 | 80×40, M=8 | 1.46 s vs 69 s sequential NumPy → **47×** |
| same, A100 | 80×40, float32 | **no faster than T4** (latency-bound: one Python thread drives the GPU) |
| float64, T4 → A100 | 160×96, M=100 | **6.4×** (fp64-throughput-bound, opposite regime) |
| float64 → mixed float32, T4 | 160×96, M=100, 400 evals | **2.46×** (4.14 h → 1.67 h) |

Two lessons worth carrying: the first wall was kernel-launch overhead, not FLOPs;
and the work is *latency*-bound in float32 (a slice moves 1.6 MB in 7.6 ms against
a ~5 µs memory floor) but *throughput*-bound in float64, so "which card" has a
regime-dependent answer.

Numerical care that was needed to get there: float32 Krylov scalars and then the
whole BiCGSTAB recurrence had to be promoted to float64 (fp32 produced a NaN
transport solve at M ≥ 100 on a T4), and a mid-search **holdout guard** evaluates
the incumbent on 40 disjoint scenarios every 8 sweeps (14 s per check). That guard
is what distinguishes a certificate from a training-set artefact: a 43%-cheaper
policy found with `M_train = 16` looked compliant on its training tail and failed
out of sample (0.925 < 0.94 gate).

---

## 3b. Figures

`DOCS/FIGURES.md` indexes the article figure set produced by
`SCRIPTS/make_figures.py` (`FIGURES/*.png`, 300 dpi): the 1D target and Peclet
family, the conductivity realization with the certified well layout and the KL
truncation, the plume with and without pumping, the compliance traces and
peak-concentration distributions, the search trajectories, the tail of the chance
constraint, the reliability-vs-sample-size certificate, and the measured compute
costs. Every panel is generated from the production code paths or from a
finished run's artifacts; only the cost panel plots hard-coded numbers, and it
says so.

---

## 4. Reproducing

```
python3 SCRIPTS/run_optimization.py --config configs/opt/hydrology_gpu.yaml \
    --stage opt --backend torch --dtype float64        # ~4 h on a T4
python3 SCRIPTS/check_tail.py --metrics runs/colab_multi/<run>/r0/artifacts.zip \
    --config runs/colab_multi/<run>/r0/config_r0.yaml --m-scen 400 --ladder
python3 SCRIPTS/check_tail.py --metrics runs/colab_multi/<run>/r0/artifacts.zip \
    --seed-base 900000 --m-scen 400          # nothing the run ever saw
python3 SCRIPTS/check_tail3d.py --run out3d_colab/plume3d_size_mid_gpu \
    --m-scen 400 --json out3d_colab/plume3d_size_mid_gpu/cert/fresh_400.json
python3 tests/run_tests.py                              # 293 tests
```

---

## 5. What is *not* established

1. **Global optimality.** Every certified policy is a local optimum of a
   non-convex program, and the 2D restart experiment is the evidence that this
   matters: at M_train = 40 both random restarts converged to the *same*
   attractor from two different seeds, one of them 3% worse, so restarts sample
   the basin rather than proving it is the only one. A 3D restart probe is in
   flight; until it lands, "the single-well 3D design is the best design" is
   not a claim this document makes.
2. **2.5D / 3D.** The 3D results are separate work with their own runner,
   config and document (`SCRIPTS/run_3d.py`, `configs/opt/full3d.yaml`,
   `DOCS/3D_results.md`); the paper preset validates at 32×16×8 = 4096 cells with
   `q*` on 4 of 9 wells and test reliability 0.969. What is not established is
   the *analytical* side: `physics/fv3d.py` has no Ogata–Banks/Thiem gate yet, so
   its evidence is agreement with the batched torch path
   (`tests/test_fv3d_torch_colab.py`) and with the 2D kernels by extrusion, not
   agreement with a closed-form solution. The 2.5D axisymmetric bridge
   (`DOCS/3D_plan.md` phase 3D-0) is untouched.
3. **Long-horizon supply reliability.** All certificates are for the fixed
   deterministic source/gradient used in this config; source-strength and
   gradient uncertainty are implemented (`random_source` / `random_gradient`) but
   off in these runs.
4. **Surrogate-in-the-loop.** The FNO/PINN surrogates were never used inside the
   optimization loop; all optimization results use the finite-volume model.
5. **What the 3D audit does and does not certify.** It certifies one policy, at
   one geometry, on one seed base. It does not establish that the single-well
   design is globally optimal, nor that 400 fields would separate it from other
   certified 3D candidates (none have been found yet). The `size_mid` versus
   `full` comparison is **not** a resolution study: the two grids have different
   KL bases (n_kl 2062 vs 819), so they are two different aquifers. And the 3D
   movie is one scenario's maximum-intensity projection, which answers *where*,
   not *how much* — the numbers in §2.4 come from `stack_stats`, not the image.
