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
| PINN | hard (`eps = 1e-3`) | **1.35e-2** | **8.0e-3** | residual error is the measure-zero inlet corner |
| FNO | — | see `DOCS/Conclusions.md` §5 | — | normalized coordinates were required |

The hard-constraint answer `u = (1-ξ)s(τ) + ξ(1-ξ)s(τ)·NN` encodes the inlet
discontinuity exactly, which is why the corner error disappears from the interior
metric. Full narrative and the FNO negative-then-fixed story: `DOCS/Conclusions.md`.

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

Combining the two runs would be wrong (different scenarios), so each policy was
re-evaluated on 400 fresh scenarios (`SCRIPTS/check_tail.py --m-scen 400`):

| policy | violations | reliability | 95% Wilson CI | gate 0.94 |
| :--- | ---: | ---: | :--- | :--- |
| float64 `q*` (3 wells) | 1/400 | 0.998 | **[0.986, 1.000]** | PASS |
| float32 `q*` (2 wells) | 7/400 | 0.983 | **[0.964, 0.991]** | PASS |

The interval is the point of this table. A bare `0.975` at M = 40 is consistent
with a true reliability anywhere in ~[0.87, 1.00]; `399/400` is not. Sample-size
ladders are non-monotone (the same single violating scenario enters the tail at
different rates), which is why a single M is never quoted without the interval:

| M | 40 | 60 | 100 | 200 | 300 | 400 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| violations, float64 policy | 0 | 0 | 1 | 1 | 1 | 1 |
| reliability | 1.000 | 1.000 | 0.990 | 0.995 | 0.997 | 0.998 |
| Wilson low | 0.912 | 0.940 | 0.946 | 0.972 | 0.981 | 0.986 |
| violations, float32 policy | 0 | 0 | 1 | 3 | 6 | 7 |
| reliability | 1.000 | 1.000 | 0.990 | 0.985 | 0.980 | 0.983 |
| Wilson low | 0.912 | 0.940 | 0.946 | 0.957 | 0.957 | 0.964 |

**A fully independent draw.** The 400 scenarios above continue from the run's
train seed, so the first 100 of them *are* the training fields and the first
ladder column is the run checking itself. Re-running with `--seed-base 900000`
(nothing the optimizer ever saw, float32 policy) gives

| M | violations | reliability | 95% Wilson CI | gate 0.94 |
| :--- | ---: | ---: | :--- | :--- |
| 100 | 4/100 | 0.960 | [0.902, 0.984] | **inconclusive** |
| 200 | 5/200 | 0.975 | [0.943, 0.989] | PASS |
| 400 | 9/400 | 0.978 | [0.958, 0.988] | PASS |

with J = 428,509 (0.02% from the reported 428,592) and CVaR-5%(G) = +0.616 rather
than +0.005. So the certificate holds on unseen scenarios, and the tail is worse
than the run claimed: the run's own CVaR ≈ 0 is a training-set statement, while a
fresh draw puts the mean of the worst 5% at +0.6 m³. The 100-scenario sample is
the useful part of this table -- it is the sample size the search itself used, and
it is *not* enough to certify anything: 4 violations out of 100 straddles the gate
and only the larger draws resolve it.

Both policies clear the 0.94 gate with certainty, but not equally: the float64
policy violates in 1 of 400 scenarios, the float32 policy in 7 of 400. A point
estimate of 0.975 at M = 40 cannot distinguish them -- both look identical there
-- and the difference only becomes visible at M ≥ 200. **That is the argument for
large out-of-sample evaluation**: policy ranking is not resolvable at the sample
size the search itself used.

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

## 4. Reproducing

```
python3 SCRIPTS/run_optimization.py --config configs/opt/hydrology_gpu.yaml \
    --stage opt --backend torch --dtype float64        # ~4 h on a T4
python3 SCRIPTS/check_tail.py --metrics runs/colab_multi/<run>/r0/artifacts.zip \
    --config runs/colab_multi/<run>/r0/config_r0.yaml --m-scen 400 --ladder
python3 SCRIPTS/check_tail.py --metrics runs/colab_multi/<run>/r0/artifacts.zip \
    --seed-base 900000 --m-scen 400          # nothing the run ever saw
python3 tests/run_tests.py                              # 153 tests
```

---

## 5. What is *not* established

1. **Global optimality.** Both policies are local optima of a non-convex program;
   the 1.2% spread suggests a flat basin, not that nothing better exists. A
   multi-start sweep at M = 100 is in flight.
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
