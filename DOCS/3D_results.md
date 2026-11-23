# 3D Results — Stochastic Optimization and Remediation

Validation results for `full3D_colab_run.py` (self-contained numpy/scipy
runner; TPFA Darcy, diagonal Bear dispersion, `K_h` with `kv_ratio` vertical
anisotropy, partially penetrating wells; cross terms and density coupling
deliberately omitted per `3D_plan.md` §1).

All runs use CVaR chance constraints (α=0.05 on concentration, α_h=0.01 on
head), penalty weights ρ_g = ρ_h = 1e6, and pattern search with multiple
starts. The reported `q*` is selected on the training archive and scored
out-of-sample on the test archive.

## Presets validated

| preset | grid (nx·ny·nz) | n_kl | scenarios (tr/te) | wells | result |
|---|---|---|---|---|---|
| smoke | 12·6·3 = 216 | 164 | 6 / 12 | 4 | degenerate: already compliant, q*=0 |
| full | 24·12·6 = 1728 | 819 | 24 / 48 | 6 | **q\*=[12.89, 12.7, 11.91, 12.5, 0, 0]**, reliability 0.979 |
| paper | 32·16·8 = 4096 | 1419 | 64 / 128 | 9 | **q\*=[12.5, 12.5, 12.5, 12.5, 0×5]**, reliability 0.969 |
| size_mid | 40·20·10 = 8000 | 2062 | 24 / 48 | 6 | **q\*=[3.1, 0, 1.6, 12.5, 12.5, 12.5]**, reliability 0.875 — *fails* the gate; see below |
| size_mid (GPU, torch) | 40·20·10 = 8000 | 2062 | 100 / 48 | 6 | **q\*=[0, 0, 0, 37.3, 0, 0]**, in-run 0.938 but **certified 0.9675 [0.945, 0.981] on 400 fresh fields — PASS** |

## Paper preset (headline)

- **q\*** = `[12.5, 12.5, 12.5, 12.5, 0, 0, 0, 0, 0]` m³/d — total 50 m³/d
  across 4 of 9 wells, clustered around the plume centroid. (Note: these are
  *equal* rates, in contrast to the `full` preset's unequal 12.9/12.7/11.9/12.5,
  because the pattern search converged on the same 12.5 rate for all four
  active wells.)
- **train J** = 547 605.94; **test J** = 547 665.31 (gap 1.1e-4, no overfitting).
- **test reliability = 0.969** (≥ 1−α−1% = 0.94 gate) vs **no-pump reliability 0.0**.
- **test mean cleanup τ = 110.6 d** vs no-pump τ = 3650 d (never cleans up).
- test violation probability 0.031, mean head margin H = 10.03 m (above h_min).
- mass-balance error ~7e-15 (machine precision).

### Gates (all pass)

`KL_retained_variance≥0.95` ✓ · `reproducible` ✓ · `mass_balance≤1%` ✓ ·
`test_reliability≥1−α−1%` ✓ · `optimization_improved_vs_no_pump` ✓ ·
`no_pump_reliability = 0.0` ✓

### Cross-machine reproducibility

The paper preset's `f* = 5.476e+05` from the q₀=0 start matches the value
obtained in a partial Colab run of the same preset (`f=5.476e+05`), so the
single-file runner reproduces across machines and runtimes.

## Notes

- The `smoke` preset is degenerate on purpose: the source is weak enough that
  the no-pumping plume already satisfies the limit (q*=0 is correct, and
  `optimization_improved_vs_no_pump = False` there is expected).
- `--n-starts` is now a CLI override (default from preset: 1/1/2/3 for
  smoke/quick/full/paper).
- Run locally: `python full3D_colab_run.py --preset paper --budget 15 --n-starts 2`.
  The paper preset took ~20 min on a 4-core laptop (calibration 19.5 s per
  64-scenario archive evaluation).
- On Colab the same preset calibrates at ~14.9 s/eval on a 2-core runtime; use
  `colab run --timeout 0 full3D_colab_run.py --preset paper --budget 20 --n-starts 2 --zip`
  (CPU runtime — the workload is sparse LU, not GEMM, so a GPU buys nothing).

**Local CPU rerun after the merge (2026-11-21).** The paper preset was re-run on
the assembled tree on a 4-core laptop (`python full3D_colab_run.py --preset paper
--budget 15 --n-starts 2`, 1014.8 s, 49 evaluations) and reproduced this document
exactly: `q* = [12.5, 12.5, 12.5, 12.5, 0×5]`, test `J = 547,665.31`, reliability
`0.96875`, mass-balance error `9.7e-15`, all six gates true. That is a third
machine and a different BLAS after the 3D and opt branches were merged, so the
agreement is not just the same binary.

**Rewritten as thesis Chapter 6 (2026-11-22).** `TEX/results_3d.tex` is now
seven sections: the model and its seven gates, the paper preset (with the
baselines and the reproducibility table), *eight thousand cells and the search
that breaks* (this run, the fresh 128-draw certificate, the α·M mechanism, the
cost of a converged start), *what the plume movie shows* (the contact sheet and
the final-frame statistics), *the wall is the search, not the solver* (the
progressive-hedging line: degenerate on the one-stage model, substantive with
recourse, and what has and has not been verified), and the limits. The earlier
note on this chapter follows.

**Written up as thesis Chapter 6 (2026-11-21).** `TEX/results_3d.tex` turns this
into prose (model, solver gates, presets, baselines, reproducibility, limits) and
adds one caveat that the run's own gate does not express: the test reliability
0.969 is 124/128, whose Wilson interval is [0.922, 0.988]. The point estimate
clears the 0.94 gate; the interval does not. By the reporting rule established in
`DOCS/Results.md` §2.2 the 3D policy is therefore *not* certified, and doing so
needs ~400 test scenarios (~2 h at 18.6 s/eval on one core).

## The size_mid run (8,000 cells): a bigger grid is not a better search

`--preset size_mid`, run directory `out3d_colab/plume3d_size_mid`. Chosen as the
step between `full` (1,728 cells) and `paperGPU` (32,768 cells): 40x20x10 cells at
dx = dy = 5 m, dz = 2 m, so the 20 m correlation length is resolved by 4 cells
(it is 2.4 at `full`) and a 30 %-of-thickness screen spans 3 layers instead of
1.8 — the vertical structure of a partially penetrating well is finally
representable.

| quantity | value |
|---|---|
| grid | 40x20x10 = 8,000 cells (5.0 x 5.0 x 2.0 m), n_kl = 2062 (95.0 % variance) |
| scenarios | 24 train / 48 test, nsteps = 100 |
| search | pattern search, 2 starts, 13 candidates/sweep, **budget-clamped to 107 evaluations per start** |
| wall clock | 4,486 s (74.8 min), 25.3 s per evaluation |
| q\* | [3.125, 0, 1.5625, 12.5, 12.5, 12.5] m³/d — 42.19 total, 5 of 6 active |
| train J | 519,263 (reliability 0.958, CVaR-5%(G) = +0.036) |
| test J | 519,493 (reliability **0.875**, CVaR-5%(G) = +0.533, mean tau 456 d) |
| baselines | no pump 303,639 (reliability 0.000, mean G 16.95); uniform half-capacity 475,393 (reliability 0.167) |
| mass balance | max error 0.0 (identical operator at both dtypes) |
| gates | 5 of 6 pass; **`test_reliability >= 1-alpha-1%` fails** |

**The run is honest about its own failure.** Its test draw gives 0.875, and an
independent 128-scenario draw (seeds 900000+, `SCRIPTS/check_tail3d.py
--m-scen 128 --ladder`) gives 9/128 = 0.930 with a 95 % Wilson interval
[0.872, 0.963] — `INCONCLUSIVE` at the 0.94 gate, with the point estimate below
it. The nested ladder is flat: 1/16, 1/24, 3/40, 4/64, 7/100, 9/128. So the policy
is not an artifact of one draw: it genuinely violates about 7 % of scenarios, and
the failure mode is the one already known from 2D — **M_train = 24 estimates the
tail from roughly one scenario** (`alpha*M = 1.2`), so the search cannot see the
cost of the violation it is buying.

Two things this run establishes, and one it does not:

- The 8,000-cell configuration is *mechanically* sound end to end: KL at 95 %
  variance, mass balance exact to float precision, reproducible, and the movie
  renders. `per_eval` is 25.3 s, 3.4x the paper preset for 2x the cells, i.e.
  cost grows roughly as n^1.9 — sparse 3D LU fill-in, as expected.
- The budget is what limits the *search*, not the node count: 90 min buys 214
  evaluations, i.e. ~10 sweeps per start with the compass step never halving.
  A converged 3D search at this geometry needs M_train ≈ 100. **Correction
  (2026-11-22):** the per-scenario cost is 25.333/24 = 1.06 s, not the 0.775 s
  quoted above (25.333 s is per 24-scenario evaluation), so M=100 is ~105 s per
  evaluation and 300 evaluations per start is 8–9 h per start on this machine,
  2 starts ≈ 18 h. The paper preset's 18.63 s per 64-scenario evaluation gives
  0.29 s per scenario, so the ratio is 3.6x cost for 1.95x cells — the claimed
  ~n^1.9 holds, the hours do not. That is the honest price of a certified mid-size 3D
  policy; it should be run on the GPU backend or overnight, not inside a 90-minute
  budget.
- **Not** a resolution study. The `full` and `size_mid` runs use different KL
  bases (819 vs 2062 modes) and therefore different realisations of K; comparing
  their plume peaks or J would compare two different aquifers, which is exactly
  the error that made a correct 2D run look irreproducible for an afternoon. A
  resolution claim needs the same realisation on two grids, which the KL
  truncation does not give.

### What the movie shows (quantitative, from `movies/fields.npz`)

`SCRIPTS/plume_movie3d.stack_stats` on the saved stacks, final frame, 32 frames:

| | untreated | under q\* |
|---|---|---|
| peak (mg/L) | 44.8 | 6.0 |
| mean (mg/L) | 2.94 | 0.19 |
| cells above 1 mg/L | 35.7 % | 6.3 % |

The policy reduces the plume by roughly an order of magnitude but does not
confine it, which is exactly the picture the 0.93 reliability implies: on the
violating scenarios the residual plume still reaches the compliance point. The
same panel pair in a picture: `movies/plume3d_contact.png`.

## The same problem on a GPU: 15x faster, a different policy, and the same tail problem

`SCRIPTS/colab3d.py launch --preset size_mid --gpu T4 --dtype float64 --m-train 100
--max-evals 200 --n-starts 1 --budget 60` (run `runs/colab3d/20261122-1349`).
Same geometry, same KL, same seeds as the numpy `size_mid` run above — the only
differences are the backend and a training set the geometry can afford.

| | numpy `size_mid` | torch on a T4 |
|---|---|---|
| train / test scenarios | 24 / 48 | **100** / 48 |
| cost per 24-scenario evaluation | 25.3 s | — |
| cost per 100-scenario evaluation | ~105 s (25.333 s per 24-scenario evaluation = 1.06 s/scenario) | **6.2-7.1 s** (0.071 s/scenario) |
| archive build (148 scenarios) | 27.6 s | **4.7 s** |
| search | 2 starts x 107 evals, budget-truncated | 1 start, **converged at nit=23** (161 evals) |
| wall clock | 4,486 s | **1,057 s** (search 1,003 s) |
| q\* | [3.125, 0, 1.5625, 12.5, 12.5, 12.5] (42.19 m³/d, 5 active) | **[0, 0, 0, 37.27, 0, 0]** (37.27 m³/d, 1 active) |
| train J | 519,263 (rel 0.958) | 501,060 (rel 0.980) |
| test J | 519,493 (rel 0.875) | 501,261 (rel **0.938**) |
| CVaR-5%(G) test | +0.533 | +0.139 |
| mass balance | 0.0 | 0.0 |
| gates | 5/6 (`test_reliability` fails) | 5/6 (`test_reliability` fails, 0.9375 vs 0.94) |

Three findings, in decreasing order of importance:

1. **The GPU is worth ~15x here, and the win is batching, not arithmetic — which
   is what makes M_train=100 affordable at all.** One
   full-archive evaluation of 100 scenarios costs ~6.2 s against the ~105 s the NumPy
   backend needs for the same work. The T4's fp64 throughput is weak, so this is
   not floating-point throughput: it is one batched Krylov solve replacing 100
   sparse factorisations. The 3D `splu` is the expensive part (cost ~n^1.9 in
   cell count), which is exactly the regime where a matrix-free batched solver
   wins — and it already wins on **CPU** torch (7.75 s vs 18.5 s per 8-scenario
   evaluation), which is the honest way to say it: the backend matters more than
   the card.
2. **M_train = 100 buys the certificate that M_train = 24 cannot.** This is the
   headline. Both policies were re-evaluated on the same 400 fresh fields
   (seeds 900000+, `SCRIPTS/check_tail3d.py --m-scen 400`):

   | policy | violations | reliability | 95% Wilson | verdict | fresh J |
   |---|---|---|---|---|---|
   | GPU, M_train=100 | 13/400 | **0.9675** | [0.945, 0.981] | **PASS** | 501,050 |
   | NumPy, M_train=24 | 37/400 | 0.9075 | [0.875, 0.932] | **FAIL** | 519,433 |

   The GPU policy is the first *certified* 3D policy at this geometry: its
   interval's lower bound clears the 0.94 gate. The NumPy policy is *rejected* —
   the upper bound is 0.932 — which is the honest reading of a search that
   estimated the tail from `alpha*M_train = 1.2` scenarios. It is also 3.5%
   cheaper (501,050 vs 519,433) and much cleaner on the plume (final peak 3.71 vs
   5.95 mg/L, 3.9% vs 6.3% of cells above the limit).

   Sample size is what decides this, and 128 was not enough: on a 128-field draw
   the same two policies give 5/128 = 0.961 [0.912, 0.983] and 9/128 = 0.930
   [0.872, 0.963] — both `INCONCLUSIVE`, though the ordering is already right.
   Note too that the GPU run's *own* test row (0.9375) fails the gate while its
   fresh draw (0.9675) passes: at M=48 the interval is 0.83-0.98, so an in-run
   row can neither certify nor condemn a policy. Only the 400-draw audit does.
3. **The optimum is not unique, and single-well policies are competitive.** The
   GPU search converged on one well at 37.27 m³/d while the NumPy search spread
   42.19 m³/d over five wells; both sit on the constraint (CVaR ≈ +0.02 train)
   and both are ~0.96-0.98 reliable. A policy-multiplicity statement needs the
   same 400-draw audit for each candidate, not a comparison of their in-run rows.

Krylov health on the GPU (recorded in `metrics.json` because a stalled transport
solve silently poisons J): flow CG 217 iterations to 8.6e-12, converged, not
stalled; transport 100 solves x 100 steps at ~0.07 s per scenario, `converged:
true`, `stalled: false`. The movie rendered 32 frames at 0.6 s per capture.

## Plume movie

Every `full3D_colab_run.py` run now writes `<run>/movies/` with
`plume3d.mp4`, `plume3d.gif`, `plume3d_contact.png` and the raw `fields.npz`
(untreated beside `q*`, same scenario, shared colour scale). The stage is
unconditional and self-guarded, rendered before `metrics.json` is written, and its
paths are recorded under `metrics["movie"]`. Specification: `DOCS/3D_movie.md`.
Measured cost on the `full` preset: ~0.1 s per captured policy, ~4 s to render 24
frames, ~1 s to encode -- under 2 % of the run.
