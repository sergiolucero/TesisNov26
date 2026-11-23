# 3D Results — Stochastic Optimization and Remediation

Validation results for `full3D_colab_run.py` (self-contained numpy/scipy
runner; TPFA Darcy, diagonal Bear dispersion, `K_h` with `kv_ratio` vertical
anisotropy, partially penetrating wells; cross terms and density coupling
deliberately omitted per `3D_plan.md` §1).

All runs use CVaR chance constraints (α=0.05 on concentration, α_h=0.01 on
head), penalty weights ρ_g = ρ_h = 1e6, and pattern search with multiple
starts. The reported `q*` is selected on the training archive and scored
out-of-sample on the test archive.

## Correction (2026-11-23): every 3D run before this date pumped the wrong cells

**Read this before quoting any 3D number in this file.** The well kernels were
written into transposed cells, so every search below solved a different geometry
than the one documented. The numbers are internally consistent — they are valid
results for *some* well layout — but they are not results for the layout in the
table. Fixed; superseded numbers are marked in the table.

What was wrong. Fields are stored ``(nz, ny, nx)``: ``Grid3D.index`` is
``k*nx*ny + j*nx + i`` and ``FlowModel3D._build_matrix`` reshapes the index array
the same way. ``Grid3D.centres()`` returned ``np.meshgrid(x, y, z,
indexing="ij")``, which is ``(nx, ny, nz)``. ``gaussian_well_weights_3d`` builds
its kernel from those arrays and ends with ``(w / w.sum()).ravel()``, so the value
intended for cell ``(i, j, k)`` landed at flat position
``i*ny*nz + j*nz + k`` instead of ``(k*ny + j)*nx + i``. On a cubic test grid a
well asked for at ``(3.5, 2.5, 2.0)`` landed at ``(0.5, 2.5, 3.5)`` — x and z
swapped. Nothing in a run could reveal it, because the coordinates reported in
``metrics.json`` and ``policy.json`` are the *requested* ones, and the z-screen
mask was applied to a coordinate array that was itself transposed.

How it surfaced. Not from the runs: from verifying ``physics/fv3d.py`` against
analytics (§ *Verification* below). Chasing an Ogata--Banks mismatch ended in the
axis conventions, and the fix to ``centres()`` was checked with a
one-line probe that asks where the well kernel actually peaks. Two regression
tests now hold it: ``test_well_kernel_lands_on_the_requested_cell`` and
``test_centres_are_in_field_order``.

What changed. The corrected runs find *cheaper* policies because the wells now
sit under the plume rather than scattered across it:

| geometry | search | q\* (m³/d) | total | train J | in-run test rel. | 400 fresh fields [95% Wilson] |
|---|---|---|---|---|---|---|
| size_mid, 8000 cells | ~~M=100, transposed~~ | ~~1 well, 37.3~~ | ~~37.27~~ | ~~501,060~~ | ~~0.938~~ | ~~0.9675 [0.945, 0.981]~~ |
| size_mid, 8000 cells | M=24, fixed | [0, 0, 12.5, 0, 0, 8.59] | **21.09** | 441,298 | 0.958 | 0.9375 [0.9094, 0.9573] **INCONCLUSIVE** |
| size_mid, 8000 cells | M=40, fixed | [0.05, 12.5, 0, 0, 10.06, 0] | 22.61 | 446,742 | 0.979 | **0.9700 [0.9483, 0.9828] PASS** |
| size_mid, 8000 cells | M=100, fixed | [0.2, 12.5, 0, 0, 9.77, 0] | 22.46 | **446,286** | 0.979 | **0.9700 [0.9483, 0.9828] PASS** |
| paper, 4096 cells | ~~M=64, transposed~~ | ~~4 wells at 12.5~~ | ~~50.0~~ | ~~547,606~~ | ~~0.969~~ | — (not certified) |
| paper, 4096 cells | M=64, fixed, T4 | [0, 10.64, 0, 0, 0, 12.79, 0, 0, 0] | 23.44 | **449,972** | 0.953 | **0.9725 [0.9514, 0.9846] PASS** |
| paper, 4096 cells | M=64, fixed, NumPy, 3 starts | [17.05, 0…, 8.42] | 25.47 | 457,397 | **0.984** | 0.9475 [0.9211, 0.9654] **INCONCLUSIVE** |

Three consequences worth stating plainly.

1. **The headline 3D numbers move.** The size_mid policy is 11 % cheaper
   (446,286 vs 501,060) at the same certified reliability, and the paper-preset
   policy is 18 % cheaper (449,972 vs 547,606) with a different shape: two wells
   carrying ~11-13 m³/d instead of four at 12.5. Every earlier 3D number in this
   file is superseded, including the ones quoted in the article's first draft.
2. **The sample-size story survives, but with a different boundary, and one of my
   own claims does not.** Under the corrected geometry the M=40 search finds the
   M=100 policy to 0.1 % and both certify at 0.970 — so the earlier reading,
   "M_train=40 under-pumps and fails the audit", was an artefact of the
   transposed wells and is withdrawn. What remains is a weaker but real boundary,
   and it is *below* where the 2D rule put it: `alpha*M_train = 1.2` (M=24) is
   inconclusive at 0.9375 with a point estimate under the gate, while
   `alpha*M_train = 2.0` (M=40) passes. The cheapest policy is again the least
   reliable (441,298 at 21.09 m3/d versus 446,286) — cheaper by 1.1 %, audited
   0.037 lower — so cost and reliability are still anti-correlated, just with a
   much smaller spread than the transposed geometry suggested.
   The 2D dose-response (0.912 vs 0.978/0.980 on 400 draws) is untouched.
3. **Policy multiplicity is still real, and the in-run row is decisively not a
   certificate.** At the paper geometry two searches with the same seeds and
   different hardware found *different* designs: wells {2,6} at 10.64+12.79
   (449,972, audited **0.9725**) and wells {1,9} at 17.05+8.42 (457,397). The
   second is the one with the *best* in-run reliability of any 3D run here
   (0.984) and it audits at **0.9475 — inconclusive**, point estimate below the
   gate. That is the cleanest demonstration in this file that a search's own test
   row cannot settle anything, in either direction.

What survives unchanged: mass balance (~1e-15), KL truncation and reproducibility,
NumPy/torch parity, the batching and cost conclusions, the movie, and the
methodology (audit any policy on fresh fields; the in-run test row is not a
certificate).

## Verification of the transport operator against analytics

The reason the bug above was found is that ``physics/fv3d.py`` was checked
against solutions that do not depend on our own code. Two of those checks are now
permanent tests (`tests/test_fv3d_reference.py`):

- **Pulse moments.** A pulse's spatial variance must grow as ``2Dt``. Measured:
  longitudinal ``D = alpha_l v + v dx/2`` (the upwind term, to 10 %), transverse
  ``D = alpha_t v`` on both axes (to 25 %, read before the no-flow walls bound the
  variance), centroid at the pore velocity ``v = K grad / phi``. This is a direct
  measurement of the dispersion tensor with no inlet involved.
- **The front against Ogata--Banks.** A mass-flux inlet is not a Dirichlet inlet.
  The classical formula carries a reflected term
  ``exp(vx/D) erfc((x+vt)/2 sqrt(Dt))`` that a pure advective inlet has no
  analogue for, and at ``vx/D`` large the two differ by a shift of ``D/v`` — 5 to
  10 cells here, which does not shrink with ``dx``. Compared against the correct
  reference, the swept front ``0.5 C0 erfc((x - vt)/2 sqrt(Dt))``, the interior
  relative L2 is **1.1e-2 at dx = 1 m** and **4.0e-3 at dx = 0.5 m**
  (``3.4e-3`` once ``v dx/2`` is added to ``D``), i.e. first order in ``dx`` and
  inside the G3.1 gate of 1.5e-2. The earlier "0.41 shape error" was the wrong
  reference, not a bug in the solver; no amplitude fit could have fixed it
  because the discrepancy is a shift.

## Presets validated

| preset | grid (nx·ny·nz) | n_kl | scenarios (tr/te) | wells | result |
|---|---|---|---|---|---|
| smoke | 12·6·3 = 216 | 164 | 6 / 12 | 4 | degenerate: already compliant, q*=0 |
| full | 24·12·6 = 1728 | 819 | 24 / 48 | 6 | ~~q\*=[12.89, 12.7, 11.91, 12.5, 0, 0]~~ (transposed wells) |
| paper | 32·16·8 = 4096 | 1419 | 64 / 128 | 9 | **q\*=[0, 10.64, 0, 0, 0, 12.79, 0, 0, 0]** (fixed), J 449,972, **certified 0.9725 [0.951, 0.985]** |
| paper | 32·16·8 = 4096 | 1419 | 64 / 128 | 9 | ~~q\*=[12.5 ×4, 0×5], reliability 0.969~~ (transposed wells) |
| size_mid | 40·20·10 = 8000 | 2062 | 24 / 48 | 6 | ~~q\*=[3.1, 0, 1.6, 12.5, 12.5, 12.5], reliability 0.875~~ (transposed wells) |
| size_mid | 40·20·10 = 8000 | 2062 | 100 / 48 | 6 | **q\*=[0.2, 12.5, 0, 0, 9.77, 0]** (fixed), J 446,286, **certified 0.9700 [0.948, 0.983]** |
| size_mid | 40·20·10 = 8000 | 2062 | 40 / 48 | 6 | **q\*=[0.05, 12.5, 0, 0, 10.06, 0]** (fixed), J 446,742, **certified 0.9700** |
| size_mid (GPU, torch) | 40·20·10 = 8000 | 2062 | 100 / 48 | 6 | ~~q\*=[0, 0, 0, 37.3, 0, 0], certified 0.9675~~ (transposed wells) |

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

   | policy | M_train | train J | violations | reliability | 95% Wilson | verdict |
   |---|---|---|---|---|---|---|
   | GPU, cheapest found | 40 | **490,433** | 38/400 | 0.9050 | [0.872, 0.930] | **FAIL** |
   | NumPy, 90-min budget | 24 | 519,263 | 37/400 | 0.9075 | [0.875, 0.932] | **FAIL** |
   | GPU, certified | 100 | 501,360 | 13/400 | **0.9675** | [0.945, 0.981] | **PASS** |

   The three rows are a dose-response, and it points the wrong way for the cheap
   policies. The M=40 search (240 evaluations, 631 s, `--m-train 40`) found a
   *cheaper* policy than the certified one — a single well at 34.23 m3/d, train J
   490,433, i.e. 2.2% below the M=100 result — and it is **less reliable**, not
   more: 38/400 violations, 0.9050, upper bound 0.930, rejected. So the ordering
   by cost runs 519k, 490k, 501k and the ordering by reliability runs 0.9075,
   0.9050, 0.9675: the cheapest policy is the second-least reliable, and only the
   most expensive one passes. `alpha*M_train` is 1.2, 2.0 and 5.0 respectively —
   the gate opens somewhere between two and five expected tail samples, which is
   the quantitative version of the 2D rule "require `alpha*M_train >= 5`".

   The certified policy is also much cleaner on the plume than the two rejected
   ones (final peak 3.71 vs 5.95 mg/L, 3.9% vs 6.3% of cells above the limit).

   Sample size is what decides this, and 128 was not enough: on a 128-field draw
   the same two policies give 5/128 = 0.961 [0.912, 0.983] and 9/128 = 0.930
   [0.872, 0.963] — both `INCONCLUSIVE`, though the ordering is already right.
   Note too that the GPU run's *own* test row (0.9375) fails the gate while its
   fresh draw (0.9675) passes: at M=48 the interval is 0.83-0.98, so an in-run
   row can neither certify nor condemn a policy. Only the 400-draw audit does.
3. **A higher-pumping start does not beat the empty one.** The M=40 probe ran
   two starts. The first (q0 = zeros) converged to the single well at 34.23 m3/d
   (f = 490,433 at nit=17, 119 evaluations). The second (uniform 0.5*q_max, q0
   summing to 150 m3/d) descended 866,875 -> 821,250 -> 775,625 -> 730,000 ->
   684,375 -> 638,750 -> 547,554 before its budget ran out, ending at 3 active
   wells and 50 m3/d. Both of those starts are *deterministic*: this file's start
   family is `[zeros, uniform 0.5*q_max, then random only if n_starts > 2]`, so
   the pair says the uniform start loses, not that random starts cannot win.
   Random-start evidence in three dimensions comes from the paper preset instead
   (three starts, 4 wells at 12.5 m3/d each, certified 0.969 at M_test = 128), a
   genuinely different design — so multiplicity across *searches* is real even
   though within one search the lower-pumping attractor wins.
4. **The optimum is not unique, and single-well policies are competitive.** The
   GPU search converged on one well at 37.27 m³/d while the NumPy search spread
   42.19 m³/d over five wells; both sit on the constraint (CVaR ≈ +0.02 train)
   and both are ~0.96-0.98 reliable. A policy-multiplicity statement needs the
   same 400-draw audit for each candidate, not a comparison of their in-run rows.

**A note on the museum piece above.** The run that produced these numbers was
reproduced bit-for-bit on a second VM the same evening (identical trajectory,
`f* = 501,360.88`, `q* = [0,0,0,37.27,0,0]`), which is what makes it a
measurement rather than an anecdote. That second VM was reclaimed by Colab
~45 min after launch, while it was certifying — the third reclaim in three
launches, all inside an hour — so its figures and zip were never downloaded.
The policy, the movie and every number quoted here survive because they had been
collected from the first run; the lesson is in `RUNTIMES.txt`: download the
artifact the moment it exists, and certify on the CPU.

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
