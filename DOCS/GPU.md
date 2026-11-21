# GPU usage: what actually runs on the Colab GPU

Short answer to "does this code use the Colab GPU?": **it did not.** This
document records the audit that established that, what changed, and how to
verify on your own runtime that a GPU is now doing real work.

## 1. Audit (before)

| Where | Device | Why |
|---|---|---|
| `SCRIPTS/solve_1d_pinn.py` | CUDA | torch autograd MLP, but a 2x64x4x1 net on ~1.5k collocation points |
| `SCRIPTS/solve_1d_fno.py` | CUDA | torch FNO, but 1D, 12 modes, 20 channels, batch 8 |
| `SCRIPTS/hydrology.py` | NumPy | dataset generation |
| `physics/fv2d.py` | NumPy + `scipy.sparse.linalg.splu` | 2D flow/transport |
| `stoch/*.py` (random fields, wells, objective, SAA, chance, **adjoint**, optimizer) | NumPy | everything above the solver |
| `SCRIPTS/run_2d.py`, `run_optimization.py`, `seed_sweep.py` | NumPy | never imported torch |

So the only GPU work was two **1D toy models**: a 4-layer MLP with 2 inputs and
an FNO on 64-point 1D signals. Those kernels are launch-bound, not
throughput-bound: a T4 does not beat a decent CPU on them, and `N_f`/`batch`
sizes were far too small to fill a GPU. Meanwhile the stage that actually costs
time -- `run_optimization.py --stage opt`, which exceeded **1800 s** on one
vCPU (`RUNTIMES.txt`) -- is 100 % SciPy sparse LU.

Two further gaps confirmed the diagnosis:

1. `SCRIPTS/collab_run.py --gpu N` mapped to `--device cuda:N`, but the `opt`
   stage never received it: `run_opt()` built its command line without
   `--device`. Selecting a GPU had no effect on the expensive stage.
2. There was no AMP, no `torch.compile`, no pinned-memory/batched transfers, no
   CUDA-graph capture, and no batched ensemble: every scenario was solved in a
   serial Python loop.

**Conclusion: paying for Colab Pro bought nothing** -- the accelerator was idle
for ~all of the wall clock that mattered.

## 2. What changed

A second, batched backend for the 2D flow + transport + SAA stack. The NumPy
path is untouched and remains the reference.

| File | Role |
|---|---|
| `physics/fv2d_torch.py` | batched 5-point finite-volume flow and transport: matrix-free stencils, batched CG (Darcy, SPD) and BiCGSTAB (transport, nonsymmetric), Jacobi preconditioning, warm-started time loop |
| `stoch/objective_torch.py` | batched cost functional, cleanup time and its sigmoid surrogate, CVaR -- transcribed term-by-term from `stoch/objective.py` |
| `stoch/saa_torch.py` | `TorchArchive`: the whole scenario ensemble resident on the device; `evaluate(q)` (drop-in) and `evaluate_many(Q)` (batched candidates) |
| `stoch/optimize.py` | `pattern_search_batched`: one compass sweep per batched call |
| `SCRIPTS/run_optimization.py` | `--backend {numpy,torch}`, `--device`, `--dtype`, `--krylov-rtol/maxiter`, `--[no-]batched-search` |
| `SCRIPTS/collab_run.py` | `--opt-backend {auto,numpy,torch}`, `--opt-dtype`; the `opt` stage now receives the device |
| `SCRIPTS/bench_backend.py` | measures both backends on *your* runtime (archive build, per-evaluation, compass sweep, agreement, projection) and prints a verdict |
| `tests/test_fv2d_torch.py`, `tests/test_saa_torch.py` | 28 parity tests (flow, velocity, transport, stencil transpose, objective, summary, optimizer, backend selection) |

### Why matrices are not batched with `splu`

There is no CUDA sparse LU in PyTorch, so the batched backend does not try to
replicate `splu`. It changes the execution model instead:

* the operator depends on the candidate rate vector but **not** on the time
  level, so it is assembled **once** and reused for all `nsteps`;
* each implicit-Euler step is a **batched** Krylov solve whose batch axis holds
  `S` candidates x `M` scenarios, so one launch does `S*M` solves;
* the time loop is **warm-started** from the previous concentration field, which
  is what keeps the iteration count low (measured: 8-9 BiCGSTAB iterations per
  step at `rtol=1e-11`, versus ~17 cold).

### Accuracy

`splu` is a direct solve; Krylov is not, so the accuracy contract is explicit:

| Quantity | NumPy reference | batched float64 | batched float32 |
|---|---|---|---|
| relative field / objective error | - | ~1e-11 | ~1e-5 |
| mass-balance error | ~1e-15 | ~1e-10 | ~1e-6 |
| default Solver tolerance | - | `rtol=1e-11` | `rtol=1e-6` |

`tests/test_*_torch.py` enforce the float64 column, and the O1-O4 gates in
`run_optimization.py` deliberately stay on the NumPy backend: they certify
solver properties (exact discrete adjoint vs. central differences at 1e-4,
mass balance at 1e-2), not throughput.

## 3. How to use it

### On Colab (GPU runtime)

```python
# one command: the opt stage picks up the GPU automatically
!python SCRIPTS/collab_run.py --models opt --opt-stage opt \
    --opt-config configs/opt/benchmark.yaml --opt-dtype float32 --gpu 0
```

or directly:

```python
!python SCRIPTS/run_optimization.py --config configs/opt/benchmark.yaml \
    --stage opt --backend torch --device cuda --dtype float32
```

Notes:

* `--dtype float32` is the throughput mode (~2x memory bandwidth, half the
  footprint). Use `float64` when you want the same numbers as the NumPy run.
* The gates (`--stage gates`) always run on the NumPy reference; asking for
  `--opt-stage gates --opt-backend torch` prints a note and does that.
* `--no-batched-search` keeps the sequential pattern search on the GPU archive
  (useful to separate the two speedups: batching over scenarios vs. batching
  over candidates).
* `SLSQP` + discrete adjoint is NumPy-only: `TorchArchive.problems` raises with
  a clear message rather than silently returning a wrong gradient.

### Without a GPU

```bash
python SCRIPTS/bench_backend.py --quick --device cpu      # parity + CPU timings
python SCRIPTS/run_optimization.py --quick --stage opt --backend torch --device cpu
```

Both work, both reproduce the NumPy answer exactly (see below), and both are
**slower** than the SciPy path -- as they should be. A CPU has nothing to
parallelise across `S*M` systems, and the batched loop pays Python overhead that
`splu` does not.

## 4. Measured results

### Cross-backend agreement (this machine, CPU, float64)

| Check | Result |
|---|---|
| `run_optimization.py --quick --stage opt` (NumPy) | q\\* = [0.1, 0.0, 0.0, 9.38], train J = 378906.45, test J = 378822.07, 13.8 s |
| same with `--backend torch --device cpu` | q\\* = [0.1, 0.0, 0.0, 9.38], train J = 378906.45, test J = 378822.07, 56.7 s |

Identical answer to the last digit, both matching the value recorded in
`RUNTIMES.txt`. The batched backend is 1.6-4x slower here (22.0 s vs 13.8 s on
an idle box, 56.7 s on a loaded one) because this machine has no accelerator;
see `DOCS/Conclusions.md` §9 for what that control does and does not prove.

### Batched scaling on one CPU core

`SCRIPTS/bench_backend.py --device cpu`, float64, `benchmark.yaml`:

| geometry | backend | archive build | one evaluation | compass of 18 |
|---|---|---|---|---|
| 80x40, `nsteps=100`, `M=8` | NumPy `splu` | 0.25 s | 3.85 s | 69 s (18 x 3.85) |
| 80x40, `nsteps=100`, `M=8` | torch, CPU | 37.2 s | 10.5 s | **168 s** |
| 30x15, `nsteps=40`, `M=8` (`--quick`) | NumPy `splu` | 0.05 s | 0.11 s | 2.0 s |
| 30x15, `nsteps=40`, `M=8` (`--quick`) | torch, CPU | 0.18 s | 1.14 s | 2.0 s |

Two things to read from this:

1. **On a CPU the batching is a straight loss** (0.37x the NumPy throughput at
   the full geometry). The batch axis has nothing to parallelise across, so one
   batched call costs ~`S` times one candidate, and the Python-level Krylov loop
   adds overhead that `splu` does not have. This is why batched search is
   enabled automatically only on an accelerator.
2. **The arithmetic is unchanged by batching; the parallelism is new.** The
   compass sweep is 16x one candidate at the full geometry (it is
   arithmetic-bound there) -- on a GPU that 16x still has to be done, but it is
   done as *one* batch of `18 x M` systems instead of 18 sequential `M`-system
   loops on a single core. The comparison that matters is GPU-seconds per
   batched call versus CPU-seconds for 18 sequential calls.

The Krylov cost is well behaved (measured at the full geometry, 9 wells):
24 BiCGSTAB iterations per time step (worst 34) with warm starts, and 345 CG
iterations for the batched Darcy solve (one-time per archive).

### Measured on a T4 (the estimate above, now replaced by a measurement)

`SCRIPTS/new_high_colab.py --launch --gpu-class turing --accelerator T4
bench forward --preset paper --verify`, on a rented **Tesla T4 (sm_75,
15.6 GB, 40 SMs)**, torch 2.14.1+cu130, float32, paper geometry
(80x40, `nsteps=100`, 9 wells, `M=8`, compass `S=18`):

| candidates `S` | wall clock | ms per (cand·scen) | Mcell·step/s | peak VRAM |
|---|---|---|---|---|
| 1 | 1.41 s | 176.7 | 1.8 | 0.01 GB |
| 18 | 1.46 s | 10.16 | 31.5 | 0.08 GB |
| 72 | 4.68 s | 8.12 | 39.4 | 0.28 GB |
| 288 | 17.58 s | 7.63 | **41.9** | 1.11 GB |

Matmul: fp32 **7.0** TFLOP/s, bf16 4.2 TFLOP/s (bf16 *slower* on a T4, as
expected for a bandwidth-bound workload on Turing). `forward --verify` passed
at `rel_max = 4.42e-05` against the NumPy reference.

Two things follow that the estimate got only partly right.

**Batching is the whole win, and it saturates.** Going from `S=1` to `S=18`
(one compass sweep) cuts the per-slice cost **17x**, 176.7 ms -> 10.2 ms; but
`S=18 -> S=288` adds only 1.3x more (10.2 -> 7.6 ms) while VRAM goes from
0.08 GB to 1.11 GB. The compass sweep is where the value is; there is little
left to gain by batching further.

**A full `M=40` compass sweep: 7.3 s if extrapolated from `M=8`, ~1.9 s
measured directly** (i.e. ~0.10 s per SAA evaluation).
18 candidates x 40 scenarios x 10.16 ms extrapolates to 7.3 s, but
that 10.16 ms/slice was itself derived at `M=8` where a 1.5 s fixed overhead is
amortised over only 144 slices. Measuring at `M=40` directly (see "It is the
fixed overhead" below) gives **1.86 s**. The extrapolated budgets below are
therefore conservative upper bounds; the real ones are ~4x smaller.

18 candidates are 18 SAA evaluations, so a budget of `max_evals` costs
`max_evals/18` sweeps:

| run | SAA evaluations | compass sweeps | T4, per-sweep measurement | T4, `ms/slice` extrapolation |
|---|---|---|---|---|
| bounded (1 start, 200 evals) | 200 | 12 | **~22 s** | 24 min |
| paper (3 starts x 1500 evals) | 4500 | 250 | **~7.8 min** | 9.1 h |

The right-hand column is the conservative one and it is ~20x too pessimistic:
it multiplies a per-slice cost measured at `M=8` (where 1.5 s of fixed overhead
is spread over only 144 slices) by `M=40`, and then charges it per evaluation
rather than per sweep. The per-sweep column uses the directly measured 1.86 s
at `M=40`. See `DOCS/Conclusions.md` section 9.

Compare **like for like**, at `M=8` and the same 80x40 geometry: the NumPy
`splu` backend takes 69 s for a sequential compass of 18, the T4 takes 1.46 s
for the same sweep batched -- a **47x** speedup. At the `M=40` the optimizer
actually uses, one sweep is 1.86 s, i.e. **~0.10 s per evaluation**; the CPU
figure it replaces is the 33.2 s/eval recorded in `runs/runtimes.jsonl` at
160x80 (contended, and a different geometry, so treat that ratio as indicative
rather than exact).

So the GPU is worth having and the paper run fits inside one session: minutes,
not hours, at the measured per-sweep cost. The 9.1 h figure in the conservative
column is what `--max-seconds` splitting or `SCRIPTS/multi_colab.py` across
several VMs was being planned for; on the measured basis that machinery is a
safety net, not a requirement. Against ~21 h for the same budget on one CPU
core, this is the whole case for renting the GPU.

**The workload is latency bound, not bandwidth bound — contrary to the
module docstring's premise.** At `S=288` a slice moves 1.58 MB in 7.6 ms, i.e.
~208 MB/s. The T4's HBM does ~320 GB/s, so the memory floor for that slice is
about **5 us**, three orders of magnitude below the measured 7.6 ms. The time
therefore goes into the Krylov iterations (24 BiCGSTAB steps x 100 time steps,
each a short dependent sparse solve), not into moving bytes. That has a
consequence for tuning: float32-vs-float64 and other traffic reductions buy
little here, while reducing the *number* of Krylov iterations (better
preconditioning, a looser tolerance) or increasing batching further are the
levers that would actually pay. The `bench` numbers above are the evidence.

### But float64 flips it: the A100 is 6.4x the T4

The A100 verdict above is a **float32** verdict, and it only holds while the work
is launch-bound. At 160x96 the float64 path (which section 6 shows is mandatory
at this geometry) is throughput-bound instead, and there the bigger card wins by
a wide margin. Same config, same M_train=100 / M_test=40, same seeds, same
9-well layout, same `--opt-backend torch --opt-dtype float64`:

| | T4 (15.6 GB, 40 SM) | A100-SXM4-80GB | ratio |
| :--- | ---: | ---: | ---: |
| archives (100 train + 40 test) | 38.0 s | **7.6 s** | 5.0x |
| flow CG (800 iterations, one-time) | ~30 s | ~6 s | 5x |
| one SAA evaluation (M=100) | 38.1 s | **5.89 s** | 6.5x |
| one 19-candidate compass sweep | 680 s | **106 s** | 6.4x |
| per candidate*scenario | 0.381 s | **0.059 s** | 6.5x |
| best objective at 100 evals | 821,250 | 821,250 | identical |
| gates | pass | pass | - |

The trajectory is bit-for-bit the same (both took `f = 821250` at the 100-eval
mark), so the 6.4x is not bought with a wrong answer. The gap versus the ~32x
fp64 FLOPS ratio (0.3 vs 9.7 TFLOPS) shows the workload is still partly
latency/bandwidth-bound, but fp64 throughput now clearly matters - the opposite
of the float32 finding.

Practical rule: **choose the card by dtype.** float32 screening -> the T4 is as
good as anything and 6x cheaper per hour. float64 (mandatory at 160x96 and
above) -> A100/H100 pay off as soon as their hourly price is under ~6x the T4's;
if they cost more than that, they are break-even on wall clock but still buy 85
GB of VRAM (M_train 400 fits) and 6.4x lower latency per experiment.

This also re-prices the queue: the "finish the M=100 search" run (400 evals) is
**~40 min on an A100** versus ~4 h on a T4.

### It is the fixed overhead, not the hardware (measured on A100)

A100 vs T4 on the identical full geometry is the decisive comparison, and it
does **not** favour the bigger card at the batch size the optimizer uses:

| compass `S=18`, `M=8` | T4 | A100-SXM4 |
|---|---|---|
| ms per (cand·scen) | 10.16 | 10.85 |
| best throughput (`S=288`) | 41.9 Mcell·step/s | 234.2 Mcell·step/s |
| matmul fp32 / bf16 | 7.0 / 4.2 TFLOP/s | 94.0 / 140.1 TFLOP/s |

The A100 has 13x the fp32 matmul throughput and 5.6x the peak batched
throughput, yet it is **no faster on one compass sweep** (10.85 vs 10.16 ms).
Scaling `M` at fixed `S=18` on the A100 explains why:

```
S=18, M      wall      ms/slice
1            1.413 s   78.50
4            1.606 s   22.31
8            1.664 s   11.56
40           1.856 s    2.58

linear fit:  wall = 1.516 s fixed + 8.9 ms per scenario
```

**A compass sweep costs 1.52 s of fixed overhead plus 8.9 ms per scenario. At
`M=40` that is 81% pure overhead** -- the hardware is idle for most of the call.
The work is a sequence of ~100 time steps x ~24 BiCGSTAB iterations, each a
short dependent sparse solve issued separately, so the GPU spends its time in
launch and host<->device synchronisation rather than arithmetic. Buying a
faster card does not fix that; it is why an A100 ties a T4 here.

**Consequence for the plan.** Removing the per-step launches (one batched
solve for the whole time history, or a persistent kernel) would take the paper
run from roughly **8 min to under 2 min on the A100** -- a ~5x win from
software (the same 81 % overhead share, applied to the corrected basis), against the
0x that a better card delivered. That, not a bigger GPU, is the next thing to
try.

**This also corrects the `bench` extrapolation.** `bench` derives ms/slice as
`wall/(S*M)` at `M=8`, where the 1.5 s fixed cost is spread over only 144
slices, so it reports 10.85 ms/slice and implies 7.8 s for an `M=40`
evaluation. Measured directly at `M=40` the same sweep takes **1.86 s**. The
bench therefore **over-estimates the cost of a realistic evaluation by ~4x**;
its per-slice figure is only meaningful once `S*M` is large enough to amortise
the fixed term (S=288 rows above), and `recommended_m_train` should be read
with that in mind.

**Caveat on `bytes_per_slice`.** Measured 1,583,808 B versus an analytic
153,600 B -- 10x higher than the model. The analytic value counts one
`nx*ny*itemsize` field; the measured peak includes the dozen Krylov work
vectors and allocator fragmentation. The driver uses the *measured* value for
sizing, which is why `recommended_m_train` stayed at 40 rather than the
analytic sizing suggesting far more headroom.

### What must be measured on your runtime

Throughput claims have to come from the GPU itself, so `SCRIPTS/bench_backend.py`
exists instead of a table in this file. On the Colab runtime of your choice:

```python
!python SCRIPTS/bench_backend.py --device cuda --dtype float32 --out out/bench_gpu.json
```

It prints (a) the cost of one SAA evaluation for both backends, (b) the cost of
a full compass sweep, (c) the relative difference in the objective (so a
"speedup" cannot come from a wrong answer), and (d) a projection of the full
`optim.max_evals` budget. The verdict line says whether the GPU pays for itself
at that size -- if it does not, increase `nx*ny`, `time.nsteps` or `saa.M_train`
rather than upgrading the subscription.

Rule of thumb, from the structure of the work:

* `splu` cost per solve grows **superlinearly** with `nx*ny` (fill-in) while the
  stencil matvec grows **linearly** in `nnz`. The crossover therefore moves
  towards the GPU as the grid grows: the batched backend is already competitive
  at `80x40` (`3.2e3` cells) on paper, and clearly ahead at `160x80`
  (`1.28e4`, the `benchmark_160x80.yaml` geometry).
* batching pays off in proportion to `nsteps * M * S`. At `--quick` sizes
  (450 cells, `nsteps=40`, `M=8`) a GPU will look useless -- do not judge a
  subscription on the quick run.
* kernel-launch overhead sets a floor of roughly `15 ms x nsteps x iterations`
  per batched call, which is why the recommendation is "measure, do not
  extrapolate".

## 5. Remaining NumPy-only work

Honest scope list -- these still run on the CPU:

* `run_2d.py` (single deterministic forward/optimize) -- a handful of solves,
  dominated by process start-up.
* Verification gates O1-O4, by design.
* `stoch/adjoint.py` (NumPy discrete adjoint) and therefore `SLSQP`.
* Random-field generation (`KLExpansion`) -- one `eigh` of an `nx*nx` matrix per
  axis, milliseconds.
* Plotting/reporting.

If a future run becomes bottlenecked in the discrete adjoint (SLSQP at large
`M`), the same stencil machinery supports it: `Stencil5.apply_transpose` is the
exact transpose of the transport operator (`tests/test_fv2d_torch.py::
TestAdvectionStencil` checks that against the sparse transpose), so the adjoint
recursion is ready to be batched.

## 6. The float32 transport breakdown: found and fixed

At 160x96 the batched float32 transport used to lose the residual entirely for a
few compass candidates, the first sweep returned a non-finite objective, and the
only workaround was float64 (2.8x more expensive per candidate*scenario on a
T4). The cause is now identified and the fix is in `physics/fv2d_torch.py`.

**Root cause: the scalar reductions, not the stencil.** A BiCGSTAB iteration
needs `<rhat, r>`, `||r||`, `||s||` and the `beta`/`omega` ratios, all of which
sum ~1e4-1e5 similar-magnitude terms. In float32 that reduction error is what
decides whether the residual can reach 1e-5 at all, and it is device-dependent:
on a T4 the batched solve then stalls, the per-iteration residual becomes NaN,
and the max-over-batch convergence test poisons every candidate in the batch.

**Fix: promote only the scalars to float64** (`_dot`, `_flat_norm`) while the
work vectors and the stencil matvec stay float32, so the cost centre is
untouched. Module switch `SCALAR_FP64` (env `TESIS_SCALAR_FP64=0`) keeps the old
behaviour for A/B measurement.

Measured on a T4, 160x96, M=40, S=19, nsteps=100, float32, 4 repetitions each
(the failure is deterministic per setting, and reproducible both ways):

| setting | transport residual | converged | finite candidates |
| :--- | :--- | :--- | :--- |
| `TESIS_SCALAR_FP64=0` | `nan` | False (stalled) | 18/19, 4/4 reps |
| `TESIS_SCALAR_FP64=1` (default) | 9.99e-06 | True (33 it/solve) | 19/19, 4/4 reps |

The healthy iteration count is unchanged (33.2 iters/solve with and without the
promotion when the plain path happens to survive), so the fix costs accuracy
nothing and buys the whole float32 mode back:

| T4, 160x96, batched | float64 | float32 (fixed) |
| :--- | ---: | ---: |
| per candidate*scenario | 0.381 s | **0.138 s** |
| one 19-candidate sweep, M=40 | 272 s | 105 s |
| one 19-candidate sweep, M=100 | 680 s | ~262 s |

An end-to-end float32 optimization run at M_train=M_test=M_holdout=40 now
completes with finite objectives, passing gates and a clean metrics file
(527.9 s, 105 s per sweep).

**Scope limit, measured.** The promotion fixes M=40 but not M=100: a float32 run
at M=100 (batch of 1900 systems) still lost one of 19 candidates to a NaN
residual after 28 iterations, i.e. at that batch size the float32 recurrence
itself, not just the reductions, is not accurate enough. So the honest rule is

| batch M (160x96) | float32 |
| :--- | :--- |
| <= 40 | usable (4/4 clean runs, 2.8x cheaper than float64) |
| 100 | still unsafe -- use float64 |

Going further would mean keeping the whole BiCGSTAB recurrence (vectors p, v, s,
t and rho/alpha/omega) in float64 and using the float32 stencil only as a
preconditioned operator; that is a bigger change and is not done.

The failure was nevertheless *cheap* this time: the warning fired, and
`_finite_or_raise` killed the run inside the first sweep (8 min of VM) instead of
letting it burn 45 min on NaN objectives -- which is what the guard is for.

### Mixed-precision recurrence: 2.5x, trajectory-identical (M=100)

The next step up from promoting the scalars is promoting the *recurrence*:
`MIXED_KRYLOV` (env `TESIS_MIXED_KRYLOV=0` to disable) keeps `x, r, rhat, v, p,
s, t` and `rho/alpha/omega/beta` in float64 while casting only each matvec input
down to float32 and the result back up -- so the stencil apply stays float32 and
the ~8 work vectors pay about 10% extra elementwise traffic.

Measured on a T4 at 160x96, M=100, S=19, nsteps=100, float32:

| | per candidate*scenario | one 19-candidate sweep |
| :--- | ---: | ---: |
| float64 | 0.381 s | ~700 s |
| float32, fp64 scalars only | 0.138 s | -- (NaN at M=100) |
| float32, mixed recurrence | **0.152 s** | **288 s** |

and the quality check that matters: the mixed float32 run reproduced the float64
best-value trajectory **bit-for-bit** over its first four sweeps
(`1.04938e6 -> 1.00375e6 -> 958125 -> 912500`, 76 evaluations), with no
non-finite objective -- the exact sequence the float64 T4 and A100 runs produced.

**Completed end-to-end.** Run `20261120-111849` is the same job as the float64
certifying run (`M_train=100`, `M_test=40`, `M_holdout=40`, 1 start, 400
evaluations) in mixed float32:

| | float64 (`20261118-114831`) | mixed float32 (`20261120-111849`) |
| :--- | ---: | ---: |
| wall clock | 4.14 h (14,756 s) | **1.67 h (5,996 s)** |
| evaluations | 401 | 401 |
| q* | `[0,9.72,5.86,0,3.38,0,0,0,0]`, 3 wells | `[0,9.33,8.16,0,0,0,0,0,0]`, 2 wells |
| J (train/test) | 433,945 / 434,008 | **428,592 / 428,579** |
| CVaR-5% G | +0.018 / -0.007 | +0.005 / +0.009 |
| reliability | 0.990 / 0.975 | 0.990 / 0.975 |
| holdout | 1.000 | 0.975 |
| mass balance | 4.2e-11 | 2.3e-05 |
| gates | all pass | all pass |

So the mixed path is **2.46x faster with no loss of quality**, and the two runs
agree to 1.2% in cost while picking *different well subsets*. That is a result in
its own right: near the optimum the objective is flat along the direction
"which two or three of the nine wells carry the duty", so multi-start here is a
search over near-equivalent policies rather than over cost.

Two honest caveats. (1) The float32 mass balance is 2.3e-05 versus 4.2e-11 --
that is the fp32 accumulation floor, and it passes the 1% gate with five orders
of margin but means mass-conservation *identities* must stay float64. (2) The
float32 search accepted/rejected on a noisier objective, so it converged at
`nit~21` instead of `nit=28`: fine-step compass decisions are decided by
differences of order the float32 noise, which is why it landed on the other
policy. Use float64 when you need to *reproduce* a specific trajectory.

**Update (2026-11-21): still not safe at M=100 over a long search.** Run
`20261121-111857` used the mixed path and lost a candidate 174 evaluations in, at
start 2 iter 7 -> 8: `worst residual nan, 51 iterations, stalled=True`, then
`RuntimeError: objective is not finite for candidate(s) [5]`. Start 1 had already
completed 250 clean evaluations, so this is not the "first sweep" failure the
promotion fixed -- it is a low-probability NaN that a 400-evaluation run survived
(`20261120-111849`) and a 500-evaluation one did not. Treat mixed float32 at
M=100 as *usually* fine, occasionally fatal: screening yes, anything you will
report no.

Two consequences in the driver. (1) `--zip-results` now zips in a `finally`: this
run uploaded nothing, because the zip was written only after a *successful*
dispatch, so the completed start 1 died with the VM. (2) `--allow-nonfinite`
(opt-in, off by default) turns one diverging candidate into a rejected move --
`+inf` can never pass the compass acceptance test `fc < best - 1e-12` -- while an
entire non-finite sweep stays fatal, because that is the batched solve diverging
and not one unlucky candidate.

**A second fix that mattered as much.** The first float32 run finished its
optimization and then died in `plot_distributions`: `np.histogram` refuses to
build 20 bins over the tiny data range of float32 objective values, and that
`ValueError` aborted the run *after* the search succeeded and *before*
`metrics.json` was written, i.e. an entire GPU run produced no numbers. The plot
helper now casts to float64 first (`tests/test_report_float32.py`).

## 7. Three complete runs on the 160x96 hydrology (T4, float64)

Section 6 says float64 is the only trustworthy mode at this geometry; these runs
prove it end to end and then isolate the *statistical* limit that a GPU does not
fix. All use `configs/opt/hydrology_gpu.yaml` (ly 120 m, `sigma_G=0.65`,
`ell=15 m`, 160x96 = 15,360 cells, dx = dy = 1.25 m), `--opt-backend torch
--opt-dtype float64`, M_test 40, seeds train 7000 / test 700000, and the same
9-well layout.

| run | M_train | budget | wall | q* (m3/d) | J (train) | test reliability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `20261117-171411` | 16 | 2 x 126 evals (7 sweeps) | 26 min | `[25.5,0.3,4.9,4.0,6.7,35.1,12.0,3.8,9.8]`, 102.1 | 737,707 | 1.000 |
| `20261118-041443` | 16 | 2 x 300 evals (19/18 sweeps) | 60 min | `[0,0.3,4.9,4.0,6.7,0,0,0,0]`, 15.9 | 423,264 | **0.925 (gate FAILS)** |
| `20261118-052228` | **100** | 1 x 200 evals (12 sweeps) | 2.15 h | `[0,9.72,9.72,9.72,9.72,0,0,9.72,9.72]`, 58.3 | **577,917** | **1.000 (all gates pass)** |
| `20261118-114831` | **100** | 1 x 400 evals (**28 sweeps, converged**) | 4.14 h | `[0,9.72,5.86,0,3.38,0,0,0,0]`, 19.0 | **433,945** | **0.975, holdout 1.000 (all gates pass)** |
| `20261120-111849` | **100** | 1 x 400 evals, **float32 mixed** | **1.67 h** | `[0,9.33,8.16,0,0,0,0,0,0]`, 17.5 | **428,592** | **0.975, holdout 0.975 (all gates pass)** |

The first two runs found the same first four well rates; the longer search then
shut the large pumps off, cutting J by 43% — admissible only because the
16-scenario tail of the *training* set said so. That policy does not survive
held-out data. The 100-scenario run instead settled on six wells at 9.72 m3/d
each, J = 577,917, plume peak 42.1 -> 0.446 mg/L on the plotted scenario, and it
passes every gate (O0, gap 1e-12, mass balance 7.7e-11, reliability 1.000).

### The converged policy is certified, and it is barely more expensive

Run `20261118-114831` is the one that answers the research question. The compass
step went 12.5 -> 6.25 -> 3.12 -> 1.56 -> 0.78 -> 0.39 -> 0.195 -> 0.098 and the
loop stopped at `nit=28` with 401 evaluations (4.14 h on a T4), i.e. it converged
rather than running out of budget:

| | value |
| :--- | :--- |
| q* | `[0, 9.72, 5.86, 0, 3.38, 0, 0, 0, 0]` m3/d = 18.96 total, 3 active wells |
| J | 433,945 train / 434,008 test (gap 0.015%) |
| CVaR-5% G | +0.018 train / -0.007 test |
| reliability | 0.990 train / **0.975 test** (gate needs >= 0.94) |
| cleanup time | 36 d train / 91 d test |
| holdout | 3 in-run checks, all reliability **1.000** (min = 1.000) |
| mass balance | 4.2e-11 |
| gates | all pass, including the new `holdout_reliability` gate |

Two things matter here. First, the solution sits exactly on the constraint
(`CVaR ~ 0` from both sides), which is what an optimum of a chance-constrained
problem must do - this is not a policy that happens to be safe, it is the
cheapest one that is safe. Second, **the certified policy costs only 2.5% more J
than the uncertifiable one**: the M=16 run's 423,264 buys a policy that fails
out of sample (0.925 on its own test set), while 433,945 is certified by 100
training scenarios, 40 held-out test scenarios and 3 in-run holdout checks.
Cheap-but-fragile was never actually cheap.

This also settles the sample-size question quantitatively: 400 evaluations per
start at M_train=100 is enough to converge at this geometry, and the holdout
guard (40 scenarios, `optim.holdout_every=8`, 14 s per check on a T4) is what
would have caught the failure mode while the run was still alive.

### Independent verification (`SCRIPTS/check_tail.py`)

A run's own `metrics.json` should not be the only witness, so every policy is
re-evaluated with the NumPy reference backend on fresh scenarios:

| policy | recomputed J | CVaR-5% G | reliability | run's own rows |
| :--- | :--- | :--- | :--- | :--- |
| `20261118-041443` q* (M=16, 300 evals) | 422,900 (M=100) | **+1.920** | 0.950 (5/100) | train -0.045 / 1.000, test +1.863 / 0.925 |
| `20261118-052228` q* (M=100, 200 evals) | 577,916.7 | **-1.000** | 1.000 (0/100) | train and test agree exactly |
| `20261120-111849` q* (M=100, 400 evals, float32) | 428,509 (M=400) | **+0.616** | 0.978 (9/400) | train 428,592 / 0.990, test 428,579 / 0.975 |

The third row is the certified 160x96 policy of section 7, re-derived on seeds
900,000+ -- scenarios no run ever touched. Its J reproduces to 0.02%, its
reliability clears the 0.94 gate with a Wilson interval of [0.958, 0.988], and its
CVaR tail is *not* the +0.005 the run reported: the run measures its own training
tail, a fresh draw puts the mean of the worst 5% at +0.6. Both readings are
correct and they answer different questions, which is why `--seed-base` exists.

Two things worth reading off this table. The M=100 policy is compliant at every
sample size checked (M = 16, 20, 40, 100: zero violations), so it is certified
rather than lucky. The M=16 policy sits *on* the boundary and the answer flips
with the sample: its first 16 training scenarios give CVaR -0.045 and zero
violations (which is exactly what the run reported - the run was right), its
40 held-out scenarios give 0.925 reliability (gate fails at < 0.94), and 100
fresh scenarios give CVaR +1.920 with 5 violations, i.e. 0.950 - back above the
gate. A policy whose certificate depends on which 40-100 scenarios you draw is
not certifiable; that is the real argument for `M_train >= 100`, stronger than
"the M=16 run overfit".

**A verifier bug worth remembering.** The first version of `check_tail.py`
(and `probe_krylov.py`) rebuilt the grid by hand as `Grid(ny, nx, lx, ly)`, but
`physics.fv2d.Grid` is `(nx, ny, lx, ly)`. The result was a transposed 160x96
grid with dx = 2.08 m / dy = 0.75 m instead of 1.25 m / 1.25 m, `n_kl` 1392
instead of 1529, and a completely different scenario set - which made a correct
run look unreproducible for an afternoon. Both scripts now import
`run_optimization.make_grid/make_kl/base_aquifer/make_forward/make_cost`
instead of re-deriving them, and the M=16 run reproduces to the last digit
(`J 423,264.1`, `CVaR -0.045`, first 16 scenarios). Rule: a verification tool
must reconstruct the run through the run's own constructors, never by hand.

### Planning rules that came out of these runs

1. **Budget first.** At 160x96 the M=100 search needs 28 sweeps to converge
   (401 evaluations, step 12.5 -> 0.098); at 200 evaluations it was still taking
   full 12.5 steps. Budget `max_evals >= 400` per start, and note the sweeps get
   cheaper as the step shrinks (~700 s at 12.5, ~310 s at 0.78) because most
   moves hit the capacity bounds.
2. **Then scenarios.** `alpha*M_train >= 5`, otherwise the chance constraint is
   fitted to one realization. At `alpha=0.05` that means `M_train >= 100`; 16
   (`alpha*M = 0.8`, i.e. the degenerate one-sample tail) produced a policy whose
   certificate flips between 0.925 and 0.950 depending on the draw, and the
   paper config's 40 (tail of 2) is thin.
3. **Judge adequacy on the constraint, not on J.** `rel_se_J` was 0.02-0.05% in
   every run (the J spread collapses once the captured mass saturates), so the
   O2 Monte-Carlo gate passes even when the CVaR estimate is badly wrong.
4. **Cost scales with M, not with S.** A 19-candidate sweep took 105 s at M=16
   and 680 s at M=100 (6.5 s per evaluation per 100 scenarios); the batch is
   free, the scenarios are not.

float32 at this geometry was broken when these runs were made. Section 6 now
records the fix (float64 accumulators for the Krylov scalars), which restores
float32 at 0.138 s per candidate*scenario instead of 0.381 s. Both runs above
were float64 and are unaffected; the same queue in float32 is ~2.8x cheaper.
