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
`RUNTIMES.txt`. The batched backend is ~4x slower here because this box has no
accelerator.

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

**A full `M=40` evaluation: 7.3 s if extrapolated from `M=8`, ~1.9 s measured
directly.** 18 candidates x 40 scenarios x 10.16 ms extrapolates to 7.3 s, but
that 10.16 ms/slice was itself derived at `M=8` where a 1.5 s fixed overhead is
amortised over only 144 slices. Measuring at `M=40` directly (see "It is the
fixed overhead" below) gives **1.86 s**. The extrapolated budgets below are
therefore conservative upper bounds; the real ones are ~4x smaller.

| run | evaluations | T4 wall clock |
|---|---|---|
| bounded (1 start, 200 evals) | 200 | **24 min** |
| paper (3 starts x 1500 evals) | 4500 | **9.1 h** |

Compare **like for like**, at `M=8` and the same 80x40 geometry: the NumPy
`splu` backend takes 69 s for a sequential compass of 18, the T4 takes 1.46 s
for the same sweep batched -- a **47x** speedup. At the `M=40` the optimizer
actually uses, one evaluation is 7.3 s on the T4; the CPU figure it replaces is
the 33.2 s/eval recorded in `runs/runtimes.jsonl` at 160x80 (contended, and a
different geometry, so treat that ratio as indicative rather than exact).

So the GPU is worth having, but it does not put the paper run inside one
session: 9.1 h exceeds a T4 session, which needs `--max-seconds` splitting or
`SCRIPTS/multi_colab.py` across several VMs.

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
run from **2.3 h to 0.45 h on the A100** -- a 5x win from software, against the
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
