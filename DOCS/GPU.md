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
| `SCRIPTS/bench_backend.py` | measures both backends on *your* runtime and prints a verdict |
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

### Batched scaling (CPU, float64, 32x16 grid, 24 steps, 8 scenarios)

| candidates `S` | wall clock | Krylov iterations/solve |
|---|---|---|
| 1 | 0.69 s | 17.2 |
| 2 | 0.73 s | 17.9 |
| 4 | 1.43 s | 18.5 |
| 8 | 4.40 s | 19.1 |

The important column is the middle one: **8 candidates cost ~6x one candidate,
not 8x, and the iteration count barely moves.** That sub-linearity is what a
GPU converts into throughput -- one compass sweep (18 candidates at the
benchmark's 9 wells) becomes a single large kernel batch instead of 18 serial
`M`-scenario Python loops.

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

Rule of thumb, from the structure of the work: the batched backend wins when
`nx*ny >= 10^4` and `nsteps*M >= 10^3`, i.e. exactly the full-benchmark
geometry (`80x40`, `nsteps=100`, `M_train=40`). At the `--quick` sizes the grid
is 450 cells and the GPU will look useless; do not judge Pro on the quick run.

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
