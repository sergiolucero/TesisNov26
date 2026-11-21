# SCRIPTS — usage (scripts in `SCRIPTS/`, this file in `DOCS/`)

Every runnable entry point in the project lives in `SCRIPTS/`; this file and
the rest of the write-ups live in `DOCS/`. The repository root holds only
`README.md`, the `physics/` and `stoch/` packages, `configs/`, `tests/` and
the generated `out/` / `figs/` / `runs/` trees.

All paths below are relative to the **repository root**, which is what the
scripts assume:
`SCRIPTS/<script>.py` adds `..` to `sys.path` itself, so `import physics` and
`import stoch` work regardless of where you launch it from, but the *default*
data paths (`out/dataset.npz`, `figs/`, `runs/…`) are relative to your cwd.
Run from the root unless you are passing absolute paths.

---

## 0. Interpreter

Use a Python that actually has torch. On the original measurement host that was
`/root/base/bin/python3`; on a normal desktop it is usually `python3` in a venv
with `pip install -r requirements.txt`.

```bash
python3 -c "import torch, numpy, scipy, yaml, matplotlib; print(torch.__version__)"
```

Prefix long jobs with `python3 -u` so progress lines reach the log immediately
(`HUMAN.md` in this directory, gotcha #4 — a killed run otherwise shows nothing).

---

## 1. At a glance

| Script | Purpose | Runs in |
|---|---|---|
| `hydrology.py` | 1D advection–dispersion scenarios + FNO dataset + Ogata–Banks reference figures | seconds – minutes |
| `solve_1d_pinn.py` | 1D PINN solver (Adam → L-BFGS, soft/hard BC) | minutes (GPU: seconds/minutes) |
| `solve_1d_fno.py` | 1D Fourier neural operator over the hydrology dataset | minutes |
| `inspect_residual.py` | PDE-residual maps of the PINN at init and after training | minutes |
| `run_2d.py` | 2D forward solve **and** the stochastic optimizer | 6 s – 20 s (quick) |
| `run_optimization.py` | Gate suite (O0–O4) and SAA/CVaR optimization workflow (`--backend numpy\|torch`) | 2 s – hours |
| `plot_policy_plume.py` | Re-render the contamination figure from a stored `policy.json` (no rerun); `--all` sweeps every test scenario | seconds – minutes |
| `plot_plume_movie.py` | Animate `C(x,y,t)` as GIF/MP4 from `forward_fields.npz` or a stored policy run | seconds – minutes |
| `bench_backend.py` | Measure NumPy vs batched-GPU backend on *this* runtime and print a verdict | minutes |
| `libcolab.py` | Notebook helper: `modrun(nx, ny)` → `run_2d.py --mode optimize` | as `run_2d.py` |
| `nruns.py` | Loop `run_2d.py --quick` over N seeds | minutes |
| `seed_sweep.py` | Run any command across seeds, report median/IQR, compare arms | per command |
| `runtimes_report.py` | Read `runs/runtimes.jsonl` as a table or a per-script summary | instant |
| `collab_run.py` | GPU/Colab dispatcher with `smoke`/`quick`/`full` presets and `--gpu/--nx/--ny` | preset-dependent |
| `new_high_colab.py` | Same stages tuned for the **high-end** Colab GPUs (A100/H100/G4): measures the card, batches the transport sweep, drives `--backend torch` | minutes – hours |
| `run_colab.sh` | Shell wrapper: GPU check + pip install + `collab_run.py` | preset-dependent |
| `multi_colab.py` | Drive **several** Colab VMs from your laptop for large runs (`logs --source remote` reads a live run's VM log) | hours |
| `app.py` | Flask viewer: figures, plots, fleet health and Colab run status (`/`, `/plots`, `/fleet`, `/runs`, `/api/colab`) | long-running |
| `colab_status.py` | Read-only status of current/recent Colab runs (`--json`, `--live-logs`; backs `/runs`) | seconds |
| `run_app.sh` | start/stop/status for `app.py` (port 5001) | — |
| `deploy_do.sh` | Provision a DigitalOcean droplet, rsync the repo, run the job, fetch results | minutes – hours |
| `remote_run.sh` | The script `deploy_do.sh` executes **on** the droplet | as `SIM_MODE` |
| `fleet_bootstrap.sh` | Idempotent node setup (Tailscale + Herdr + env) | minutes |
| `fleet_status.sh` | Health probe (`--json` for the dashboard) | seconds |
| `fleet-tag.sh` | Tag fleet nodes | seconds |
| `gh_setup.sh` | Git/GitHub auth + remote setup on a node | seconds |
| `psums.py` | Trivial CPU benchmark (compute-unit sanity check) | seconds |
| `psums_test.sh` | `colab --auth adc run --gpu <variant> SCRIPTS/psums.py` wrapper | seconds |
| `generate_sample_dataset.py` | Writes a synthetic `out/dataset.npz` (placeholder data) | instant |
| `generate_structured_sample_dataset.py` | Same, with a proper `scenario{i}_*` layout | instant |
| `fno10_detailed_error_plots.py` | Error-panel figure **with placeholder arrays** — edit before use | instant |

---

## 2. The 1D pipeline

### 2.1 `hydrology.py` — data

Analytical 1D ADE (Ogata–Banks) plus a heterogeneous numerical dataset for the
FNO. Produces figures, one rich single-scenario archive, and a multi-scenario
dataset.

```bash
# defaults: L=100 T=500 v=0.1 D=0.5 C0=1, single scenario 256x256,
#           dataset 128x128 over 32 scenarios
python3 -u SCRIPTS/hydrology.py

# what the presets actually use (small)
python3 -u SCRIPTS/hydrology.py --nx 64 --nt 64 --dataset-nx 64 --dataset-nt 64 \
    --n-scenarios 8 --dataset out/small/dataset.npz --figdir out/small/figs
```

| Flag | Default | Meaning |
|---|---|---|
| `--nx`, `--nt` | 256, 256 | grid of the single-scenario demo archive |
| `--dataset-nx`, `--dataset-nt` | 128, 128 | grid of the **multi-scenario dataset** (this is the FNO's input) |
| `--n-scenarios` | 32 | number of scenarios in the dataset |
| `--L --T --v --D --C0` | 100 500 0.1 0.5 1 | physical/numerical settings |
| `--scenario --dataset --figdir` | `out/scenario.npz`, `out/dataset.npz`, `figs` | outputs |

The two grid flags are the ones people get wrong: `--nx/--nt` size the demo
plot, `--dataset-nx/--dataset-nt` size the dataset that the FNO trains on.

### 2.2 `solve_1d_pinn.py` — the PINN

```bash
# reference run (what the paper figures use)
python3 -u SCRIPTS/solve_1d_pinn.py --hard-bc --eps 1e-3 \
    --outdir out/pinn_hardbc --save-model out/pinn_hardbc/model.pt

# smoke: proves the code path without waiting (~10 s on CPU)
python3 -u SCRIPTS/solve_1d_pinn.py --hard-bc --adam-steps 200 --lbfgs-steps 0 \
    --N-f 256 --N-ic 64 --N-bc 64 --nx-test 64 --nt-test 64 \
    --outdir out/pinn_smoke --save-model out/pinn_smoke/model.pt

# resume Adam from a checkpoint
python3 -u SCRIPTS/solve_1d_pinn.py --resume out/pinn_hardbc/checkpoint.pt \
    --ckpt-every 1000 --outdir out/pinn_hardbc
```

| Flag group | Flags | Notes |
|---|---|---|
| problem | `--pe` (20), `--tau-max` (0.5) | Péclet number and time horizon |
| network | `--hidden` (64), `--layers` (4) | MLP width/depth |
| BC | `--hard-bc`, `--eps` (1e-3) | hard ansatz satisfies IC/outlet exactly |
| schedule | `--lr` (1e-3), `--adam-steps` (12000), `--lbfgs-steps` (200) | |
| sampling | `--N-f` (1500), `--N-ic` (150), `--N-bc` (150), `--N-f-fix` (3000) | collocation / boundary points |
| losses | `--w-pde --w-ic --w-bc` (1, 10, 10) | weights |
| tau ramp | `--tau-ramp --ramp-tau 0.05 --ramp-power 1.0` | curriculum on the time window |
| eval | `--nx-test` (256), `--nt-test` (256) | only the reported grid |
| io | `--outdir` (`out/pinn`), `--save-model`, `--ckpt-*`, `--config`, `--device`, `--dtype`, `--seed` | CLI flags beat `--config` values |

Outputs: `history.json`, model, plots, and a `metrics` block used by
`seed_sweep.py` (e.g. `metrics.regions.full`).

### 2.3 `solve_1d_fno.py` — the FNO

Trains on the `hydrology.py` dataset. The grid comes from the dataset; the only
resolution knob you own is `--coarse-factor` (resample *down*).

```bash
python3 -u SCRIPTS/solve_1d_fno.py --dataset out/dataset.npz \
    --epochs 300 --modes1 12 --modes2 12 --width 20 \
    --outdir out/fno --save-model out/fno/model.pt
```

Two ablations worth knowing, because both were real bugs once:
`--no-coords` drops the `(xi, tau)` input channels — without them the operator
only sees a *constant* input, has only the DC Fourier mode and cannot represent
a front at all; `--no-normalize` drops per-channel normalization.

### 2.4 `inspect_residual.py` — is the PINN actually solving the PDE?

Plots `|R|` at initialization and after training, so a model that reports a
nice loss but violates the PDE is visible.

```bash
python3 -u SCRIPTS/inspect_residual.py --hard-bc --eps 1e-3 \
    --model out/pinn_hardbc/model.pt --nx 128 --nt 128 --outdir out/residual
```

---

## 3. 2D forward and stochastic optimization

### 3.1 `run_2d.py` — single 2D case

```bash
python3 -u SCRIPTS/run_2d.py --quick                     # forward + optimize, small
python3 -u SCRIPTS/run_2d.py --mode forward --no-plot    # one forward solve
python3 -u SCRIPTS/run_2d.py --mode optimize --config configs/opt/benchmark.yaml \
    --out out/run2d/bench
```

`--quick` shrinks the grid and scenario counts (30x15, few SAA samples);
`--rate` sets the pumping as a fraction of `wells.q_max`; `--seed` picks the
conductivity realization.

### 3.2 `run_optimization.py` — the workflow + gates

```bash
python3 -u SCRIPTS/run_optimization.py --quick                 # ~15 s sanity run
python3 -u SCRIPTS/run_optimization.py --stage gates           # ~75 s on the full benchmark
python3 -u SCRIPTS/run_optimization.py --stage opt --nx 40 --ny 20 \
    --out out/opt/small --config configs/opt/benchmark_small.yaml
```

| `--stage` | What it does | Cost (full benchmark) |
|---|---|---|
| `o0` | deterministic forward + Thiem/velocity checks + well layout | ~0.1 s |
| `o1` | KL random-field checks | seconds |
| `o2` | objective/mass-balance checks | seconds |
| `o3` | discrete-adjoint gradient vs finite differences (`rel error <= 1e-4`) | ~10 s |
| `o4` | SAA vs CVaR reformulation comparison | ~30 s |
| `gates` | O0–O4 with pass/fail summary | **74 s** (measured) |
| `opt` | the actual multi-start SAA optimization | **> 30 min; 19–33 h** at 3 starts × 1500 evals (**minutes** with `--backend torch` on a GPU) |
| `all` | everything above | hours |

The `opt` stage writes `figures/well_layout.png`, `convergence.png`,
`plume_scenario0.png`, `test_distributions.png` and `contamination.png` — the
last one is the one to show anyone who asks what the policy buys: the same test
scenario with every well **off** (peak concentration, source, compliance and
observation points, `creg` contour), the same field **under `q*`** with each
active well labelled in m³/d, and the breakthrough curves at the compliance
point (untreated dashed, treated solid) against `creg` with the compliance
window shaded. Each field panel keeps its own colour scale — a shared one makes
the cleaned panel look empty — and the peaks are stated in the titles and the
supra-title. Note the field is zero at `t = 0` by construction (the source
injects over time), so "untreated" means the incoming state at the end of the
horizon, not the first time step. `SCRIPTS/plot_policy_plume.py --run <dir>`
re-renders the identical figure from a stored `policy.json` without rerunning
the optimization.

`--nx` / `--ny` override `domain.nx`/`domain.ny` **after** the config is loaded,
so they beat `--quick`'s built-in 30x15. The effective grid is recorded in
`metrics.json` (`config.domain`), which is how you verify what actually ran.

#### Backend: NumPy reference vs batched GPU

Every stage runs on the SciPy reference by default. The `opt` stage can
instead use the batched torch backend, which is where a GPU finally earns its
keep (see [`GPU.md`](GPU.md) for the audit that motivated it):

```bash
# GPU: one compass sweep per kernel batch, float32 throughput mode
python3 -u SCRIPTS/run_optimization.py --stage opt \
    --config configs/opt/benchmark.yaml \
    --backend torch --device cuda --dtype float32

# CPU: same backend, same answer, but slower than SciPy -- useful for parity
python3 -u SCRIPTS/run_optimization.py --quick --stage opt \
    --backend torch --device cpu --dtype float64
```

| Flag | Default | Meaning |
|---|---|---|
| `--backend` | `numpy` | `numpy` (SciPy `splu`) or `torch` (batched Krylov, GPU) |
| `--device` | `auto` | `torch` device: `auto`, `cuda`, `cuda:N`, `cpu`, `mps` |
| `--dtype` | `float64` | working precision; `float32` is the fast screening mode |
| `--krylov-rtol` | per dtype | `1e-11` (float64) / `1e-6` (float32) |
| `--krylov-maxiter` | `4000` | cap per time step |
| `--[no-]batched-search` | on for a GPU | batched compass sweep vs. sequential pattern search |
| `--allow-nonfinite` | off | one candidate whose objective is non-finite is rejected as a move instead of ending the run (a whole non-finite sweep stays fatal); the count lands in `metrics['stages']['opt']['nonfinite_candidates']` |

The choice is recorded in `metrics.json` under `backend` (including the GPU
name, the flow-solve Krylov report and whether every solve converged) and in
`runs/runtimes.jsonl`. The O1-O4 gates stay on NumPy by design: they certify
solver properties, not throughput. `SLSQP`/the discrete adjoint is NumPy-only
and says so if asked otherwise.

Measure first on the runtime you are paying for:

```bash
python3 -u SCRIPTS/bench_backend.py --device cuda --dtype float32 --out out/bench_gpu.json
```

Cost model for the `opt` stage (measured, single-threaded numpy, x86):

```
seconds_per_eval ≈ 1.5e-6 × nx × ny × nsteps × M_train
total wall       ≈ n_starts × max_evals × seconds_per_eval
```

`max_evals` is per start. `optim.method: slsqp` (adjoint gradients, one
`fun()` call per iteration) is typically an order of magnitude cheaper than the
default pattern search, which costs `2 × n_wells` evaluations per iteration.

### 3.3 `plot_plume_movie.py` — the plume evolution movie

Renders the stored `C_all` sequence (or solves transport again for a stored
policy) as an animated GIF. Pillow is enough for `.gif`; `.mp4`/`.mov` needs
ffmpeg on `PATH`.

```bash
# fastest: reuse the fields already written by a forward run
python3 -u SCRIPTS/plot_plume_movie.py \
    --fields out/run_2d/forward/forward_fields.npz

# rebuild the exact test scenario and animate q*
python3 -u SCRIPTS/plot_plume_movie.py --run out/opt_run --fps 12

# same scenario with every well off, for the before/after comparison
python3 -u SCRIPTS/plot_plume_movie.py --run out/opt_run --nopump
```

Output defaults to `<run>/figures/plume_movie.gif` (or `_nopump.gif`). The
colour scale is fixed for the whole movie. The auto scale is the larger of the
final-frame peak and the 99th percentile of all cells, so a one-cell source
spike cannot turn the late-time plume black; clipping is named on the
colourbar. Pass `--vmax` to force a scale. `--stride N` drops frames but always
keeps the final state, and `--scenario` picks a different test realisation.

### 3.4 Notebook helpers

```python
# libcolab.py — rewrite the small benchmark grid and run the optimizer
from SCRIPTS.libcolab import modrun      # or: python3 SCRIPTS/libcolab.py 32 16
modrun(32, 16)
```

`nruns.py` is the blunt version: `run_2d.py --quick` for seeds 10..14
(`N=2 python3 SCRIPTS/nruns.py` to change the count). For anything you intend to
report, use `seed_sweep.py` instead.

---

## 4. Experiment harnesses

### 4.1 `seed_sweep.py` — seeds, medians, arm comparison

Runs a command once per seed, reads one scalar from each run's JSON, and prints
per-seed values plus median/IQR/min-max, then a paired comparison across arms.

```bash
./SCRIPTS/seed_sweep.py --arm base --seeds 0-4 --metric metrics.regions.full \
    --outdir runs/sweeps/base -- \
    python3 -u SCRIPTS/solve_1d_pinn.py --hard-bc --eps 1e-3 \
        --adam-steps 10000 --lbfgs-steps 250 \
        --outdir {outdir} --save-model {outdir}/model.pt

# two arms, compared on shared seeds
./SCRIPTS/seed_sweep.py --seeds 0-4 --metric metrics.regions.full \
    --outdir runs/sweeps/cmp \
    --arm base -- python3 -u SCRIPTS/solve_1d_pinn.py --hard-bc --eps 1e-3 \
        --outdir {outdir}/{arm}/s{seed} --save-model {outdir}/{arm}/s{seed}/model.pt \
    --arm ramp -- python3 -u SCRIPTS/solve_1d_pinn.py --hard-bc --eps 1e-3 \
        --tau-ramp --ramp-tau 0.05 --ramp-power 1.0 \
        --outdir {outdir}/{arm}/s{seed} --save-model {outdir}/{arm}/s{seed}/model.pt
```

Placeholders: `{seed}`, `{arm}`, `{outdir}`, `{manifest}`. `--seed N` is appended
automatically when the command has no `{seed}`/`--seed`. Use `--timeout` for a
per-run wall clock and `--continue-on-error` to survive a bad seed. Exit status:
0 all runs ok, 1 a run failed.

### 4.2 `collab_run.py` — one Colab/GPU session

```bash
python3 SCRIPTS/collab_run.py --models smoke --preset smoke --gpu cpu   # verify the box
python3 SCRIPTS/collab_run.py --models pinn --preset quick --device cuda
python3 SCRIPTS/collab_run.py --models dataset fno --preset full --fno-epochs 300
python3 SCRIPTS/collab_run.py --models opt --preset full --opt-stage gates
python3 SCRIPTS/collab_run.py --models pinn --gpu 1 --nx 512            # second GPU, 512-point eval grid
python3 SCRIPTS/collab_run.py --models opt --nx 40 --ny 20 --opt-config configs/opt/small.yaml
```

| `--preset` | dataset (nx×nt, scenarios) | PINN (Adam steps, points) | FNO (epochs) | opt |
|---|---|---|---|---|
| `smoke` | 64×64, 8 | 100 Adam, small sampling | 3 | stage `o0` when `smoke` is in `--models` |
| `quick` | 128×128, 16 | 1500 Adam | 50 | `--quick` |
| `full` | 256×256, 32 | 12000 Adam | 300 | full config |

`--models` accepts `dataset pinn fno opt tests` plus the aliases `smoke`, `all`
(`dataset+pinn+fno+opt`) and `both` (`pinn+fno`); they run in that fixed order.
`--nx/--ny/--nt` override the grid for whatever runs: for 1D they size the
dataset *and* the PINN eval grid (`--ny` is the time axis; `--nt` wins over it),
for the 2D optimizer they override `domain.nx/domain.ny`.
`--gpu` takes an index (`0`), `cuda:1`, `cpu`, `mps` or `auto`; it sets both
the torch `--device` and `CUDA_VISIBLE_DEVICES` for the subprocesses.
`--no-quick-opt` forbids `--quick` from reaching the optimizer, so an explicit
large grid survives a `smoke`/`quick` preset.

Outputs go to `--out-root` (default `out/colab`), and `--zip-results` packs
`out-root` + `figs` into `out/colab_results.zip`. Works with out-roots outside
the repo (Colab scratch, mounted Drive) — the zip names are relative to each
folder, so entries stay `out/...` and `figs/...`.

### 4.3 `run_colab.sh` — Colab cell wrapper

```bash
# inside a Colab cell
!bash SCRIPTS/run_colab.sh                                   # quick, dataset+pinn
!bash SCRIPTS/run_colab.sh --models fno --preset full --fno-epochs 300
COLAB_MODELS="pinn fno" COLAB_PRESET=smoke bash SCRIPTS/run_colab.sh
```

Env: `COLAB_PRESET` (quick), `COLAB_MODELS` (`"dataset pinn"`), `PYTHON`
(`python3`), `SKIP_INSTALL=1`, `SKIP_GPU_CHECK=1`. Writes `/tmp/tesis-colab.exit`
and `/tmp/tesis-colab.done` for the `ubuntu-hub` dispatcher.

### 4.4 `multi_colab.py` — many VMs, large runs

For jobs that cannot finish before a Colab session is reclaimed. It plans the
run locally (resolving the *effective* config with `stoch.config`), estimates
the cost, bundles the repo, provisions one VM per replica, streams each run's
log home, downloads the artifacts, and stops the VMs.

```bash
./SCRIPTS/multi_colab.py launch --nx 80 --ny 40 --estimate --dry-run   # cost only
./SCRIPTS/multi_colab.py launch --gpu none --nx 80 --ny 40 \
    --opt-stage opt --set optim.max_evals=250 --set optim.n_starts=1 \
    --set saa.M_train=40 --max-hours 6
./SCRIPTS/multi_colab.py launch --replicas 3 --gpu L4 --nx 80 --ny 40 \
    --opt-stage opt --set optim.max_evals=250 --set optim.n_starts=1
./SCRIPTS/multi_colab.py status | logs -f | collect | stop
```

Requires the Colab CLI (`pip install google-colab-cli`) and ADC:
`gcloud auth application-default login` (or a service-account key in
`GOOGLE_APPLICATION_CREDENTIALS`). Every CLI call uses `--auth adc`; override
with `--auth oauth2` (after `colab auth`) or `COLAB_AUTH=...`.
Replica `k` gets `optim.seed = --seed-base + k`; compare `train.mean_J` from
each `r<k>/artifacts.zip`. Run state lives in `runs/colab_multi/<run-id>/`.
Use `--gpu none` for the optimizer (CPU-bound), a GPU only for PINN/FNO.

### 4.5 Runtime records — `runs/runtimes.jsonl`

Every entry point appends **one JSON line per run** to `runs/runtimes.jsonl`
(written by `stoch/runtimes.py`): timestamp, host, script, argv, git revision,
wall clock in seconds/hours, and the fields that identify the run — grid,
`nsteps`, `M_train`, method, starts, budget, evaluations, seconds per
evaluation, outdir, status. It is append-only, one line per run, and safe to
`tail -f`.

```bash
./SCRIPTS/runtimes_report.py                     # last 20 runs as a table
./SCRIPTS/runtimes_report.py --all --summary     # mean wall clock and s/eval per grid
./SCRIPTS/runtimes_report.py --stage opt --since 2026-11-17
./SCRIPTS/runtimes_report.py --json | jq -r '[.grid|join("x"), .s_per_eval] | @tsv'
tail -f runs/runtimes.jsonl                      # while something is running
```

`run_optimization.py` also takes `--runtime-log PATH` (default
`runs/runtimes.jsonl`, `''` disables) so a scratch run can keep its own record;
`multi_colab.py` writes a per-replica `runtimes.jsonl` inside its run
directory. `RUNTIMES.txt` stays the hand-curated table of canonical
measurements — the JSONL is the raw record it is distilled from.

Measured cost per objective evaluation (this log's `s_per_eval`, `M_train=40`,
`nsteps=100`, one core):

| grid | cells | s/eval |
|---|---|---|
| 40x20 | 800 | 0.47 |
| 80x40 | 3 200 | 2.5 |
| 80x80 | 6 400 | ~5.8 |
| 160x80 | 12 800 | ~14 |

That is close to `cells^1.2`, not the linear-in-cells guess, which is why the
old `1.5e-6 * nx * ny * nsteps * M_train` estimate was ~6x pessimistic at
80x80 and can be used only as an upper bound.

### 4.6 `new_high_colab.py` — the A100/H100/G4 path

`collab_run.py` is sized for a T4 and never selects the batched GPU backend of
the optimizer. `new_high_colab.py` profiles the card (compute capability, VRAM,
SM count), maps it onto a class (`v5` = A100, `v6` = H100/Blackwell G4, plus
`ada`/`turing` fallbacks), and runs the same physics with the numbers that
class can actually use: `float32` + a 1e-6 Krylov tolerance, one compass sweep
per kernel batch, and `M_train`/`max_evals` sized from a measured peak-memory
budget instead of guessed.

```bash
python3 SCRIPTS/new_high_colab.py bench forward --verify            # what does this card do?
python3 SCRIPTS/new_high_colab.py opt --assume v5 --preset hi --zip  # paper-scale SAA, retuned
python3 SCRIPTS/new_high_colab.py all --preset paper                 # reproduce the published numbers
python3 SCRIPTS/new_high_colab.py bench --preset smoke --nx 24 --ny 12 --plan  # plumbing, no compute
python3 SCRIPTS/new_high_colab.py --launch --gpu-class v6 --accelerator H100 \
    bench forward opt --zip                                          # rent + ship + run + fetch
```

Stages are `bench` (throughput/VRAM probe → `bench.json`), `forward` (batched
source plume, `--verify` compares against the NumPy/`splu` reference), `opt`
(`run_optimization.py --backend torch` on a card-retuned config), and
`pinn`/`fno` (delegated to `collab_run.py` with card-scaled sizes). Presets:
`smoke` (quick config), `paper` (config values, published numbers), `hi`
(default: card-class scenarios/budget) and `push` (2x grid, 1.5x `nsteps` —
a resolution study, not comparable with the paper). Artifacts land in `--out`
(`report.json`, `tuning.json`, `bench.json`, `forward.npz`, `opt/`), one record
per run goes to `runs/runtimes.jsonl`, and `--launch` reuses the `colab` CLI
(`colab new/upload/exec/download/stop`) so the whole cycle runs from the
laptop. Full card-class table and tuning rationale: the module docstring.

**Order matters, and it costs money if you get it wrong.** `bench` writes
`bench.json`, and that measurement is what sizes `M_train`/`max_evals` for
`opt` — run it first, on the same card, or the later stages fall back to the
analytic estimate. `forward --verify` is the correctness gate: it compares the
torch kernels against the NumPy/`splu` reference, so a wrong kernel cannot hide
behind a large batch.

```bash
# before spending compute units: dry-run prints the exact colab commands
python3 SCRIPTS/new_high_colab.py --launch --gpu-class v6 --accelerator H100 --dry-run bench
# cheap probes, in this order
python3 SCRIPTS/new_high_colab.py bench --preset smoke --nx 24 --ny 12 --nsteps 20
python3 SCRIPTS/new_high_colab.py forward --verify
# long run: cap the remote cell, and remember --keep leaves the VM billing
python3 SCRIPTS/new_high_colab.py --launch --keep --session tesis-high --max-seconds 21600 opt
colab --auth adc url -s tesis-high   # re-attach after a disconnect; `colab --auth adc stop` when done
```

Gotchas verified on this host: `--launch` needs the `colab` CLI on `PATH`
(`~/.local/bin/colab`); `--device auto` is resolved to `cpu` when no CUDA card
is present, so CPU smoke runs work but are ~1 s per transport solve, and `bench`
caps its batch sweep at 4 slices there. Summarise what the log actually says
with `python3 SCRIPTS/runtimes_report.py --summary` (mean wall clock and s/eval
per script/grid), and use `SCRIPTS/bench_backend.py` for the separate question
of whether the torch backend is ahead at all — see [`GPU.md`](GPU.md).

---

## 5. Serving and dashboards

```bash
./SCRIPTS/run_app.sh start         # http://localhost:5001  (/ , /plots, /fleet, /runs, /api/colab)
./SCRIPTS/run_app.sh status
./SCRIPTS/run_app.sh stop
PORT=8080 ./SCRIPTS/run_app.sh start
python3 SCRIPTS/app.py --port 5000            # foreground
```

`app.py` serves the repo's `out/` as static files, `figs/` as `/figs/<name>`,
renders `templates/*.html`, and shells out to `SCRIPTS/fleet_status.sh --json`
for `/fleet` (15 s timeout so the page never hangs). `/runs` reads
`runs/colab_multi/*/plan.json` and `out/high/*` through
`SCRIPTS/colab_status.py`, shows per-replica progress and log tails, and only
touches the network when you ask it to: `?live=1` runs `colab sessions`, and
`?live_logs=1` pulls a live replica's `run.log` off the VM (bounded per
replica, one retry) — without it a run in flight only shows the frozen local
header, since the streaming `exec` returns nothing until the run ends.
`run_app.sh` keeps its pid/log in `SCRIPTS/app.pid` / `SCRIPTS/app.log`.

`/api/colab` (alias `/runs.json`) is the same snapshot as JSON for scripts and
monitors, plus a `pipeline` block whose `state` is one word
(`running`/`idle`/`finished`/`failed`/`stopped`/…):

```bash
curl -sf localhost:5001/api/colab | jq '.pipeline.state, .pipeline.progress_pct'
curl -sf 'localhost:5001/api/colab?limit=1'      # newest run only
curl -sf 'localhost:5001/api/colab?live=1'       # also probe `colab sessions`
curl -sf 'localhost:5001/api/colab?live_logs=1'  # real progress from the VM
curl -sf 'localhost:5001/api/colab?run=<run-id>' # one run, 404 if unknown
curl -sf 'localhost:5001/api/colab/<run-id>'     # full detail: replicas, progress, tails
curl -sf 'localhost:5001/runs/log/multi/<run-id>?replica=0&remote=1'  # live tail
```

A non-200 means the call itself failed (500 → `ok:false` + `error`, 404 →
unknown run id), so `curl -sf` is a valid health check. Replies carry
`Cache-Control: no-store`.

---

## 6. Remote and fleet operations

```bash
# provisioning helper on a fresh node (Tailscale + Herdr + toolchain)
sudo ./SCRIPTS/fleet_bootstrap.sh --role hub
./SCRIPTS/fleet_bootstrap.sh --verify

# health probe, for scripts and the dashboard
./SCRIPTS/fleet_status.sh
./SCRIPTS/fleet_status.sh --json
./SCRIPTS/fleet-tag.sh --help

# DigitalOcean: create droplet, rsync repo, run remote_run.sh, fetch results
./SCRIPTS/deploy_do.sh --sim smoke
./SCRIPTS/deploy_do.sh --sim pinn --gpu --size gpu-l40x1-48gb --keep
./SCRIPTS/deploy_do.sh --ip 203.0.113.10 --sim smoke
```

`remote_run.sh` runs **on** the droplet: it installs apt/venv/torch, then runs
`hydrology.py`, `solve_1d_pinn.py` and/or `solve_1d_fno.py` according to
`SIM_MODE` (`smoke|pinn|fno|all|none`), writing `/tmp/tesis-run.exit` and
`/tmp/tesis-run.done`. `deploy_do.sh` polls those sentinels and rsyncs `out/`
back into `do_results/`.

`herdr-server.service` and `fleet-acl.hujson` are installed by
`fleet_bootstrap.sh`; `gh_setup.sh` configures git/GitHub auth on a node.

---

## 7. Checks and small utilities

```bash
# the whole unit suite (49 tests, ~0.5 s) — also runs via collab_run --models tests
python3 -m unittest discover -s tests -p "test_*.py"
python3 tests/run_tests.py

# compute-unit sanity check on a rented VM
colab --auth adc run --gpu T4 SCRIPTS/psums.py
```

`generate_sample_dataset.py` and `generate_structured_sample_dataset.py` write
placeholder `out/dataset.npz` files (the second one in the real
`scenario{i}_*` layout). `fno10_detailed_error_plots.py` still generates its
arrays with `np.random` — replace the load section before trusting any figure
it produces.

---

## 8. Conventions and gotchas

- **Relative defaults are cwd-relative.** `out/`, `figs/`, `runs/`, and the
  default `--config configs/opt/benchmark.yaml` assume you are in the repo root.
  Scripts resolve their own *imports* from the repo root, not their data paths.
- **`python3 -u`** for anything long; progress lines are flushed but Python's
  stdout is block-buffered through a pipe.
- **Grid flags split by stage.** `--nx/--nt` size the 1D demo; `--dataset-nx/--dataset-nt`
  size the FNO dataset; `--nx/--ny` in `run_optimization.py` override the 2D
  domain, and `--nx/--ny/--nt` in `collab_run.py` fan out to the right script.
- **`--quick` beats your grid**, unless you pass the grid *after* it:
  `apply_quick()` force-sets 30x15 and small SAA counts. `run_optimization.py
  --nx/--ny` is applied after loading, and `collab_run.py --no-quick-opt`
  stops `--quick` from being forwarded at all.
- **A config path that does not exist is now an error** (`stoch.config.load_config`
  raises `FileNotFoundError`) instead of silently running the defaults.
- **`--stage opt` is the expensive one.** Gates first, then `opt`; keep the
  unbuffered log (`tee` it somewhere durable) because the optimizer writes
  `policy.json`/`metrics.json` only when it finishes.
- **Effective configs are recorded.** `metrics.json` carries the loaded config
  (`config.domain`, `config.time.nsteps`, …), so a run documents its own grid.

---

## 9. Wall-clock reference

Measured on a 1 vCPU container (`RUNTIMES.txt`); see that file for the full
table and the caveat that it predates the move into `SCRIPTS/`.

| Command | Time |
|---|---|
| unit suite | 2.1 s |
| `run_2d.py --quick` (forward / optimize / all) | 6 s / 16 s / 22 s |
| `run_optimization.py --quick` | 15 s |
| `run_optimization.py --stage gates` (full benchmark) | 74 s |
| `run_optimization.py --stage opt` (full benchmark) | **> 1800 s** (19–33 h extrapolated) |
