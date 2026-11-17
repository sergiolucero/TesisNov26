# HUMAN.md — read this first when you come back

Welcome back. This is the "what is this, where did I leave off, and what will
bite me" file. The other docs are for the project itself; this one is for you.

## TL;DR

We're debugging **physics-informed neural networks (PINNs) and a Fourier neural
operator (FNO)** for **groundwater contaminant transport** — specifically the
1D advection–dispersion equation (ADE), verified against the Ogata–Banks (1961)
analytical solution.

The 1D sandbox **works**: the PINN matches the analytical solution to ~8e-3
relative L2 in the interior. The remaining error is a measure-zero corner
singularity at (ξ,τ)=(0,0), **not a bug**. We're now moving from "does it work"
to "make it scale / productionize."

## If you read nothing else, read these gotchas

These are the things that already cost real time. Don't re-learn them.

1. **Use `/root/base/bin/python3` for everything.** The default `python3` has
   **no torch**. `python3 -c "import torch"` fails; `/root/base/bin/python3`
   has `torch 2.15.0+cpu`, `numpy 2.5.4`, `scipy`, `matplotlib`.
2. **This box has ~961 MB RAM and NO swap.** The OOM killer has killed
   `python3` here before (see `dmesg`). That's why `inspect_residual.py`
   evaluates the PDE residual in *chunks*. Do not "simplify" the chunking away
   when moving to a big machine — keep it, it costs nothing.
3. **Exit code 143 = killed by SIGTERM, not a crash.** Training runs are capped
   by `timeout 600`. 143 means the 10-minute wall hit (or someone ran `kill`).
   It does **not** mean the math failed.
4. **"No output" = stdout buffering, not silence.** Python block-buffers stdout
   when it's not a terminal, so a SIGTERM kill discards un-flushed prints.
   **Always run with `python -u`** (or `PYTHONUNBUFFERED=1`) for anything that
   might get killed or backgrounded.
5. **The corner singularity is a metric artifact.** The step-inlet problem is
   discontinuous at (0,0): IC says u=0, inlet BC says u=1. The reference
   returns 1 there; any continuous ansatz returns 0. So a bright |error| spot
   at the corner is expected. **Always report the corner-excluded interior
   error alongside the full-grid error.** Don't chase the full-grid number.
6. **CLI flags beat `--config` values** in `solve_1d_pinn.py`. And every run
   writes `run_config.json` into `--outdir` — that file is your reproducible
   record; read it, don't guess.

## Where I left off (as of the last session)

- **Phase 1 (1D sandbox): DONE ✅.** Best result: hard-BC ansatz,
  `s(τ)=1-exp(-τ/eps)`, `eps=1e-3` → **rel-L2 1.35e-2 full / 8.0e-3 interior**.
  Model at `out/pinn_hardbc_exp/model.pt`. See `Actions.md` and `Results.md`.
- **Phase 2 (BC verification): DONE ✅** — the hard-BC sweep found the best
  config; residual-inspection script `inspect_residual.py` was written and
  smoke-tested, but **not yet run at full 256×256 on the trained model**.
- **Growth/portability:** wrote `GROWTH.md` (scaling + dockerization plan) and
  upgraded `solve_1d_pinn.py` with `--device`, `--dtype`, `--config`, and
  `run_config.json` manifest output. It's hardware-agnostic now (cpu/cuda/mps).
- **Not done yet:** Phase 4 (FNO), longer Phase 3 optimizer experiments,
  TensorBoard/gradient logging, and actually building/running the Docker images.
- **Portability progress:** `solve_1d_pinn.py` now has resumable checkpoints,
  SIGTERM checkpointing, `--batch-size`/`--grad-accum`, `--tau-ramp`, and
  config/manifest support. `Dockerfile.cpu`, `Dockerfile.gpu`,
  `docker-compose.yml`, `.dockerignore`, and `runs/best_hardbc.json` exist.

## The files (and what each one is)

| File | Purpose |
|---|---|
| `plan.md` | 4-phase debugging plan (the roadmap) |
| `Actions.md` | Running log of what's been done + results tables |
| `Results.md` | Results table (longer runs) — mostly superseded by Actions.md |
| `GROWTH.md` | How to scale this to bigger hardware + dockerize |
| `HUMAN.md` | this file |
| `hydrology.py` | Generates synthetic data + Ogata–Banks reference solution |
| `solve_1d_pinn.py` | The PINN solver (train + evaluate + plots) |
| `inspect_residual.py` | Plots PDE residual magnitude abs(R) at init and after training |
| `app.py` | Flask figure viewer |
| `run_app.sh` | start/stop/status for the viewer (port **5001**) |
| `formulation.tex` / `.pdf` | Math write-up (ADE, reference, losses) |
| `out/` | All run artifacts (`pinn*/`, models, figures, `dataset.npz`) |
| `figs/` | Data-generator demo figures |

## The commands I actually use

```bash
cd /root/TESIS

# Train the PINN (best config so far):
/root/base/bin/python3 -u SCRIPTS/solve_1d_pinn.py \
  --hard-bc --eps 1e-3 --save-model out/pinn_hardbc_exp/model.pt \
  --outdir out/pinn_hardbc_exp

# Inspect the PDE residual on the trained model (the immediate next step):
PYTHONUNBUFFERED=1 timeout 1200 /root/base/bin/python3 -u SCRIPTS/inspect_residual.py \
  --hard-bc --model out/pinn_hardbc_exp/model.pt --outdir out/residual
# (if too slow/heavy: add --nx 128 --nt 128)

# Generate/refresh data:
/root/base/bin/python3 -u SCRIPTS/hydrology.py

# Viewer (port 5001):
./SCRIPTS/run_app.sh start      # then http://localhost:5001
./SCRIPTS/run_app.sh stop
```

Config-driven run (new):

```bash
/root/base/bin/python3 -u SCRIPTS/solve_1d_pinn.py --config runs/best.yaml
# override a config value on the CLI:
/root/base/bin/python3 -u SCRIPTS/solve_1d_pinn.py --config runs/best.yaml --dtype float64
```

## Conventions

- **Run discipline:** every training run is capped at `timeout 600` (10 min).
  If a run needs longer, raise it explicitly — don't just let it be killed.
- **Reproducibility:** pass `--seed`, use `--config`, and check the
  `run_config.json` that lands in `--outdir` (it records config + git SHA +
  metrics).
- **Metrics:** always report full-grid rel-L2 **and** the corner-excluded
  interior error. The interior is the physically meaningful number.
- **Output:** use `python -u` everywhere; the scripts already line-buffer
  `inspect_residual.py` as a safety net, but don't rely on it.

## Next actions (pick one and go)

1. **Phase 1 wrap-up:** update `formulation.tex` to report the corner-excluded
   error alongside full-grid, recompile the PDF.
2. **Docker smoke test:** build the CPU image on a host with Compose v2
   (`docker compose`; not installed on this box):
   `docker compose --profile cpu run --rm train solve_1d_pinn.py --config runs/best_hardbc.json --adam-steps 1 --lbfgs-steps 0 --nx-test 16 --nt-test 16`.
3. **Phase 3 experiment:** run the hard-BC model with `--tau-ramp` (or a small
   RAR implementation) and compare full/interior errors plus residual figures.
4. **Phase 4 (FNO):** train on `out/dataset.npz`, test Fourier modes /
   resolution invariance / per-channel normalization.
5. **Production hardening:** add TensorBoard/gradient logging and/or MLflow;
   consider a Zarr/HDF5 data path before large FNO datasets.

## One-liner answers to questions you'll probably ask

- "Why is the error ~1.35e-2 but the interior is 8e-3?" → corner singularity,
  metric artifact (gotcha #5).
- "Why did my script die with 143 and print nothing?" → `timeout` SIGTERM +
  buffering (gotchas #3 and #4).
- "Which python do I use?" → `/root/base/bin/python3` (gotcha #1).
- "Where's the best model?" → `out/pinn_hardbc_exp/model.pt`.
- "Why is the residual script so slow?" → double autodiff on 65k points; drop
  `--nx/--nt` or raise the timeout.
