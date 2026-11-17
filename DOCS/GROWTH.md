# GROWTH.md — Scaling & Productionization Plan

The current environment is a deliberate, tiny sandbox for *debugging the
physics and the PyTorch graph*, not for throughput. This file records what is
true today, what to change in the code so it transfers cleanly to larger
hardware, and a concrete dockerization path.

## 1. Current constraints (and why the code looks the way it does)

| Aspect | Today | Consequence |
|---|---|---|
| Compute | CPU-only, `torch 2.15.0+cpu` | no `cuda` / `mps` code paths exercised |
| RAM | ~961 MiB, **no swap** | OOM killer has already killed `python3`; `inspect_residual.py` evaluates the double-backward residual in chunks for this reason |
| Python | `/root/base/bin/python3` (3.12.3) | system `python3` has **no torch**; do not assume it does |
| Run discipline | `timeout 600` per run | 10-min cap; SIGTERM (exit 143) kills runs, and block-buffered stdout was losing all prints — now mitigated with `python -u` / line buffering |
| Orchestration | shell scripts (`run_app.sh`), `app.pid`, `app.log` | fine for one box, not reproducible elsewhere |
| Data | local `.npz` files under `out/` | fine for 32 scenarios, not for large/versioned datasets |

The goal: **keep the cheap-box debug loop, but make the exact same code run on
a GPU workstation / cluster / container without modification.**

---

## 2. Guiding principles

1. **Hardware-agnostic code.** No hardcoded `cpu`, no assumptions about
   available RAM. Device and dtype are chosen once, from CLI/env.
2. **Bounded memory always.** Chunked autodiff (already done in
   `inspect_residual.py`), batched collocation, checkpointed fields — never
   build an `nx*nt` autograd graph in one shot, even when RAM is "plenty".
3. **Reproducible runs.** Seed + full config + code version → identical
   artifacts. A run is only useful if it can be replayed on bigger hardware.
4. **Containerized delivery.** The image, not the host, owns the Python
   environment and the exact dependency versions.

---

## 3. Code changes that unlock bigger hardware

These are small, backwards-compatible edits that pay off immediately on a GPU
box.

### 3.1 Device & dtype selection (`solve_1d_pinn.py`, `inspect_residual.py`)

Add a single helper (or a tiny `devices.py`) and thread it through:

```python
import torch

def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

- `solve_1d_pinn.py`: add `--device auto` and `--dtype float32|float64`;
  call `model.to(device)` and move collocation tensors to `device`.
- `inspect_residual.py`: same `--device` flag; the chunked `residual_grid`
  already works on any device — chunks just live on GPU instead of CPU.
- Keep `torch.set_default_dtype(torch.float32)` for the cheap box; allow
  `float64` for high-`Pe` / steep-front cases where single precision limits
  accuracy.

### 3.2 Resumable training & checkpointing

`solve_1d_pinn.py` currently saves only `model.state_dict()`. On a long GPU run
you want to stop/resume and to survive a crash:

- Save a **run checkpoint**: `state_dict`, `optim.state_dict()`, `step`,
  `history`, and the parsed config (see 3.3).
- Add `--resume <checkpoint.pt>` to continue Adam / L-BFGS from where it left
  off.
- Save at a cadence (`--ckpt-every N`) and on SIGTERM so `timeout`/preemption
  still yields a usable artifact.

### 3.3 Config-driven runs (exit the CLI-typo era)

CLI flags are fine for a handful of knobs, but growth means sweeps. Add a
`--config path/to.yaml` that overrides defaults, and always dump the *effective*
config next to the outputs:

```yaml
# run.yaml
pe: 20.0
tau_max: 0.5
hidden: 64
layers: 4
hard_bc: true
eps: 1.0e-3
lr: 1.0e-3
adam_steps: 20000
lbfgs_steps: 500
device: auto
dtype: float32
seed: 0
```

Emit `run_config.json` (and a short git SHA) beside `model.pt` and the figures.
Then "reproduce this run on the GPU" becomes one command:

```bash
python -u solve_1d_pinn.py --config runs/best_hardbc.yaml
```

### 3.4 Batching & gradient accumulation on big machines

On a GPU you can raise `--N-f` (collocation points) by orders of magnitude, but
the optimizer should keep a stable *effective batch size*:

- Add `--batch-size` + `--grad-accum N` so the physical gradient is averaged
  over `N` sub-batches. This keeps memory bounded and makes runs comparable
  across hardware.

---

## 4. Data & pipeline scaling

`hydrology.py` is already parameterized (`v_range`, `D_range`, `n_scenarios`,
`nx`, `nt`) — that is the right shape, just turn the dials up.

- **More scenarios / higher resolution:** the 32-scenario 128×128 `dataset.npz`
  is a smoke test. Target ~10³–10⁴ scenarios at 256²/512² for FNO (Phase 4).
- **Move off `.npz` for big data.** `np.savez_compressed` re-reads everything
  into RAM. Switch to:
  - **Zarr** (compressed, chunked, parallel, streaming) or **HDF5** via `h5py`
    once a single dataset exceeds RAM or you want random access.
  - Keep `.npz` as the small "unit test" path so the cheap-box loop still works.
- **Parallel generation.** `generate_dataset` is embarrassingly parallel — use
  `multiprocessing`/`joblib` locally, or map over scenarios in a cluster job.
- **Version datasets.** Name artifacts by content, not "latest": e.g.
  `dataset_pe20_n1024_256x256_v1.zarr`. Store generation parameters inside the
  archive (already done via the `Scenario` payload).

---

## 5. Dockerization

Containerize so the Python environment is pinned, the code is immutable, and the
same image runs on the laptop, the GPU box, and CI.

**Current status:** `Dockerfile.cpu`, `Dockerfile.gpu`, `docker-compose.yml`,
`.dockerignore`, and `runs/best_hardbc.json` have been created from this plan.
The remaining task is to build/smoke-test the images on a Docker-capable host.

### 5.1 CPU image (works today, on this box)

```dockerfile
# syntax=docker/dockerfile:1
# Dockerfile.cpu — build with:  docker build -f Dockerfile.cpu -t tesis:1d .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Install torch (CPU wheels) first so dependency resolution is stable.
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2" \
 && pip install "numpy>=1.26" "scipy>=1.11" "matplotlib>=3.8" "flask>=3.0"

COPY . .

# Non-root runtime user.
RUN useradd -m tesis && chown -R tesis:tesis /app
USER tesis

# /app/out and /app/figs are expected volumes.
VOLUME ["/app/out", "/app/figs"]

ENTRYPOINT ["python", "-u"]
CMD ["solve_1d_pinn.py", "--hard-bc", "--save-model", "out/pinn_hardbc_exp/model.pt"]
```

### 5.2 GPU image (the future box)

Use the official PyTorch CUDA image so CUDA/toolkit/cuDNN versions match:

```dockerfile
# Dockerfile.gpu — build with:  docker build -f Dockerfile.gpu -t tesis:1d-gpu .
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app
RUN pip install "numpy>=1.26" "scipy>=1.11" "matplotlib>=3.8" "flask>=3.0"
COPY . .
RUN useradd -m tesis && chown -R tesis:tesis /app
USER tesis
VOLUME ["/app/out", "/app/figs"]
ENTRYPOINT ["python", "-u"]
CMD ["solve_1d_pinn.py", "--hard-bc", "--device", "cuda"]
```

Requires `nvidia-container-toolkit` on the host:

```bash
nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
docker run --gpus all --rm tesis:1d-gpu solve_1d_pinn.py --device cuda --save-model out/model.pt
```

### 5.3 Compose with CPU/GPU profiles

```yaml
# docker-compose.yml
services:
  train:
    build:
      context: .
      dockerfile: Dockerfile.cpu
    profiles: ["cpu"]
    volumes:
      - ./out:/app/out
      - ./figs:/app/figs
    command: ["solve_1d_pinn.py", "--hard-bc", "--save-model", "out/model.pt"]

  train-gpu:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    profiles: ["gpu"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./out:/app/out
      - ./figs:/app/figs
    command: ["solve_1d_pinn.py", "--hard-bc", "--device", "cuda", "--save-model", "out/model.pt"]

  viewer:
    build:
      context: .
      dockerfile: Dockerfile.cpu
    profiles: ["serve"]
    command: ["app.py", "--host", "0.0.0.0", "--port", "5000"]
    ports:
      - "127.0.0.1:5000:5000"   # bind localhost only by default
    volumes:
      - ./out:/app/out
      - ./figs:/app/figs
```

Usage:

```bash
docker compose --profile cpu  run --rm train
docker compose --profile gpu  run --rm train-gpu
docker compose --profile serve up viewer
```

### 5.4 Docker notes specific to this repo

- **`PYTHONUNBUFFERED=1` is mandatory** in images — it is the containerized
  equivalent of `python -u` and prevents the "killed, no output" failure mode.
- **Mount `out/` and `figs/` as volumes.** Never write results into the image
  layer; they must survive container teardown and be inspectable on the host.
- **Pin the base image tag**, not `latest`. A `pytorch/pytorch` tag change is a
  silent CUDA version change.
- **Don't ship `app.log` / `app.pid` / `.ipynb_checkpoints`** — add a
  `.dockerignore`:

  ```dockerignore
  .git
  __pycache__
  .ipynb_checkpoints
  out
  figs
  *.log
  *.pid
  *.npz
  ```

---

## 6. Reproducibility & experiment tracking

As runs multiply (and move between machines), ad-hoc notes in `Actions.md`
won't scale.

- **Every run emits a manifest:** `run_config.json` + git SHA + seed + device +
  timings + final metrics (`rel-L2` full/interior, region errors). `train()`
  already computes these; persist them alongside `model.pt`.
- **Track artifacts centrally** with **MLflow** (self-hosted, GPU-friendly) or
  **Weights & Biases**. Log: config, loss curves, `rel-L2`, residual figures,
  and the `model.pt` artifact. The existing `history` dict is a drop-in source.
- **Seeds everywhere.** `solve_1d_pinn.py` and `inspect_residual.py` already
  take `--seed`; `hydrology.py` seeds its RNG. Add `--seed` to any new script
  and record it in the manifest.
- **Metric convention:** always report *both* full-grid rel-L2 **and** the
  corner-excluded interior error (the corner singularity at (0,0) is a
  measure-zero artifact — see `Actions.md`). Bigger hardware will tempt you to
  chase the full-grid number; the interior metric is the physically meaningful
  one.

---

## 7. Production hardening of the viewer (`app.py`)

The `app.log` on the current box shows the public Flask viewer being scanned
with exploit attempts. Before this becomes a "deployment":

- **Bind localhost** (done in the compose example via `127.0.0.1:5000:5000`),
  or put it behind a reverse proxy (nginx/Caddy) with TLS.
- **Run non-root** (the Dockerfiles already do `USER tesis`).
- **Flask dev server is not production** — use `waitress`/`gunicorn` in the
  image `CMD`, not `app.run()`.
- The viewer is **read-only**; it must never expose `solve_1d_pinn.py` or any
  write endpoint.

---

## 8. Roadmap (tied to `plan.md`)

| Stage | Where | What |
|---|---|---|
| **Now** | this box | Phase 1/2 debug loop; chunked residual; `-u` output discipline |
| **Next** | this box | formulation write-up update; Phase 3 tau-ramp/RAR experiment |
| **Short term** | CPU container | build/smoke-test `Dockerfile.cpu` + compose; reproduce current best run inside the container |
| **Medium term** | GPU workstation | `Dockerfile.gpu`; Phase 3 (dynamic weighting, Adam→L-BFGS, gradient logging); scale collocation via 3.4 |
| **Medium term** | GPU + Zarr | Phase 4 FNO: larger dataset, modes/resolution-invariance checks; MLflow tracking |
| **Long term** | cluster / 2D–3D | extend to heterogeneous permeability & 2D/3D ADE; distributed data generation; distributed FNO training |

### Rule of thumb for "more capable hardware"

> Make the **same** run go *faster* first (GPU, larger batch, float64), and only
> then make it *bigger* (more scenarios, 2D/3D, deeper nets). The cheap box is
> for correctness; the big box is for scale — never debug numerics on the big
> box if the small one can already reproduce the failure.
