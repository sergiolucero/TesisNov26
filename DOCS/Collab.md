# TESIS Google Colab GPU Run Guide

This project can be run in a Google Colab notebook with a GPU by using the helper script `SCRIPTS/collab_run.py`.

> Note: Google spells the product **Colab**. This repository also includes this file as `Collab.md` to match the requested name.
>
> **Before paying for Pro, read [`GPU.md`](GPU.md).** Until the batched backend
> was added, the only code that touched the GPU was the two 1D toy models, while
> the expensive optimization stage ran on SciPy sparse LU with the accelerator
> idle. `GPU.md` records that audit, the new `--backend torch` path, and how to
> measure on your own runtime whether the GPU actually pays off
> (`SCRIPTS/bench_backend.py`).

## 1. Start a GPU runtime

In Colab:

1. Open a new notebook.
2. Select **Runtime → Change runtime type**.
3. Choose **Hardware accelerator: GPU**.
4. Run:

```python
!nvidia-smi
```

If a GPU is attached, the command prints the GPU model.

## 2. Upload or copy the project

### Option A — use the prepared zip from this repository

Upload `static/TESIS_colab.zip` to Colab, then run:

```python
!unzip -q TESIS_colab.zip -d TESIS
%cd TESIS
```

If the zip contains the top-level folder already, use `%cd TESIS/TESIS` instead.

### Option B — upload files manually

Upload the repository files to `/content/TESIS`, then run:

```python
%cd /content/TESIS
```

## 3. Install requirements

Colab usually already has a CUDA-enabled PyTorch installed. The requirements file intentionally does not pin a CUDA wheel, so it is safe to run:

```python
!pip install -r requirements.txt
```

If PyTorch is missing for any reason, install the CUDA wheel recommended at <https://pytorch.org/get-started/locally/> for the current Colab CUDA version.

## 4. Check the runner and GPU

```python
!python SCRIPTS/collab_run.py --help
!python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY
```

## 5. Run models/experiments

`SCRIPTS/collab_run.py` dispatches the project entry points and lets you choose one or more models.

### Which stages use the GPU

| Stage | Backend | Notes |
|---|---|---|
| `dataset` | CPU (NumPy) | Ogata-Banks + dataset generation |
| `pinn`, `fno` | GPU (torch) | 1D models: launch-bound, so a GPU barely helps |
| `opt` | **GPU (torch) when available** | `--opt-backend auto` (default) selects the batched backend on a GPU runtime |
| gates (`o1`-`o4`) | CPU (NumPy) | they certify the solver, not throughput |

```python
# force the batched backend and float32 throughput mode
!python SCRIPTS/collab_run.py --models opt --opt-stage opt \
    --opt-config configs/opt/benchmark.yaml --opt-backend torch --opt-dtype float32 --gpu 0

# is the GPU actually earning its keep?
!python SCRIPTS/bench_backend.py --device cuda --dtype float32 --out out/bench_gpu.json
```

### Fast smoke run

Use this first to verify that the environment works:

```python
!python SCRIPTS/collab_run.py --models smoke --zip-results
```

This generates a small dataset, runs short PINN and FNO jobs, runs a short optimization stage, and writes `out/colab_results.zip`.

### PINN only

```python
# Quick GPU PINN run
!python SCRIPTS/collab_run.py --models pinn --preset quick --device cuda

# Longer/default PINN run
!python SCRIPTS/collab_run.py --models pinn --preset full --device cuda --hard-bc

# Custom Adam steps
!python SCRIPTS/collab_run.py --models pinn --preset quick --pinn-steps 3000 --device cuda
```

Outputs are written under `out/colab/pinn/`.

### FNO only

The FNO needs a dataset. The runner will generate it automatically if missing.

```python
# Quick FNO run
!python SCRIPTS/collab_run.py --models dataset fno --preset quick --device cuda

# Longer/default FNO run
!python SCRIPTS/collab_run.py --models dataset fno --preset full --device cuda --fno-epochs 300
```

Outputs are written under `out/colab/fno/`.

### PINN + FNO

```python
!python SCRIPTS/collab_run.py --models both --preset quick --device cuda --zip-results
```

### Stochastic optimization

```python
# Gate checks using smaller quick settings
!python SCRIPTS/collab_run.py --models opt --preset quick --opt-stage gates

# Final optimization stage, quick settings
!python SCRIPTS/collab_run.py --models opt --preset quick --opt-stage opt

# Full benchmark configuration
!python SCRIPTS/collab_run.py --models opt --preset full --opt-stage all
```

Outputs are written under `out/colab/opt/`.

### Unit tests

```python
!python SCRIPTS/collab_run.py --models tests
```

## 6. Save outputs from Colab

Zip the output folder:

```python
!python SCRIPTS/collab_run.py --models tests --zip-results
```

Or zip results without rerunning models:

```python
!zip -r out/colab_results.zip out/colab figs
```

Download in Colab:

```python
from google.colab import files
files.download('out/colab_results.zip')
```

Optional Google Drive sync:

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r out/colab /content/drive/MyDrive/TESIS_colab_outputs
```

## 7. Main files

- `SCRIPTS/collab_run.py` — Colab-friendly dispatcher for dataset, PINN, FNO, optimization, and tests.
- `SCRIPTS/run_colab.sh` — shell wrapper that checks the GPU, installs deps, and calls the dispatcher.
- `SCRIPTS/multi_colab.py` — drives several Colab VMs from your laptop (`launch/status/logs/collect/stop`) for runs too large for one notebook session.
- `SCRIPTS/new_high_colab.py` — the same stages tuned for the high-end Colab GPUs (A100 = v5, H100/G4 = v6): measures the card, batches the transport sweep and drives the optimizer's `--backend torch`; `--launch` rents and drives the VM for you.
- `requirements.txt` — Python dependencies.
- `SCRIPTS/hydrology.py` — Generates 1D ADE scenarios/datasets.
- `SCRIPTS/solve_1d_pinn.py` — PyTorch PINN model.
- `SCRIPTS/solve_1d_fno.py` — PyTorch FNO model.
- `SCRIPTS/run_optimization.py` — stochastic pump-and-treat optimization workflow.

Every script lives in `SCRIPTS/`; see `USAGE.md` (this directory) for the full
inventory.

## Troubleshooting

- If `torch.cuda.is_available()` is `False`, restart Colab with **Runtime → Change runtime type → GPU**.
- If installation upgrades PyTorch to a CPU-only build, reinstall the CUDA PyTorch wheel from the official PyTorch instructions.
- If a full run exceeds Colab limits, use `--preset quick`, lower `--pinn-steps`, or lower `--fno-epochs`.
- If you see stale outputs, remove them with `!rm -rf out/colab` before rerunning.
