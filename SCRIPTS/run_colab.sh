#!/usr/bin/env bash
# run_colab.sh
# ============
# Entry point for the ephemeral Google Colab T4 node described in DOCS/NewPlan.md
# (§1 "Burst GPU Compute", §2 bootstrap).
#
# This is the thin shell wrapper around `collab_run.py`: it checks the GPU,
# installs the Python requirements once, and dispatches the requested
# experiment preset. Everything experiment-specific stays in `collab_run.py`;
# see DOCS/Collab.md and DOCS/USAGE.md for the notebook workflow.
#
# Usage (inside a Colab cell):
#   !bash SCRIPTS/run_colab.sh                                   # quick preset, dataset+pinn
#   !bash SCRIPTS/run_colab.sh --models fno --preset full --fno-epochs 300
#   COLAB_MODELS="pinn fno" COLAB_PRESET=smoke bash SCRIPTS/run_colab.sh
#
# Environment variables:
#   COLAB_PRESET   smoke | quick | full          (default: quick)
#   COLAB_MODELS   space-separated model names   (default: "dataset pinn")
#   PYTHON         interpreter to use            (default: python3)
#   SKIP_INSTALL   1 = do not pip install        (default: 0)
#   SKIP_GPU_CHECK 1 = tolerate a missing GPU    (default: 0)
#
# Sentinel files (polled by the dispatcher on `ubuntu-hub`):
#   /tmp/tesis-colab.exit  exit code of this script
#   /tmp/tesis-colab.done  touched when the script exits

set -Eeuo pipefail

# cd to the repository root (this script lives in SCRIPTS/).
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

COLAB_PRESET=${COLAB_PRESET:-quick}
COLAB_MODELS=${COLAB_MODELS:-"dataset pinn"}
PYTHON=${PYTHON:-python3}
SKIP_INSTALL=${SKIP_INSTALL:-0}
SKIP_GPU_CHECK=${SKIP_GPU_CHECK:-0}
JOB_DONE=${TESIS_COLAB_DONE:-/tmp/tesis-colab.done}
JOB_EXIT=${TESIS_COLAB_EXIT:-/tmp/tesis-colab.exit}

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

finish() {
  local rc=$?
  trap - EXIT
  set +e
  printf '%s\n' "$rc" > "$JOB_EXIT"
  touch "$JOB_DONE"
  exit "$rc"
}
trap finish EXIT

log() { printf '\n[colab] %s\n' "$*"; }
die() { printf '[colab] error: %s\n' "$*" >&2; exit 2; }

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
log "host=$(hostname) date=$(date -u +%FT%TZ)"
log "preset=$COLAB_PRESET models=$COLAB_MODELS python=$PYTHON"

command -v "$PYTHON" >/dev/null 2>&1 || die "$PYTHON not found; set PYTHON=/usr/bin/python3"
[[ -f SCRIPTS/collab_run.py ]] || die "SCRIPTS/collab_run.py missing; are you in the repository root?"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  log "nvidia-smi not available"
fi

"$PYTHON" - <<'PY' || true
try:
    import torch
    print(f"[colab] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[colab] gpu={torch.cuda.get_device_name(0)} cuda={torch.version.cuda}")
except Exception as exc:
    print(f"[colab] torch import failed: {exc}")
PY

if [[ "$SKIP_GPU_CHECK" != "1" ]]; then
  if ! "$PYTHON" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    log "WARNING: no CUDA device visible. Set SKIP_GPU_CHECK=1 to silence, or"
    log "         Runtime -> Change runtime type -> GPU in Colab."
    # Colab CPU-only is slow but still valid for smoke runs; keep going.
  fi
fi

# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------
if [[ "$SKIP_INSTALL" == "1" ]]; then
  log "SKIP_INSTALL=1; not touching pip"
elif "$PYTHON" -c 'import torch' >/dev/null 2>&1; then
  log "torch already importable; installing requirements only"
  "$PYTHON" -m pip install -q -r requirements.txt
else
  log "torch missing; installing requirements (this pulls a wheel)"
  "$PYTHON" -m pip install -q -r requirements.txt
fi

# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------
# Word-split the model list on purpose: it is a user-provided list of names.
read -r -a models <<< "$COLAB_MODELS"
args=(--models "${models[@]}" --preset "$COLAB_PRESET" "$@")
# Trailing "$@" wins over the defaults above, so callers can override freely.
log "running: SCRIPTS/collab_run.py ${args[*]}"
"$PYTHON" -u SCRIPTS/collab_run.py "${args[@]}"

log "complete; artifacts under out/colab"
