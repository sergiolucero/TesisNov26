#!/usr/bin/env bash
# remote_run.sh
# =============
# Executed on the DigitalOcean droplet by deploy_do.sh.
#
# It installs the Python environment, then runs the requested simulation and
# writes a completion sentinel so the local deploy script can poll for it.
#
# Environment variables (set by deploy_do.sh):
#   SIM_MODE        smoke | pinn | fno | all | none
#   DEVICE          auto | cpu | cuda
#   TORCH_INDEX_URL PyTorch wheel index (CPU or CUDA)
#   FNO_EPOCHS      Epochs for the FNO run
#   VENV            Virtualenv path (default: ./.venv)
#
# Sentinel files:
#   /tmp/tesis-run.exit  exit code of this script
#   /tmp/tesis-run.done  touched when the script exits

set -Eeuo pipefail

SIM_MODE=${SIM_MODE:-pinn}
DEVICE=${DEVICE:-auto}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}
FNO_EPOCHS=${FNO_EPOCHS:-300}
VENV=${VENV:-$PWD/.venv}
JOB_DONE=${TESIS_JOB_DONE:-/tmp/tesis-run.done}
JOB_EXIT=${TESIS_JOB_EXIT:-/tmp/tesis-run.exit}

SIM_MODE=${SIM_MODE,,}

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

log() { printf '\n[remote] %s\n' "$*"; }

# This script lives in SCRIPTS/; the project root is its parent.
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

log "host=$(hostname) date=$(date -u +%FT%TZ)"
log "pwd=$PWD"
log "mode=$SIM_MODE device=$DEVICE torch_index=$TORCH_INDEX_URL"
log "cpu=$(nproc 2>/dev/null || echo '?') mem=$(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo '?') disk=$(df -h . | awk 'NR==2{print $4}')"

# --------------------------------------------------------------------------
# System packages
# --------------------------------------------------------------------------
if [[ "$(id -u)" -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  log "installing system packages"
  apt-get update -qq
  apt-get install -y -qq \
    python3 python3-venv python3-pip python3-dev \
    build-essential git curl ca-certificates libgomp1
elif ! command -v python3 >/dev/null 2>&1; then
  log "python3 not found and apt-get is unavailable"
  exit 2
fi

# --------------------------------------------------------------------------
# Python environment
# --------------------------------------------------------------------------
if [[ ! -x "$VENV/bin/python" ]]; then
  log "creating virtualenv at $VENV"
  python3 -m venv "$VENV"
fi

log "upgrading pip"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

log "installing PyTorch from $TORCH_INDEX_URL"
"$VENV/bin/pip" install --index-url "$TORCH_INDEX_URL" "torch>=2.2"

log "installing project requirements"
"$VENV/bin/pip" install -r requirements.txt

log "checking torch"
"$VENV/bin/python" - <<'PY'
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

mkdir -p out/remote_pinn out/remote_fno out/remote_smoke

# --------------------------------------------------------------------------
# Simulations
# --------------------------------------------------------------------------
run_hydrology() {
  log "generating synthetic hydrology data"
  "$VENV/bin/python" -u SCRIPTS/hydrology.py \
    --scenario out/scenario.npz \
    --dataset out/dataset.npz \
    2>&1 | tee out/hydrology.log
}

run_pinn() {
  log "training the 1D PINN"
  "$VENV/bin/python" -u SCRIPTS/solve_1d_pinn.py \
    --config runs/best_hardbc.json \
    --outdir out/remote_pinn \
    --save-model out/remote_pinn/model.pt \
    --device "$DEVICE" \
    2>&1 | tee out/remote_pinn/run.log
}

run_smoke() {
  log "running the PINN smoke test (1 Adam step, 16x16 evaluation)"
  "$VENV/bin/python" -u SCRIPTS/solve_1d_pinn.py \
    --hard-bc --eps 1e-3 \
    --adam-steps 1 --lbfgs-steps 0 \
    --N-f 64 --N-ic 16 --N-bc 16 --N-f-fix 64 \
    --nx-test 16 --nt-test 16 \
    --device "$DEVICE" \
    --outdir out/remote_smoke \
    --save-model out/remote_smoke/model.pt \
    2>&1 | tee out/remote_smoke/run.log
}

run_fno() {
  log "training the 1D FNO"
  "$VENV/bin/python" -u SCRIPTS/solve_1d_fno.py \
    --dataset out/dataset.npz \
    --epochs "$FNO_EPOCHS" \
    --outdir out/remote_fno \
    --save-model out/remote_fno/model.pt \
    --device "$DEVICE" \
    2>&1 | tee out/remote_fno/run.log
}

case "$SIM_MODE" in
  smoke)
    run_hydrology
    run_smoke
    ;;
  pinn)
    run_hydrology
    run_pinn
    ;;
  fno)
    run_hydrology
    run_fno
    ;;
  all)
    run_hydrology
    run_pinn
    run_fno
    ;;
  none)
    log "environment only; no simulation requested"
    ;;
  *)
    log "unknown SIM_MODE='$SIM_MODE' (use smoke|pinn|fno|all|none)"
    exit 2
    ;;
esac

log "simulation complete; artifacts are under $PWD/out"
