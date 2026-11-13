#!/usr/bin/env bash
# deploy_do.sh
# =============
# Pack the current TESIS folder, create a DigitalOcean droplet, copy the
# archive to it, install the Python environment, run simulations, and (by
# default) download the artifacts and destroy the droplet.
#
# Requirements on the machine running this script:
#   curl, jq, tar, ssh, scp, ssh-keygen, bash 4+
#
# Requirements on DigitalOcean:
#   A personal access token with read/write access to Droplets and SSH keys.
#   Export it as DIGITALOCEAN_TOKEN (recommended) or pass --token-file.
#
# Quick start:
#   export DIGITALOCEAN_TOKEN=dop_v1_...
#   ./deploy_do.sh --sim pinn
#
# Cheap smoke test (1 Adam step, 16x16 evaluation):
#   ./deploy_do.sh --sim smoke
#
# Keep the droplet and print its IP:
#   ./deploy_do.sh --sim pinn --keep
#
# Use an already-created droplet (skip creation and destruction):
#   ./deploy_do.sh --ip 203.0.113.10 --sim smoke
#
# GPU (the caller must supply a GPU size/image available in their account):
#   ./deploy_do.sh --gpu --size gpu-l40x1-48gb --image gpu-ubuntu-2204 --sim pinn
#
# Author: TESIS deployment helper

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$SCRIPT_DIR}

# --------------------------------------------------------------------------
# Defaults / configuration
# --------------------------------------------------------------------------
TOKEN=${DIGITALOCEAN_TOKEN:-${DO_API_TOKEN:-}}
TOKEN_FILE=""
SSH_KEY_PATH=${DO_SSH_KEY:-"$HOME/.ssh/tesis_do_ed25519"}
SSH_KEY_ID=${DO_SSH_KEY_ID:-}
SSH_USER=${DO_SSH_USER:-root}
SSH_PORT=${DO_SSH_PORT:-22}

NAME="tesis-$(date -u +%Y%m%d-%H%M%S)"
REGION=${DO_REGION:-nyc3}
SIZE=${DO_SIZE:-s-2vcpu-4gb}
IMAGE=${DO_IMAGE:-ubuntu-24-04-x64}
GPU=0
TORCH_INDEX_URL=${TORCH_INDEX_URL:-}

REMOTE_DIR=${DO_REMOTE_DIR:-/root/tesis}
SIM_MODE=${DO_SIM:-pinn}
DEVICE=${DO_DEVICE:-auto}
FNO_EPOCHS=${DO_FNO_EPOCHS:-300}

KEEP=0
KEEP_ON_FAILURE=1
DOWNLOAD=1
LOCAL_RESULTS=${DO_RESULTS_DIR:-$ROOT/do_results}
EXISTING_IP=""
RUN_TIMEOUT=${DO_RUN_TIMEOUT:-21600}   # 6 hours; 0 disables the timeout
WAIT_SECONDS=${DO_WAIT_SECONDS:-600}   # droplet boot + SSH wait
DRY_RUN=0

EXTRA_EXCLUDES=()
WORKDIR=""
DROPLET_ID=""
IP=""
CURL_CONFIG=""

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
log()  { printf '[deploy] %s\n' "$*" >&2; }
die()  { printf '[deploy] ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"; }

usage() {
  cat <<'EOF'
Usage: ./deploy_do.sh [options]

Packs this folder, creates a DigitalOcean droplet, scp's the archive, installs
the Python environment, runs the requested simulation, and downloads results.

DigitalOcean token:
  export DIGITALOCEAN_TOKEN=dop_v1_...
  or pass --token-file FILE.

Options:
  --token-file FILE       Read the DO token from FILE.
  --name NAME             Droplet name (default: tesis-UTC timestamp).
  --region REGION         DO region slug (default: nyc3).
  --size SIZE             DO size slug (default: s-2vcpu-4gb).
  --image IMAGE           DO image slug (default: ubuntu-24-04-x64).
  --ssh-key PATH          Private SSH key (default: ~/.ssh/tesis_do_ed25519;
                          generated if missing).
  --ssh-key-id ID         Use an already-registered DO SSH-key fingerprint.
  --ssh-user USER         SSH user (default: root).
  --ssh-port PORT         SSH port (default: 22).
  --remote-dir DIR        Remote working directory (default: /root/tesis).
  --sim MODE              Smoke test: smoke|pinn|fno|all|none (default: pinn).
  --device DEVICE         auto|cpu|cuda passed to the solver (default: auto).
  --gpu                   Install a CUDA PyTorch wheel (also auto-detected when
                          --size starts with "gpu-").
  --torch-index-url URL   Override the PyTorch wheel index.
  --fno-epochs N          Epochs for --sim fno/all (default: 300).
  --ip ADDRESS            Use this existing instance; do not create/destroy.
  --results-dir DIR       Local results directory (default: ./do_results).
  --no-download           Do not download artifacts.
  --keep                  Keep the droplet after a successful run.
  --destroy-on-failure    Destroy the droplet even if the remote run fails.
  --timeout SECONDS       Remote run timeout (default: 21600; 0 = none).
  --wait SECONDS          Boot/SSH wait timeout (default: 600).
  --exclude PATTERN       Extra tar exclude pattern (repeatable).
  --dry-run               Print the plan and exit.
  -h, --help              Show this help.

Examples:
  export DIGITALOCEAN_TOKEN=dop_v1_...
  ./deploy_do.sh --sim smoke
  ./deploy_do.sh --sim pinn --keep
  ./deploy_do.sh --sim all --size s-4vcpu-8gb --results-dir ./do_results
  ./deploy_do.sh --ip 203.0.113.10 --sim smoke
EOF
}

cleanup() {
  local rc=$?
  set +e

  if [[ -n "${DROPLET_ID:-}" && "${KEEP}" -eq 0 ]]; then
    if [[ $rc -eq 0 || "${KEEP_ON_FAILURE}" -eq 0 ]]; then
      log "destroying droplet ${DROPLET_ID}"
      do_api DELETE "droplets/${DROPLET_ID}" >/dev/null 2>&1 \
        || log "warning: could not destroy droplet ${DROPLET_ID}; delete it in the DO console"
    else
      log "keeping droplet ${DROPLET_ID} for debugging (remote exit ${rc})"
      log "delete it with: doctl compute droplet delete ${DROPLET_ID}  (or the DO console)"
    fi
  elif [[ -n "${DROPLET_ID:-}" && "${KEEP}" -eq 1 ]]; then
    log "keeping droplet ${DROPLET_ID} at ${IP:-unknown IP} (--keep)"
  fi

  [[ -n "${WORKDIR:-}" && -d "${WORKDIR:-}" ]] && rm -rf "$WORKDIR"
  trap - EXIT
  exit "$rc"
}
trap cleanup EXIT

# --------------------------------------------------------------------------
# Parse arguments
# --------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --token-file)       TOKEN_FILE=$2; shift 2 ;;
    --name)             NAME=$2; shift 2 ;;
    --region)           REGION=$2; shift 2 ;;
    --size)             SIZE=$2; shift 2 ;;
    --image)            IMAGE=$2; shift 2 ;;
    --ssh-key)          SSH_KEY_PATH=$2; shift 2 ;;
    --ssh-key-id)       SSH_KEY_ID=$2; shift 2 ;;
    --ssh-user)         SSH_USER=$2; shift 2 ;;
    --ssh-port)         SSH_PORT=$2; shift 2 ;;
    --remote-dir)       REMOTE_DIR=$2; shift 2 ;;
    --sim)              SIM_MODE=$2; shift 2 ;;
    --device)           DEVICE=$2; shift 2 ;;
    --gpu)              GPU=1; shift ;;
    --torch-index-url)  TORCH_INDEX_URL=$2; shift 2 ;;
    --fno-epochs)       FNO_EPOCHS=$2; shift 2 ;;
    --ip)               EXISTING_IP=$2; shift 2 ;;
    --results-dir)      LOCAL_RESULTS=$2; shift 2 ;;
    --no-download)      DOWNLOAD=0; shift ;;
    --keep)             KEEP=1; shift ;;
    --destroy-on-failure) KEEP_ON_FAILURE=0; shift ;;
    --timeout)          RUN_TIMEOUT=$2; shift 2 ;;
    --wait)             WAIT_SECONDS=$2; shift 2 ;;
    --exclude)          EXTRA_EXCLUDES+=("$2"); shift 2 ;;
    --dry-run)          DRY_RUN=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    *)                  die "unknown argument: $1 (use --help)" ;;
  esac
done

# Normalize inputs.
NAME=${NAME,,}
NAME=${NAME//[^a-z0-9_.-]/-}
SIM_MODE=${SIM_MODE,,}
case "$SIM_MODE" in
  smoke|pinn|fno|all|none) ;;
  *) die "--sim must be one of smoke|pinn|fno|all|none (got '$SIM_MODE')" ;;
esac

if [[ -n "$TOKEN_FILE" ]]; then
  [[ -r "$TOKEN_FILE" ]] || die "cannot read --token-file: $TOKEN_FILE"
  TOKEN=$(<"$TOKEN_FILE")
  TOKEN=${TOKEN#export }
  TOKEN=${TOKEN#DIGITALOCEAN_TOKEN=}
  TOKEN=${TOKEN#DO_API_TOKEN=}
  TOKEN=${TOKEN//[$'\r\n']/}
fi
TOKEN=${TOKEN//[$'\r\n']/}

if [[ "$SIZE" == gpu-* ]]; then
  GPU=1
fi
if (( GPU )); then
  TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}
  if [[ "$SIZE" != gpu-* ]]; then
    log "warning: --gpu with size '$SIZE'; CUDA will only work if the image has drivers"
  fi
  if [[ "$DEVICE" == "auto" ]]; then
    DEVICE="cuda"
  fi
else
  TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}
fi

# Dry run needs no token and has no side effects.
if (( DRY_RUN )); then
  cat <<EOF
[deploy] DRY RUN — no droplet will be created.

  source folder : $ROOT
  droplet name  : $NAME
  region/size   : $REGION / $SIZE
  image         : $IMAGE
  ssh key       : $SSH_KEY_PATH
  ssh user/port : $SSH_USER / $SSH_PORT
  remote dir    : $REMOTE_DIR
  simulation    : $SIM_MODE (device=$DEVICE)
  torch index   : $TORCH_INDEX_URL
  results dir   : $LOCAL_RESULTS
  keep droplet  : $KEEP
  timeout       : ${RUN_TIMEOUT}s

  archive       : <tmp>/tesis.tar.gz (excludes caches/logs/do_results)
  remote job    : nohup bash remote_run.sh > /tmp/tesis-run.log
  results       : scp <droplet>:$REMOTE_DIR/out -> $LOCAL_RESULTS/$NAME
EOF
  exit 0
fi

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
need curl
need jq
need tar
need ssh
need scp
need ssh-keygen

if [[ -z "$EXISTING_IP" && -z "$TOKEN" ]]; then
  die "set DIGITALOCEAN_TOKEN (or DO_API_TOKEN), or pass --token-file FILE"
fi

KEY_DIR=$(dirname "$SSH_KEY_PATH")
mkdir -p "$KEY_DIR"
if [[ "$KEY_DIR" == "$HOME/.ssh" ]]; then
  chmod 700 "$KEY_DIR" 2>/dev/null || true
fi
if [[ ! -f "$SSH_KEY_PATH" ]]; then
  log "generating SSH key at $SSH_KEY_PATH"
  ssh-keygen -q -t ed25519 -N "" -C "tesis-deploy $(date -u +%FT%TZ)" -f "$SSH_KEY_PATH"
fi
chmod 600 "$SSH_KEY_PATH" 2>/dev/null || true
if [[ ! -f "$SSH_KEY_PATH.pub" ]]; then
  ssh-keygen -y -f "$SSH_KEY_PATH" > "$SSH_KEY_PATH.pub"
fi

WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/tesis-do.XXXXXX")
SSH_KNOWN_HOSTS="$WORKDIR/known_hosts"
ARCHIVE="$WORKDIR/tesis.tar.gz"
CURL_CONFIG="$WORKDIR/curl.cfg"
umask 077
{
  printf 'header = "Authorization: Bearer %s"\n' "$TOKEN"
  printf 'header = "Content-Type: application/json"\n'
} > "$CURL_CONFIG"
chmod 600 "$CURL_CONFIG"

SSH_OPTS=(
  -i "$SSH_KEY_PATH"
  -p "$SSH_PORT"
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="$SSH_KNOWN_HOSTS"
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o LogLevel=ERROR
)
SCP_OPTS=(
  -i "$SSH_KEY_PATH"
  -P "$SSH_PORT"
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="$SSH_KNOWN_HOSTS"
  -o ConnectTimeout=15
  -o LogLevel=ERROR
)

# --------------------------------------------------------------------------
# DigitalOcean API
# --------------------------------------------------------------------------
DO_API="https://api.digitalocean.com/v2"

do_api() {
  local method=$1 path=$2 data=${3-}
  local args=(
    --config "$CURL_CONFIG"
    -sS
    --fail-with-body
    -X "$method"
    "$DO_API/$path"
  )
  # Retry only idempotent requests.  Retrying POST /droplets could create
  # duplicate droplets if the response is lost after the server commits.
  if [[ "$method" == "GET" || "$method" == "DELETE" ]]; then
    args+=(--retry 3 --retry-delay 2)
  fi
  if [[ -n "$data" ]]; then
    args+=(--data "$data")
  fi
  curl "${args[@]}"
}

if [[ -n "$EXISTING_IP" ]]; then
  IP="$EXISTING_IP"
  log "using existing instance at $IP; creation and destruction are disabled"
else
  log "checking DigitalOcean token"
  if ! ACCOUNT=$(do_api GET account); then
    die "DigitalOcean API token check failed (see response above)"
  fi
  jq -e '.account.email' >/dev/null <<<"$ACCOUNT" \
    || die "unexpected /account response: $ACCOUNT"

  if [[ -z "$SSH_KEY_ID" ]]; then
    PUB_KEY=$(<"$SSH_KEY_PATH.pub")
    LOCAL_FP=$(ssh-keygen -lf "$SSH_KEY_PATH.pub" | awk '{print $2}')
    KEY_TYPE=$(awk '{print $1}' <<<"$PUB_KEY")
    KEY_DATA=$(awk '{print $2}' <<<"$PUB_KEY")

    log "looking up SSH key in DigitalOcean"
    if ! KEYS=$(do_api GET "account/keys?per_page=200"); then
      die "could not list DigitalOcean SSH keys"
    fi
    SSH_KEY_ID=$(jq -r \
      --arg fp "$LOCAL_FP" \
      --arg kt "$KEY_TYPE" \
      --arg kd "$KEY_DATA" '
        .ssh_keys[]?
        | select(
            .fingerprint == $fp
            or ((.public_key | split(" ")) as $p
                | ($p[0] == $kt and $p[1] == $kd))
          )
        | .fingerprint
      ' <<<"$KEYS" | head -n 1 || true)

    if [[ -z "$SSH_KEY_ID" ]]; then
      log "registering SSH key with DigitalOcean"
      KEY_BODY=$(jq -n \
        --arg name "tesis-deploy-$NAME" \
        --arg pk "$PUB_KEY" \
        '{name:$name, public_key:$pk}')
      if ! KEY_RESP=$(do_api POST account/keys "$KEY_BODY"); then
        die "could not register SSH key"
      fi
      SSH_KEY_ID=$(jq -r '.ssh_key.fingerprint // empty' <<<"$KEY_RESP")
      [[ -n "$SSH_KEY_ID" ]] || die "unexpected SSH-key response: $KEY_RESP"
    fi
  fi
  log "using SSH key fingerprint $SSH_KEY_ID"

  log "creating droplet $NAME ($SIZE, $IMAGE, $REGION)"
  DROPLET_BODY=$(jq -n \
    --arg name "$NAME" \
    --arg region "$REGION" \
    --arg size "$SIZE" \
    --arg image "$IMAGE" \
    --arg key "$SSH_KEY_ID" \
    '{
       name: $name,
       region: $region,
       size: $size,
       image: $image,
       ssh_keys: [$key],
       backups: false,
       ipv6: false,
       monitoring: true,
       tags: ["tesis-deploy"]
     }')
  if ! DROPLET_RESP=$(do_api POST droplets "$DROPLET_BODY"); then
    die "could not create droplet"
  fi
  DROPLET_ID=$(jq -r '.droplet.id // empty' <<<"$DROPLET_RESP")
  [[ -n "$DROPLET_ID" ]] || die "unexpected droplet response: $DROPLET_RESP"
  log "created droplet id=$DROPLET_ID"

  log "waiting for droplet to boot"
  STATUS=""
  for ((i = 0; i < (WAIT_SECONDS / 5) + 1; i++)); do
    if DROPLET_STATE=$(do_api GET "droplets/$DROPLET_ID" 2>/dev/null); then
      STATUS=$(jq -r '.droplet.status // empty' <<<"$DROPLET_STATE" 2>/dev/null || true)
      IP=$(jq -r '.droplet.networks.v4[]? | select(.type == "public") | .ip_address' \
             <<<"$DROPLET_STATE" 2>/dev/null | head -n 1 || true)
      if [[ "$STATUS" == "active" && -n "$IP" ]]; then
        break
      fi
    fi
    log "  status=${STATUS:-unknown}; waiting..."
    sleep 5
  done
  [[ -n "$IP" ]] || die "droplet did not become active with a public IP"
  log "droplet is active at $IP"
fi

log "waiting for SSH at $SSH_USER@$IP:$SSH_PORT"
SSH_OK=0
for ((i = 0; i < (WAIT_SECONDS / 5) + 1; i++)); do
  if ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" true >/dev/null 2>&1; then
    SSH_OK=1
    break
  fi
  sleep 5
done
(( SSH_OK )) || die "SSH did not become available at $IP"
log "SSH is up"

# --------------------------------------------------------------------------
# Pack and copy
# --------------------------------------------------------------------------
TAR_EXCLUDES=(
  --exclude=./.git
  --exclude=./.venv
  --exclude='./.venv/*'
  --exclude=./__pycache__
  --exclude='*/__pycache__'
  --exclude=./.ipynb_checkpoints
  --exclude='*/.ipynb_checkpoints'
  --exclude='*.pyc'
  --exclude='*.aux'
  --exclude='*.out'
  --exclude='*.toc'
  --exclude='*.fls'
  --exclude='*.fdb_latexmk'
  --exclude='*.synctex.gz'
  --exclude=./.pytest_cache
  --exclude='*/.pytest_cache'
  --exclude=./app.log
  --exclude=./*.pid
  --exclude=./Nov04.zip
  --exclude=./do_results
  --exclude='./do_results/*'
)
for pattern in "${EXTRA_EXCLUDES[@]}"; do
  TAR_EXCLUDES+=(--exclude="$pattern")
done

log "packing $ROOT"
tar "${TAR_EXCLUDES[@]}" -czf "$ARCHIVE" -C "$ROOT" .
log "archive size: $(du -h "$ARCHIVE" | awk '{print $1}')"

REMOTE_ARCHIVE="/tmp/tesis-$NAME.tar.gz"
log "copying archive to $IP:$REMOTE_ARCHIVE"
scp "${SCP_OPTS[@]}" "$ARCHIVE" "$SSH_USER@$IP:$REMOTE_ARCHIVE"

REMOTE_DIR_Q=$(printf '%q' "$REMOTE_DIR")
REMOTE_ARCHIVE_Q=$(printf '%q' "$REMOTE_ARCHIVE")
log "unpacking into $REMOTE_DIR"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" \
  "mkdir -p $REMOTE_DIR_Q && tar -xzf $REMOTE_ARCHIVE_Q -C $REMOTE_DIR_Q"

# --------------------------------------------------------------------------
# Run remotely (detached, so an SSH hiccup does not kill the simulation)
# --------------------------------------------------------------------------
REMOTE_LOG="/tmp/tesis-run.log"
REMOTE_DONE="/tmp/tesis-run.done"
REMOTE_EXIT="/tmp/tesis-run.exit"

log "starting remote simulation: mode=$SIM_MODE device=$DEVICE"
REMOTE_CMD="cd $REMOTE_DIR_Q && rm -f $REMOTE_DONE $REMOTE_EXIT && \
nohup env \
  SIM_MODE=$(printf '%q' "$SIM_MODE") \
  DEVICE=$(printf '%q' "$DEVICE") \
  TORCH_INDEX_URL=$(printf '%q' "$TORCH_INDEX_URL") \
  FNO_EPOCHS=$(printf '%q' "$FNO_EPOCHS") \
  PYTHONUNBUFFERED=1 \
  MPLBACKEND=Agg \
  bash remote_run.sh > $REMOTE_LOG 2>&1 & echo \$!"
REMOTE_PID=$(ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" "$REMOTE_CMD")
log "remote pid: $REMOTE_PID"

START=$SECONDS
LAST_REPORT=0
JOB_FAILED=0
JOB_EXIT=1

while true; do
  if ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" "test -f $REMOTE_DONE" >/dev/null 2>&1; then
    break
  fi

  ELAPSED=$((SECONDS - START))
  if (( RUN_TIMEOUT > 0 && ELAPSED > RUN_TIMEOUT )); then
    log "remote run exceeded ${RUN_TIMEOUT}s; leaving it running and keeping the droplet"
    JOB_FAILED=1
    JOB_EXIT=124
    break
  fi

  if (( ELAPSED - LAST_REPORT >= 60 )); then
    LINE=$(ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" \
      "tail -n 1 $REMOTE_LOG 2>/dev/null" 2>/dev/null || true)
    log "running ${ELAPSED}s: ${LINE:-no log line yet}"
    LAST_REPORT=$ELAPSED
  fi
  sleep 10
done

if [[ "$JOB_FAILED" -eq 0 ]]; then
  JOB_EXIT=$(ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" \
    "cat $REMOTE_EXIT 2>/dev/null || echo 1" 2>/dev/null || echo 1)
  if [[ "$JOB_EXIT" == "0" ]]; then
    log "remote simulation finished successfully"
  else
    log "remote simulation failed with exit code $JOB_EXIT"
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" \
      "tail -n 80 $REMOTE_LOG 2>/dev/null" 2>/dev/null || true
    JOB_FAILED=1
  fi
fi

# --------------------------------------------------------------------------
# Download artifacts
# --------------------------------------------------------------------------
if (( DOWNLOAD )); then
  mkdir -p "$LOCAL_RESULTS/$NAME"
  log "downloading remote log to $LOCAL_RESULTS/$NAME/remote-run.log"
  scp "${SCP_OPTS[@]}" "$SSH_USER@$IP:$REMOTE_LOG" \
    "$LOCAL_RESULTS/$NAME/remote-run.log" 2>/dev/null \
    || log "warning: could not download remote log"

  RESULTS_ARCHIVE="/tmp/tesis-results-$NAME.tar.gz"
  log "packing remote results"
  ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" \
    "cd $REMOTE_DIR_Q && \
     (tar -czf $RESULTS_ARCHIVE out figs 2>/dev/null || \
      tar -czf $RESULTS_ARCHIVE out 2>/dev/null || true)"

  if ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" "test -f $RESULTS_ARCHIVE" >/dev/null 2>&1; then
    log "downloading results to $LOCAL_RESULTS/$NAME/results.tar.gz"
    scp "${SCP_OPTS[@]}" "$SSH_USER@$IP:$RESULTS_ARCHIVE" \
      "$LOCAL_RESULTS/$NAME/results.tar.gz"
    tar -xzf "$LOCAL_RESULTS/$NAME/results.tar.gz" -C "$LOCAL_RESULTS/$NAME"
    log "results extracted under $LOCAL_RESULTS/$NAME"
  else
    log "warning: no result archive was produced"
  fi
fi

if (( JOB_FAILED )); then
  log "remote run did not complete successfully"
  exit 1
fi

log "success"
if (( KEEP )); then
  log "droplet kept: id=$DROPLET_ID ip=$IP"
fi
exit 0
