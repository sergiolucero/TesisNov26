#!/usr/bin/env bash
# SCRIPTS/fleet_bootstrap.sh
# ==========================
# Idempotent Tailscale bootstrap for the fleet described in NewPlan.md (§1-§2).
#
# One script, one role per node:
#
#   ubuntu-hub    master control + heavy CPU hub   (herdr, bare repo, Ollama)
#   do-relay      always-on web/LaTeX droplet      (docker compose stack)
#   surface-ctrl  Windows tablet control plane     (SSH / VS Code Remote)
#   colab-t4      ephemeral Colab GPU node         (run_colab.sh)
#
# The script only performs safe, repeatable actions: it installs Tailscale if
# missing, joins the tailnet with the node's hostname, and prints the per-role
# next step. It never touches experiment data and never runs the workloads
# themselves.
#
# Usage:
#   ./SCRIPTS/fleet_bootstrap.sh --role hub              # ubuntu-hub
#   ./SCRIPTS/fleet_bootstrap.sh --role relay --dry-run  # do-relay
#   ./SCRIPTS/fleet_bootstrap.sh --role surface          # surface-ctrl
#   ./SCRIPTS/fleet_bootstrap.sh --verify                # check this node
#
# Auth:
#   Pass an auth key with --authkey or TS_AUTHKEY. Without one, `tailscale up`
#   prints a login URL that must be opened in a browser (fine for interactive
#   nodes, awkward on the headless droplet).
#
# Options:
#   --role ROLE      hub | relay | surface | colab (default: hub)
#   --hostname NAME  Tailscale hostname (default: role mapping below)
#   --authkey KEY    Tailscale auth key (default: $TS_AUTHKEY)
#   --no-ssh         do not enable Tailscale SSH
#   --dry-run        print the commands instead of running them
#   --verify         only report status; change nothing
#   -h, --help       this text

set -Eeuo pipefail

ROLE="hub"
HOSTNAME_ARG=""
AUTHKEY="${TS_AUTHKEY:-}"
ENABLE_SSH=1
DRY_RUN=0
VERIFY=0

usage() { sed -n '2,/^set -Eeuo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

log()  { printf '\n[fleet] %s\n' "$*"; }
warn() { printf '[fleet] warning: %s\n' "$*" >&2; }
die()  { printf '[fleet] error: %s\n' "$*" >&2; exit 2; }

have() { command -v "$1" >/dev/null 2>&1; }

# Run a command, or print it under --dry-run.
run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] %s\n' "$*"
  else
    printf '[fleet] $ %s\n' "$*"
    "$@"
  fi
}

# --------------------------------------------------------------------------
# Argument parsing (no getopt: this must run on macOS and Git-Bash too)
# --------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --role)      ROLE=${2:-}; shift 2 ;;
    --hostname)  HOSTNAME_ARG=${2:-}; shift 2 ;;
    --authkey)   AUTHKEY=${2:-}; shift 2 ;;
    --no-ssh)    ENABLE_SSH=0; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --verify)    VERIFY=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    *)           die "unknown argument '$1' (try --help)" ;;
  esac
done

case "$ROLE" in
  hub)     DEFAULT_HOST=ubuntu-hub ;;
  relay)   DEFAULT_HOST=do-relay ;;
  surface) DEFAULT_HOST=surface-ctrl ;;
  colab)   DEFAULT_HOST=colab-t4 ;;
  *)       die "unknown role '$ROLE' (use hub|relay|surface|colab)" ;;
esac
NODE_NAME=${HOSTNAME_ARG:-$DEFAULT_HOST}

OS=$(uname -s)

log "role=$ROLE node=$NODE_NAME os=$OS dry_run=$DRY_RUN verify=$VERIFY"

# --------------------------------------------------------------------------
# Verify mode: report only
# --------------------------------------------------------------------------
if [[ "$VERIFY" == "1" ]]; then
  if ! have tailscale; then
    die "tailscale is not installed on $NODE_NAME"
  fi
  log "tailscale version"
  tailscale version || true
  log "tailnet status (first 15 lines)"
  tailscale status 2>&1 | head -15 || true
  if ipv4=$(tailscale ip -4 2>/dev/null); then
    log "tailnet IPv4: $ipv4"
  else
    warn "no tailnet IPv4; the node is not up"
  fi
  log "expected hostname for role '$ROLE': $DEFAULT_HOST (actual: $(hostname))"
  exit 0
fi

# --------------------------------------------------------------------------
# 1. Install Tailscale if missing
# --------------------------------------------------------------------------
install_tailscale() {
  case "$OS" in
    Linux)
      if have apt-get; then
        log "installing Tailscale from the official script (apt host)"
      fi
      if have curl; then
        run sh -c 'curl -fsSL https://tailscale.com/install.sh | sh'
      elif have wget; then
        run sh -c 'wget -qO- https://tailscale.com/install.sh | sh'
      else
        die "curl/wget not found; install one and re-run"
      fi
      ;;
    Darwin)
      if have brew; then
        run brew install --cask tailscale
        run open -a Tailscale
      else
        die "install Homebrew, or get Tailscale from https://tailscale.com/download/mac"
      fi
      ;;
    MINGW*|MSYS*|CYGWIN*)
      if have winget; then
        run winget install --id tailscale.tailscale -e --accept-package-agreements
      else
        die "install winget, or get Tailscale from https://tailscale.com/download/windows"
      fi
      ;;
    *)
      die "unsupported OS '$OS'; see https://tailscale.com/download"
      ;;
  esac
}

if have tailscale; then
  log "tailscale already installed: $(tailscale version 2>/dev/null | head -1)"
else
  log "tailscale not found"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] would install tailscale for %s\n' "$OS"
  else
    install_tailscale
  fi
fi

# `tailscaled` needs a moment on a fresh install before `tailscale up` works.
if [[ "$DRY_RUN" != "1" ]] && have tailscale; then
  for _ in 1 2 3 4 5; do
    tailscale status >/dev/null 2>&1 && break
    sleep 2
  done
fi

# --------------------------------------------------------------------------
# 2. Join the tailnet under the fleet hostname
# --------------------------------------------------------------------------
UP_ARGS=(up "--hostname=$NODE_NAME")
[[ "$ENABLE_SSH" == "1" ]] && UP_ARGS+=(--ssh)
[[ -n "$AUTHKEY" ]] && UP_ARGS+=("--authkey=$AUTHKEY")

if [[ "$DRY_RUN" == "1" ]]; then
  # Do not leak the key into shell history or CI logs.
  printf '[dry-run] tailscale %s\n' "${UP_ARGS[*]/--authkey=*/--authkey=REDACTED}"
else
  log "joining the tailnet as $NODE_NAME (ssh=$ENABLE_SSH, authkey=${AUTHKEY:+set}${AUTHKEY:-unset})"
  tailscale "${UP_ARGS[@]}"
fi

# --------------------------------------------------------------------------
# 3. Role-specific next step
# --------------------------------------------------------------------------
case "$ROLE" in
  hub)
    cat <<'EOF'

[fleet] next: hub duties
  - master bare repo:   git clone --bare <origin> ~/tesis.git
  - colab dispatch:     ./run_colab.sh  (sentinel /tmp/tesis-colab.done)
  - heavy CPU work:     python -u run_optimization.py --stage gates
  - local inference:    ollama serve
EOF
    ;;
  relay)
    cat <<'EOF'

[fleet] next: relay duties
  - docker stack:  docker compose up -d      (TeX Live / Jupyter / Flask)
  - TeX rebuild:   see TEX/ (latexmk) and deploy_do.sh
  - remote runs:   ./remote_run.sh           (sentinel /tmp/tesis-run.done)
EOF
    ;;
  surface)
    cat <<'EOF'

[fleet] next: control plane
  - ssh into the hub:   ssh <user>@ubuntu-hub
  - VS Code Remote-SSH: add ubuntu-hub and do-relay as hosts
  - keep NewPlan.md and the literature notes here
EOF
    ;;
  colab)
    cat <<'EOF'

[fleet] next: ephemeral GPU node
  - this node is short-lived; do not expect its hostname to persist
  - run the experiments with:  bash run_colab.sh --preset quick
  - see Collab.md for the notebook steps
EOF
    ;;
esac

log "done. verify from any node with: tailscale status"
