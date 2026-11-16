#!/usr/bin/env bash
# SCRIPTS/fleet_status.sh
# =======================
# One-shot health snapshot for the fleet described in NewPlan.md (§5).
#
# Checks, in order: host/load/memory, Tailscale, Herdr, repo state, job
# sentinels, and disk. Read-only: it never starts, stops or repairs anything.
#
# Usage:
#   ./SCRIPTS/fleet_status.sh              # human summary
#   ./SCRIPTS/fleet_status.sh --json       # machine-readable (for scripts/tablet)
#   ./SCRIPTS/fleet_status.sh --role hub   # label the node with its fleet role
#   ./SCRIPTS/fleet_status.sh --help
#
# Environment:
#   REPO          repository root (default: parent of this script)
#   SENTINEL_DIR  where jobs write <job>.done/.exit (default: /tmp)
#   JOBS          sentinel job names (default: "tesis-run tesis-colab")
#
# Exit status: 0 if no failing check, 1 if any check failed (bad sentinel exit,
# disk >= 95%, or an unreachable tailnet when --role is given).

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
SENTINEL_DIR=${SENTINEL_DIR:-/tmp}
JOBS=${JOBS:-"tesis-run tesis-colab"}
JSON=0
ROLE=""

# herdr installs to ~/.local/bin, which non-interactive shells may not have.
PATH="$HOME/.local/bin:$PATH"
export PATH

usage() { sed -n '2,/^set -Eeuo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json) JSON=1; shift ;;
    --role) ROLE=${2:-}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'fleet_status: unknown argument %s (try --help)\n' "$1" >&2; exit 2 ;;
  esac
done

have() { command -v "$1" >/dev/null 2>&1; }

FAILS=0
WARNS=0
ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
fail() { printf '  [fail] %s\n' "$*"; }
info() { printf '  %s\n' "$*"; }

json_str() {
  local s=${1//\\/\\\\}
  s=${s//\"/\\\"}
  s=${s//$'\n'/\\n}
  printf '"%s"' "$s"
}

# --------------------------------------------------------------------------
# Collect
# --------------------------------------------------------------------------
HOST=$(hostname 2>/dev/null || echo unknown)
NOW=$(date -u +%FT%TZ)
UPTIME=$(uptime -p 2>/dev/null || echo unknown)
LOAD=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo "? ? ?")
MEM_AVAIL_MB=$(free -m 2>/dev/null | awk '/^Mem:/{print $7}' || true)
DISK_PCT=$(df -Pk "$REPO" 2>/dev/null | awk 'NR==2{gsub("%","",$5); print $5}' || true)
[[ -n "${MEM_AVAIL_MB:-}" ]] || MEM_AVAIL_MB="?"
[[ -n "${DISK_PCT:-}" ]] || DISK_PCT="?"

TS_INSTALLED=0
TS_IP=""
TS_UP=0
if have tailscale; then
  TS_INSTALLED=1
  TS_IP=$(tailscale ip -4 2>/dev/null | head -1 || true)
  [[ -n "$TS_IP" ]] && TS_UP=1
fi

HERDR_INSTALLED=0
HERDR_VERSION=""
HERDR_SERVER="not running"
if have herdr; then
  HERDR_INSTALLED=1
  HERDR_VERSION=$(herdr --version 2>/dev/null | head -1 || true)
  HERDR_STATUS=$(herdr status 2>/dev/null || true)
  if grep -q 'status: running' <<<"$HERDR_STATUS"; then
    HERDR_SERVER=running
  fi
fi

GIT_OK=0
GIT_BRANCH=""
GIT_LAST=""
GIT_DIRTY=""
GIT_AHEAD=""
GIT_BEHIND=""
if git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  GIT_OK=1
  GIT_BRANCH=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
  GIT_LAST=$(git -C "$REPO" log -1 --format='%h %s' 2>/dev/null || true)
  GIT_DIRTY=$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
  if git -C "$REPO" rev-parse --abbrev-ref '@{u}' >/dev/null 2>&1; then
    read -r GIT_BEHIND GIT_AHEAD < <(git -C "$REPO" rev-list --left-right --count '@{u}...HEAD' 2>/dev/null || echo "? ?") || true
  fi
fi

# Job sentinels: "<name> <done|missing> <exit> <log line>"
JOBS_FOUND=()
JOBS_STATE=()
for job in $JOBS; do
  done_file="$SENTINEL_DIR/$job.done"
  exit_file="$SENTINEL_DIR/$job.exit"
  log_file="$SENTINEL_DIR/$job.log"
  if [[ -f "$done_file" ]]; then
    code=$(cat "$exit_file" 2>/dev/null || echo "?")
    last=$(tail -n 1 "$log_file" 2>/dev/null | cut -c1-80 || true)
    JOBS_FOUND+=("$job")
    JOBS_STATE+=("done exit=$code ${last:+| $last}")
  else
    JOBS_FOUND+=("$job")
    JOBS_STATE+=("missing")
  fi
done

# --------------------------------------------------------------------------
# Assess health once, so human and JSON output agree on the result.
# --------------------------------------------------------------------------
if [[ "$TS_INSTALLED" == "1" && "$TS_UP" == "1" ]]; then :; else FAILS=$((FAILS + 1)); fi

if [[ "$HERDR_INSTALLED" == "0" ]]; then
  if [[ "$ROLE" == "hub" ]]; then FAILS=$((FAILS + 1)); else WARNS=$((WARNS + 1)); fi
elif [[ "$HERDR_SERVER" != "running" && "$ROLE" == "hub" ]]; then
  WARNS=$((WARNS + 1))
fi

if [[ "$GIT_OK" == "1" && "$GIT_DIRTY" != "0" ]]; then
  WARNS=$((WARNS + 1))
fi

for i in "${!JOBS_FOUND[@]}"; do
  state=${JOBS_STATE[$i]}
  [[ "$state" == missing ]] && continue
  code=${state#done exit=}
  code=${code%% *}
  [[ "$code" == "0" ]] || FAILS=$((FAILS + 1))
done

if [[ "$DISK_PCT" =~ ^[0-9]+$ ]]; then
  if (( DISK_PCT >= 95 )); then FAILS=$((FAILS + 1))
  elif (( DISK_PCT >= 85 )); then WARNS=$((WARNS + 1)); fi
else
  WARNS=$((WARNS + 1))
fi

# --------------------------------------------------------------------------
# Human output
# --------------------------------------------------------------------------
if [[ "$JSON" == "0" ]]; then
  printf 'fleet status  %s\n' "$NOW"
  printf '  host: %s%s\n' "$HOST" "${ROLE:+  role: $ROLE}"
  printf '  uptime: %s   load: %s   mem available: %s MB   disk used: %s%%\n' \
    "$UPTIME" "$LOAD" "$MEM_AVAIL_MB" "$DISK_PCT"
  echo

  echo "Tailscale"
  if [[ "$TS_INSTALLED" == "1" ]]; then
    if [[ "$TS_UP" == "1" ]]; then ok "up, tailnet IPv4 $TS_IP"
    else fail "installed but not up (run: sudo tailscale up)"; fi
  else
    fail "not installed (run: SCRIPTS/fleet_bootstrap.sh --role ${ROLE:-hub})"
  fi
  echo

  echo "Herdr"
  if [[ "$HERDR_INSTALLED" == "1" ]]; then
    ok "installed (${HERDR_VERSION:-version unknown})"
    if [[ "$HERDR_SERVER" == "running" ]]; then ok "server running"
    elif [[ "$ROLE" == "hub" ]]; then warn "server not running (start: herdr server)"
    else info "server not running (expected off-hub)"; fi
  else
    if [[ "$ROLE" == "hub" ]]; then fail "not installed (see NewPlan.md §2B)"
    else warn "not installed"; fi
  fi
  echo

  echo "Repository"
  if [[ "$GIT_OK" == "1" ]]; then
    info "$REPO"
    ok "branch $GIT_BRANCH at $GIT_LAST"
    if [[ "$GIT_DIRTY" == "0" ]]; then ok "worktree clean"
    else warn "$GIT_DIRTY uncommitted change(s)"; fi
    if [[ -n "$GIT_AHEAD" && -n "$GIT_BEHIND" ]]; then
      info "vs upstream: $GIT_AHEAD ahead, $GIT_BEHIND behind"
    fi
  else
    warn "$REPO is not a git checkout"
  fi
  echo

  echo "Job sentinels ($SENTINEL_DIR)"
  for i in "${!JOBS_FOUND[@]}"; do
    job=${JOBS_FOUND[$i]}
    state=${JOBS_STATE[$i]}
    if [[ "$state" == missing ]]; then
      info "$job: no sentinel (not running or not finished)"
    else
      code=${state#done exit=}
      code=${code%% *}
      if [[ "$code" == "0" ]]; then ok "$job: $state"
      else fail "$job: $state"; fi
    fi
  done
  echo

  echo "Disk"
  if [[ "$DISK_PCT" =~ ^[0-9]+$ ]]; then
    if (( DISK_PCT >= 95 )); then fail "${DISK_PCT}% used on $REPO"
    elif (( DISK_PCT >= 85 )); then warn "${DISK_PCT}% used on $REPO"
    else ok "${DISK_PCT}% used on $REPO"; fi
  else
    warn "could not read disk usage for $REPO"
  fi

  echo
  if (( FAILS > 0 )); then
    printf 'RESULT: %d fail, %d warn\n' "$FAILS" "$WARNS"
  else
    printf 'RESULT: healthy (%d warn)\n' "$WARNS"
  fi
  exit $(( FAILS > 0 ))
fi

# --------------------------------------------------------------------------
# JSON output
# --------------------------------------------------------------------------
jobs_json=""
for i in "${!JOBS_FOUND[@]}"; do
  job=${JOBS_FOUND[$i]}
  state=${JOBS_STATE[$i]}
  [[ -n "$jobs_json" ]] && jobs_json+=","
  if [[ "$state" == missing ]]; then
    jobs_json+="{\"job\":$(json_str "$job"),\"state\":\"missing\",\"exit\":null,\"last\":\"\"}"
  else
    code=${state#done exit=}
    code=${code%% *}
    last=""
    [[ "$state" == *" | "* ]] && last=${state#* | }
    jobs_json+="{\"job\":$(json_str "$job"),\"state\":\"done\",\"exit\":$(json_str "$code"),\"last\":$(json_str "$last")}"
  fi
done

printf '{'
printf '"host":%s,' "$(json_str "$HOST")"
printf '"role":%s,' "$(json_str "$ROLE")"
printf '"time":%s,' "$(json_str "$NOW")"
printf '"uptime":%s,' "$(json_str "$UPTIME")"
printf '"load":%s,' "$(json_str "$LOAD")"
printf '"mem_available_mb":%s,' "$(json_str "$MEM_AVAIL_MB")"
printf '"disk_used_pct":%s,' "$(json_str "$DISK_PCT")"
printf '"tailscale":{"installed":%s,"up":%s,"ipv4":%s},' \
  "$TS_INSTALLED" "$TS_UP" "$(json_str "$TS_IP")"
printf '"herdr":{"installed":%s,"version":%s,"server":%s},' \
  "$HERDR_INSTALLED" "$(json_str "$HERDR_VERSION")" "$(json_str "$HERDR_SERVER")"
printf '"repo":{"git":%s,"path":%s,"branch":%s,"head":%s,"dirty":%s,"ahead":%s,"behind":%s},' \
  "$GIT_OK" "$(json_str "$REPO")" "$(json_str "$GIT_BRANCH")" "$(json_str "$GIT_LAST")" \
  "$(json_str "$GIT_DIRTY")" "$(json_str "$GIT_AHEAD")" "$(json_str "$GIT_BEHIND")"
printf '"jobs":[%s],' "$jobs_json"
printf '"fails":%d,"warns":%d' "$FAILS" "$WARNS"
printf '}\n'

exit $(( FAILS > 0 ))
