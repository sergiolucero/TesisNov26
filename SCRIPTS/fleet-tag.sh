#!/usr/bin/env bash
# SCRIPTS/fleet-tag.sh
# ====================
# Apply the TESIS fleet ACL tags (DOCS/NewPlan.md §6) to a node.
#
# The policy these tags implement lives in SCRIPTS/fleet-acl.hujson. This
# script only assigns a tag to the node it runs on; it never edits the policy.
#
# Usage:
#   ./SCRIPTS/fleet-tag.sh                       # show this node + suggested tag
#   ./SCRIPTS/fleet-tag.sh --dry-run --role hub  # print the exact command
#   sudo ./SCRIPTS/fleet-tag.sh --role hub --yes # actually tag the node
#   ./SCRIPTS/fleet-tag.sh --show                # read back the current tags
#
#   ./SCRIPTS/fleet-tag.sh --role hub|relay|surface|worker [--yes] [--dry-run]
#   ./SCRIPTS/fleet-tag.sh --tag tag:custom-name [--yes]
#   ./SCRIPTS/fleet-tag.sh --show
#
# Roles map to the tags declared in fleet-acl.hujson:
#   hub     -> tag:server-hub       (runs herdr, bare repo; may SSH to relay)
#   relay   -> tag:server-relay     (public HTTPS; never initiates to the hub)
#   surface -> tag:client-surface   (tablet control plane)
#   worker  -> tag:worker           (ephemeral Colab/DO; outbound only)
#
# WHY THIS IS A SCRIPT AND NOT A ONE-LINER
#   1. `tailscale set` has NO --advertise-tags flag (checked on 1.104.x); tags
#      are only settable with `tailscale up --advertise-tags=...`.
#   2. `tailscale up` with any flag demands the COMPLETE settings set: an
#      unspecified flag whose default differs from the node's current value
#      makes it fail ("flags must be the complete set of desired settings").
#      So we read the live prefs and re-state every non-default one.
#   3. Tagging drops the node's user ownership. If the node authenticates with
#      an auth key this is seamless; if it is user-owned (the default for an
#      interactive `tailscale up`), the control plane will want re-auth — and
#      re-auth over SSH can drop the connection. Hence --yes is required and
#      the warning below.
#
# SAFETY
#   * Dry run by default: prints the command and exits 0 without changing
#     anything. Nothing is applied unless --yes is passed.
#   * The ACL must already contain the matching `tagOwners` entry, or the
#     control plane rejects the tag. Verify step 1 of fleet-acl.hujson first.
#   * Never run this on a node while it is your only path to that node.
#
# Exit status: 0 ok/dry-run, 1 the change failed, 2 bad usage.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

ROLE=""
TAG=""
SHOW=0
YES=0
DRY_RUN=1        # must opt in to changing anything

log()  { printf '[tag] %s\n' "$*"; }
warn() { printf '[tag] warning: %s\n' "$*" >&2; }
die()  { printf '[tag] error: %s\n' "$*" >&2; exit 2; }

usage() { sed -n '2,/^set -Eeuo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role)    ROLE=${2:-}; shift 2 ;;
    --tag)     TAG=${2:-}; shift 2 ;;
    --show)    SHOW=1; shift ;;
    --yes|-y)  YES=1; DRY_RUN=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)         die "unknown argument '$1' (try --help)" ;;
  esac
done

have() { command -v "$1" >/dev/null 2>&1; }
have tailscale || die "tailscale is not installed on this node"

# --------------------------------------------------------------------------
# Read live state. `tailscale status --json` works unprivileged; prefs need
# root, so fall back gracefully when we cannot read them.
# --------------------------------------------------------------------------
status_json=$(tailscale status --json 2>/dev/null || true)
[[ -n "$status_json" ]] || die "cannot read 'tailscale status --json'"

read -r CUR_TAGS CUR_HOST CUR_USER <<<"$(printf '%s' "$status_json" | python3 -c '
import json, sys
d = json.load(sys.stdin)
s = d.get("Self") or {}
tags = ",".join(s.get("Tags") or []) or "-"
print(tags, s.get("HostName") or "-", s.get("UserID") or "-")
')"

if [[ "$SHOW" == "1" ]]; then
  log "host: $CUR_HOST"
  if [[ "$CUR_TAGS" == "-" ]]; then
    log "tags: (none — node is untagged / user-owned)"
  else
    log "tags: $CUR_TAGS"
  fi
  exit 0
fi

# --------------------------------------------------------------------------
# Resolve the tag to apply.
# --------------------------------------------------------------------------
case "$ROLE" in
  hub)     ROLE_TAG="tag:server-hub" ;;
  relay)   ROLE_TAG="tag:server-relay" ;;
  surface) ROLE_TAG="tag:client-surface" ;;
  worker)  ROLE_TAG="tag:worker" ;;
  "")      ROLE_TAG="" ;;
  *)       die "unknown role '$ROLE' (use hub|relay|surface|worker)" ;;
esac

if [[ -n "$ROLE" && -n "$TAG" ]]; then
  die "pass either --role or --tag, not both"
fi
TARGET=${TAG:-$ROLE_TAG}

if [[ -z "$TARGET" ]]; then
  # No role given: infer from the hostname so a bare invocation is still useful.
  case "$CUR_HOST" in
    ubuntu-hub)        TARGET="tag:server-hub" ;;
    do-relay)          TARGET="tag:server-relay" ;;
    surface-ctrl)      TARGET="tag:client-surface" ;;
    colab-t4|*worker*) TARGET="tag:worker" ;;
    *)                 TARGET="" ;;
  esac
  if [[ -z "$TARGET" ]]; then
    warn "cannot infer a role from hostname '$CUR_HOST'"
    warn "pass --role hub|relay|surface|worker (or --tag tag:...)"
    warn "tags must be declared in SCRIPTS/fleet-acl.hujson tagOwners"
    exit 2
  fi
  log "inferred $TARGET from hostname '$CUR_HOST'"
fi

# Normalize: tailscale accepts a bare name and adds "tag:".
[[ "$TARGET" == tag:* ]] || TARGET="tag:$TARGET"

log "host:     $CUR_HOST"
if [[ "$CUR_TAGS" == "-" ]]; then
  log "current:  (none)"
else
  log "current:  $CUR_TAGS"
fi
log "target:   $TARGET"

if [[ "$CUR_TAGS" == "$TARGET" ]]; then
  log "already tagged $TARGET — nothing to do"
  exit 0
fi
if [[ "$CUR_TAGS" != "-" && "$CUR_TAGS" != "$TARGET" ]]; then
  warn "node already has tags: $CUR_TAGS"
  warn "tagging replaces the set; pass every tag in one --advertise-tags list"
fi

# --------------------------------------------------------------------------
# Build the complete `tailscale up` invocation.
#
# Every setting we can read is re-stated, because `up` with flags requires the
# complete set. Prefs need root; if unreadable we still emit the safe subset
# and let the operator compare against `tailscale debug prefs`.
# --------------------------------------------------------------------------
PREF_FLAGS=()
prefs=$( (tailscale debug prefs 2>/dev/null) || true )
if [[ -z "$prefs" ]]; then
  # Do NOT fall through with an empty flag set: `tailscale up` with any flag
  # resets unspecified settings to their defaults. Refuse instead of risking it.
  die "cannot read current prefs (needs root), so the flag set would be
       incomplete and could reset this node's settings. Re-run with sudo, or
       grant operation once with: sudo tailscale set --operator=\$USER"
fi
if ! PREF_LINES=$(printf '%s' "$prefs" | python3 -c '
import json, sys
p = json.load(sys.stdin)
out = []
# Emit only flags that differ from the `tailscale up` defaults, so the command
# stays readable. Flags verified against `tailscale up --help` (v1.104.x).
if p.get("AcceptDNS") is False:        out.append("--accept-dns=false")
if p.get("AcceptRoutes") is True:      out.append("--accept-routes")
if p.get("ShieldsUp") is True:         out.append("--shields-up")
if p.get("ExitNodeIP"):                out.append("--exit-node=" + str(p["ExitNodeIP"]))
if p.get("Hostname"):                  out.append("--hostname=" + str(p["Hostname"]))
if p.get("OperatorUser"):              out.append("--operator=" + str(p["OperatorUser"]))
nm = p.get("NetfilterMode")
# 0=off, 1=nodivert, 2=on. The `up` default is on, so only emit when not on.
if nm == 0:     out.append("--netfilter-mode=off")
elif nm == 1:   out.append("--netfilter-mode=nodivert")
if p.get("RunSSH") is True:            out.append("--ssh")
if p.get("AdvertiseExitNode") is True: out.append("--advertise-exit-node")
routes = p.get("AdvertiseRoutes") or []
if routes: out.append("--advertise-routes=" + ",".join(routes))
print("\n".join(out))
'); then
  die "could not parse 'tailscale debug prefs'; refusing to build a command"
fi
if [[ -n "$PREF_LINES" ]]; then
  mapfile -t PREF_FLAGS <<<"$PREF_LINES"
fi

CMD=(tailscale up --advertise-tags="$TARGET" "${PREF_FLAGS[@]}")

# --------------------------------------------------------------------------
# Report / execute.
# --------------------------------------------------------------------------
echo
log "command (run as root):"
printf '  sudo %s\n' "${CMD[*]}"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  cat <<EOF
[tag] DRY RUN — nothing changed. Before applying:

  1. Is $TARGET declared in SCRIPTS/fleet-acl.hujson tagOwners?
     If not, the control plane will reject the tag. Paste the policy first.
  2. Tagging drops this node's user ownership. If it was claimed by a user
     login (not an auth key), expect a re-auth prompt; over SSH that can drop
     the session. Have console access to the node or use an auth key:
       sudo tailscale up --advertise-tags=$TARGET --auth-key=tskey-auth-...
  3. Apply when ready:
       sudo ./SCRIPTS/fleet-tag.sh --role ${ROLE:-<role>} --yes

Verify afterwards:
  ./SCRIPTS/fleet-tag.sh --show
  ./SCRIPTS/fleet_status.sh --role ${ROLE:-hub}     # Tags line goes [ok]
  tailscale ping ${CUR_HOST}
EOF
  exit 0
fi

[[ "$YES" == "1" ]] || die "internal: --yes required to apply"
[[ "$(id -u)" == "0" ]] || die "must run as root: sudo $0 --role ${ROLE:-?} --yes"

log "applying: ${CMD[*]}"
if ! "${CMD[@]}"; then
  die "failed; the previous tag set is unchanged. Re-run 'tailscale status' to check"
fi

echo
log "done. reading back:"
tailscale status --json 2>/dev/null | python3 -c '
import json, sys
s = (json.load(sys.stdin).get("Self") or {})
tags = s.get("Tags") or []
print("[tag] tags now:", ",".join(tags) if tags else "(none)")
' || true
log "verify reachability from another node before tagging the next one"
