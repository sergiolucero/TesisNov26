#!/usr/bin/env bash
# run_app.sh — run the Flask figure viewer in the background on port 5001.
#
# Usage:
#   ./SCRIPTS/run_app.sh [start|stop|status]
#   PORT=5001 PYTHON=python3 ./SCRIPTS/run_app.sh start
#
# Defaults to "start" when no argument is given. Writes app.pid and app.log
# next to this script. The viewer itself is SCRIPTS/app.py; it serves the
# repository's out/ and figs/ directories, so the repo root is the cwd.

set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname -- "$HERE")"
cd "$ROOT"

PORT="${PORT:-5001}"
PY="${PYTHON:-python3}"
LOG="$HERE/app.log"
PIDFILE="$HERE/app.pid"

is_running() {
  [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

case "${1:-start}" in
  start)
    if is_running; then
      echo "already running (pid $(cat "$PIDFILE")) on port $PORT"
      exit 0
    fi
    nohup "$PY" "$HERE/app.py" --host 0.0.0.0 --port "$PORT" > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 1
    if is_running; then
      echo "started http://localhost:$PORT (pid $(cat "$PIDFILE"), log $LOG)"
    else
      echo "failed to start; see $LOG" >&2
      rm -f "$PIDFILE"
      exit 1
    fi
    ;;
  stop)
    if is_running; then
      kill "$(cat "$PIDFILE")"
      rm -f "$PIDFILE"
      echo "stopped"
    else
      echo "not running"
      rm -f "$PIDFILE"
    fi
    ;;
  status)
    if is_running; then
      echo "running (pid $(cat "$PIDFILE")) on port $PORT"
    else
      echo "not running"
    fi
    ;;
  *)
    echo "usage: $0 [start|stop|status]" >&2
    exit 1
    ;;
esac
