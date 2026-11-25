#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
VENV_DIR="./venv"                  # Path to your Python virtual environment
FLASK_MODULE="app:app"             # module:callable for Gunicorn (from app.py)
FLASK_PORT=5000
JUPYTER_PORT=8888
NOTEBOOK_DIR="$(pwd)"
LOG_DIR="./logs"

mkdir -p "$LOG_DIR"

# 1. Resolve the private Tailscale IPv4 address
if ! command -v tailscale &> /dev/null; then
    echo "Error: tailscale CLI not found." >&2
    exit 1
fi

TS_IP=$(tailscale ip -4 | head -n 1)
if [[ -z "$TS_IP" ]]; then
    echo "Error: Could not retrieve Tailscale IPv4. Is tailscaled running and authenticated?" >&2
    exit 1
fi

# 2. Activate virtual environment if present
if [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
fi

# 3. Trap Ctrl+C / SIGTERM to cleanly stop both background processes
cleanup() {
    echo -e "\nShutting down Flask and Jupyter Lab..."
    [[ -n "${FLASK_PID:-}" ]] && kill "$FLASK_PID" 2>/dev/null || true
    [[ -n "${JUPYTER_PID:-}" ]] && kill "$JUPYTER_PID" 2>/dev/null || true
    wait
    echo "Stopped."
}
trap cleanup SIGINT SIGTERM EXIT

# 4. Start Flask via Gunicorn bound strictly to Tailscale IP
echo "Starting Flask on http://${TS_IP}:${FLASK_PORT} ..."
gunicorn \
    --bind "${TS_IP}:${FLASK_PORT}" \
    --workers 2 \
    --access-logfile "${LOG_DIR}/flask_access.log" \
    --error-logfile "${LOG_DIR}/flask_error.log" \
    "$FLASK_MODULE" &
FLASK_PID=$!

# 5. Start Jupyter Lab bound strictly to Tailscale IP
echo "Starting Jupyter Lab on http://${TS_IP}:${JUPYTER_PORT} ..."
jupyter lab \
    --ServerApp.ip="$TS_IP" \
    --ServerApp.port="$JUPYTER_PORT" \
    --ServerApp.open_browser=False \
    --ServerApp.root_dir="$NOTEBOOK_DIR" \
    --ServerApp.allow_remote_access=True \
    > "${LOG_DIR}/jupyter.log" 2>&1 &
JUPYTER_PID=$!

# Brief pause to let Jupyter print its auth token URL to the log
sleep 2
JUPYTER_URL=$(grep -m 1 -oE "http://${TS_IP}:${JUPYTER_PORT}/lab\?token=[a-zA-Z0-9]+" "${LOG_DIR}/jupyter.log" || echo "http://${TS_IP}:${JUPYTER_PORT}/lab (check ${LOG_DIR}/jupyter.log for token)")

echo "------------------------------------------------------------"
echo "Services live on your Tailnet:"
echo "  Flask App   -> http://${TS_IP}:${FLASK_PORT}"
echo "  Jupyter Lab -> ${JUPYTER_URL}"
echo "Press Ctrl+C to stop both."
echo "------------------------------------------------------------"

# Keep script alive while children run
wait
