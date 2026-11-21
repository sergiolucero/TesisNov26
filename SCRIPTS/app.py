"""
app.py
======

Minimal Flask viewer for the TESIS groundwater-transport PINN/FNO figures.

Serves the ``out/`` directory as Flask's static folder and renders a homepage
that walks through every figure with a short discussion of what to expect.

Routes:
    /            annotated figure homepage
    /plots       plot gallery + single-plot viewer
    /fleet       fleet health dashboard (``SCRIPTS/fleet_status.sh --json``)
    /runs        current and recent Colab runs (``SCRIPTS/colab_status.py``)
    /api/colab   same Colab snapshot as JSON, for scripts and monitors
                 (alias: /runs.json; ``/api/colab/<run_id>`` for one run)

Run:
    python SCRIPTS/app.py            # http://localhost:5000
    python SCRIPTS/app.py --port 8080

Author: TESIS (groundwater remediation PINN/FNO debugging)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
)

BASE = os.path.dirname(os.path.abspath(__file__))   # SCRIPTS/
ROOT = os.path.dirname(BASE)                        # repository root
OUT = os.path.join(ROOT, "out")      # static folder (PINN run outputs)
FIGS = os.path.join(ROOT, "figs")    # data-generator demo figures
TEMPLATES = os.path.join(ROOT, "templates")
FLEET_STATUS = os.path.join(BASE, "fleet_status.sh")

# Fleet status is a shell-out with a hard timeout: the dashboard must never
# hang because a tailnet call or a disk is slow.
FLEET_TIMEOUT = 15.0

# Colab run status comes from the local run directories; the `colab sessions`
# probe is optional and bounded (see /runs?live=1 in colab_status.py).
if BASE not in sys.path:
    sys.path.insert(0, BASE)
from colab_status import collect_runs, log_text  # noqa: E402

DEFAULT_RUNS_LIMIT = 8
DEFAULT_RUNS_REFRESH = 20
DEFAULT_LIVE_TIMEOUT = 8.0

# File types treated as browsable plots by the /plots gallery.
PLOT_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".pdf"}

app = Flask(__name__, static_folder=OUT, static_url_path="/out",
            template_folder=TEMPLATES)


@app.route("/figs/<path:filename>")
def figs(filename: str):
    """Serve the data-generator demo figures from ``figs/``."""
    return send_from_directory(FIGS, filename)


# --------------------------------------------------------------------------
# Plot browser: walks ``out/`` and exposes every plot through a gallery
# (/plots) and a single-plot viewer with previous/next navigation.
# --------------------------------------------------------------------------
def _plot_files() -> list[str]:
    """Return every plot under ``OUT`` as a sorted, POSIX-style relpath."""
    plots: list[str] = []
    for root, dirs, files in os.walk(OUT):
        dirs.sort()
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in PLOT_EXTS:
                rel = os.path.relpath(os.path.join(root, name), OUT)
                plots.append(rel.replace(os.sep, "/"))
    return plots


def _safe_plot_path(relpath: str) -> str | None:
    """Normalize ``relpath`` and reject anything outside ``OUT``."""
    rel = os.path.normpath(relpath.replace("\\", "/").lstrip("/"))
    if rel.startswith("..") or os.path.isabs(rel):
        return None
    if not os.path.isfile(os.path.join(OUT, rel)):
        return None
    return rel.replace(os.sep, "/")


def _fleet_snapshot(role: str) -> dict:
    """Run ``SCRIPTS/fleet_status.sh --json`` and parse it.

    Returns the parsed snapshot plus an ``error`` key (None when the script
    ran). ``fleet_status.sh`` exits 1 when a check fails, but still prints
    valid JSON, so a non-zero exit is not treated as an error by itself.
    """
    cmd = ["bash", FLEET_STATUS, "--json"]
    if role:
        cmd += ["--role", role]
    if not os.path.isfile(FLEET_STATUS):
        return {"error": f"not found: {FLEET_STATUS}"}
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FLEET_TIMEOUT,
            cwd=BASE,
        )
    except FileNotFoundError:
        return {"error": "bash is not available on this host"}
    except subprocess.TimeoutExpired:
        return {"error": f"fleet_status.sh timed out after {FLEET_TIMEOUT:g}s"}

    try:
        snapshot = json.loads(proc.stdout)
    except json.JSONDecodeError:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else "no output"
        return {"error": f"fleet_status.sh produced no JSON ({tail})"}

    snapshot["error"] = None
    return snapshot


@app.route("/fleet")
def fleet():
    """Fleet health dashboard fed by ``SCRIPTS/fleet_status.sh --json``.

    Rendered server-side so the tablet needs no JavaScript and a stale tailnet
    still shows the last successful checks rather than a blank page. Pass
    ``?role=hub`` to flag a stopped Herdr server as a warning.
    """
    role = (request.args.get("role") or os.environ.get("FLEET_ROLE", "")).strip()
    snapshot = _fleet_snapshot(role)
    return render_template(
        "fleet.html",
        snapshot=snapshot,
        error=snapshot.get("error"),
        role=role,
        status_script=os.path.relpath(FLEET_STATUS, BASE),
    )


# --------------------------------------------------------------------------
# Colab run status: current and recent `multi_colab.py` / `new_high_colab.py`
# runs.  Everything is read from the run directories; the live VM probe is an
# explicit opt-in because it needs the Colab CLI and the network.
# --------------------------------------------------------------------------
def _int_value(raw, default: int, low: int, high: int) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def _flag(raw) -> bool:
    """Interpret a query/env value as a boolean switch (``0/1``, ``yes``…)."""
    return str(raw if raw is not None else "").strip().lower() in {
        "1", "true", "yes", "on"}


def _colab_params() -> tuple[bool, int, float, bool]:
    """Shared ``(live, limit, live_timeout, live_logs)`` parsing for the routes."""
    live = _flag(request.args.get("live")
                 or os.environ.get("COLAB_RUNS_LIVE", ""))
    limit = _int_value(request.args.get("limit",
                                        os.environ.get("COLAB_RUNS_LIMIT",
                                                       DEFAULT_RUNS_LIMIT)),
                       DEFAULT_RUNS_LIMIT, 1, 50)
    live_timeout = float(_int_value(
        os.environ.get("COLAB_RUNS_LIVE_TIMEOUT", DEFAULT_LIVE_TIMEOUT),
        int(DEFAULT_LIVE_TIMEOUT), 1, 30))
    live_logs = _flag(request.args.get("live_logs")
                      or os.environ.get("COLAB_RUNS_LIVE_LOGS", ""))
    return live, limit, live_timeout, live_logs


@app.route("/runs")
def colab_runs():
    """Status of the current and most recent Colab runs.

    ``?live=1`` additionally runs ``colab sessions`` (bounded timeout),
    ``?live_logs=1`` pulls live replicas' ``run.log`` off the VM (also
    bounded) so a foreground run shows real ``iter/evals`` progress instead of
    ``quiet``, and ``?refresh=0`` disables the meta refresh.  Rendering is
    server-side so the page works without JavaScript.
    """
    live, limit, live_timeout, live_logs = _colab_params()

    refresh_raw = request.args.get("refresh")
    if refresh_raw is None:
        refresh_raw = "0" if live else os.environ.get(
            "COLAB_RUNS_REFRESH", DEFAULT_RUNS_REFRESH)
    refresh = _int_value(refresh_raw, DEFAULT_RUNS_REFRESH, 0, 600)
    if 0 < refresh < 5:
        refresh = 5

    error = None
    try:
        snapshot = collect_runs(limit=limit, live=live,
                                live_timeout=live_timeout,
                                live_logs=live_logs)
    except Exception as exc:                       # keep the dashboard up
        snapshot = None
        error = f"{type(exc).__name__}: {exc}"
    return render_template("runs.html", snap=snapshot, error=error,
                           refresh=refresh, live=live, live_logs=live_logs,
                           status_script=os.path.relpath(
                               os.path.join(BASE, "colab_status.py"), ROOT))


@app.route("/runs/log/<kind>/<run_id>")
def colab_log(kind: str, run_id: str):
    """Tail of one replica's driver log, served as plain text.

    ``?remote=1`` fetches the VM's own ``run.log`` for a replica that is still
    running -- the local copy is frozen until the run cell returns.
    """
    replica = _int_value(request.args.get("replica", 0), 0, 0, 999)
    remote = _flag(request.args.get("remote"))
    found = log_text(kind, run_id, replica, remote=remote)
    if found is None:
        abort(404)
    text, _ = found
    return Response(text, mimetype="text/plain")


# --------------------------------------------------------------------------
# Machine-readable twin of /runs, for cron jobs, tailnet probes and the phone.
# Same snapshot, no HTML: ``curl -sf localhost:5001/api/colab | jq .pipeline``.
# --------------------------------------------------------------------------
def _json_response(payload, status: int = 200) -> Response:
    """JSON reply with caching disabled -- pollers must never see a stale run."""
    resp = jsonify(payload)
    resp.status_code = status
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _pipeline(snapshot: dict) -> dict:
    """One-glance answer to "is the current Colab pipeline running?".

    The run list is newest-first, so the first *active* run is the pipeline
    that is in flight; with nothing active, the newest run says how the last
    attempt ended.  ``state`` is one word, safe to match on in a script.
    """
    runs = snapshot.get("runs") or []
    current = [run for run in runs if run.get("is_current")]
    focus = current[0] if current else (runs[0] if runs else None)

    if not runs:
        state = "idle"
    elif current:
        state = "running"
    else:
        state = focus.get("state", "unknown")

    if focus is None:
        return {"state": state, "active": 0, "run": None,
                "progress_pct": None, "headline": snapshot.get("headline")}

    rep_progress = [rep.get("progress", {}).get("pct")
                    for rep in focus.get("replicas") or []
                    if rep.get("progress", {}).get("pct") is not None]
    compact = {
        "run_id": focus.get("run_id"),
        "kind": focus.get("kind"),
        "state": focus.get("state"),
        "tone": focus.get("tone"),
        "started": focus.get("started"),
        "age_text": focus.get("age_text"),
        "elapsed_h": focus.get("elapsed_h"),
        "gpu": focus.get("gpu"),
        "work": focus.get("work"),
        "dir": focus.get("dir"),
        "error": focus.get("error"),
        "artifacts": focus.get("artifacts_count"),
        "replicas": len(focus.get("replicas") or []),
    }
    return {
        "state": state,
        "active": len(current),
        "run": compact,
        "progress_pct": max(rep_progress) if rep_progress else None,
        "headline": snapshot.get("headline"),
    }


@app.route("/api/colab")
@app.route("/runs.json")
def colab_api():
    """JSON snapshot of the current and recent Colab runs.

    Same data as ``/runs`` and ``colab_status.py --json``, plus a ``pipeline``
    block (``state`` is ``running`` / ``idle`` / ``finished`` / ``failed`` / …)
    so a monitor can decide with one field. Query parameters:

    ``?live=1``      also probe ``colab sessions`` (bounded, network)
    ``?live_logs=1`` pull live replicas' run.log off the VM (bounded, network)
    ``?limit=N``     keep at most N runs (1-50, default 8)
    ``?run=<id>``    restrict the ``runs`` list to one run id (``counts`` and
                     ``total`` still describe every run on disk)
    """
    live, limit, live_timeout, live_logs = _colab_params()
    wanted = (request.args.get("run") or "").strip()
    try:
        snapshot = collect_runs(limit=limit if not wanted else 1000,
                                live=live, live_timeout=live_timeout,
                                live_logs=live_logs)
    except Exception as exc:                       # keep monitors up, but say so
        return _json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                               "pipeline": {"state": "error", "active": 0,
                                            "run": None}},
                              status=500)

    if wanted:
        snapshot["runs"] = [run for run in snapshot["runs"]
                            if run.get("run_id") == wanted]
        snapshot["shown"] = len(snapshot["runs"])
        if not snapshot["runs"]:
            return _json_response({"ok": False, "error": f"unknown run: {wanted}",
                                   "pipeline": {"state": "unknown", "active": 0,
                                                "run": None}},
                                  status=404)

    payload = {"ok": True, "error": None, **snapshot}
    payload["pipeline"] = _pipeline(snapshot)
    payload["endpoint"] = {
        "page": url_for("colab_runs"),
        "logs": "/runs/log/<kind>/<run_id>?replica=<n>",
    }
    return _json_response(payload)


@app.route("/api/colab/<run_id>")
def colab_run_api(run_id: str):
    """One run's full snapshot, with its replicas, progress and log tails.

    The run's own ``error`` (its failure line, if any) stays inside ``run`` so
    the top-level ``error`` always means "the API call failed".
    """
    try:
        snapshot = collect_runs(limit=1000)
    except Exception as exc:
        return _json_response({"ok": False,
                               "error": f"{type(exc).__name__}: {exc}"}, status=500)
    for run in snapshot["runs"]:
        if run.get("run_id") == run_id:
            return _json_response({"ok": True, "error": None,
                                   "generated": snapshot["generated"],
                                   "run": run})
    return _json_response({"ok": False, "error": f"unknown run: {run_id}"}, status=404)


@app.route("/plots")
@app.route("/plots/")
def plots():
    """Gallery of every plot found under ``out/``, grouped by directory."""
    groups: dict[str, list[str]] = {}
    for rel in _plot_files():
        groups.setdefault(os.path.dirname(rel) or ".", []).append(rel)
    return render_template("plots.html", groups=groups,
                           total=sum(len(v) for v in groups.values()))


@app.route("/plots/<path:relpath>")
def plot_view(relpath: str):
    """Single-plot viewer with links to the previous/next plot."""
    rel = _safe_plot_path(relpath)
    if rel is None:
        abort(404)
    files = _plot_files()
    try:
        idx = files.index(rel)
    except ValueError:
        abort(404)
    return render_template(
        "plot.html",
        rel=rel,
        index=idx + 1,
        total=len(files),
        prev=files[idx - 1] if idx > 0 else None,
        next=files[idx + 1] if idx < len(files) - 1 else None,
        src=url_for("static", filename=rel),
    )


# --------------------------------------------------------------------------
# Curated homepage content: one section per experiment, one entry per figure.
# --------------------------------------------------------------------------
INTRO = (
    "Dimensionless 1D advection–dispersion equation (homogeneous, isotropic, "
    "non-reactive aquifer): <em>u</em><sub>τ</sub> + <em>u</em><sub>ξ</sub> = "
    "(1/Pe) <em>u</em><sub>ξξ</sub>, with <em>u</em>(ξ,0)=0, "
    "<em>u</em>(0,τ)=1, <em>u</em>(1,τ)=0, Pe=20, τ<sub>max</sub>=0.5. "
    "The analytical reference is Ogata–Banks (1961)."
)

CORNER_NOTE = (
    "The step-inlet problem is discontinuous at the corner (ξ,τ)=(0,0): the "
    "initial condition says u=0 while the inlet BC says u=1. The reference "
    "returns 1 there, but any continuous ansatz returns 0. So a bright |error| "
    "spot confined to that corner is a <strong>measure-zero metric artifact, "
    "not a bug</strong> — it is what inflates the full-grid rel-L2 vs. the "
    "interior error."
)

SECTIONS = [
    {
        "id": "ground-truth",
        "title": "1. Ground truth — data generator (Ogata–Banks)",
        "summary": (
            "Synthetic reference fields from <code>hydrology.py</code>. These "
            "define the target the PINN must reproduce."
        ),
        "figures": [
            {
                "src": "/figs/field.png",
                "title": "field.png — full u(ξ,τ) field",
                "discussion": (
                    "Heatmap of u(ξ,τ). Expect u≈0 along the bottom (τ=0, the "
                    "initial condition) and u≈1 along the left edge (ξ=0, the "
                    "inlet). A contaminant front sweeps diagonally along "
                    "ξ≈τ; at Pe=20 it is sharp but smeared over a transition "
                    "zone of width ~2√(τ/Pe). Behind the front u≈1, ahead "
                    "u→0."
                ),
            },
            {
                "src": "/figs/profiles.png",
                "title": "profiles.png — concentration snapshots",
                "discussion": (
                    "Six u-vs-ξ slices at increasing τ. Each is an S-shaped "
                    "erfc profile (≈1 at ξ=0 decaying to ≈0 at large ξ). As τ "
                    "grows the front moves right and widens (dispersion "
                    "accumulates over time)."
                ),
            },
        ],
    },
    {
        "id": "soft-bc",
        "title": "2. Soft-BC baseline (out/pinn)",
        "summary": (
            "Plain MLP (tanh hidden, sigmoid output); IC/inlet/outlet are "
            "enforced as soft loss terms. <strong>rel-L2 = 2.43e-2</strong> "
            "(full), 1.64e-2 interior, 3.79e-2 in the left boundary layer."
        ),
        "figures": [
            {
                "src": "/out/pinn/loss.png",
                "title": "loss.png — Adam loss history (semilogy)",
                "discussion": (
                    "Total, PDE, IC and BC losses. Expect a fast drop in the "
                    "first ~1000 iterations then a noisy plateau; the noise is "
                    "expected because collocation points are re-sampled every "
                    "step. The PDE term typically dominates the total."
                ),
            },
            {
                "src": "/out/pinn/comparison.png",
                "title": "comparison.png — PINN vs reference vs |error|",
                "discussion": (
                    "Left: PINN u<sub>θ</sub>; middle: Ogata–Banks reference; "
                    "right: |error|. The two field panels should look alike, "
                    "with the error concentrated near the left boundary layer "
                    "(ξ<0.05) and the corner — the soft BCs are the weak "
                    "point."
                ),
            },
            {
                "src": "/out/pinn/profiles.png",
                "title": "profiles.png — PINN (solid) vs reference (dashed)",
                "discussion": (
                    "Solid/dashed curves should overlap for τ away from 0; the "
                    "mismatch is largest near ξ=0 at early τ, where the soft "
                    "inlet BC has to balance against the IC."
                ),
            },
        ],
    },
    {
        "id": "hardbc-rational",
        "title": "3. Hard-BC, rational ramp (out/pinn_hardbc)",
        "summary": (
            "Hard-constrained ansatz u = (1−ξ)s + ξ(1−ξ)s·NN with "
            "s(τ)=τ/(τ+eps), eps=1e-3. IC and outlet BC exact, inlet "
            "approximate. <strong>rel-L2 = 2.02e-2</strong>, left layer "
            "1.91e-2 (improved), interior 1.76e-2 (slightly worse than soft)."
        ),
        "figures": [
            {
                "src": "/out/pinn_hardbc/loss.png",
                "title": "loss.png — Adam loss history (semilogy)",
                "discussion": (
                    "Only the total and PDE losses are meaningful; IC and BC "
                    "are enforced architecturally so their loss is identically "
                    "zero."
                ),
            },
            {
                "src": "/out/pinn_hardbc/comparison.png",
                "title": "comparison.png — PINN vs reference vs |error|",
                "discussion": (
                    "The rational ramp tightens the left boundary layer but is "
                    "slower to reach u≈1 at the inlet, so a thin inlet strip "
                    "still shows up in the error panel."
                ),
            },
            {
                "src": "/out/pinn_hardbc/profiles.png",
                "title": "profiles.png — PINN (solid) vs reference (dashed)",
                "discussion": (
                    "Overlap is good except at the smallest τ, where the "
                    "smoothing width eps of the inlet ramp is visible."
                ),
            },
        ],
    },
    {
        "id": "hardbc-exp",
        "title": "4. Hard-BC, exponential ramp — BEST (out/pinn_hardbc_exp)",
        "summary": (
            "Hard ansatz with s(τ)=1−exp(−τ/eps), eps=1e-3. "
            "<strong>rel-L2 = 1.35e-2</strong> full grid, "
            "<strong>8.0e-3 interior</strong> (target ≤1e-2). The full-grid "
            "number is dominated by the corner singularity."
        ),
        "figures": [
            {
                "src": "/out/pinn_hardbc_exp/loss.png",
                "title": "loss.png — Adam loss history (semilogy)",
                "discussion": (
                    "Only total + PDE are non-zero (IC/BC are architectural). "
                    "Expect the usual fast drop then noisy plateau."
                ),
            },
            {
                "src": "/out/pinn_hardbc_exp/comparison.png",
                "title": "comparison.png — PINN vs reference vs |error|",
                "discussion": (
                    "The error panel should be dark almost everywhere, with a "
                    "bright spot only at the bottom-left corner (ξ,τ)→(0,0) — "
                    "the expected corner singularity (see note above)."
                ),
            },
            {
                "src": "/out/pinn_hardbc_exp/profiles.png",
                "title": "profiles.png — PINN (solid) vs reference (dashed)",
                "discussion": (
                    "Solid and dashed curves should be nearly indistinguishable "
                    "away from τ=0; only the far-left edge at the earliest τ "
                    "may deviate slightly."
                ),
            },
        ],
    },
    {
        "id": "hardbc-eps4",
        "title": "5. Hard-BC, exponential ramp eps=1e-4 — unstable "
                 "(out/pinn_hardbc_eps4)",
        "summary": (
            "Same ansatz but a 10× steeper inlet ramp. "
            "<strong>rel-L2 = 5.05e-2</strong>, interior 3.80e-2. The steep "
            "s(τ) near τ=0 destabilizes training — a cautionary data point "
            "showing eps matters."
        ),
        "figures": [
            {
                "src": "/out/pinn_hardbc_eps4/loss.png",
                "title": "loss.png — Adam loss history (semilogy)",
                "discussion": (
                    "Expect a noisier, slower-converging PDE curve than the "
                    "eps=1e-3 run, reflecting the harder near-inlet gradient."
                ),
            },
            {
                "src": "/out/pinn_hardbc_eps4/comparison.png",
                "title": "comparison.png — PINN vs reference vs |error|",
                "discussion": (
                    "The error panel shows a wider, more visible mismatch band "
                    "near the inlet/left boundary layer, not just the corner."
                ),
            },
            {
                "src": "/out/pinn_hardbc_eps4/profiles.png",
                "title": "profiles.png — PINN (solid) vs reference (dashed)",
                "discussion": (
                    "More visible separation between solid and dashed curves at "
                    "early τ near ξ=0 compared to the eps=1e-3 run."
                ),
            },
        ],
    },
]


@app.route("/")
def index():
    return render_template("index.html", intro=INTRO, corner=CORNER_NOTE,
                           sections=SECTIONS)


def main() -> None:
    ap = argparse.ArgumentParser(description="Flask viewer for TESIS figures")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
