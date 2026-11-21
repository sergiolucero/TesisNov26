#!/usr/bin/env python3
"""colab_status.py — read-only status of current and recent Colab runs.

Why this exists
---------------
The Colab drivers write everything needed to answer "what is running?" to
disk, but in three different places:

* ``multi_colab.py launch`` -> ``runs/colab_multi/<run-id>/plan.json`` +
  ``r<k>/run.log`` / ``cli.log`` / ``artifacts.zip``;
* ``new_high_colab.py --launch`` -> ``out/high/<run-id>/cli.log`` +
  ``artifacts.zip``;
* the live VM list is only available from ``colab sessions``.

This module normalizes those sources into one JSON-able snapshot so the Flask
page (``/runs``) and the command line show the same numbers.  It never writes
to a run directory and never talks to Colab unless ``--live`` is requested.

State is inferred, not guessed from a single timestamp: an explicit exit
marker wins, then the persisted ``plan.json`` result, then the last CLI stage
and the age of the newest log line.  A run whose log has been silent for
``QUIET_SECONDS`` is still reported, but as ``quiet`` instead of ``running``,
because a long transport solve can legitimately print nothing for minutes.

That local log is also why a perfectly healthy run can look silent: ``colab
exec`` only streams a cell's output home when the cell *returns*, and the run
cell holds the kernel for hours.  ``--live-logs`` closes that gap by pulling
the VM's own ``run.log`` with ``colab download`` -- the filesystem API, which
answers immediately -- and feeding it through the same parser, so ``quiet``
becomes real ``iter/evals`` progress.  Never ask a *running* session for a log
with ``colab exec``: that cell queues behind the run for as long as it lasts.

Usage
-----
    ./SCRIPTS/colab_status.py                    # human summary
    ./SCRIPTS/colab_status.py --json             # the /runs page snapshot
    ./SCRIPTS/colab_status.py --live --json      # also `colab sessions`
    ./SCRIPTS/colab_status.py --live-logs        # real progress from the VM
    ./SCRIPTS/colab_status.py --limit 20         # more history

Exit status: 0 when at least one run directory was found (even if all runs
failed), 1 when there is nothing to show, 2 on a usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent        # SCRIPTS/
ROOT = HERE.parent                            # repository root
DEFAULT_MULTI = ROOT / "runs" / "colab_multi"
DEFAULT_HIGH = ROOT / "out" / "high"

TAIL_BYTES = 160 * 1024        # enough for the last iterations, cheap to parse
LIVE_TIMEOUT = 8.0             # `colab sessions` must never hang the dashboard
QUIET_SECONDS = 10 * 60        # no output for this long -> "quiet", not "running"
STARTING_SECONDS = 5 * 60      # a plan without any CLI activity is still starting
REMOTE_LOG_TIMEOUT = 30.0      # `colab download` of a remote log, per replica
REMOTE_LOG_MAX = 400 * 1024    # tail of the remote log that is re-parsed
REMOTE_LOG_STATES = {"running", "quiet", "starting", "provisioning"}

RUN_MARKER = "[multi] run exit="
PROVISION_MARKER = "[multi] provision exit="
HIGH_RUN_MARKER = "[high] run exit="
HIGH_PROVISION_MARKER = "[high] provision exit="

CURRENT_STATES = {"running", "quiet", "provisioning", "starting",
                  "pending", "detached"}

STATE_TONES = {
    "finished": "ok",
    "running": "ok",
    "quiet": "warn",
    "detached": "warn",
    "provisioning": "warn",
    "starting": "warn",
    "pending": "warn",
    "interrupted": "warn",
    "failed": "fail",
    "dry-run": "idle",
    "stopped": "idle",
    "unknown": "idle",
}

# multi_colab's persisted result states -> the words the page uses.
RESULT_STATES = {
    "finished": ("finished", "ok"),
    "detached": ("detached", "warn"),
    "pending": ("pending", "warn"),
    "interrupted": ("interrupted", "warn"),
    "run-failed": ("failed", "fail"),
    "setup-failed": ("failed", "fail"),
    "upload-failed": ("failed", "fail"),
    "provision-failed": ("failed", "fail"),
    "start-failed": ("failed", "fail"),
}

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
ITER_RE = re.compile(r"\biter\s+(\d+)\b[^\n]*?\bevals=\s*(\d+)")
EVAL_RE = re.compile(r"\beval\s+(\d+)\b")
START_RE = re.compile(r"\bopt start\s+(\d+)/(\d+)")
HIGH_EVALS_RE = re.compile(r"\b(\d+)\s+evaluations\b")


# ---------------------------------------------------------------------------
# Small text/time helpers
# ---------------------------------------------------------------------------
def _clean(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "")


def _tail_text(path: Path, max_bytes: int = TAIL_BYTES) -> tuple[str, bool]:
    """Return (decoded tail, truncated).  A partial first line is dropped."""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            raw = fh.read()
    except OSError:
        return "", False
    text = _clean(raw.decode("utf-8", errors="replace"))
    truncated = size > max_bytes
    if truncated:
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
    return text, truncated


def _last_lines(text: str, count: int) -> list[str]:
    lines = [ln.rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln.strip()]
    return lines[-count:]


def _last_meaningful_line(text: str) -> str:
    """Last non-empty line, preferring a driver log line over CLI decoration."""
    lines = _last_lines(text, 40)
    for line in reversed(lines):
        if re.search(r"\[(opt|high|collab_run|multi|colab)\]", line):
            return line[:400]
    return lines[-1][:400] if lines else ""


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _file_age(path: Path, now: float) -> float | None:
    mtime = _mtime(path)
    return None if mtime is None else max(0.0, now - mtime)


def _human_age(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    return f"{seconds // 86400}d {seconds % 86400 // 3600}h"


def _human_hours(hours: float | None) -> str:
    if hours is None:
        return "—"
    seconds = max(0.0, float(hours) * 3600.0)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.2f}h"


def _human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.0f} KB"
    return f"{num_bytes / 1e6:.1f} MB"


def _parse_started(value: str | None, fallback: float) -> float:
    """Parse plan.json's local timestamp or a 14-digit run id."""
    if value:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return time.mktime(time.strptime(value, fmt))
            except (ValueError, OverflowError):
                pass
        if re.fullmatch(r"\d{14}", value):
            try:
                return time.mktime(time.strptime(value, "%Y%m%d%H%M%S"))
            except (ValueError, OverflowError):
                pass
    return fallback


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace(os.sep, "/")
    except (ValueError, OSError):
        return str(path)


def _marker_exit(text: str, marker: str) -> int | None:
    for line in reversed(text.splitlines()):
        if marker in line:
            found = re.search(r"-?\d+", line.split(marker, 1)[1])
            if found:
                return int(found.group(0))
    return None


def _all_dry_run(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return bool(lines) and all(ln.startswith("# dry-run:") for ln in lines)


def _has_cli_failure(text: str) -> bool:
    """Whether the *last* CLI call failed, not an earlier retried attempt."""
    needles = ("Connection was lost", "RuntimeError",
               "Traceback (most recent call last)",
               "Session terminated unexpectedly")
    last_failure = max(text.rfind(needle) for needle in needles)
    if last_failure == -1:
        return False
    last_section = text.rfind("\n### ")
    return last_failure > last_section


def _failure_line(text: str) -> str | None:
    for line in reversed(_last_lines(text, 80)):
        low = line.lower()
        if any(word in low for word in ("error", "failed", "lost", "traceback")):
            return line[:400]
    return None


def _last_cli_stage(text: str) -> str | None:
    """Which lifecycle call the last ``### colab ...`` line represents."""
    for line in reversed(text.splitlines()):
        if not line.startswith("### "):
            continue
        command = line[4:].strip()
        if re.search(r"\bupload\b", command):
            return "upload"
        if re.search(r"\bdownload\b", command):
            return "download"
        if re.search(r"\bstop\b", command):
            return "stop"
        if " -f " in command:
            return "run" if re.search(r"run\.py\s*$", command) else "provision"
        if re.search(r"\blog\b", command):
            return "log"
        if re.search(r"\bexec\b", command):
            return "exec"
        if re.search(r"\bnew\b", command):
            return "new"
        if re.search(r"\bsessions\b", command):
            return "sessions"
    return None


def _activity_age(paths: list[Path], now: float) -> float | None:
    ages = [age for path in paths if (age := _file_age(path, now)) is not None]
    return min(ages) if ages else None


def _state_tone(state: str) -> str:
    return STATE_TONES.get(state, "idle")


def _quiet(state: str, activity_age: float | None) -> str:
    if state == "running" and activity_age is not None \
            and activity_age > QUIET_SECONDS:
        return "quiet"
    return state


# ---------------------------------------------------------------------------
# Progress parsing
# ---------------------------------------------------------------------------
def _progress(text: str, cfg: dict | None = None) -> dict[str, Any]:
    """Best-effort stage/iteration progress from a driver log."""
    cfg = cfg or {}
    opt = cfg.get("optim") or {}
    n_starts = _int(opt.get("n_starts"))
    max_iter = _int(opt.get("max_iter"))
    max_evals = _int(opt.get("max_evals"))

    start = None
    for match in START_RE.finditer(text):
        start = (int(match.group(1)), int(match.group(2)))
    starts_total = start[1] if start else n_starts
    total_evals = (starts_total * max_evals
                   if starts_total and max_evals else None)
    iteration = None
    evals = None
    for match in ITER_RE.finditer(text):
        iteration = int(match.group(1))
        evals = int(match.group(2))
    if evals is None:
        for match in EVAL_RE.finditer(text):
            evals = int(match.group(1))
    if evals is None:
        for match in HIGH_EVALS_RE.finditer(text):
            evals = int(match.group(1))

    stage = None
    stage_matches = re.findall(r"=== running (\w+)", text)
    if stage_matches:
        stage = stage_matches[-1]
    else:
        high_stages = re.findall(r"=== stage (\w+) ===", text)
        if high_stages:
            stage = high_stages[-1]
        elif re.search(r"\bstage opt\b", text):
            stage = "opt"
        elif re.search(r"\bopt: config for this card\b", text):
            stage = "opt"

    pct = None
    if total_evals and evals is not None:
        pct = max(0.0, min(100.0, 100.0 * evals / total_evals))
    elif start:
        done = (start[0] - 1)
        if max_iter and iteration is not None:
            done += iteration / max(1, max_iter)
        pct = max(0.0, min(100.0, 100.0 * done / max(1, start[1])))

    bits: list[str] = []
    if stage:
        bits.append(f"stage {stage}")
    if start:
        bits.append(f"start {start[0]}/{start[1]}")
    if iteration is not None:
        bits.append(f"iter {iteration}" + (f"/{max_iter}" if max_iter else ""))
    if evals is not None:
        bits.append(f"{evals}" + (f"/{total_evals} evals" if total_evals else " evals"))
    return {
        "stage": stage,
        "iteration": iteration,
        "evals": evals,
        "max_evals": total_evals,
        "pct": None if pct is None else round(pct, 1),
        "text": " · ".join(bits) or "—",
    }


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# multi_colab.py runs
# ---------------------------------------------------------------------------
def _multi_replica(rep: dict, results: dict[int, dict], run_dir: Path,
                   plan: dict, now: float,
                   remote: tuple[str, float] | None = None) -> dict[str, Any]:
    """One replica's view.  ``remote`` is ``(log text, fetched_at)`` from the VM."""
    index = int(rep.get("index", 0))
    session = rep.get("session") or f"r{index}"
    rdir = Path(rep.get("local_dir") or run_dir / f"r{index}")
    if not rdir.exists():
        rdir = run_dir / f"r{index}"
    run_log = rdir / "run.log"
    cli_log = rdir / "cli.log"
    history = rdir / "history.md"
    artifact = rdir / "artifacts.zip"
    run_text, truncated = _tail_text(run_log)
    source = "local"
    if remote is not None:
        # The VM's own run.log.  For a live run this is the only fresh
        # telemetry there is, so it replaces the local file rather than
        # merging with it (the local file is just the frozen exec header).
        run_text, source = remote[0], "remote"
        truncated = False
    cli_text, _ = _tail_text(cli_log)
    activity_age = _activity_age([run_log, cli_log, history, artifact], now)
    if remote is not None:
        # A successful pull *is* proof of life, even if run.log is unchanged.
        activity_age = max(0.0, now - remote[1])

    result = results.get(index) or {}
    state: str
    if result.get("state"):
        state, _ = RESULT_STATES.get(str(result["state"]),
                                     (str(result["state"]), "idle"))
    elif _all_dry_run(cli_text):
        state = "dry-run"
    elif (exit_code := _marker_exit(run_text, RUN_MARKER)) is not None:
        state = "finished" if exit_code == 0 else "failed"
    elif _has_cli_failure(cli_text) or _has_cli_failure(run_text):
        state = "failed"
    else:
        stage = _last_cli_stage(cli_text)
        # The run cell's own `### ... exec ... run.py` marker is written to
        # run.log, not cli.log, so a run.log with any content means the run
        # has started even while the last cli.log stage still reads provision.
        if stage == "run" or (run_text.strip() and stage != "stop"):
            state = "running"
        elif stage in {"new", "upload", "provision", "exec"}:
            state = "provisioning"
        elif stage == "stop":
            state = "stopped"
        elif cli_text.strip():
            state = "starting"
        else:
            plan_age = _file_age(run_dir / "plan.json", now) or 0.0
            state = "starting" if plan_age < STARTING_SECONDS else "unknown"
    state = _quiet(state, activity_age)

    exit_code = result.get("exit")
    if exit_code is None:
        exit_code = _marker_exit(run_text, RUN_MARKER)
    elapsed_h = result.get("elapsed_h")
    progress = _progress(run_text, plan.get("config"))
    error = None
    if state == "failed":
        error = _failure_line(cli_text) or _failure_line(run_text)
        if not error and exit_code is not None:
            error = f"exit {exit_code}"
        error = error or "failed"

    artifacts_present = artifact.is_file()
    return {
        "index": index,
        "session": session,
        "state": state,
        "state_label": state,
        "tone": _state_tone(state),
        "elapsed_h": elapsed_h,
        "elapsed_text": _human_hours(elapsed_h) if elapsed_h is not None else "—",
        "exit": exit_code,
        "vm": result.get("vm"),
        "activity_age_s": None if activity_age is None else round(activity_age, 1),
        "activity_text": ("no output yet" if activity_age is None
                          else f"last output {_human_age(activity_age)} ago"),
        "progress": progress,
        "log_source": source,
        "last_line": _last_meaningful_line(run_text) or _last_meaningful_line(cli_text),
        "tail": _last_lines(run_text, 8) or _last_lines(cli_text, 8),
        "log_path": _rel(run_log if run_log.exists() else cli_log, ROOT),
        "log_exists": run_log.exists() or cli_log.exists(),
        "artifacts": {
            "present": artifacts_present,
            "bytes": artifact.stat().st_size if artifacts_present else 0,
            "text": _human_size(artifact.stat().st_size) if artifacts_present else "none",
            "path": _rel(artifact, ROOT) if artifacts_present else None,
        },
        "error": error,
        "truncated": truncated,
    }


def _multi_run(plan_path: Path, now: float, live_logs: bool = False,
               log_timeout: float = REMOTE_LOG_TIMEOUT) -> dict[str, Any] | None:
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    run_dir = plan_path.parent
    started_ts = _parse_started(plan.get("created"), plan_path.stat().st_mtime)
    results = {int(r.get("index", -1)): r for r in plan.get("results") or []}
    replicas = [_multi_replica(rep, results, run_dir, plan, now)
                for rep in (plan.get("replicas") or [])]
    if live_logs:
        replicas = _with_remote_logs(plan, results, run_dir, replicas, now,
                                     log_timeout)
    states = [rep["state"] for rep in replicas]

    if not replicas:
        state = "unknown"
    elif all(s == "finished" for s in states):
        state = "finished"
    elif all(s == "dry-run" for s in states):
        state = "dry-run"
    elif any(s == "failed" for s in states):
        state = "failed"
    elif any(s in {"running", "quiet"} for s in states):
        state = "running" if any(s == "running" for s in states) else "quiet"
    elif any(s in {"provisioning", "starting"} for s in states):
        state = "provisioning" if any(s == "provisioning" for s in states) else "starting"
    elif all(s == "detached" for s in states):
        state = "detached"
    elif all(s in {"stopped", "unknown"} for s in states):
        state = "stopped" if any(s == "stopped" for s in states) else "unknown"
    else:
        state = "unknown"

    estimate = plan.get("estimate") or {}
    known_elapsed = max((r.get("elapsed_h") or 0.0) for r in results.values()) \
        if results else None
    replica_ages = [rep["activity_age_s"] for rep in replicas
                    if rep["activity_age_s"] is not None]
    activity_age = min(replica_ages) if replica_ages else _file_age(plan_path, now)
    elapsed_h = _elapsed_hours(started_ts, now, state, activity_age,
                               known_elapsed)
    artifacts = [rep["artifacts"] for rep in replicas if rep["artifacts"]["present"]]
    total_bytes = sum(a["bytes"] for a in artifacts)
    cfg = plan.get("config") or {}
    domain = cfg.get("domain") or {}
    timing = cfg.get("time") or {}
    grid = ""
    if domain.get("nx") and domain.get("ny"):
        grid = f"{domain['nx']}×{domain['ny']}"
        if timing.get("nsteps"):
            grid += f"×{timing['nsteps']}"
    work = " · ".join(bit for bit in (
        ",".join(plan.get("models") or []),
        plan.get("preset"),
        grid,
    ) if bit)

    return {
        "kind": "multi_colab",
        "kind_label": "multi_colab.py",
        "run_id": plan.get("run_id") or run_dir.name,
        "started": time.strftime("%Y-%m-%d %H:%M", time.localtime(started_ts)),
        "started_ts": started_ts,
        "age_s": max(0.0, now - started_ts),
        "age_text": _human_age(now - started_ts),
        "state": state,
        "state_label": state,
        "tone": _state_tone(state),
        "gpu": plan.get("gpu") or "cpu",
        "models": ",".join(plan.get("models") or []),
        "preset": plan.get("preset"),
        "grid": grid,
        "work": work or "—",
        "estimate_h": estimate.get("hours"),
        "estimate_text": (_human_hours(estimate["hours"]) + "/replica"
                          if estimate.get("hours") is not None else "—"),
        "elapsed_h": elapsed_h,
        "elapsed_text": _human_hours(elapsed_h),
        "dir": _rel(run_dir, ROOT),
        "replicas": replicas,
        "artifacts_text": (f"{len(artifacts)}/{len(replicas)} · {_human_size(total_bytes)}"
                           if artifacts else "none"),
        "artifacts_count": len(artifacts),
        "error": next((rep["error"] for rep in replicas if rep["error"]), None),
        "log_tail": replicas[0]["tail"] if replicas else [],
    }


# ---------------------------------------------------------------------------
# new_high_colab.py --launch runs
# ---------------------------------------------------------------------------
def _high_argv(rdir: Path) -> list[str]:
    try:
        text = (rdir / "run.py").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        if "SCRIPTS/new_high_colab.py" not in line:
            continue
        command = line.split("SCRIPTS/new_high_colab.py", 1)[1]
        try:
            return shlex.split(command)
        except ValueError:
            return command.split()
    return []


def _elapsed_hours(started_ts: float, now: float, state: str,
                   activity_age: float | None,
                   known: float | None = None) -> float | None:
    """Wall clock for display: known result > live age > last-output time."""
    if known is not None:
        return float(known)
    if state in {"dry-run", "unknown", "pending"}:
        return None
    if state in CURRENT_STATES:
        return max(0.0, (now - started_ts) / 3600.0)
    if activity_age is not None:
        return max(0.0, (now - activity_age - started_ts) / 3600.0)
    return max(0.0, (now - started_ts) / 3600.0)


def _high_run(rdir: Path, now: float) -> dict[str, Any] | None:
    cli_log = rdir / "cli.log"
    artifact = rdir / "artifacts.zip"
    if not (cli_log.exists() or (rdir / "run.py").exists()):
        return None
    text, truncated = _tail_text(cli_log)
    activity_age = _activity_age([cli_log, artifact, rdir / "run.py"], now)

    if _all_dry_run(text):
        state = "dry-run"
    elif (exit_code := _marker_exit(text, HIGH_RUN_MARKER)) is not None:
        state = "finished" if exit_code == 0 else "failed"
    elif (setup := _marker_exit(text, HIGH_PROVISION_MARKER)) is not None and setup != 0:
        state = "failed"
    elif _has_cli_failure(text):
        state = "failed"
    else:
        stage = _last_cli_stage(text)
        if stage == "run":
            state = "running"
        elif stage in {"new", "upload", "provision", "exec"}:
            state = "provisioning"
        elif stage == "download":
            state = "unknown"
        elif stage == "stop":
            state = "stopped"
        else:
            state = "starting"
    state = _quiet(state, activity_age)

    argv = _high_argv(rdir)
    stages = list(dict.fromkeys(
        tok for tok in argv if not tok.startswith("-")
        and tok in {"bench", "forward", "opt", "pinn", "fno", "all"}))
    preset = _flag_value(argv, "--preset")
    nx, ny = _flag_value(argv, "--nx"), _flag_value(argv, "--ny")
    grid = f"{nx}×{ny}" if nx and ny else ""
    work = " · ".join(bit for bit in (" ".join(stages), preset, grid) if bit)

    session = _last_match(text, r"-s\s+([A-Za-z0-9][\w.-]*)") or f"high-{rdir.name}"
    gpu = _last_match(text, r"--gpu\s+([A-Za-z0-9]+)") or "unknown"
    started_ts = _parse_started(rdir.name, _mtime(rdir / "cli.log") or now)
    run_exit = _marker_exit(text, HIGH_RUN_MARKER)
    artifacts_present = artifact.is_file()
    progress = _progress(text)
    high_elapsed = _elapsed_hours(started_ts, now, state, activity_age)
    error = None
    if state == "failed":
        error = _failure_line(text) or (f"exit {run_exit}" if run_exit is not None
                                        else "failed")

    replica = {
        "index": 0,
        "session": session,
        "state": state,
        "state_label": state,
        "tone": _state_tone(state),
        "elapsed_h": None,
        "elapsed_text": "—",
        "exit": run_exit,
        "vm": None,
        "activity_age_s": None if activity_age is None else round(activity_age, 1),
        "activity_text": ("no output yet" if activity_age is None
                          else f"last output {_human_age(activity_age)} ago"),
        "progress": progress,
        "last_line": _last_meaningful_line(text),
        "tail": _last_lines(text, 8),
        "log_path": _rel(cli_log, ROOT),
        "log_exists": cli_log.exists(),
        "artifacts": {
            "present": artifacts_present,
            "bytes": artifact.stat().st_size if artifacts_present else 0,
            "text": _human_size(artifact.stat().st_size) if artifacts_present else "none",
            "path": _rel(artifact, ROOT) if artifacts_present else None,
        },
        "error": error,
        "truncated": truncated,
    }
    return {
        "kind": "high",
        "kind_label": "new_high_colab.py --launch",
        "run_id": rdir.name,
        "started": time.strftime("%Y-%m-%d %H:%M", time.localtime(started_ts)),
        "started_ts": started_ts,
        "age_s": max(0.0, now - started_ts),
        "age_text": _human_age(now - started_ts),
        "state": state,
        "state_label": state,
        "tone": _state_tone(state),
        "gpu": gpu,
        "models": " ".join(stages),
        "preset": preset,
        "grid": grid,
        "work": work or "—",
        "estimate_h": None,
        "estimate_text": "—",
        "elapsed_h": high_elapsed,
        "elapsed_text": _human_hours(high_elapsed),
        "dir": _rel(rdir, ROOT),
        "replicas": [replica],
        "artifacts_text": (f"1/1 · {replica['artifacts']['text']}"
                           if artifacts_present else "none"),
        "artifacts_count": 1 if artifacts_present else 0,
        "error": error,
        "log_tail": replica["tail"],
    }


def _last_match(text: str, pattern: str) -> str | None:
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None


def _flag_value(argv: list[str], flag: str) -> str | None:
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


# ---------------------------------------------------------------------------
# Live sessions
# ---------------------------------------------------------------------------
def live_sessions(binary: str = "colab", auth: str | None = None,
                  timeout: float = LIVE_TIMEOUT) -> dict[str, Any]:
    """Run ``colab sessions`` once, with a hard timeout and no shell."""
    found = shutil.which(binary)
    if found is None:
        return {"ok": False, "command": f"{binary} sessions",
                "output": "", "error": f"`{binary}` is not on PATH"}
    command = [found]
    if auth and auth != "none":
        command += ["--auth", auth]
    command.append("sessions")
    try:
        proc = subprocess.run(command, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "command": shlex.join(command), "output": "",
                "error": f"timed out after {timeout:g}s"}
    except OSError as exc:
        return {"ok": False, "command": shlex.join(command), "output": "",
                "error": str(exc)}
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if len(output) > 8000:
        output = output[:8000] + "\n… (truncated)"
    return {
        "ok": proc.returncode == 0,
        "command": shlex.join(command),
        "output": output,
        "error": None if proc.returncode == 0 else f"exit {proc.returncode}",
    }


# ---------------------------------------------------------------------------
# Remote logs (filesystem API, never the kernel)
# ---------------------------------------------------------------------------
def remote_log_text(session: str, remote_path: str, *, binary: str = "colab",
                    auth: str | None = "oauth2",
                    timeout: float = REMOTE_LOG_TIMEOUT,
                    max_bytes: int = REMOTE_LOG_MAX) -> tuple[str | None, str]:
    """Fetch one remote file with ``colab download``; return ``(text, error)``.

    ``colab exec '!tail -n 40 run.log'`` is the obvious way to read a remote
    log and the wrong one: while the driver's ``%%bash`` run cell holds the
    kernel, an ``exec`` cell queues behind it -- i.e. for hours.  ``download``
    goes through the filesystem API, is read-only, and answers in a second, so
    it is safe to call against a session that is busy running.

    Nothing is written outside a temporary directory, so this is safe for the
    dashboard, whose other promise is that it never touches a run directory.
    """
    found = shutil.which(binary)
    if found is None:
        return None, f"`{binary}` is not on PATH"
    if not session:
        return None, "no session name in plan.json"
    with tempfile.TemporaryDirectory(prefix="colab-status-") as tmp:
        target = Path(tmp) / "remote.log"
        command = [found]
        if auth and auth != "none":
            command += ["--auth", auth]
        command += ["download", "-s", session, remote_path, str(target)]
        try:
            proc = subprocess.run(command, capture_output=True, text=True,
                                  timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return None, f"`{binary} download` timed out after {timeout:g}s"
        except OSError as exc:
            return None, str(exc)
        if proc.returncode != 0 or not target.is_file():
            noise = ((proc.stderr or "") + (proc.stdout or "")).strip()
            last = noise.splitlines()[-1] if noise else f"exit {proc.returncode}"
            return None, last[:300]
        try:
            raw = target.read_bytes()
        except OSError as exc:
            return None, str(exc)
    return _clean(raw[-max_bytes:].decode("utf-8", errors="replace")), ""


def _remote_root(rep: dict, run_id: str) -> str:
    """Directory on the VM that holds ``run.log``/``artifacts.zip``.

    ``remote_config`` (``/content/tesis_<id>/repo/config_r0.yaml``) is the
    plan's own record of where the bundle was unpacked, so prefer it over
    re-deriving the naming convention, and fall back when it is absent.
    """
    remote_config = str(rep.get("remote_config") or "")
    if remote_config.startswith("/content/"):
        parent = posixpath.dirname(posixpath.dirname(remote_config))
        if parent.startswith("/content/"):
            return parent
    return f"/content/tesis_{run_id}"


def _with_remote_logs(plan: dict, results: dict[int, dict], run_dir: Path,
                      replicas: list[dict[str, Any]], now: float,
                      log_timeout: float) -> list[dict[str, Any]]:
    """Re-read live replicas from the VM so ``quiet`` means nothing again."""
    reps = plan.get("replicas") or []
    fetched: dict[int, tuple[str, float]] = {}
    for rep, view in zip(reps, replicas):
        if view["state"] not in REMOTE_LOG_STATES:
            continue
        index = int(rep.get("index", 0))
        remote = f"{_remote_root(rep, plan.get('run_id') or run_dir.name)}/run.log"
        text, _err = remote_log_text(
            rep.get("session") or "", remote,
            binary=os.environ.get("COLAB_BIN", "colab"),
            auth=os.environ.get("COLAB_AUTH", "oauth2"),
            timeout=log_timeout,
        )
        if text and text.strip():
            fetched[index] = (text, time.time())
    if not fetched:
        return replicas
    return [_multi_replica(rep, results, run_dir, plan, now,
                           remote=fetched.get(int(rep.get("index", 0))))
            for rep in reps]


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------
def collect_runs(root: Path = ROOT, limit: int = 8, live: bool = False,
                 live_timeout: float = LIVE_TIMEOUT,
                 multi_dir: Path | None = None,
                 high_dir: Path | None = None,
                 live_logs: bool = False,
                 log_timeout: float = REMOTE_LOG_TIMEOUT) -> dict[str, Any]:
    """Return the JSON-able snapshot used by ``/runs`` and ``--json``."""
    root = Path(root)
    multi_dir = Path(multi_dir) if multi_dir else root / "runs" / "colab_multi"
    high_dir = Path(high_dir) if high_dir else root / "out" / "high"
    now = time.time()
    runs: list[dict[str, Any]] = []

    for path in sorted(multi_dir.glob("*/plan.json")):
        run = _multi_run(path, now, live_logs=live_logs, log_timeout=log_timeout)
        if run:
            runs.append(run)
    for path in sorted(high_dir.iterdir() if high_dir.is_dir() else []):
        if path.is_dir():
            run = _high_run(path, now)
            if run:
                runs.append(run)

    runs.sort(key=lambda r: r["started_ts"], reverse=True)
    total = len(runs)
    runs = runs[:max(0, limit)]
    for run in runs:
        run["is_current"] = run["state"] in CURRENT_STATES

    counts = {"running": 0, "finished": 0, "failed": 0, "other": 0}
    for run in runs:
        if run["state"] in CURRENT_STATES:
            counts["running"] += 1
        elif run["state"] == "finished":
            counts["finished"] += 1
        elif run["state"] == "failed":
            counts["failed"] += 1
        else:
            counts["other"] += 1

    if not runs:
        headline = "No Colab runs found"
        subhead = f"Looked in {_rel(multi_dir, root)}/ and {_rel(high_dir, root)}/."
    else:
        newest = runs[0]
        parts = []
        if counts["running"]:
            parts.append(f"{counts['running']} in progress")
        if counts["finished"]:
            parts.append(f"{counts['finished']} finished")
        if counts["failed"]:
            parts.append(f"{counts['failed']} failed")
        if counts["other"]:
            parts.append(f"{counts['other']} other")
        headline = ", ".join(parts) or "No runs"
        subhead = (f"Newest: {newest['run_id']} — {newest['state']} "
                   f"({newest['started']}, {newest['age_text']} ago).")

    snapshot: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_ts": now,
        "root": str(root),
        "multi_dir": _rel(multi_dir, root),
        "high_dir": _rel(high_dir, root),
        "headline": headline,
        "subhead": subhead,
        "counts": counts,
        "total": total,
        "shown": len(runs),
        "runs": runs,
        "has_current": any(run["is_current"] for run in runs),
        "live": None,
        "live_logs": bool(live_logs),
    }
    if live:
        snapshot["live"] = live_sessions(
            os.environ.get("COLAB_BIN", "colab"),
            os.environ.get("COLAB_AUTH", "oauth2"),
            live_timeout,
        )
    return snapshot


# ---------------------------------------------------------------------------
# Log access for the Flask page
# ---------------------------------------------------------------------------
def log_text(kind: str, run_id: str, replica: int = 0, root: Path = ROOT,
             max_bytes: int = 400_000,
             remote: bool = False) -> tuple[str, str] | None:
    """Return ``(text, relative path)`` for one run's log, or None.

    The path always comes from the snapshot (never from the URL), so a crafted
    ``run_id`` cannot escape the run roots.  Only the tail is returned.

    ``remote=True`` asks the VM for its own ``run.log`` when the replica is
    still live, because the local copy of a live run is only the exec header;
    it falls back to the local file (with a note) if the pull fails.
    """
    wanted = {"multi": "multi_colab", "high": "high"}.get(kind)
    if wanted is None:
        return None
    snapshot = collect_runs(root=root, limit=1000)
    for run in snapshot["runs"]:
        if run["kind"] != wanted or run["run_id"] != run_id:
            continue
        for rep in run["replicas"]:
            if rep["index"] != replica:
                continue
            note = ""
            if remote and wanted == "multi_colab" \
                    and rep["state"] in REMOTE_LOG_STATES:
                remote_path = f"{_remote_run_root(root, run, replica)}/run.log"
                text, err = remote_log_text(
                    rep.get("session") or "", remote_path,
                    binary=os.environ.get("COLAB_BIN", "colab"),
                    auth=os.environ.get("COLAB_AUTH", "oauth2"),
                )
                if text is not None:
                    return text, f"{rep['log_path']} (remote)\n"
                note = (f"[colab_status] remote log unavailable ({err}); "
                        f"showing the local copy, which a live run does not "
                        f"update until its cell returns.\n\n")
            path = (Path(root) / rep["log_path"]).resolve()
            try:
                path.relative_to(Path(root).resolve())
            except ValueError:
                return None
            if not path.is_file():
                return None
            text, truncated = _tail_text(path, max_bytes)
            if truncated:
                text = (f"[colab_status] showing the last {max_bytes} bytes "
                        f"of {rep['log_path']}\n\n" + text)
            return note + text, rep["log_path"]
    return None


def _remote_run_root(root: Path, run: dict[str, Any], replica: int) -> str:
    """Remote run root for one replica, read back out of its ``plan.json``."""
    plan_file = Path(root) / str(run.get("dir") or ".") / "plan.json"
    try:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        plan = {}
    for rep in plan.get("replicas") or []:
        if int(rep.get("index", -1)) == replica:
            return _remote_root(rep, str(run.get("run_id") or ""))
    return f"/content/tesis_{run.get('run_id') or ''}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_human(snapshot: dict[str, Any]) -> None:
    print(f"Colab runs @ {snapshot['generated']}")
    print(f"  {snapshot['headline']}")
    print(f"  {snapshot['subhead']}")
    if not snapshot["runs"]:
        return
    print()
    header = f"{'run':<18} {'kind':<10} {'state':<13} {'gpu':<6} {'elapsed':>8}  work"
    print(header)
    print("-" * len(header))
    for run in snapshot["runs"]:
        print(f"{run['run_id']:<18} {run['kind']:<10} {run['state']:<13} "
              f"{run['gpu']:<6} {run['elapsed_text']:>8}  {run['work']}")
        for rep in run["replicas"]:
            line = (f"    r{rep['index']} {rep['session']}: {rep['state']} "
                    f"{rep['progress']['text']}")
            if rep.get("log_source") == "remote":
                line += " [remote log]"
            if rep["last_line"]:
                line += f" | {rep['last_line']}"
            print(line)
        if run["error"]:
            print(f"    error: {run['error']}")
    live = snapshot.get("live")
    if live:
        print(f"\n$ {live['command']}")
        if live["error"]:
            print(f"  {live['error']}")
        if live["output"]:
            print(live["output"])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--json", action="store_true", help="emit the snapshot as JSON")
    ap.add_argument("--limit", type=int, default=8,
                    help="how many recent runs to include (default: 8)")
    ap.add_argument("--live", action="store_true",
                    help="also run `colab sessions` (network, bounded timeout)")
    ap.add_argument("--live-timeout", type=float, default=LIVE_TIMEOUT,
                    help="timeout for `colab sessions` (default: %(default)s s)")
    ap.add_argument("--live-logs", action="store_true",
                    help="pull live replicas' run.log off the VM with "
                         "`colab download` (network, bounded per replica)")
    ap.add_argument("--log-timeout", type=float, default=REMOTE_LOG_TIMEOUT,
                    help="timeout per remote log pull (default: %(default)s s)")
    ap.add_argument("--multi-dir", default=str(DEFAULT_MULTI),
                    help="multi_colab run root (default: runs/colab_multi)")
    ap.add_argument("--high-dir", default=str(DEFAULT_HIGH),
                    help="new_high_colab --launch run root (default: out/high)")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.limit < 1:
        print("colab_status.py: --limit must be >= 1", file=sys.stderr)
        return 2
    snapshot = collect_runs(
        ROOT, limit=args.limit, live=args.live, live_timeout=args.live_timeout,
        multi_dir=Path(args.multi_dir).expanduser(),
        high_dir=Path(args.high_dir).expanduser(),
        live_logs=args.live_logs, log_timeout=args.log_timeout,
    )
    if args.json:
        print(json.dumps(snapshot, indent=2, sort_keys=False))
    else:
        _print_human(snapshot)
    return 0 if snapshot["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
