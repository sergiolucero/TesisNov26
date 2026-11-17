"""Machine-written runtime log.

Why this exists
---------------
Wall-clock numbers are the cheapest thing to record and the first thing to be
lost: a run prints "done in 898.1 s" to a terminal, the terminal closes, and the
next planning session re-derives the cost model from scratch (see
``DOCS/USAGE.md`` §9 and ``RUNTIMES.txt``, which is a hand-curated table).
Every entry point therefore appends one JSON line per run to
``runs/runtimes.jsonl`` -- one record per *run*, not per iteration, so the file
stays readable.

What it records
---------------
The timestamp, host, script, argv, git revision, wall clock in seconds and
hours, and whatever run-specific fields the caller passes (grid, method, budget,
evaluations, outdir, status). Nothing else: no environment, no timings of inner
stages. ``runs/runtimes.jsonl`` is append-only and safe to ``tail -f``.

Usage
-----
From a script::

    from stoch.runtimes import log_runtime, human_duration

    t0 = time.time()
    ...
    log_runtime(ROOT / "runs/runtimes.jsonl", "run_optimization.py",
                time.time() - t0, grid=(cfg["domain"]["nx"], cfg["domain"]["ny"]),
                stage="opt", status="ok")

From a shell::

    tail -3 runs/runtimes.jsonl | python3 -m json.tool --json-lines
    python3 -c "import json;[print(r['when'], r['script'], round(r['wall_clock_s'],1))
                for r in map(json.loads, open('runs/runtimes.jsonl'))]"

The writer never raises on a filesystem error (a read-only checkout must not
fail a run because a log line could not be appended); it prints a warning
instead.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["log_runtime", "human_duration", "DEFAULT_LOG"]

DEFAULT_LOG = Path("runs") / "runtimes.jsonl"


def human_duration(seconds: float) -> str:
    """`898.3 s` -> `15m 0s` (or `1h 02m` past an hour)."""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _git_rev() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def log_runtime(path: str | Path, script: str, wall_clock_s: float,
                argv: list[str] | None = None, **fields) -> Path | None:
    """Append one JSON record to `path`; return it (or None if the write failed).

    `fields` are recorded verbatim, so callers decide what identifies the run
    (grid, method, budget, evaluations, outdir, status). Values must be
    JSON-serialisable; numpy scalars are converted when they arrive as
    ``.item()``-capable objects.
    """
    path = Path(path)
    record: dict = {
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "script": Path(script).name,
        "argv": [str(a) for a in (sys.argv[1:] if argv is None else argv)],
        "pid": os.getpid(),
        "wall_clock_s": round(float(wall_clock_s), 3),
        "wall_clock_h": round(float(wall_clock_s) / 3600.0, 4),
        "wall_clock_human": human_duration(wall_clock_s),
    }
    rev = _git_rev()
    if rev:
        record["git"] = rev
    for key, value in fields.items():
        record[key] = value.item() if hasattr(value, "item") else value
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=float) + "\n")
    except OSError as exc:  # a read-only checkout must not fail the run
        print(f"[runtimes] WARNING: could not append to {path}: {exc}")
        return None
    print(f"[runtimes] {record['script']} {record['wall_clock_human']} "
          f"-> {path}")
    return path
