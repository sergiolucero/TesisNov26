#!/usr/bin/env python3
"""Read the machine-written runtime log (`runs/runtimes.jsonl`).

Why this exists
---------------
Every entry point appends one JSON line per run to ``runs/runtimes.jsonl`` (see
``stoch/runtimes.py``). JSON lines are easy to append and awkward to read, so
this prints them as a table with the derived number that actually matters when
planning the next run: **seconds per objective evaluation**.

What it does
------------
Reads the log, filters with ``--script``/``--stage``/``--since``, and prints one
row per record: when, script, grid, method, evaluations, wall clock, s/eval and
status. ``--json`` re-emits the selected records verbatim for further
processing, and ``--summary`` aggregates columns instead of listing rows (mean
wall clock and s/eval per script+grid).

Usage
-----
    ./SCRIPTS/runtimes_report.py                     # last 20 runs
    ./SCRIPTS/runtimes_report.py --all
    ./SCRIPTS/runtimes_report.py --stage opt --summary
    ./SCRIPTS/runtimes_report.py --script run_optimization.py --json | jq .
    ./SCRIPTS/runtimes_report.py --log /tmp/run/runtimes.jsonl

Exit status: 0 if at least one record was printed, 1 if the log is missing or
empty (so a shell `||` fallback works), 2 on a usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent        # SCRIPTS/
ROOT = HERE.parent                            # repository root
DEFAULT_LOG = ROOT / "runs" / "runtimes.jsonl"


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"error: no runtime log at {path} "
                         f"(runs are appended there automatically)")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue   # a truncated tail line must not break the report
    return records


def grid_str(rec: dict) -> str:
    grid = rec.get("grid")
    if not grid or grid[0] is None:
        return "-"
    cells = [str(g) for g in grid if g is not None]
    return "x".join(cells)


def select(records: list[dict], args) -> list[dict]:
    out = records
    if args.script:
        out = [r for r in out if args.script in str(r.get("script", ""))]
    if args.stage:
        out = [r for r in out if str(r.get("stage", "")) == args.stage]
    if args.since:
        out = [r for r in out if str(r.get("when", "")) >= args.since]
    return out if args.all else out[-args.tail:]


def table(rows: list[dict]) -> None:
    head = f"{'when':20s} {'script':26s} {'grid':9s} {'method':8s} " \
           f"{'evals':>6s} {'wall':>9s} {'s/eval':>7s}  status"
    print(head)
    print("-" * len(head))
    for r in rows:
        wall = r.get("wall_clock_human") or f"{r.get('wall_clock_s', 0):.1f} s"
        evals = r.get("n_evaluations")
        per = r.get("s_per_eval")
        per_s = f"{per:.2f}" if isinstance(per, (int, float)) else "-"
        print(f"{str(r.get('when', ''))[:19]:20s} "
              f"{str(r.get('script', ''))[:26]:26s} "
              f"{grid_str(r):9s} {str(r.get('method') or '-'):8s} "
              f"{(str(evals) if evals is not None else '-'):>6s} "
              f"{wall:>9s} {per_s:>7s}  {r.get('status', '-')}")


def summary(rows: list[dict]) -> None:
    keys: dict[tuple, list[dict]] = {}
    for r in rows:
        keys.setdefault((str(r.get("script", "")), grid_str(r),
                         str(r.get("method") or "-")), []).append(r)
    print(f"{'script':26s} {'grid':9s} {'method':8s} {'n':>3s} "
          f"{'mean wall':>10s} {'mean s/eval':>11s}")
    print("-" * 72)
    for (script, grid, method), group in sorted(keys.items()):
        walls = [g.get("wall_clock_s", 0.0) for g in group]
        pers = [g["s_per_eval"] for g in group
                if isinstance(g.get("s_per_eval"), (int, float))]
        mean_wall = sum(walls) / len(walls)
        per = f"{sum(pers) / len(pers):.2f}" if pers else "-"
        print(f"{script[:26]:26s} {grid:9s} {method:8s} {len(group):3d} "
              f"{mean_wall:10.1f} {per:>11s}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--log", default=str(DEFAULT_LOG),
                    help="runtime log to read (default: runs/runtimes.jsonl)")
    ap.add_argument("--tail", type=int, default=20, help="show the last N rows")
    ap.add_argument("--all", action="store_true", help="show every row")
    ap.add_argument("--script", default=None, help="substring of the script name")
    ap.add_argument("--stage", default=None, help="exact stage value")
    ap.add_argument("--since", default=None,
                    help="ISO timestamp prefix, e.g. 2026-11-17")
    ap.add_argument("--summary", action="store_true",
                    help="aggregate instead of listing rows")
    ap.add_argument("--json", action="store_true",
                    help="emit the selected records as JSON lines")
    args = ap.parse_args()

    rows = select(load(Path(args.log)), args)
    if not rows:
        print("no matching runtime records", file=sys.stderr)
        raise SystemExit(1)
    if args.json:
        for r in rows:
            print(json.dumps(r, default=float))
    elif args.summary:
        summary(rows)
    else:
        table(rows)
        print(f"\n{len(rows)} record(s) from {args.log}")


if __name__ == "__main__":
    main()
