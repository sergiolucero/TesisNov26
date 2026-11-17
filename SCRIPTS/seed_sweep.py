#!/usr/bin/env python3
"""seed_sweep.py — multi-seed sweeps with median/IQR reporting.

Why this exists
---------------
Phase 3 (Actions.md) showed that at *fixed* hyperparameters the 1D PINN
baseline's full-grid relative L2 error ranges from 1.17e-2 to 8.14e-2 across
seeds — a 7x spread. Any experiment measured on a single seed is therefore
noise: a lucky seed 0 made `--tau-ramp` look like a 32% improvement that did
not replicate (ramp won 2/5 seeds, median ratio 1.31 the *wrong* way).

So: never compare single runs. Sweep seeds and compare the MEDIAN, with the
spread reported alongside it so a claimed improvement can be judged against
the run-to-run noise it has to beat.

What it does
------------
Runs a command once per seed, injecting ``--seed N``, and collects one scalar
metric per run from that run's JSON manifest (``--metric``, a dotted path like
``metrics.regions.full``). Prints a per-seed table, then median / IQR / min-max
per arm, and finally a paired comparison between arms on the seeds they share.

Usage
-----
    # one arm (baseline), 5 seeds
    ./SCRIPTS/seed_sweep.py \
        --arm base --seeds 0-4 --metric metrics.regions.full \
        --outdir runs/sweeps/base -- \
        python3 -u SCRIPTS/solve_1d_pinn.py --hard-bc --eps 1e-3 \
            --adam-steps 10000 --lbfgs-steps 250 \
            --outdir {outdir} --save-model {outdir}/model.pt

    # two arms, compared on the shared seeds
    ./SCRIPTS/seed_sweep.py --seeds 0-4 --metric metrics.regions.full \
        --outdir runs/sweeps/cmp \
        --arm base -- python3 -u SCRIPTS/solve_1d_pinn.py --hard-bc --eps 1e-3 \
            --outdir {outdir}/{arm}/s{seed} --save-model {outdir}/{arm}/s{seed}/model.pt \
        --arm ramp -- python3 -u SCRIPTS/solve_1d_pinn.py --hard-bc --eps 1e-3 \
            --tau-ramp --ramp-tau 0.05 --ramp-power 1.0 \
            --outdir {outdir}/{arm}/s{seed} --save-model {outdir}/{arm}/s{seed}/model.pt

Placeholders substituted per run: ``{seed}``, ``{arm}``, ``{outdir}``,
``{manifest}``. ``--seed N`` is appended automatically if the command does not
already contain a ``{seed}``/``--seed`` token.

Exit status: 0 if every run produced a metric, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics as st
import subprocess
import sys
import time
from typing import Any


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def parse_seeds(spec: str) -> list[int]:
    """Parse ``0-4`` / ``0,2,5`` / ``0-3,7`` into a list of ints."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            if not (lo_s.strip() and hi_s.strip()):
                raise SystemExit(f"bad seed range '{part}': expected LO-HI")
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError:
                raise SystemExit(
                    f"bad seed range '{part}': not an integer range") from None
            if hi < lo:
                raise SystemExit(f"bad seed range '{part}': end < start")
            out.extend(range(lo, hi + 1))
        else:
            try:
                out.append(int(part))
            except ValueError:
                raise SystemExit(f"bad seed '{part}': not an integer") from None
    if not out:
        raise SystemExit(f"no seeds parsed from '{spec}'")
    return out


def dig(obj: Any, dotted: str) -> Any:
    """Fetch ``a.b.c`` from nested dicts, or raise KeyError."""
    cur = obj
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(dotted)
        cur = cur[key]
    return cur


def median_iqr(values: list[float]) -> dict[str, float]:
    """Median and inter-quartile range. IQR is the spread a claim must beat."""
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return {}
    if n == 1:
        return {"median": vals[0], "q1": vals[0], "q3": vals[0], "iqr": 0.0}

    def quantile(p: float) -> float:
        # Linear interpolation, matching numpy's default method.
        pos = (n - 1) * p
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return vals[lo] * (1 - frac) + vals[hi] * frac

    q1, q3 = quantile(0.25), quantile(0.75)
    return {
        "median": quantile(0.5),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": vals[0],
        "max": vals[-1],
        "mean": st.fmean(vals),
    }


def find_manifest(outdir: str, explicit: str | None) -> str | None:
    """Locate the run manifest: explicit path, else the first *.json in outdir."""
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    candidates = ["run_config.json", "summary.json", "metrics.json"]
    for name in candidates:
        p = os.path.join(outdir, name)
        if os.path.isfile(p):
            return p
    if os.path.isdir(outdir):
        for name in sorted(os.listdir(outdir)):
            if name.endswith(".json"):
                return os.path.join(outdir, name)
    return None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a command across seeds and report median/IQR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring for examples.",
    )
    ap.add_argument("--seeds", default="0-4",
                    help="seed list/range, e.g. 0-4 or 0,2,5 (default: 0-4)")
    ap.add_argument("--metric", required=True,
                    help="dotted path to a scalar in the run JSON, "
                         "e.g. metrics.regions.full")
    ap.add_argument("--arm", action="append", default=[],
                    metavar="NAME", help="start a named arm; the command "
                                         "follows after '--'")
    ap.add_argument("--outdir", default="runs/sweeps/sweep",
                    help="root directory for per-run output")
    ap.add_argument("--manifest", default=None,
                    help="explicit manifest path template "
                         "(default: auto-detect inside {outdir})")
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="per-run wall-clock limit in seconds (0 = none)")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep going when a run fails (default: stop)")
    # Everything after the final '--' that is not part of an --arm block.
    args, rest = ap.parse_known_args()

    # Rebuild arms: each `--arm NAME -- cmd...` block.
    argv = sys.argv[1:]
    arms: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--arm":
            name = argv[i + 1]
            j = i + 2
            if j >= len(argv) or argv[j] != "--":
                raise SystemExit(f"--arm {name}: expected '--' before the command")
            j += 1
            cmd: list[str] = []
            while j < len(argv) and argv[j] != "--arm":
                cmd.append(argv[j])
                j += 1
            if not cmd:
                raise SystemExit(f"--arm {name}: empty command")
            arms.append((name, cmd))
            i = j
        else:
            i += 1

    if not arms:
        # Single-arm form: everything after the first bare '--' is the command.
        if "--" in argv:
            cmd = argv[argv.index("--") + 1:]
        else:
            cmd = []
        if not cmd:
            raise SystemExit("no command given; use --arm NAME -- cmd, "
                             "or pass the command after '--'")
        arms = [("run", cmd)]

    seeds = parse_seeds(args.seeds)
    os.makedirs(args.outdir, exist_ok=True)

    t_start = time.time()
    results: dict[str, dict[int, float]] = {}
    failures: list[str] = []

    for arm_name, cmd_tpl in arms:
        results[arm_name] = {}
        for seed in seeds:
            run_dir = os.path.join(args.outdir, arm_name, f"s{seed}")
            os.makedirs(run_dir, exist_ok=True)
            manifest = (args.manifest
                        .replace("{seed}", str(seed))
                        .replace("{arm}", arm_name)
                        .replace("{outdir}", run_dir)) if args.manifest else None

            cmd = [
                c.replace("{seed}", str(seed))
                 .replace("{arm}", arm_name)
                 .replace("{outdir}", run_dir)
                 .replace("{manifest}", manifest or "")
                for c in cmd_tpl
            ]
            # Only inject --seed when the template does not already specify it.
            if not any(("{seed}" in c or c == "--seed") for c in cmd_tpl):
                cmd += ["--seed", str(seed)]

            label = f"{arm_name}/s{seed}"
            print(f"[sweep] {label}: {shlex.join(cmd)}", flush=True)
            t0 = time.time()
            try:
                proc = subprocess.run(
                    cmd, timeout=(args.timeout or None), check=False)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = 124
            except FileNotFoundError as exc:
                print(f"[sweep] {label}: cannot execute: {exc}", file=sys.stderr)
                failures.append(label)
                if not args.continue_on_error:
                    return 1
                continue
            dt = time.time() - t0

            if rc != 0:
                print(f"[sweep] {label}: exit {rc} after {dt:.1f}s",
                      file=sys.stderr)
                failures.append(label)
                if not args.continue_on_error:
                    return 1
                continue

            mpath = find_manifest(run_dir, manifest)
            if not mpath:
                print(f"[sweep] {label}: no manifest found in {run_dir}",
                      file=sys.stderr)
                failures.append(label)
                if not args.continue_on_error:
                    return 1
                continue
            try:
                with open(mpath, encoding="utf-8") as f:
                    value = dig(json.load(f), args.metric)
                results[arm_name][seed] = float(value)
                print(f"[sweep] {label}: {args.metric} = {float(value):.4e} "
                      f"({dt:.1f}s)", flush=True)
            except (KeyError, TypeError, ValueError) as exc:
                print(f"[sweep] {label}: metric '{args.metric}' "
                      f"unreadable in {mpath}: {exc}", file=sys.stderr)
                failures.append(label)
                if not args.continue_on_error:
                    return 1

    # ----------------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------------
    print()
    print(f"seed sweep — metric {args.metric}, seeds {args.seeds}")
    print(f"root: {args.outdir}")
    print()

    width = max(6, max((len(a) for a, _ in arms), default=6))
    header = f"{'seed':>5} " + " ".join(f"{a:>{width}}" for a, _ in arms)
    print(header)
    print("-" * len(header))
    all_seeds = sorted({s for r in results.values() for s in r})
    for seed in all_seeds:
        cells = []
        for arm_name, _ in arms:
            v = results[arm_name].get(seed)
            cells.append(" " * (width - 10) + f"{v:.4e}" if v is not None
                         else f"{'-':>{width}}")
        print(f"{seed:>5} " + " ".join(cells))
    print()

    print(f"{'arm':<{width}} {'n':>3} {'median':>11} {'IQR':>11} "
          f"{'min':>11} {'max':>11}")
    print("-" * (width + 52))
    for arm_name, _ in arms:
        vals = list(results[arm_name].values())
        if not vals:
            print(f"{arm_name:<{width}} {0:>3} {'(no successful runs)':>40}")
            continue
        s = median_iqr(vals)
        print(f"{arm_name:<{width}} {len(vals):>3} {s['median']:>11.4e} "
              f"{s['iqr']:>11.4e} {s['min']:>11.4e} {s['max']:>11.4e}")
    print()

    # Paired comparison on shared seeds — the only honest way to compare arms.
    if len(arms) > 1:
        base_name = arms[0][0]
        print(f"paired comparison vs '{base_name}' (shared seeds only):")
        for arm_name, _ in arms[1:]:
            shared = sorted(set(results[base_name]) & set(results[arm_name]))
            if not shared:
                print(f"  {arm_name}: no shared successful seeds")
                continue
            ratios = [results[arm_name][s] / results[base_name][s]
                      for s in shared]
            better = sum(1 for r in ratios if r < 1)
            print(f"  {arm_name}: n={len(shared)} "
                  f"median ratio={st.median(ratios):.2f} "
                  f"(<1 means {arm_name} better) "
                  f"| better on {better}/{len(shared)} seeds")
        print()
        print("  A difference is only meaningful if it exceeds the spread above.")
        print("  Prefer n>=5 per arm; with n<5 report it as preliminary.")

    if failures:
        print(f"\n[sweep] {len(failures)} failed run(s): {', '.join(failures)}")
    print(f"\n[sweep] total wall time {(time.time() - t_start) / 60:.1f} min")

    # Persist the summary next to the runs.
    summary = {
        "metric": args.metric,
        "seeds": seeds,
        "arms": {a: {str(s): v for s, v in r.items()} for a, r in results.items()},
        "stats": {a: median_iqr(list(r.values())) for a, r in results.items()
                  if r},
        "failures": failures,
    }
    spath = os.path.join(args.outdir, "sweep_summary.json")
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[sweep] summary -> {spath}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
