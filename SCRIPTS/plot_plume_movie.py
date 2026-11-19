#!/usr/bin/env python3
"""plot_plume_movie.py — animate the 2D plume evolution.

Two inputs are supported:

``--fields FILE``
    Read ``C_all`` (and, when present, ``times`` / ``wells`` / ``q``) from a
    ``forward_fields.npz`` written by ``run_2d.py --mode forward``.  This is
    the fast path: no PDE solve, just rendering.

``--run DIR``
    Read ``policy.json`` and ``run_config.json`` from an optimization run,
    rebuild the exact test scenario the optimizer used, apply ``q*``, and solve
    transport while storing every field.  ``--nopump`` animates the same
    scenario with every well off.

The colour scale is fixed over the whole sequence so the plume's growth and
cleanup remain visible; a per-frame scale would make it look stationary.  The
default auto scale is ``max(final-frame peak, 99th percentile of all cells)``,
which keeps the late-time plume visible without letting a one-cell source spike
set the whole range; pass ``--vmax`` to override it.  The regulatory contour and
well/compliance/source markers are drawn on every frame.

Examples
--------
    python3 SCRIPTS/plot_plume_movie.py --fields out/run_2d/forward_fields.npz

    python3 SCRIPTS/plot_plume_movie.py --run out/opt_run --fps 12
    python3 SCRIPTS/plot_plume_movie.py --run out/opt_run --nopump --out out/opt_run/figures/plume_nopump.gif

``.gif`` needs only Pillow; ``.mp4``/``.mov`` additionally needs ffmpeg.
Exit status: 0 ok, non-zero on bad input.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

from physics.fv2d import Grid, transport_run
from stoch.config import default_config, load_config
from stoch.random_fields import ScenarioGenerator
from stoch.report import animate_plume
from stoch.saa import Archive

# The run builders are shared with run_2d.py / run_optimization.py so the
# reconstructed scenario is bit-for-bit the one the optimizer solved.
sys.path.insert(0, str(HERE))
from run_optimization import (  # noqa: E402
    base_aquifer,
    make_cost,
    make_forward,
    make_grid,
    make_kl,
)


def log(msg: str) -> None:
    print(f"[movie] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def _merge(base: dict, upd: dict) -> None:
    """Recursively overlay a partial config (same semantics as load_config)."""
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v


def _config_from_run(run: Path, explicit: str | None) -> tuple[dict, str]:
    """Config precedence: --config, embedded run snapshot, benchmark file."""
    if explicit:
        return load_config(explicit), explicit
    manifest = run / "run_config.json"
    if manifest.is_file():
        with open(manifest, encoding="utf-8") as f:
            m = json.load(f)
        snapshot = m.get("config")
        if isinstance(snapshot, dict) and snapshot:
            cfg = default_config()
            _merge(cfg, snapshot)
            return cfg, f"{manifest} (embedded config)"
        extra = m.get("extra") or {}
        path = extra.get("config_path")
        if not path and isinstance(snapshot, dict):
            path = snapshot.get("_path")
        if path and Path(path).is_file():
            return load_config(path), path
    fallback = ROOT / "configs" / "opt" / "benchmark.yaml"
    return load_config(fallback), str(fallback)


def _config_from_fields(fields: Path, explicit: str | None) -> tuple[dict, str]:
    if explicit:
        return load_config(explicit), explicit
    fallback = ROOT / "configs" / "opt" / "benchmark.yaml"
    return load_config(fallback), str(fallback)


def _grid_for_fields(fields: Path, ncell: int, cfg: dict) -> Grid:
    """Recover geometry from the sibling metrics file when it is consistent."""
    metrics = fields.with_name("forward_metrics.json")
    if metrics.is_file():
        with open(metrics, encoding="utf-8") as f:
            g = (json.load(f).get("grid") or {})
        nx, ny = int(g.get("nx", 0)), int(g.get("ny", 0))
        if nx * ny == ncell and nx > 1 and ny > 1:
            d = cfg["domain"]
            lx = float(g.get("dx", d["lx"] / nx)) * nx
            ly = float(g.get("dy", d["ly"] / ny)) * ny
            log(f"geometry from {metrics.name}: {nx}x{ny}, {lx:g}x{ly:g} m")
            return Grid(nx, ny, lx, ly)

    d = cfg["domain"]
    if int(d["nx"]) * int(d["ny"]) == ncell:
        return Grid(int(d["nx"]), int(d["ny"]),
                    float(d["lx"]), float(d["ly"]))

    raise SystemExit(
        f"error: C_all has {ncell} cells but the config grid is "
        f"{d['nx']}x{d['ny']}; pass --config with the matching domain "
        f"(or run next to forward_metrics.json)"
    )


def _load_fields(args) -> dict:
    path = Path(args.fields)
    if not path.is_file():
        raise SystemExit(f"error: fields file not found: {path}")
    cfg, cfg_src = _config_from_fields(path, args.config)
    with np.load(path, allow_pickle=False) as data:
        if "C_all" not in data:
            raise SystemExit(
                f"error: {path} has no C_all; expected forward_fields.npz "
                "from run_2d.py --mode forward"
            )
        C = np.asarray(data["C_all"], dtype=float)
        if C.ndim != 2:
            raise SystemExit(f"error: C_all must be 2-D, got shape {C.shape}")
        times = (np.asarray(data["times"], dtype=float) if "times" in data
                 else np.arange(C.shape[0], dtype=float) * float(cfg["time"]["dt"]))
        wells = (np.asarray(data["wells"], dtype=float) if "wells" in data
                 else np.empty((0, 2)))
        q = np.asarray(data["q"], dtype=float) if "q" in data else None

    grid = _grid_for_fields(path, C.shape[1], cfg)
    fwd = make_forward(cfg)
    cost = make_cost(cfg)
    pumping = q is not None and bool(np.any(q > 1e-9))
    log(f"fields: {path} ({C.shape[0]} steps, {grid.nx}x{grid.ny})")
    log(f"config: {cfg_src}")
    return {
        "grid": grid,
        "C_fields": C,
        "times": times,
        "wells": np.atleast_2d(wells) if wells.size else wells,
        "q": q,
        "creg": cost.creg,
        "source_xy": fwd.source_xy,
        "compliance_xy": fwd.compliance_xy,
        "obs_xy": fwd.obs_xy,
        "non_pumping": not pumping,
    }


def _load_run(args) -> dict:
    run = Path(args.run)
    if not run.is_dir():
        raise SystemExit(f"error: run directory not found: {run}")
    policy_path = Path(args.policy) if args.policy else run / "policy.json"
    if not policy_path.is_file():
        raise SystemExit(f"error: policy not found: {policy_path}")

    with open(policy_path, encoding="utf-8") as f:
        policy = json.load(f)
    q = np.asarray(policy["q"], dtype=float).ravel()
    wells = np.atleast_2d(np.asarray(policy["wells"], dtype=float))
    if q.size != wells.shape[0]:
        raise SystemExit(
            f"error: policy has {q.size} rates for {wells.shape[0]} wells"
        )

    cfg, cfg_src = _config_from_run(run, args.config)
    grid = make_grid(cfg)
    fwd = make_forward(cfg)
    cost = make_cost(cfg)
    kl = make_kl(cfg, grid)
    rf = cfg["random_field"]
    gen = ScenarioGenerator(
        grid, kl,
        random_source=bool(rf["random_source"]),
        random_gradient=bool(rf["random_gradient"]),
    )

    # Same deterministic test archive as run_optimization.py: seeds are
    # <test seed> + i for i in range(M_test).
    n_test = int(cfg["saa"]["M_test"])
    seeds = [int(cfg["seed"]["test"]) + i for i in range(n_test)]
    test_sc = [gen.sample(s) for s in seeds]
    archive = Archive(grid, base_aquifer(cfg, grid), wells, test_sc, fwd)
    idx = min(max(args.scenario, 0), len(archive.problems) - 1)
    if idx != args.scenario:
        log(f"scenario {args.scenario} out of range; using {idx}")
    prob = archive.problems[idx]

    q_use = np.zeros_like(q) if args.nopump else q
    t0 = time.time()
    res = transport_run(
        grid, prob.flow.aquifer, prob.flow, prob.resp, q_use,
        fwd.dt, fwd.nsteps,
        source_idx=prob.source_idx, source_rate=prob.source_rate,
        source_steps=fwd.source_steps, record_cells=archive.compliance_cells,
        store_fields=True,
    )
    log(f"policy: {policy_path}")
    log(f"config: {cfg_src}")
    log(f"scenario {idx}/{len(archive.problems) - 1}, "
        f"{'all wells off' if args.nopump else 'q*'}, "
        f"transport solved in {time.time() - t0:.1f}s")
    if not args.nopump:
        active = int((q > 1e-9).sum())
        log(f"q*: {active}/{q.size} wells active, total {q.sum():.1f} m3/d")
    return {
        "grid": grid,
        "C_fields": res.C_fields,
        "times": res.times,
        "wells": wells,
        "q": q_use,
        "creg": cost.creg,
        "source_xy": fwd.source_xy,
        "compliance_xy": fwd.compliance_xy,
        "obs_xy": fwd.obs_xy,
        "non_pumping": bool(args.nopump),
    }


def _auto_vmax(C: np.ndarray) -> float:
    """Display scale that keeps the end state visible.

    A single early source-cell spike can be orders of magnitude above the
    final plume and would make most of the movie black.  Taking the larger of
    the final-frame peak and the 99th percentile of every cell keeps the
    evolution readable; the true peak is still reported and any clipping is
    named on the colourbar.
    """
    peak = float(np.nanmax(C)) if C.size else 0.0
    if peak <= 0:
        return 1.0
    final_peak = float(np.nanmax(C[-1])) if C.shape[0] else 0.0
    p99 = float(np.nanpercentile(C, 99.0))
    return max(min(max(final_peak, p99), peak), peak * 1e-6)


def _output_path(args, spec: dict) -> Path:
    if args.out:
        p = Path(args.out)
    elif args.run:
        p = Path(args.run) / "figures" / (
            "plume_movie_nopump.gif" if spec["non_pumping"]
            else "plume_movie.gif"
        )
    else:
        p = Path(args.fields).with_name("plume_movie.gif")
    if p.suffix.lower() not in {".gif", ".mp4", ".m4v", ".mov"}:
        p = p.with_suffix(".gif")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fields", help="forward_fields.npz containing C_all")
    src.add_argument("--run", help="optimization run directory")
    ap.add_argument("--policy", default=None,
                    help="policy.json override (with --run)")
    ap.add_argument("--config", default=None,
                    help="config YAML override; otherwise the run snapshot")
    ap.add_argument("--scenario", type=int, default=0,
                    help="test-scenario index for --run (default: 0)")
    ap.add_argument("--nopump", action="store_true",
                    help="with --run: animate all wells off instead of q*")
    ap.add_argument("--out", default=None,
                    help="output .gif/.mp4 (default: <run>/figures/plume_movie.gif)")
    ap.add_argument("--fps", type=float, default=10.0, help="frames per second")
    ap.add_argument("--stride", type=int, default=1,
                    help="render every Nth stored step (last frame always kept)")
    ap.add_argument("--dpi", type=int, default=100, help="output resolution")
    ap.add_argument("--vmax", type=float, default=None,
                    help="fixed colour maximum [mg/L]; default: max(final "
                         "peak, 99th percentile), clipped above if needed")
    ap.add_argument("--title", default=None, help="figure title")
    args = ap.parse_args()
    if args.nopump and args.fields:
        raise SystemExit("error: --nopump is only meaningful with --run")

    spec = _load_fields(args) if args.fields else _load_run(args)
    title = args.title
    if title is None:
        title = ("Plume evolution (no pumping)" if spec["non_pumping"]
                 else "Plume evolution under pumping")
    out = _output_path(args, spec)
    vmax = args.vmax if args.vmax is not None else _auto_vmax(spec["C_fields"])

    try:
        summary = animate_plume(
            spec["grid"], spec["C_fields"], spec["times"], spec["wells"],
            spec["compliance_xy"], out,
            creg=spec["creg"], q=spec["q"],
            source_xy=spec["source_xy"], obs_xy=spec["obs_xy"],
            title=title, fps=args.fps, stride=args.stride, dpi=args.dpi,
            vmax=vmax,
        )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from None
    clip = (", clipped" if summary["peak"] > summary["vmax"] else "")
    log(f"peak C = {summary['peak']:.3g} mg/L, colour scale 0-"
        f"{summary['vmax']:.3g} mg/L{clip}")
    log(f"wrote {summary['path']} "
        f"({summary['n_frames']} frames from {summary['n_steps']} steps, "
        f"{summary['duration_s']:.0f}s at {summary['fps']:g} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
