#!/usr/bin/env python3
"""Full 3D flow/transport and pump-and-treat optimization.

The 3D counterpart of ``run_2d.py`` / ``run_optimization.py``.  It runs a
seven-point finite-volume Darcy solve and an implicit-Euler 3D ADE on a
log-normal conductivity field, places a requested number of extraction wells
inside the design plume, and (optionally) optimizes their rates with a
derivative-free compass search.

Modes
-----
``forward``
    One 3D realisation: solve flow, place wells in the no-pumping design
    plume, pump at a fixed fraction of ``q_max`` and write fields/metrics.

``optimize``
    Draw ``M_train`` random fields, evaluate the sample-average cost and run a
    bounded compass search over the well rates.  The best policy is verified
    on ``M_test`` independent fields.

``all``
    ``forward`` followed by ``optimize`` (the default).

Outputs go to ``--out`` (default ``out/full3d``): ``summary.json``,
``fields.npz`` (final concentration, K, well coordinates), ``history.json`` and
optional slice figures.  Every run appends one record to
``runs/runtimes.jsonl``.

Examples
--------
    python SCRIPTS/run_3d.py --mode forward --nx 96 --ny 48 --nz 24 \\
        --wells 12 --steps 40

    python SCRIPTS/run_3d.py --mode optimize --scenarios 4 --sweeps 2 \\
        --nx 96 --ny 48 --nz 24 --wells 12 --steps 20 --max-evals 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from physics.fv3d import (
    Aquifer3D,
    FlowModel3D,
    Grid3D,
    flow_responses3d,
    transport_run3d,
)
from stoch.random_fields3d import KLExpansion3D, Scenario3D, ScenarioGenerator3D
from stoch.runtimes import log_runtime
from stoch.wells3d import place_wells3d

RUNTIME_LOG = ROOT / "runs" / "runtimes.jsonl"
DEFAULT_CONFIG = ROOT / "configs" / "opt" / "full3d.yaml"


def log(msg: str) -> None:
    print(f"[3d] {time.strftime('%H:%M:%S')} {msg}", flush=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def default_config() -> dict:
    return {
        "domain": {"lx": 240.0, "ly": 120.0, "lz": 40.0,
                   "nx": 120, "ny": 60, "nz": 30},
        "aquifer": {
            "k0": 10.0, "phi": 0.30, "R": 1.0, "kv_ratio": 0.2,
            "alpha_l": 8.0, "alpha_th": 1.0, "alpha_tv": 0.2, "decay": 0.0,
        },
        "flow": {"grad": 1e-3},
        "random_field": {
            "sigma": 0.5, "ell_x": 24.0, "ell_y": 20.0, "ell_z": 8.0,
            "n_kl": 0, "target_variance": 0.95,
            "random_source": False, "random_gradient": False,
        },
        "wells": {
            "n": 12, "method": "farthest", "q_max": 50.0, "Q_max": 300.0,
            "eps": 1.5, "eps_z": 1.5, "plume_threshold": 0.05,
        },
        "source": {"x": 30.0, "y": 60.0, "z": 20.0,
                   "q": 10.0, "c": 100.0, "days": 365.0},
        "time": {"dt": 36.5, "nsteps": 100, "design_steps": 20},
        "saa": {"M_train": 8, "M_test": 8},
        "optim": {"method": "pattern", "sweeps": 3, "max_evals": 150,
                  "q_start_fraction": 0.5, "seed": 0},
        "cost": {"c_e": 0.05},
        "seed": {"train": 1000, "test": 500000},
        "plot": True,
    }


def deep_update(base: dict, upd: dict) -> dict:
    for key, value in upd.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[str | Path] = None) -> dict:
    cfg = default_config()
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        deep_update(cfg, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    return cfg


def apply_quick(cfg: dict) -> dict:
    return deep_update(cfg, {
        "domain": {"nx": 36, "ny": 18, "nz": 12},
        "wells": {"n": 6},
        "time": {"nsteps": 12, "design_steps": 6},
        "saa": {"M_train": 2, "M_test": 2},
        "optim": {"sweeps": 1, "max_evals": 24},
    })


def make_grid(cfg: dict) -> Grid3D:
    d = cfg["domain"]
    return Grid3D(int(d["nx"]), int(d["ny"]), int(d["nz"]),
                  float(d["lx"]), float(d["ly"]), float(d["lz"]))


def make_kl(cfg: dict, grid: Grid3D) -> KLExpansion3D:
    rf = cfg["random_field"]
    return KLExpansion3D(
        grid,
        sigma=float(rf["sigma"]),
        ell_x=float(rf["ell_x"]),
        ell_y=float(rf["ell_y"]),
        ell_z=float(rf["ell_z"]),
        n_kl=int(rf["n_kl"]),
        k0=float(cfg["aquifer"]["k0"]),
        target_variance=float(rf["target_variance"]),
    )


def make_aquifer(cfg: dict, K: np.ndarray) -> Aquifer3D:
    a = cfg["aquifer"]
    return Aquifer3D(
        K=K,
        phi=float(a["phi"]),
        R=float(a["R"]),
        kv_ratio=float(a["kv_ratio"]),
        alpha_l=float(a["alpha_l"]),
        alpha_th=float(a["alpha_th"]),
        alpha_tv=float(a["alpha_tv"]),
        decay=float(a["decay"]),
    )


def source_steps(cfg: dict) -> int:
    s, t = cfg["source"], cfg["time"]
    return max(1, min(int(t["nsteps"]),
                      int(round(float(s["days"]) / float(t["dt"])))))


def source_rate(cfg: dict, scale: float = 1.0) -> float:
    s = cfg["source"]
    return float(s["q"]) * float(s["c"]) * float(scale)


def project_rates(q: np.ndarray, q_max: float, Q_max: float) -> np.ndarray:
    q = np.clip(np.asarray(q, dtype=float), 0.0, q_max)
    total = float(q.sum())
    if total > Q_max > 0:
        q *= Q_max / total
    return q


# ---------------------------------------------------------------------------
# One scenario
# ---------------------------------------------------------------------------
class Problem3D:
    """A random field with cached unit-rate flow responses."""

    def __init__(self, cfg: dict, grid: Grid3D, scenario: Scenario3D,
                 well_xyz: np.ndarray) -> None:
        self.cfg = cfg
        self.grid = grid
        self.scenario = scenario
        self.aquifer = make_aquifer(cfg, scenario.K)
        grad = float(cfg["flow"]["grad"])
        self.flow = FlowModel3D(
            grid, self.aquifer,
            h_left=grad * grid.lx * scenario.gradient_scale, h_right=0.0,
        )
        self.resp = flow_responses3d(
            self.flow, well_xyz,
            eps_cells=float(cfg["wells"]["eps"]),
            eps_z_cells=float(cfg["wells"]["eps_z"]),
        )
        self.flow.release_factorization()
        self.well_xyz = np.atleast_2d(well_xyz)
        s = cfg["source"]
        self.source_idx = grid.index(float(s["x"]), float(s["y"]), float(s["z"]))
        self.rate = source_rate(cfg, scenario.source_scale)

    def transport(self, q, store_fields: bool = False):
        t = self.cfg["time"]
        return transport_run3d(
            self.grid, self.aquifer, self.flow, self.resp, q,
            dt=float(t["dt"]), nsteps=int(t["nsteps"]),
            source_idx=self.source_idx, source_rate=self.rate,
            source_steps=source_steps(self.cfg),
            store_fields=store_fields,
        )

    def design_plume(self) -> np.ndarray:
        """Concentration after ``design_steps`` with all wells off."""
        t = self.cfg["time"]
        res = transport_run3d(
            self.grid, self.aquifer, self.flow, self.resp,
            np.zeros(self.resp.hw.shape[0]),
            dt=float(t["dt"]), nsteps=int(t["design_steps"]),
            source_idx=self.source_idx, source_rate=self.rate,
            source_steps=min(source_steps(self.cfg),
                             int(t["design_steps"])),
        )
        return res.C_final.reshape(self.grid.shape)


def scenario_cost(cfg: dict, grid: Grid3D, problem: Problem3D,
                  q: np.ndarray, store_fields: bool = False) -> dict:
    """Mass-normalised cost and diagnostics for one scenario."""
    t = cfg["time"]
    res = problem.transport(q, store_fields=store_fields)
    nsteps = int(t["nsteps"])
    dt = float(t["dt"])
    T = dt * nsteps
    m_src = problem.rate * dt * min(source_steps(cfg), nsteps)
    m_src = max(m_src, 1e-30)
    m_final = float(res.mass["total"][-1])
    m_well = float(res.mass["well"][-1])
    c_e = float(cfg["cost"]["c_e"])
    Q_max = float(cfg["wells"]["Q_max"])
    J = m_final / m_src + c_e * float(np.sum(q)) / max(Q_max, 1e-30)
    return {
        "J": float(J),
        "final_mass": m_final,
        "source_mass": m_src,
        "extracted_mass": m_well,
        "extracted_fraction": m_well / m_src,
        "residual_fraction": m_final / m_src,
        "result": res,
    }


def mass_balance_error(res) -> float:
    """Relative closure error of the discrete mass budget."""
    m = res.mass
    lhs = float(m["total"][-1] - m["total"][0])
    rhs = float(m["source"][-1] - m["well"][-1] - m["outflow"][-1]
                - m["decay"][-1])
    scale = max(abs(float(m["source"][-1])), abs(float(m["total"][-1])), 1e-30)
    return abs(lhs - rhs) / scale


# ---------------------------------------------------------------------------
# Well placement + archive
# ---------------------------------------------------------------------------
def design_wells(cfg: dict, grid: Grid3D, generator: ScenarioGenerator3D,
                 n_wells: int) -> tuple[Scenario3D, np.ndarray]:
    """Place wells in the no-pumping plume of the first training scenario."""
    seed = int(cfg["seed"]["train"])
    s0 = generator.sample(seed)
    p0 = Problem3D(cfg, grid, s0, np.zeros((1, 3)))
    C0 = p0.design_plume()
    layout = place_wells3d(
        grid, C0, n_wells=n_wells,
        threshold=float(cfg["wells"]["plume_threshold"]),
    )
    log(f"design plume cells={int(layout.mask.sum())} "
        f"wells={layout.n_wells} covering_radius={layout.covering_radius:.2f} m")
    return s0, layout.wells


def build_archive(cfg: dict, grid: Grid3D, generator: ScenarioGenerator3D,
                  well_xyz: np.ndarray, n: int, seed0: int) -> list[Problem3D]:
    problems = []
    for m in range(n):
        s = generator.sample(seed0 + m)
        problems.append(Problem3D(cfg, grid, s, well_xyz))
        log(f"scenario {m + 1}/{n}: seed={s.seed} "
            f"K=[{s.K.min():.3g}, {s.K.max():.3g}] "
            f"clamp={s.clamp_fraction:.3f}")
    return problems


def mean_cost(cfg: dict, grid: Grid3D, problems: list[Problem3D],
              q: np.ndarray) -> dict:
    vals = [scenario_cost(cfg, grid, p, q) for p in problems]
    J = np.array([v["J"] for v in vals])
    return {
        "J": float(J.mean()),
        "J_worst": float(J.max()),
        "J_per_scenario": J.tolist(),
        "extracted_fraction": float(np.mean(
            [v["extracted_fraction"] for v in vals])),
        "residual_fraction": float(np.mean(
            [v["residual_fraction"] for v in vals])),
        "mass_balance_error_max": float(max(
            mass_balance_error(v["result"]) for v in vals)),
        "values": vals,
    }


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
def compass_search(cfg: dict, grid: Grid3D, problems: list[Problem3D],
                   q0: np.ndarray) -> tuple[np.ndarray, dict, list[dict]]:
    w = cfg["wells"]
    o = cfg["optim"]
    q_max = float(w["q_max"])
    Q_max = float(w["Q_max"])
    max_evals = int(o["max_evals"])
    sweeps = int(o["sweeps"])
    q = project_rates(q0, q_max, Q_max)
    best = mean_cost(cfg, grid, problems, q)
    history = [{
        "eval": 1, "q": q.tolist(), "J": best["J"],
        "J_worst": best["J_worst"],
        "extracted_fraction": best["extracted_fraction"],
    }]
    evals = 1
    step = 0.5 * q_max
    log(f"start J={best['J']:.5f} q={np.round(q, 2).tolist()}")

    for sweep in range(sweeps):
        improved = False
        for i in range(q.size):
            if evals >= max_evals:
                break
            for sign in (+1.0, -1.0):
                if evals >= max_evals:
                    break
                trial = q.copy()
                trial[i] = np.clip(trial[i] + sign * step, 0.0, q_max)
                trial = project_rates(trial, q_max, Q_max)
                if np.allclose(trial, q):
                    continue
                val = mean_cost(cfg, grid, problems, trial)
                evals += 1
                history.append({
                    "eval": evals, "q": trial.tolist(), "J": val["J"],
                    "J_worst": val["J_worst"],
                    "extracted_fraction": val["extracted_fraction"],
                })
                if val["J"] < best["J"] - 1e-12:
                    q, best = trial, val
                    improved = True
                    log(f"  eval {evals:3d} sweep {sweep} well {i} "
                        f"{'+' if sign > 0 else '-'}: J={best['J']:.5f}")
        if not improved:
            step *= 0.5
            log(f"  sweep {sweep}: no improvement, step -> {step:.2f}")
        if evals >= max_evals:
            break
    return np.asarray(q), best, history


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_fields(path: Path, grid: Grid3D, problem: Problem3D,
                res, q: np.ndarray) -> None:
    np.savez_compressed(
        path,
        C_final=res.C_final.reshape(grid.shape),
        K=problem.scenario.K,
        wells=problem.well_xyz,
        q=np.asarray(q),
        C_well=res.C_well,
        mass_total=res.mass["total"],
        mass_source=res.mass["source"],
        mass_well=res.mass["well"],
        times=res.times,
    )


def plot_slices(grid: Grid3D, C: np.ndarray, path: Path,
                k: Optional[int] = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional
        log(f"plot skipped: {exc}")
        return
    C3 = np.asarray(C).reshape(grid.shape)
    k = grid.nz // 2 if k is None else int(k)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = ax.imshow(C3[:, :, k].T, origin="lower", aspect="auto",
                   extent=[0, grid.lx, 0, grid.ly], cmap="magma")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"3D concentration, z-slice k={k}")
    fig.colorbar(im, ax=ax, label="C [mg/L]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_forward(cfg: dict, outdir: Path) -> dict:
    grid = make_grid(cfg)
    kl = make_kl(cfg, grid)
    gen = ScenarioGenerator3D(
        grid, kl,
        random_source=bool(cfg["random_field"]["random_source"]),
        random_gradient=bool(cfg["random_field"]["random_gradient"]),
    )
    log(f"3D grid {grid.nx}x{grid.ny}x{grid.nz} = {grid.n} cells, "
        f"KL modes={kl.n_kl} (retained {kl.retained:.3f})")
    n_wells = int(cfg["wells"]["n"])
    s0, wells = design_wells(cfg, grid, gen, n_wells)
    problem = Problem3D(cfg, grid, s0, wells)
    q = project_rates(np.full(n_wells, 0.3 * float(cfg["wells"]["q_max"])),
                      float(cfg["wells"]["q_max"]),
                      float(cfg["wells"]["Q_max"]))
    t0 = time.time()
    cost = scenario_cost(cfg, grid, problem, q, store_fields=True)
    res = cost["result"]
    err = mass_balance_error(res)
    elapsed = time.time() - t0
    out = outdir / "forward"
    out.mkdir(parents=True, exist_ok=True)
    save_fields(out / "fields.npz", grid, problem, res, q)
    if bool(cfg["plot"]):
        plot_slices(grid, res.C_final, out / "slice_xy.png")
    summary = {
        "mode": "forward",
        "grid": {"nx": grid.nx, "ny": grid.ny, "nz": grid.nz,
                 "lx": grid.lx, "ly": grid.ly, "lz": grid.lz,
                 "cells": grid.n},
        "kl": kl.to_dict(),
        "wells": {"n": n_wells, "xyz": wells.tolist(), "q": q.tolist()},
        "time": {"dt": float(cfg["time"]["dt"]),
                 "nsteps": int(cfg["time"]["nsteps"])},
        "source": dict(cfg["source"]),
        "metrics": {
            "J": cost["J"],
            "final_mass": cost["final_mass"],
            "source_mass": cost["source_mass"],
            "extracted_mass": cost["extracted_mass"],
            "extracted_fraction": cost["extracted_fraction"],
            "residual_fraction": cost["residual_fraction"],
            "mass_balance_error": err,
            "wall_clock_s": elapsed,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                      encoding="utf-8")
    log(f"forward done in {elapsed:.1f}s: extracted "
        f"{100 * cost['extracted_fraction']:.1f}% of source mass, "
        f"mass-balance err {err:.2e}")
    return summary


def run_optimize(cfg: dict, outdir: Path) -> dict:
    grid = make_grid(cfg)
    kl = make_kl(cfg, grid)
    gen = ScenarioGenerator3D(
        grid, kl,
        random_source=bool(cfg["random_field"]["random_source"]),
        random_gradient=bool(cfg["random_field"]["random_gradient"]),
    )
    log(f"3D grid {grid.nx}x{grid.ny}x{grid.nz} = {grid.n} cells, "
        f"KL modes={kl.n_kl} (retained {kl.retained:.3f})")
    n_wells = int(cfg["wells"]["n"])
    _, wells = design_wells(cfg, grid, gen, n_wells)

    n_train = int(cfg["saa"]["M_train"])
    n_test = int(cfg["saa"]["M_test"])
    train = build_archive(cfg, grid, gen, wells,
                          n_train, int(cfg["seed"]["train"]))
    q0 = project_rates(
        np.full(n_wells, float(cfg["optim"]["q_start_fraction"])
                * float(cfg["wells"]["q_max"])),
        float(cfg["wells"]["q_max"]), float(cfg["wells"]["Q_max"]),
    )
    t0 = time.time()
    best_q, best, history = compass_search(cfg, grid, train, q0)
    elapsed = time.time() - t0
    log(f"optimization done in {elapsed:.1f}s, J={best['J']:.5f} "
        f"(-{best['extracted_fraction']:.3f} extracted)")

    test = build_archive(cfg, grid, gen, wells,
                         n_test, int(cfg["seed"]["test"]))
    test_metrics = mean_cost(cfg, grid, test, best_q)
    log(f"test: J={test_metrics['J']:.5f} worst={test_metrics['J_worst']:.5f} "
        f"extracted={100 * test_metrics['extracted_fraction']:.1f}%")

    # final field of the best policy on the first test scenario
    best_cost = scenario_cost(cfg, grid, test[0], best_q, store_fields=True)
    out = outdir / "opt"
    out.mkdir(parents=True, exist_ok=True)
    save_fields(out / "fields.npz", grid, test[0], best_cost["result"], best_q)
    if bool(cfg["plot"]):
        plot_slices(grid, best_cost["result"].C_final, out / "slice_xy.png")
    (out / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8")
    summary = {
        "mode": "optimize",
        "grid": {"nx": grid.nx, "ny": grid.ny, "nz": grid.nz,
                 "lx": grid.lx, "ly": grid.ly, "lz": grid.lz,
                 "cells": grid.n},
        "kl": kl.to_dict(),
        "wells": {"n": n_wells, "xyz": wells.tolist(),
                  "q_best": best_q.tolist(),
                  "q_max": float(cfg["wells"]["q_max"]),
                  "Q_max": float(cfg["wells"]["Q_max"])},
        "saa": {"M_train": n_train, "M_test": n_test},
        "optim": {"sweeps": int(cfg["optim"]["sweeps"]),
                  "max_evals": int(cfg["optim"]["max_evals"]),
                  "evaluations": len(history)},
        "train": {k: v for k, v in best.items()
                  if k not in {"values", "J_per_scenario"}},
        "test": {k: v for k, v in test_metrics.items()
                 if k not in {"values", "J_per_scenario"}},
        "wall_clock_s": elapsed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                      encoding="utf-8")
    return summary


def zip_results(outdir: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(outdir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(outdir.parent))
    log(f"results zip -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--mode", choices=["forward", "optimize", "all"],
                    default="all")
    ap.add_argument("--out", default=str(ROOT / "out" / "full3d"))
    ap.add_argument("--quick", action="store_true",
                    help="small grid/scenario budget for a smoke run")
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--nz", type=int, default=None)
    ap.add_argument("--wells", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--design-steps", type=int, default=None)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--scenarios", type=int, default=None,
                    help="M_train (training scenarios)")
    ap.add_argument("--test-scenarios", type=int, default=None)
    ap.add_argument("--sweeps", type=int, default=None)
    ap.add_argument("--max-evals", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted config override, repeatable")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--zip-results", action="store_true")
    ap.add_argument("--results-zip", default=None)
    ap.add_argument("--quiet", action="store_true")
    return ap


def parse_overrides(items: list[str]) -> dict:
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set wants KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        cur = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
    return out


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.quick:
        cfg = apply_quick(cfg)
    deep_update(cfg, parse_overrides(args.set))

    d = cfg["domain"]
    if args.nx is not None:
        d["nx"] = int(args.nx)
    if args.ny is not None:
        d["ny"] = int(args.ny)
    if args.nz is not None:
        d["nz"] = int(args.nz)
    if args.wells is not None:
        cfg["wells"]["n"] = int(args.wells)
    if args.steps is not None:
        cfg["time"]["nsteps"] = int(args.steps)
    if args.design_steps is not None:
        cfg["time"]["design_steps"] = int(args.design_steps)
    if args.dt is not None:
        cfg["time"]["dt"] = float(args.dt)
    if args.scenarios is not None:
        cfg["saa"]["M_train"] = int(args.scenarios)
    if args.test_scenarios is not None:
        cfg["saa"]["M_test"] = int(args.test_scenarios)
    if args.sweeps is not None:
        cfg["optim"]["sweeps"] = int(args.sweeps)
    if args.max_evals is not None:
        cfg["optim"]["max_evals"] = int(args.max_evals)
    if args.seed is not None:
        cfg["optim"]["seed"] = int(args.seed)
        cfg["seed"]["train"] = int(args.seed)
        cfg["seed"]["test"] = int(args.seed) + 500000

    outdir = Path(args.out).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    t0 = time.time()
    summaries = {}
    status = "ok"
    try:
        if args.mode in {"forward", "all"}:
            summaries["forward"] = run_forward(cfg, outdir)
        if args.mode in {"optimize", "all"}:
            summaries["optimize"] = run_optimize(cfg, outdir)
    except Exception:
        status = "failed"
        raise
    finally:
        elapsed = time.time() - t0
        log_runtime(
            RUNTIME_LOG, "run_3d.py", elapsed,
            stage=args.mode,
            grid=(int(d["nx"]), int(d["ny"]), int(d["nz"])),
            wells=int(cfg["wells"]["n"]),
            nsteps=int(cfg["time"]["nsteps"]),
            scenarios=int(cfg["saa"]["M_train"]),
            outdir=str(outdir), status=status,
        )
        if args.zip_results:
            dest = Path(args.results_zip) if args.results_zip \
                else outdir / "full3d_results.zip"
            zip_results(outdir, dest)
    print(json.dumps({k: v.get("metrics", v.get("train"))
                      for k, v in summaries.items()}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
