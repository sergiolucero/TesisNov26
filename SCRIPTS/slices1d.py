#!/usr/bin/env python3
"""1D slices through a 3D stochastic optimum.

A policy ``q*`` is a point in ``n_wells`` dimensions, which is unplottable.  This
script projects it onto one-dimensional lines and scores every point along the
line with the same batched archive the optimizer used, so the *shape* of the
optimum becomes visible:

* **coordinate slices**  - move one well's rate while the others stay at ``q*``.
  A genuine local minimum is a minimum along every such line; a flat segment
  means that well's rate is (locally) indeterminate.
* **scale slice**        - ``q = s * q*``, the objective vs total pumping, with
  the reliability/chance-constraint crossover marked.
* **equal-split slice**  - ``q = (T/n) * ones``, the "spread it evenly" family,
  plotted on the same total-pumping axis so the optimized mix can be seen to
  beat any equal split.
* **active-split slice** - hold all other wells at ``q*`` and vary the share
  between the two largest wells at fixed combined rate.

    python3 SCRIPTS/slices1d.py --run out3d_colab/run_paperGPU
    python3 SCRIPTS/slices1d.py --run <dir> --archive train --device cuda --dtype float32

The slices use the *torch* archive (batched: one ``evaluate_many`` per slice),
because 10-20 points per slice would otherwise be 10-20 separate splu solves.
Outputs land in ``<run>/slices1d/``: three PNG figures and ``slices1d.json``
with the raw curves.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import full3D_colab_run as f3d  # noqa: E402


def rebuild(run_dir: Path):
    """Same reconstruction as SCRIPTS/verify_papergpu.py (run's own constructors)."""
    cfg = json.loads((run_dir / "run_config.json").read_text())
    pol = json.loads((run_dir / "policy.json").read_text())
    grid = f3d.Grid3D(cfg["nx"], cfg["ny"], cfg["nz"], cfg["lx"], cfg["ly"], cfg["lz"])
    aquifer = f3d.Aquifer3D(
        K=np.full((grid.nz, grid.ny, grid.nx), cfg["k0"]),
        kv_ratio=cfg["kv_ratio"], phi=cfg["phi"], R=cfg["R"],
        alpha_l=cfg["alpha_l"], alpha_th=cfg["alpha_th"], alpha_tv=cfg["alpha_tv"],
        decay=cfg["decay"],
    )
    kl = f3d.KL3D(grid, cfg["sigma"], cfg["ell_x"], cfg["ell_y"], cfg["ell_z"],
                  n_kl=cfg["n_kl"], k0=cfg["k0"], kv_ratio=cfg["kv_ratio"],
                  target_variance=cfg["target_variance"])
    fwd = f3d.ForwardConfig3D(
        dt=cfg["dt"], nsteps=cfg["nsteps"], grad=cfg["grad"],
        source_xyz=tuple(cfg["source_xyz"]),
        source_q=cfg["source_q"], source_c=cfg["source_c"],
        source_days=cfg["source_days"],
        compliance_xyz=[tuple(p) for p in cfg["compliance_xyz"]],
        obs_xyz=[tuple(p) for p in cfg["obs_xyz"]],
        eps_well=cfg["eps_well"],
    )
    cost = f3d.CostConfig3D(
        c_e=cfg["c_e"], c_w=cfg["c_w"], c_m=cfg["c_m"], c_tau=cfg["c_tau"],
        c_fail=cfg["c_fail"], creg=cfg["creg"], window_start=cfg["window_start"],
        h_min=cfg["h_min"], eps_time=cfg["eps_time"],
        alpha=cfg["alpha"], alpha_h=cfg["alpha_h"],
    )
    wells = np.atleast_2d(np.asarray(pol["wells"], dtype=float))
    q = np.asarray(pol["q"], dtype=float).ravel()
    return cfg, pol, grid, aquifer, kl, fwd, cost, wells, q


def score(archive, cost, Q, cfg):
    """Score a (K, n) candidate matrix and reduce to the scalar curves."""
    ev = archive.evaluate_many(Q, cost)
    mean_J = ev.mean_J.detach().cpu().numpy()
    G = ev.G
    H = ev.H
    cg = f3d.cvar_batch3d(G, cfg["alpha"]).detach().cpu().numpy()
    ch = f3d.cvar_batch3d(-H, cfg["alpha_h"]).detach().cpu().numpy()
    reliab = ev.reliability.detach().cpu().numpy()
    f = (mean_J
         + cfg["rho_g"] * np.clip(cg, 0.0, None) ** 2
         + cfg["rho_h"] * np.clip(ch, 0.0, None) ** 2)
    return {
        "Q": np.asarray(Q, dtype=float),
        "mean_J": mean_J,
        "f": f,
        "cvar_G": cg,
        "reliability": reliab,
        "total": np.asarray(Q, dtype=float).sum(axis=1),
    }


def _with_marker(lo, hi, n, marker):
    """Even grid that always includes ``marker`` so the curve passes through q*."""
    x = np.linspace(lo, hi, n)
    x = np.unique(np.sort(np.concatenate([x, [marker]])))
    return x


def coordinate_slices(q, q_max, n):
    out = []
    for i in range(n):
        t = _with_marker(0.0, q_max, 20, q[i])
        Q = np.tile(q, (t.size, 1))
        Q[:, i] = t
        out.append({"kind": "coordinate", "well": i, "x": t,
                    "Q": Q, "xlabel": f"well {i} rate (m3/d)",
                    "marker": q[i]})
    return out


def scale_slice(q, s_max=2.0, n=20):
    s = _with_marker(0.0, s_max, n, 1.0)
    Q = s[:, None] * q[None, :]
    return [{"kind": "scale", "x": s, "Q": Q, "xlabel": "scale s (x q*)",
             "marker": 1.0}]


def equal_split_slice(q, total_max, n_wells, n=21):
    T = np.linspace(0.0, total_max, n)
    Q = (T / n_wells)[:, None] * np.ones((1, n_wells))
    return [{"kind": "equal", "x": T, "Q": Q,
             "xlabel": "total pumping, equal split (m3/d)",
             "marker": float(q.sum())}]


def split_slice(q, n=20):
    order = np.argsort(q)[::-1]
    i, j = int(order[0]), int(order[1])
    S = float(q[i] + q[j])
    if S <= 0:
        return []
    share = q[i] / S
    alpha = _with_marker(0.0, 1.0, n, share)
    Q = np.tile(q, (alpha.size, 1))
    Q[:, i] = alpha * S
    Q[:, j] = (1.0 - alpha) * S
    return [{"kind": "split", "i": i, "j": j, "x": alpha, "Q": Q,
             "xlabel": f"share to well {i} (m3/d)",
             "marker": share}]


def make_slices(q, cfg, q_max, n_wells):
    slices = coordinate_slices(q, q_max, n_wells)
    slices += scale_slice(q)
    slices += equal_split_slice(q, total_max=max(100.0, 2.0 * q.sum()), n_wells=n_wells)
    slices += split_slice(q)
    return slices


def plot_coordinates(results, q, q_max, outdir, infeas_thresh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = [r for r in results if r["kind"] == "coordinate"]
    n = len(coords)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.4 * rows),
                             squeeze=False)
    for k, (r, ax) in enumerate(zip(coords, axes.ravel())):
        infeas = r["reliability"] < infeas_thresh
        ax.axvspan(0, q_max, color="tab:red", alpha=0.05)
        if infeas.any():
            ax.fill_between(r["x"], r["f"].min() - 1, r["f"].max() + 1,
                            where=infeas, color="tab:red", alpha=0.08,
                            label="chance infeasible")
        ax.plot(r["x"], r["f"], "k-", lw=2, label="penalized f")
        ax.plot(r["x"], r["mean_J"], "b--", lw=1.4, label="mean J")
        ax.axvline(r["marker"], color="tab:green", lw=1.5, ls=":")
        ax.set_title(f"well {r['well']} (q*={q[r['well']]:.2f})")
        ax.set_xlabel(r["xlabel"])
        ax.set_ylabel("objective")
        ax.legend(fontsize=7, loc="best")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"coordinate slices through q* = {np.round(q, 2).tolist()}")
    fig.tight_layout()
    fig.savefig(outdir / "coordinate_slices.png", dpi=150)
    plt.close(fig)


def plot_total_pumping(results, q, outdir, infeas_thresh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scale = next(r for r in results if r["kind"] == "scale")
    equal = next(r for r in results if r["kind"] == "equal")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(scale["total"], scale["f"], "k-", lw=2, label="mix q = s·q*")
    ax.plot(equal["total"], equal["f"], "tab:gray", lw=1.6, ls="--",
            label="equal split q = (T/n)·1")
    ax.axvline(q.sum(), color="tab:green", ls=":", lw=1.5)
    ax.annotate(f"q* total {q.sum():.1f} m3/d", xy=(q.sum(), scale["f"][-1]),
                xytext=(q.sum() * 1.05, scale["f"].max()),
                arrowprops=dict(arrowstyle="->", color="tab:green"), fontsize=8)
    # reliability crossover on the scale curve
    reliab = scale["reliability"]
    ok = np.nonzero(reliab >= infeas_thresh)[0]
    if ok.size:
        s_ok = scale["total"][ok[0]]
        ax.axvline(s_ok, color="tab:red", ls="-.", lw=1.2)
        ax.annotate(f"reliability ≥ {infeas_thresh:.2f} at {s_ok:.1f} m3/d",
                    xy=(s_ok, ax.get_ylim()[0]), fontsize=8, color="tab:red",
                    rotation=90, va="bottom")
    ax.set_xlabel("total pumping (m3/d)")
    ax.set_ylabel("penalized objective f")
    ax.set_title("optimized mix vs equal split, vs total pumping")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "total_pumping_slices.png", dpi=150)
    plt.close(fig)


def plot_split(results, q, outdir, infeas_thresh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = [r for r in results if r["kind"] == "split"]
    if not splits:
        return
    r = splits[0]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    infeas = r["reliability"] < infeas_thresh
    ax.fill_between(r["x"], r["f"].min() - 1, r["f"].max() + 1, where=infeas,
                    color="tab:red", alpha=0.08, label="chance infeasible")
    ax.plot(r["x"], r["f"], "k-", lw=2, label="penalized f")
    ax.plot(r["x"], r["mean_J"], "b--", lw=1.4, label="mean J")
    ax.axvline(r["marker"], color="tab:green", ls=":", lw=1.5)
    ax.set_xlabel(r["xlabel"])
    ax.set_ylabel("objective")
    ax.set_title(f"split between wells {r['i']} and {r['j']} "
                 f"(q* share {r['marker']:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "split_slice.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="out3d_colab/run_paperGPU")
    ap.add_argument("--archive", choices=["train", "test"], default="train",
                    help="archive the slices are scored on (default train, the "
                         "objective the optimizer minimized)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    ap.add_argument("--krylov-rtol", type=float, default=None)
    ap.add_argument("--krylov-maxiter", type=int, default=4000)
    ap.add_argument("--max-m", type=int, default=0,
                    help="cap scenarios (0 = all) for a quick smoke run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if not f3d._HAS_TORCH:
        raise SystemExit("slices1d needs torch (the batched backend); "
                         "pip install torch")

    run_dir = Path(args.run)
    cfg, pol, grid, aquifer, kl, fwd, cost, wells, q = rebuild(run_dir)
    outdir = Path(args.out) if args.out else run_dir / "slices1d"
    outdir.mkdir(parents=True, exist_ok=True)

    q_max = cfg["q_max"]
    n_wells = wells.shape[0]
    key = "seed_train" if args.archive == "train" else "seed_test"
    m = cfg["m_train"] if args.archive == "train" else cfg["m_test"]
    seeds = [cfg[key] + i for i in range(m)]
    if args.max_m:
        seeds = seeds[: args.max_m]

    f3d.log(f"slices1d {run_dir}: {args.archive} archive "
            f"({len(seeds)} scenarios), grid {grid.nx}x{grid.ny}x{grid.nz}, "
            f"dtype={args.dtype}")
    scenarios = f3d.sample_scenarios(grid, kl, seeds)
    archive = f3d.TorchArchive3D(grid, aquifer, wells, scenarios, fwd,
                                 device=args.device, dtype=args.dtype,
                                 rtol=args.krylov_rtol, maxiter=args.krylov_maxiter,
                                 label="slices")
    f3d.log(f"archive ready: KL n_kl={kl.n_kl}, retained={kl.retained:.4f}")

    slices = make_slices(q, cfg, q_max, n_wells)
    infeas_thresh = 1.0 - cfg["alpha"] - 0.01  # matches the reliability gate
    results = []
    for sl in slices:
        f3d.log(f"slice {sl['kind']}"
                + (f" well={sl['well']}" if sl["kind"] == "coordinate" else ""))
        r = score(archive, cost, sl["Q"], cfg)
        r.update({k: sl[k] for k in ("kind", "x", "xlabel", "marker")
                  if k in sl})
        if sl["kind"] == "coordinate":
            r["well"] = sl["well"]
        if sl["kind"] == "split":
            r["i"], r["j"] = sl["i"], sl["j"]
        results.append(r)

    plot_coordinates(results, q, q_max, outdir, infeas_thresh)
    plot_total_pumping(results, q, outdir, infeas_thresh)
    plot_split(results, q, outdir, infeas_thresh)

    payload = []
    for r in results:
        payload.append({
            "kind": r["kind"],
            **({"well": r["well"]} if r["kind"] == "coordinate" else {}),
            **({"i": r["i"], "j": r["j"]} if r["kind"] == "split" else {}),
            "x": np.asarray(r["x"]).tolist(),
            "total": np.asarray(r["total"]).tolist(),
            "mean_J": np.asarray(r["mean_J"]).tolist(),
            "f": np.asarray(r["f"]).tolist(),
            "cvar_G": np.asarray(r["cvar_G"]).tolist(),
            "reliability": np.asarray(r["reliability"]).tolist(),
        })
    (outdir / "slices1d.json").write_text(json.dumps(
        {"run": str(run_dir), "archive": args.archive, "q": q.tolist(),
         "dtype": args.dtype, "slices": payload}, indent=2))
    f3d.log(f"wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
