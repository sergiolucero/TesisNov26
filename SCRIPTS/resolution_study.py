"""Does the grid change the answer? One realisation, two grids.

Every 3D number in this repo comes from a run whose Karhunen--Loeve basis was
truncated at that run's own grid, so two runs at different resolutions are two
different aquifers and their difference is not "resolution". This script removes
that confound the only way it can be removed: draw ONE realisation on the fine
grid, upscale it by block-averaging, and then evaluate the SAME policies on both
grids. What remains when the comparison is like-for-like is a genuine resolution
effect (including the smoothing that upscaling itself introduces, which is part of
what a coarser grid is).

Run:  python3 SCRIPTS/resolution_study.py --out out3d_colab/resolution_study.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SCRIPTS"))

import full3D_colab_run as f3d  # noqa: E402
import check_tail3d as ct  # noqa: E402

# The wells of the 13,824-cell run, which is the fine grid here. Fixing one
# layout is the point: a resolution study compares representations of one
# problem, not two searches.
WELLS = None  # filled from --run's policy.json

POLICIES = {
    "no_pump":       None,   # zeros
    "uniform_half":  None,   # 0.5 q_max on every well, projected
    "bigrun_q":      None,   # the 13,824-cell incumbent
    "size_mid_q":    None,   # a 3-active-well design from the 8,000-cell run
}


def build(cfg, grid):
    aquifer = f3d.Aquifer3D(
        K=np.full((grid.nz, grid.ny, grid.nx), float(cfg["k0"])),
        kv_ratio=cfg["kv_ratio"], phi=cfg["phi"], R=cfg["R"],
        alpha_l=cfg["alpha_l"], alpha_th=cfg["alpha_th"],
        alpha_tv=cfg["alpha_tv"], decay=cfg["decay"])
    fwd = f3d.ForwardConfig3D(
        dt=cfg["dt"], nsteps=cfg["nsteps"], grad=cfg["grad"],
        source_xyz=tuple(cfg["source_xyz"]), source_q=cfg["source_q"],
        source_c=cfg["source_c"], source_days=cfg["source_days"],
        compliance_xyz=[tuple(p) for p in cfg["compliance_xyz"]],
        obs_xyz=[tuple(p) for p in cfg["obs_xyz"]], eps_well=cfg["eps_well"])
    cost = f3d.CostConfig3D(
        c_e=cfg["c_e"], c_w=cfg["c_w"], c_m=cfg["c_m"], c_tau=cfg["c_tau"],
        c_fail=cfg["c_fail"], creg=cfg["creg"],
        window_start=cfg["window_start"], h_min=cfg["h_min"],
        eps_time=cfg["eps_time"], alpha=cfg["alpha"], alpha_h=cfg["alpha_h"])
    return aquifer, fwd, cost


def summarise(ev, m):
    G = np.asarray(ev.G, float).ravel()
    J = np.asarray(ev.J, float).ravel()
    return dict(mean_J=float(J.mean()), mean_G=float(G.mean()),
                violations=int(np.sum(G > 0)), reliability=float(np.mean(G <= 0)),
                cvar_G=float(np.sort(G)[::-1][:max(1, int(np.floor(0.05 * m)))].mean()),
                mean_tau=float(np.asarray(ev.tau, float).ravel().mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out3d_colab/bigrun_nov23b",
                    help="fine-grid run whose config and wells define the problem")
    ap.add_argument("--m-scen", type=int, default=16)
    ap.add_argument("--seed-base", type=int, default=900000)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--nsteps", type=int, default=None)
    ap.add_argument("--out", default="out3d_colab/resolution_study.json")
    args = ap.parse_args()

    cfg, policy, _ = ct.load_run(Path(args.run))
    wells = np.atleast_2d(np.asarray(policy["wells"], float))
    n = wells.shape[0]
    if args.nsteps:
        cfg = dict(cfg, nsteps=int(args.nsteps))

    f = int(args.factor)
    gf = f3d.Grid3D(cfg["nx"], cfg["ny"], cfg["nz"], cfg["lx"], cfg["ly"], cfg["lz"])
    gc = f3d.Grid3D(cfg["nx"] // f, cfg["ny"] // f, cfg["nz"] // f,
                    cfg["lx"], cfg["ly"], cfg["lz"])
    print(f"fine   {gf.nx}x{gf.ny}x{gf.nz} = {gf.n} cells "
          f"(dx={gf.dx:.2f}, dz={gf.dz:.2f})")
    print(f"coarse {gc.nx}x{gc.ny}x{gc.nz} = {gc.n} cells "
          f"(dx={gc.dx:.2f}, dz={gc.dz:.2f}), factor {f}")

    kl = f3d.KL3D(gf, cfg["sigma"], cfg["ell_x"], cfg["ell_y"], cfg["ell_z"],
                  n_kl=cfg["n_kl"], k0=cfg["k0"], kv_ratio=cfg["kv_ratio"],
                  target_variance=cfg["target_variance"])
    print(f"fine KL n_kl={kl.n_kl} (retained {getattr(kl, 'retained', float('nan'))!r})")

    seeds = [args.seed_base + i for i in range(int(args.m_scen))]
    t0 = time.time()
    scen_f = f3d.sample_scenarios(gf, kl, seeds)
    scen_c = [f3d.transfer_scenarios(s, gc, f) for s in scen_f]
    print(f"scenarios: {len(seeds)} sampled on the fine grid in {time.time() - t0:.1f}s")

    lkf = np.concatenate([np.log(s.K).ravel() for s in scen_f])
    lkc = np.concatenate([np.log(s.K).ravel() for s in scen_c])
    print(f"log K: fine mean {lkf.mean():.4f} std {lkf.std():.4f} | "
          f"coarse mean {lkc.mean():.4f} std {lkc.std():.4f} "
          f"(variance kept {lkc.var() / lkf.var():.3f})")

    aqf, fwd, cost = build(cfg, gf)
    aqc, _, _ = build(cfg, gc)
    tf = time.time(); arch_f = f3d.Archive3D(gf, aqf, wells, scen_f, fwd)
    print(f"fine archive in {time.time() - tf:.1f}s")
    tc = time.time(); arch_c = f3d.Archive3D(gc, aqc, wells, scen_c, fwd)
    print(f"coarse archive in {time.time() - tc:.1f}s")

    qs = {
        "no_pump": np.zeros(n),
        "uniform_half": np.full(n, 0.5 * float(cfg["q_max"])),
        "bigrun_q": np.asarray(policy["q"], float),
        "size_mid_q": np.array([0.05, 12.5, 0.0, 0.0, 10.06, 0.0])[:n],
    }
    if qs["size_mid_q"].size < n:
        qs["size_mid_q"] = np.pad(qs["size_mid_q"], (0, n - qs["size_mid_q"].size))

    out = {"config": {"fine": [gf.nx, gf.ny, gf.nz], "coarse": [gc.nx, gc.ny, gc.nz],
                      "factor": f, "m_scen": len(seeds), "nsteps": cfg["nsteps"],
                      "seed_base": args.seed_base},
           "logK": {"fine_mean": float(lkf.mean()), "fine_std": float(lkf.std()),
                    "coarse_mean": float(lkc.mean()), "coarse_std": float(lkc.std()),
                    "variance_kept": float(lkc.var() / lkf.var())},
           "policies": {}}

    for name, q in qs.items():
        qq = np.asarray(q, float)
        s = float(qq.sum())
        if s > float(cfg["Q_max"]):
            qq = qq * float(cfg["Q_max"]) / s
        t1 = time.time()
        evf = arch_f.evaluate(qq, cost)
        tf_s = time.time() - t1
        t1 = time.time()
        evc = arch_c.evaluate(qq, cost)
        tc_s = time.time() - t1
        sf, sc = summarise(evf, len(seeds)), summarise(evc, len(seeds))
        Jf = np.asarray(evf.J, float).ravel(); Jc = np.asarray(evc.J, float).ravel()
        Gf = np.asarray(evf.G, float).ravel(); Gc = np.asarray(evc.G, float).ravel()
        r = float(np.corrcoef(Jf, Jc)[0, 1]) if Jf.size > 2 else float("nan")
        rG = float(np.corrcoef(Gf, Gc)[0, 1]) if Gf.size > 2 else float("nan")
        agree = float(np.mean((Gf > 0) == (Gc > 0)))
        dJ = 100.0 * (sc["mean_J"] - sf["mean_J"]) / sf["mean_J"]
        out["policies"][name] = dict(
            q_total=float(qq.sum()), n_active=int((qq > 1e-9).sum()),
            fine=sf, coarse=sc, dJ_pct=dJ, corr_J=r, corr_G=rG,
            violation_agreement=agree, t_fine_s=tf_s, t_coarse_s=tc_s)
        print(f"\n{name} (sum q={qq.sum():.2f}, {int((qq>1e-9).sum())} active)")
        print(f"   fine   J={sf['mean_J']:10.1f} G={sf['mean_G']:+7.3f} "
              f"viol={sf['violations']}/{len(seeds)} tau={sf['mean_tau']:7.1f}  ({tf_s:.1f}s)")
        print(f"   coarse J={sc['mean_J']:10.1f} G={sc['mean_G']:+7.3f} "
              f"viol={sc['violations']}/{len(seeds)} tau={sc['mean_tau']:7.1f}  ({tc_s:.1f}s)")
        print(f"   dJ={dJ:+.2f}%  corr(J)={r:+.4f}  corr(G)={rG:+.4f}  "
              f"violation flags agree {100*agree:.0f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
