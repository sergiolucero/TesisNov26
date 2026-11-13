#!/usr/bin/env python3
"""Colab GPU runner for TESIS experiments.

This script is a small command dispatcher so a Colab notebook can choose which
model/experiment to run without remembering each entry-point's arguments.

Examples
--------
    python collab_run.py --models smoke
    python collab_run.py --models pinn --preset quick
    python collab_run.py --models dataset fno --preset full --fno-epochs 300
    python collab_run.py --models opt --preset quick --opt-stage gates
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run a subprocess from the project root and fail fast on errors."""
    printable = " ".join(str(x) for x in cmd)
    print(f"\n[collab_run] $ {printable}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    print(f"[collab_run] done in {time.time() - t0:.1f}s", flush=True)


def cuda_report() -> None:
    """Print the GPU/torch status available to this runtime."""
    print(f"[collab_run] project root: {ROOT}")
    try:
        import torch

        print(f"[collab_run] torch: {torch.__version__}")
        print(f"[collab_run] cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[collab_run] gpu: {torch.cuda.get_device_name(0)}")
            print(f"[collab_run] cuda: {torch.version.cuda}")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[collab_run] could not import torch: {exc}")


def ensure_out(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (ROOT / "figs").mkdir(exist_ok=True)


def build_dataset(args: argparse.Namespace) -> None:
    ensure_out(args.out_root)
    if args.preset == "smoke":
        nx = nt = 64
        n_scenarios = 8
    elif args.preset == "quick":
        nx = nt = 128
        n_scenarios = 16
    else:
        nx = nt = 256
        n_scenarios = 32

    run([
        sys.executable,
        "hydrology.py",
        "--scenario", str(args.out_root / "scenario.npz"),
        "--dataset", str(args.dataset),
        "--figdir", str(args.out_root / "figs"),
        "--nx", str(nx),
        "--nt", str(nt),
        "--n-scenarios", str(n_scenarios),
    ])


def run_pinn(args: argparse.Namespace) -> None:
    ensure_out(args.out_root)
    if args.pinn_steps is not None:
        adam_steps = args.pinn_steps
    else:
        adam_steps = {"smoke": 100, "quick": 1500, "full": 12000}[args.preset]
    lbfgs_steps = 0 if args.preset == "smoke" else args.lbfgs_steps
    n_f = {"smoke": 256, "quick": 1000, "full": 1500}[args.preset]
    n_side = {"smoke": 64, "quick": 128, "full": 150}[args.preset]

    cmd = [
        sys.executable,
        "solve_1d_pinn.py",
        "--device", args.device,
        "--adam-steps", str(adam_steps),
        "--lbfgs-steps", str(lbfgs_steps),
        "--N-f", str(n_f),
        "--N-ic", str(n_side),
        "--N-bc", str(n_side),
        "--outdir", str(args.out_root / "pinn"),
        "--save-model", str(args.out_root / "pinn" / "model.pt"),
        "--ckpt-every", str(args.ckpt_every),
        "--ckpt-path", str(args.out_root / "pinn" / "checkpoint.pt"),
    ]
    if args.hard_bc:
        cmd.append("--hard-bc")
    run(cmd)


def run_fno(args: argparse.Namespace) -> None:
    ensure_out(args.out_root)
    if not args.dataset.exists():
        print(f"[collab_run] dataset not found at {args.dataset}; generating it first")
        build_dataset(args)
    epochs = args.fno_epochs if args.fno_epochs is not None else {"smoke": 3, "quick": 50, "full": 300}[args.preset]
    batch_size = {"smoke": 4, "quick": 8, "full": 8}[args.preset]

    run([
        sys.executable,
        "solve_1d_fno.py",
        "--dataset", str(args.dataset),
        "--device", args.device,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--modes1", str(args.fno_modes),
        "--modes2", str(args.fno_modes),
        "--width", str(args.fno_width),
        "--outdir", str(args.out_root / "fno"),
        "--save-model", str(args.out_root / "fno" / "model.pt"),
    ])


def run_opt(args: argparse.Namespace) -> None:
    ensure_out(args.out_root)
    cmd = [
        sys.executable,
        "run_optimization.py",
        "--config", str(args.opt_config),
        "--stage", args.opt_stage,
        "--out", str(args.out_root / "opt" / f"{args.preset}_{args.opt_stage}"),
    ]
    if args.preset in {"smoke", "quick"} or args.quick_opt:
        cmd.append("--quick")
    run(cmd)


def run_tests(args: argparse.Namespace) -> None:
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"])


def zip_results(args: argparse.Namespace) -> None:
    target = args.results_zip
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    print(f"[collab_run] zipping results to {target}")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder in [args.out_root, ROOT / "figs"]:
            if not folder.exists():
                continue
            for path in folder.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(ROOT))
    print(f"[collab_run] wrote {target} ({target.stat().st_size / 1e6:.2f} MB)")


def normalize_models(models: list[str]) -> list[str]:
    aliases = {
        "smoke": ["dataset", "pinn", "fno", "opt"],
        "all": ["dataset", "pinn", "fno", "opt"],
        "both": ["pinn", "fno"],
    }
    out: list[str] = []
    for model in models:
        for item in aliases.get(model, [model]):
            if item not in out:
                out.append(item)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["smoke"],
                    choices=["smoke", "all", "dataset", "pinn", "fno", "both", "opt", "tests"],
                    help="Which experiment(s) to run. Use multiple values for a custom sequence.")
    ap.add_argument("--preset", choices=["smoke", "quick", "full"], default="quick",
                    help="Runtime preset. smoke is a fast end-to-end check; full uses paper/default sizes.")
    ap.add_argument("--device", default="auto", help="Passed to PyTorch scripts: auto, cuda, cpu, etc.")
    ap.add_argument("--out-root", type=Path, default=ROOT / "out" / "colab", help="Output directory")
    ap.add_argument("--dataset", type=Path, default=ROOT / "out" / "colab" / "dataset.npz", help="FNO dataset path")

    ap.add_argument("--pinn-steps", type=int, default=None, help="Override PINN Adam steps")
    ap.add_argument("--lbfgs-steps", type=int, default=200, help="PINN L-BFGS polish steps for quick/full")
    ap.add_argument("--hard-bc", action="store_true", help="Use the hard-boundary-condition PINN ansatz")
    ap.add_argument("--ckpt-every", type=int, default=0, help="PINN checkpoint frequency; 0 disables")

    ap.add_argument("--fno-epochs", type=int, default=None, help="Override FNO epochs")
    ap.add_argument("--fno-modes", type=int, default=12, help="FNO Fourier modes in each dimension")
    ap.add_argument("--fno-width", type=int, default=20, help="FNO channel width")

    ap.add_argument("--opt-config", type=Path, default=ROOT / "configs" / "opt" / "benchmark.yaml")
    ap.add_argument("--opt-stage", choices=["all", "gates", "o0", "o1", "o2", "o3", "o4", "opt"], default="gates")
    ap.add_argument("--quick-opt", action="store_true", help="Force --quick for optimization")

    ap.add_argument("--zip-results", action="store_true", help="Zip out-root and figs after running")
    ap.add_argument("--results-zip", type=Path, default=ROOT / "out" / "colab_results.zip")
    return ap.parse_args()


def main() -> None:
    os.chdir(ROOT)
    args = parse_args()
    if "smoke" in args.models:
        args.preset = "smoke"
        # Keep smoke optimization very short unless the user explicitly asks.
        if args.opt_stage == "gates":
            args.opt_stage = "o0"
    cuda_report()
    print(f"[collab_run] free disk at root: {shutil.disk_usage(ROOT).free / 1e9:.1f} GB")

    dispatch = {
        "dataset": build_dataset,
        "pinn": run_pinn,
        "fno": run_fno,
        "opt": run_opt,
        "tests": run_tests,
    }
    for model in normalize_models(args.models):
        print(f"\n[collab_run] === running {model} ({args.preset}) ===")
        dispatch[model](args)

    if args.zip_results:
        zip_results(args)
    print("\n[collab_run] complete")


if __name__ == "__main__":
    main()
