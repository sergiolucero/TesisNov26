#!/usr/bin/env python3
"""Colab GPU runner for TESIS experiments.

This script is a small command dispatcher so a Colab notebook can choose which
model/experiment to run without remembering each entry-point's arguments.

Examples
--------
    python SCRIPTS/collab_run.py --models smoke
    python SCRIPTS/collab_run.py --models pinn --preset quick
    python SCRIPTS/collab_run.py --models dataset fno --preset full --fno-epochs 300
    python SCRIPTS/collab_run.py --models opt --preset quick --opt-stage gates

Grid and GPU selection
----------------------
    # run the PINN on the second GPU with a 512-point evaluation grid
    python SCRIPTS/collab_run.py --models pinn --gpu 1 --nx 512

    # 2D optimization on a coarser 40x20 domain grid, first GPU
    python SCRIPTS/collab_run.py --models opt --preset full --gpu 0 --nx 40 --ny 20

`--nx` / `--ny` override the grid for whatever is running: for the dataset and
PINN steps they set the spatial (`nx`) and time (`ny` -> `nt`) resolution, and
for the 2D optimization they override `domain.nx` / `domain.ny` of the
configuration file.  A bare `python SCRIPTS/collab_run.py --models ... --preset
...` keeps the preset defaults.

Every stage is launched as a subprocess from the repository root, so run this
from anywhere: it resolves the root from its own location (`SCRIPTS/..`) and
chdirs there before dispatching.
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

HERE = Path(__file__).resolve().parent        # SCRIPTS/
ROOT = HERE.parent                            # repository root

sys.path.insert(0, str(ROOT))
from stoch.runtimes import log_runtime  # noqa: E402  (needs ROOT on sys.path)

RUNTIME_LOG = ROOT / "runs" / "runtimes.jsonl"


def script(name: str) -> str:
    """Absolute path of a sibling entry point (hydrology.py, run_2d.py, ...)."""
    return str(HERE / name)


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run a subprocess from the project root and fail fast on errors."""
    printable = " ".join(str(x) for x in cmd)
    print(f"\n[collab_run] $ {printable}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    print(f"[collab_run] done in {time.time() - t0:.1f}s", flush=True)


def resolve_device(gpu: str | None, device: str) -> str:
    """Map --gpu onto the --device string handed to the PyTorch scripts."""
    if gpu is None:
        return device
    g = gpu.strip().lower()
    if g in {"", "auto"}:
        return device
    if g in {"cpu", "mps"}:
        return g
    if g.startswith("cuda"):
        return g
    if g.isdigit():
        return f"cuda:{g}"
    raise SystemExit(
        f"[collab_run] unrecognised --gpu {gpu!r}; "
        "use auto, cpu, mps, an index (0, 1, ...) or cuda:N"
    )


def cuda_report(device: str) -> bool | None:
    """Print the GPU/torch status available to this runtime.

    Returns whether CUDA is usable, or None when torch cannot be imported.
    """
    print(f"[collab_run] project root: {ROOT}")
    print(f"[collab_run] selected device: {device}")
    print(f"[collab_run] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    try:
        import torch

        print(f"[collab_run] torch: {torch.__version__}")
        available = torch.cuda.is_available()
        print(f"[collab_run] cuda available: {available}")
        if available:
            print(f"[collab_run] cuda: {torch.version.cuda}")
            count = torch.cuda.device_count()
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                marker = " <= selected" if device == f"cuda:{i}" else ""
                print(
                    f"[collab_run]   gpu {i}: {props.name} "
                    f"{props.total_memory / 1e9:.1f} GB{marker}"
                )
            if device.startswith("cuda:") and int(device.split(":", 1)[1]) >= count:
                raise SystemExit(
                    f"[collab_run] {device} not available; this runtime exposes "
                    f"{count} GPU(s) (0..{count - 1})"
                )
        return available
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[collab_run] could not import torch: {exc}")
        return None


def ensure_out(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (ROOT / "figs").mkdir(exist_ok=True)


# Preset-default 1D grids: nx = space, nt = time, kept square to match the
# original presets.  Any --nx/--ny/--nt on the command line wins over these.
PRESET_GRID = {
    "smoke": {"nx": 64, "nt": 64, "n_scenarios": 8},
    "quick": {"nx": 128, "nt": 128, "n_scenarios": 16},
    "full": {"nx": 256, "nt": 256, "n_scenarios": 32},
}


def grid_1d(args: argparse.Namespace) -> tuple[int, int, int]:
    """Resolve (nx, nt, n_scenarios) for the 1D hydrology/PINN pipeline.

    In 1D there is no y axis, so ``--ny`` is accepted as the name of the time
    axis (the second axis of the (nt, nx) fields); ``--nt`` spells it out.
    """
    base = PRESET_GRID[args.preset]
    nx = args.nx if args.nx is not None else base["nx"]
    nt = args.nt if args.nt is not None else args.ny if args.ny is not None else base["nt"]
    return nx, nt, base["n_scenarios"]


def build_dataset(args: argparse.Namespace) -> None:
    ensure_out(args.out_root)
    nx, nt, n_scenarios = grid_1d(args)
    print(f"[collab_run] dataset grid: nx={nx} nt={nt} scenarios={n_scenarios}")

    run([
        sys.executable,
        script("hydrology.py"),
        "--scenario", str(args.out_root / "scenario.npz"),
        "--dataset", str(args.dataset),
        "--figdir", str(args.out_root / "figs"),
        "--nx", str(nx),
        "--nt", str(nt),
        "--dataset-nx", str(nx),
        "--dataset-nt", str(nt),
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
    # The PINN is meshless, so --nx/--ny only set the evaluation grid used for
    # the reported errors.
    nx_test, nt_test, _ = grid_1d(args)

    cmd = [
        sys.executable,
        script("solve_1d_pinn.py"),
        "--device", args.device,
        "--adam-steps", str(adam_steps),
        "--lbfgs-steps", str(lbfgs_steps),
        "--N-f", str(n_f),
        "--N-ic", str(n_side),
        "--N-bc", str(n_side),
        "--nx-test", str(nx_test),
        "--nt-test", str(nt_test),
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
        script("solve_1d_fno.py"),
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
        script("run_optimization.py"),
        "--config", str(args.opt_config),
        "--stage", args.opt_stage,
        "--out", str(args.out_root / "opt" / f"{args.preset}_{args.opt_stage}"),
    ]
    # The GPU backend only covers the SAA/opt stage: the O-gates verify solver
    # properties (exact adjoint, 1e-15 mass balance) and are defined against
    # the NumPy reference on purpose.
    backend = "numpy"
    if args.opt_backend in {"auto", "torch"} and args.opt_stage == "opt" \
            and _cuda_available():
        backend = "torch"
    elif args.opt_backend == "torch" and args.opt_stage != "opt":
        print(f"[collab_run] note: the torch backend only applies to --opt-stage opt; "
              f"stage {args.opt_stage!r} runs on the NumPy reference")
    cmd += ["--backend", backend]
    if backend == "torch":
        cmd += ["--device", args.device, "--dtype", args.opt_dtype]
    if args.nx is not None:
        cmd += ["--nx", str(args.nx)]
    if args.ny is not None:
        cmd += ["--ny", str(args.ny)]
    if (args.preset in {"smoke", "quick"} or args.quick_opt) and not args.no_quick_opt:
        cmd.append("--quick")
    run(cmd)


def _cuda_available() -> bool:
    """Whether this runtime can actually run on a GPU (never raises)."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


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
                if not path.is_file():
                    continue
                # Names are relative to the *folder*, not to ROOT: --out-root
                # may legitimately live outside the repo (a Colab scratch dir,
                # a mounted Drive path), where path.relative_to(ROOT) raises.
                zf.write(path, Path(folder.name) / path.relative_to(folder))
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
    ap.add_argument("--gpu", default=None,
                    help="GPU selection: an index (0, 1, ...), cuda:N, cpu, mps or auto. "
                         "Overrides --device for the PyTorch scripts.")
    ap.add_argument("--nx", type=int, default=None,
                    help="Grid points along x. 1D: dataset/eval resolution; "
                         "opt: domain.nx of the 2D grid. Default: preset value.")
    ap.add_argument("--ny", type=int, default=None,
                    help="Grid points along y. opt: domain.ny of the 2D grid; "
                         "1D: alias for the time axis (see --nt). Default: preset value.")
    ap.add_argument("--nt", type=int, default=None,
                    help="1D time grid points (overrides --ny for the 1D pipeline).")
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
    ap.add_argument("--opt-backend", choices=["auto", "numpy", "torch"], default="auto",
                    help="Backend for the opt stage. auto = torch when a GPU is "
                         "visible (and the stage is 'opt'), numpy otherwise.")
    ap.add_argument("--opt-dtype", default="float32", choices=["float32", "float64"],
                    help="Working precision of the GPU backend; float32 is the "
                         "fast screening mode, float64 matches the NumPy reference.")
    ap.add_argument("--quick-opt", action="store_true", help="Force --quick for optimization")
    ap.add_argument("--no-quick-opt", action="store_true",
                    help="Never pass --quick to the optimizer, even for the smoke/quick "
                         "presets. Use when the config file already encodes the sizing "
                         "(e.g. driver scripts such as SCRIPTS/multi_colab.py).")

    ap.add_argument("--zip-results", action="store_true", help="Zip out-root and figs after running")
    ap.add_argument("--results-zip", type=Path, default=ROOT / "out" / "colab_results.zip")
    return ap.parse_args()


def validate(args: argparse.Namespace) -> None:
    """Reject grid sizes no downstream model can use."""
    for name in ("nx", "ny", "nt"):
        value = getattr(args, name)
        if value is not None and value < 4:
            raise SystemExit(f"[collab_run] --{name} must be >= 4 (got {value})")
    if args.nx is not None and args.ny is not None and args.nx * args.ny < 100:
        print(
            f"[collab_run] warning: {args.nx}x{args.ny} is a very coarse domain; "
            "the hex well layout and source may not resolve"
        )


def main() -> None:
    os.chdir(ROOT)
    args = parse_args()
    validate(args)
    if "smoke" in args.models:
        args.preset = "smoke"
        # Keep smoke optimization very short unless the user explicitly asks.
        if args.opt_stage == "gates":
            args.opt_stage = "o0"
    device = resolve_device(args.gpu, args.device)
    cuda_ok = cuda_report(device)
    if cuda_ok is False and device.startswith("cuda"):
        raise SystemExit(
            f"[collab_run] --gpu/--device asked for {device} but torch reports no "
            "CUDA device (use --gpu cpu, or pick a GPU runtime)"
        )
    args.device = device
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
        t_task = time.time()
        dispatch[model](args)
        log_runtime(
            RUNTIME_LOG, f"collab_run.py:{model}", time.time() - t_task,
            stage=model, preset=args.preset, device=args.device,
            gpu=args.gpu, models=" ".join(normalize_models(args.models)),
            grid=[args.nx, args.ny, args.nt], out_root=str(Path(args.out_root)),
        )

    if args.zip_results:
        zip_results(args)
    print("\n[collab_run] complete")


if __name__ == "__main__":
    main()
