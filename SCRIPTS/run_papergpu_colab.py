#!/usr/bin/env python3
"""Send ``full3D_colab_run.py`` (``paperGPU`` preset) to a fresh Colab GPU VM.

This is a thin wrapper around the ``colab`` CLI (``google-colab-cli``): it
provisions a GPU runtime, uploads the single self-contained script, runs the
``paperGPU`` preset on the batched torch backend, retrieves the output files
(``out3d_colab/run_paperGPU/`` including ``colab3d_results.zip``), and tears
the VM down -- one command, nothing else to manage.

Usage
-----
    python SCRIPTS/run_papergpu_colab.py                 # T4, float32, 60 min
    python SCRIPTS/run_papergpu_colab.py --gpu A100 --budget 90
    python SCRIPTS/run_papergpu_colab.py --gpu L4 --dtype float64
    python SCRIPTS/run_papergpu_colab.py -- --n-starts 4   # extra script args
    python SCRIPTS/run_papergpu_colab.py --dry-run          # print, don't run

Environment variables (overridden by the flags above)
------------------------------------------------------
    COLAB_AUTH    auth strategy: adc (default) | oauth2 | none
    COLAB_GPU     accelerator: T4 | L4 | G4 | H100 | A100   (default: T4)
    COLAB_DTYPE   torch dtype: float32 (default) | float64
    COLAB_BUDGET  optimizer wall-clock budget in minutes     (default: 60)

Prerequisites
-------------
    pip install google-colab-cli
    gcloud auth application-default login      # ADC, no keyring needed
    # or, if ADC is unavailable:
    colab auth && colab sessions --auth oauth2

``--timeout 0`` is always passed so a long, quiet optimization sweep cannot
trip the CLI's 30 s "no output" watchdog.  The reference one-liner is::

    colab --auth adc run --gpu T4 --timeout 0 full3D_colab_run.py \
        --preset paperGPU --backend torch --dtype float32 --budget 60 --zip
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "full3D_colab_run.py"

GPU_CHOICES = ("T4", "L4", "G4", "H100", "A100")


def parse_args(argv: list[str]) -> argparse.Namespace:
    # Everything after ``--`` is forwarded verbatim to full3D_colab_run.py.
    if "--" in argv:
        sep = argv.index("--")
        runner_argv, script_args = argv[:sep], argv[sep + 1:]
    else:
        runner_argv, script_args = argv, []

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--gpu", default=os.environ.get("COLAB_GPU", "T4"),
                    choices=GPU_CHOICES,
                    help="Colab accelerator (default: T4, or $COLAB_GPU)")
    ap.add_argument("--dtype", default=os.environ.get("COLAB_DTYPE", "float32"),
                    choices=["float32", "float64"],
                    help="torch dtype (default: float32, or $COLAB_DTYPE)")
    ap.add_argument("--budget", type=float,
                    default=float(os.environ.get("COLAB_BUDGET", "60")),
                    help="optimizer wall-clock budget in minutes "
                         "(default: 60, or $COLAB_BUDGET)")
    ap.add_argument("--auth", default=os.environ.get("COLAB_AUTH", "adc"),
                    help="colab --auth value (default: adc, or $COLAB_AUTH)")
    ap.add_argument("--keep", action="store_true",
                    help="leave the VM running after the run (for `colab exec`)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the command instead of running it")
    args = ap.parse_args(runner_argv)
    args.script_args = script_args
    return args


def main() -> int:
    args = parse_args(sys.argv[1:])

    colab = os.environ.get("COLAB_BIN") or shutil.which("colab")
    if not colab:
        print(
            "ERROR: 'colab' not found on PATH.\n"
            "Install the Colab client first:\n"
            "    pip install google-colab-cli\n"
            "then authenticate with `gcloud auth application-default login`\n"
            "or `colab auth` (and pass --auth oauth2).\n"
            "Or set COLAB_BIN=/path/to/colab.",
            file=sys.stderr,
        )
        return 2
    if not SCRIPT.exists():
        print(f"ERROR: script not found: {SCRIPT}", file=sys.stderr)
        return 2

    cmd = [colab]
    if args.auth and args.auth != "none":
        cmd += ["--auth", args.auth]
    cmd += ["run", "--gpu", args.gpu, "--timeout", "0"]
    if args.keep:
        cmd.append("--keep")
    cmd += [
        str(SCRIPT),
        "--preset", "paperGPU",
        "--backend", "torch",
        "--dtype", args.dtype,
        "--budget", str(args.budget),
        "--zip",
    ]
    cmd += args.script_args

    print(f"[papergpu-colab] $ {shlex.join(cmd)}", flush=True)
    if args.dry_run:
        return 0

    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
