"""Notebook helper: run `run_2d.py --mode optimize` on a resized domain.

This is a scratch helper for interactive Colab/Jupyter sessions, not part of
the pipeline.  It reads ``configs/opt/benchmark_small.yaml``, rewrites
``domain.nx``/``domain.ny``, writes the derived config next to the original and
launches the 2D optimizer on it.

    from libcolab import modrun
    modrun(32, 16)

Two changes from the original one-liner: it uses ``sys.executable`` (``python``
is not on PATH everywhere) and it resolves every path from the repository root,
so it behaves the same when imported from a notebook whose cwd is anywhere.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "configs/opt/benchmark_small.yaml"
DERIVED_CONFIG = ROOT / "configs/opt/benchmark_new.yaml"


def modrun(nx: int, ny: int) -> None:
    data = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    was = [data["domain"][k] for k in ("nx", "ny")]
    print("WAS", *was, "NOW:", nx, ny)
    data["domain"]["nx"] = int(nx)
    data["domain"]["ny"] = int(ny)
    DERIVED_CONFIG.write_text(yaml.safe_dump(data, sort_keys=False),
                              encoding="utf-8")
    print("RUNNING...")
    subprocess.run(
        [sys.executable, str(ROOT / "SCRIPTS" / "run_2d.py"),
         "--mode", "optimize", "--config", str(DERIVED_CONFIG)],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__.strip())
        raise SystemExit(0)
    if len(argv) != 2:
        raise SystemExit("usage: libcolab.py NX NY   (or import modrun and call it)")
    modrun(int(argv[0]), int(argv[1]))
