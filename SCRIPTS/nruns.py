"""Run `run_2d.py --quick` for a handful of seeds, back to back.

Quick sanity sweep: prints the commands it runs and a separator between them so
the per-seed output is readable.

    python SCRIPTS/nruns.py            # N=5 seeds, 10..14
    N=2 python SCRIPTS/nruns.py        # override the count

Note: sequential and unbuffered-free by design; for a real seed study with
aggregated statistics use `SCRIPTS/seed_sweep.py` instead.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N = int(os.environ.get("N", 5))

for ix in range(N):
    seed = ix + 10
    cmd = [sys.executable, str(ROOT / "SCRIPTS" / "run_2d.py"), "--quick",
           f"--seed={seed}"]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=False)
    print("-" * 79)
