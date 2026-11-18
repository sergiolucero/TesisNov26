#!/usr/bin/env python3
"""Run the full 3D stochastic pump-and-treat optimization from Colab/anywhere.

Thin driver over full3D_colab_run.py.  Edit the variables below — the budget
is the wall-clock ceiling (in minutes) the optimizer is allowed to spend; the
script calibrates one archive evaluation and clamps the evaluation count to
fit inside it.

    python -u run_full3d.py
"""

from pathlib import Path

import full3D_colab_run as f3d

# -------------------- editable settings -----------------------------------
PRESET = "full"          # smoke | quick | full | paper
BUDGET_MINUTES = 20.0    # hard wall-clock budget for the optimizer
ZIP_RESULTS = True       # write out3d_colab/<run>/colab3d_results.zip
PLOTS = True             # write PNG figures (needs matplotlib)
OUT_DIR = "out3d_colab"  # results land in <OUT_DIR>/run_<PRESET>/
# -------------------------------------------------------------------------


def main() -> None:
    cfg = f3d.build_config(PRESET, {})
    out = Path(OUT_DIR) / f"run_{PRESET}"
    f3d.run(
        cfg,
        out,
        budget_s=BUDGET_MINUTES * 60.0,
        plots=PLOTS,
        zip_results=ZIP_RESULTS,
    )


if __name__ == "__main__":
    main()
