# Seed sweeps — the project's error baselines

Produced by `SCRIPTS/seed_sweep.py`. **Read `sweep_summary.json`, not a single
run's `run_config.json`.**

## Why these exist

Phase 3 found that the 1D PINN's full-grid relative L2 error varies **7×**
between seeds at *fixed* hyperparameters (1.17e-2 … 8.14e-2). A single-seed
comparison therefore proves nothing: seed 0 made `--tau-ramp` look like a 32%
improvement, but the effect did not replicate (2/5 seeds, median ratio 1.31
the wrong way — see `../phase3_tau_ramp.json`).

**Policy:** report median-of-N (N≥5) with IQR. A lone number is `preliminary`.

## `baseline_n5/` — canonical 1D PINN baseline

Command:

```bash
./SCRIPTS/seed_sweep.py --seeds 0-4 --metric metrics.regions.full \
    --outdir runs/sweeps/baseline_n5 -- \
    python3 -u solve_1d_pinn.py --hard-bc --eps 1e-3 \
        --adam-steps 10000 --lbfgs-steps 250 --outdir {outdir}
```

Configuration: hard BC, `eps=1e-3`, Adam 10k + L-BFGS 250, Pe=20,
`tau_max=0.5`, 256×256 eval grid, CPU.

| statistic | full-grid rel L2 |
| :--- | :--- |
| **median** | **1.61e-2** |
| IQR (q1–q3) | 1.42e-2 – 3.59e-2 → **2.18e-2** |
| min / max | 1.17e-2 / 8.14e-2 |
| mean | 3.19e-2 |

Per seed: s0 1.42e-2, s1 3.59e-2, s2 1.17e-2, s3 8.14e-2, s4 1.61e-2.

**How to use it.** A new method has to beat the **median 1.61e-2 by more than
the IQR 2.18e-2** to be worth claiming — i.e. land under ~4e-3 on the median,
or beat baseline on most of the same seeds in a paired comparison. Anything
inside that band is indistinguishable from seed noise.

Note the mean (3.19e-2) is twice the median: the distribution is right-skewed
by the s3 outlier, which is why median+IQR is reported instead of mean±std.

## What is and isn't committed

Per-run `run_config.json` and `sweep_summary.json` are tracked (24 KB total —
they hold config, git SHA and metrics, so a result is reproducible from the
record). Models (`*.pt`), figures (`*.png`) and checkpoints are gitignored.
