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

## `fno_m4_cf1/` — PRELIMINARY, not comparable

Modes 4, 5 seeds, **60 epochs**. Median test rel L2 **8.57e-2**
(IQR 1.04e-2, range 5.72e-2 … 1.19e-1).

**Do not compare this to the coord-channel numbers in `Conclusions.md` §5.**
Different training budget: 60 epochs is heavily undertrained for this operator,
so the number reflects the budget, not the modes. It is kept only as evidence
that the harness runs and that the seed spread is measurable on the FNO too
(~2x, tighter than the PINN's 7x).

This was the 60-epoch predecessor of `fno_modes_120ep/` below, which is the
sweep to read for mode truncation (the resolution sweeps are still unrun).

## `fno_coords_250ep/` — the FNO at an adequate budget (5 seeds)

```bash
./SCRIPTS/seed_sweep.py --seeds 0-4 --metric metrics.rel_l2_test_mean \
    --outdir runs/sweeps/fno_coords_250ep --arm fno -- \
    python3 -u SCRIPTS/solve_1d_fno.py --dataset out/dataset.npz --epochs 250 \
        --outdir {outdir} --save-model ''
```

| statistic | test rel L2 |
| :--- | ---: |
| **median** | **1.99e-2** |
| IQR | 1.68e-2 – 2.14e-2 |
| min / max | 1.46e-2 / 4.59e-2 |

Per seed: s0 1.46e-2, s1 1.99e-2, s2 1.68e-2, s3 2.14e-2, s4 4.59e-2.

**This replaces the 1.46e-2 that `Conclusions.md` §5 used to quote** — that was
seed 0, the best of five. Paired against `fno_coords_120ep/` on the shared seeds
the 250-epoch arm wins with median ratio **0.660, 5/5 seeds**, so "more epochs
helps" survives as a claim; the absolute number does not. 64.5 min CPU.

## `fno_modes_120ep/` — mode truncation as a distribution, not a draw

```bash
./SCRIPTS/seed_sweep.py --seeds 0-4 --metric metrics.rel_l2_test_mean \
    --outdir runs/sweeps/fno_modes_120ep \
    --arm m4  -- python3 -u SCRIPTS/solve_1d_fno.py --dataset out/dataset.npz \
        --epochs 120 --modes1 4 --modes2 4 --outdir {outdir} --save-model '' \
    --arm m24 -- python3 -u SCRIPTS/solve_1d_fno.py --dataset out/dataset.npz \
        --epochs 120 --modes1 24 --modes2 24 --outdir {outdir} --save-model ''
```

| arm | median | IQR | per seed |
| :--- | ---: | ---: | :--- |
| modes 4 | 4.06e-2 | 3.99e-2 – 4.44e-2 | 2.79, 4.06, 3.99, 4.44, 7.18 e-2 |
| modes 24 | **2.47e-2** | 2.44e-2 – 2.61e-2 | 2.19, 2.47, 2.61, 2.44, 5.00 e-2 |

Modes 12 at the same budget is `fno_coords_120ep/` (median 3.02e-2), so
4 → 12 → 24 modes is 4.06 → 3.02 → 2.47 e-2: monotone, diminishing. Paired,
modes 24 beats modes 4 with median ratio **0.66, 5/5 seeds**.
