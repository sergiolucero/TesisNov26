# 3D Results — Stochastic Optimization and Remediation

Validation results for `full3D_colab_run.py` (self-contained numpy/scipy
runner; TPFA Darcy, diagonal Bear dispersion, `K_h` with `kv_ratio` vertical
anisotropy, partially penetrating wells; cross terms and density coupling
deliberately omitted per `3D_plan.md` §1).

All runs use CVaR chance constraints (α=0.05 on concentration, α_h=0.01 on
head), penalty weights ρ_g = ρ_h = 1e6, and pattern search with multiple
starts. The reported `q*` is selected on the training archive and scored
out-of-sample on the test archive.

## Presets validated

| preset | grid (nx·ny·nz) | n_kl | scenarios (tr/te) | wells | result |
|---|---|---|---|---|---|
| smoke | 12·6·3 = 216 | 164 | 6 / 12 | 4 | degenerate: already compliant, q*=0 |
| full | 24·12·6 = 1728 | 819 | 24 / 48 | 6 | **q\*=[12.89, 12.7, 11.91, 12.5, 0, 0]**, reliability 0.979 |
| paper | 32·16·8 = 4096 | 1419 | 64 / 128 | 9 | **q\*=[12.5, 12.5, 12.5, 12.5, 0×5]**, reliability 0.969 |

## Paper preset (headline)

- **q\*** = `[12.5, 12.5, 12.5, 12.5, 0, 0, 0, 0, 0]` m³/d — total 50 m³/d
  across 4 of 9 wells, clustered around the plume centroid. (Note: these are
  *equal* rates, in contrast to the `full` preset's unequal 12.9/12.7/11.9/12.5,
  because the pattern search converged on the same 12.5 rate for all four
  active wells.)
- **train J** = 547 605.94; **test J** = 547 665.31 (gap 1.1e-4, no overfitting).
- **test reliability = 0.969** (≥ 1−α−1% = 0.94 gate) vs **no-pump reliability 0.0**.
- **test mean cleanup τ = 110.6 d** vs no-pump τ = 3650 d (never cleans up).
- test violation probability 0.031, mean head margin H = 10.03 m (above h_min).
- mass-balance error ~7e-15 (machine precision).

### Gates (all pass)

`KL_retained_variance≥0.95` ✓ · `reproducible` ✓ · `mass_balance≤1%` ✓ ·
`test_reliability≥1−α−1%` ✓ · `optimization_improved_vs_no_pump` ✓ ·
`no_pump_reliability = 0.0` ✓

### Cross-machine reproducibility

The paper preset's `f* = 5.476e+05` from the q₀=0 start matches the value
obtained in a partial Colab run of the same preset (`f=5.476e+05`), so the
single-file runner reproduces across machines and runtimes.

## Notes

- The `smoke` preset is degenerate on purpose: the source is weak enough that
  the no-pumping plume already satisfies the limit (q*=0 is correct, and
  `optimization_improved_vs_no_pump = False` there is expected).
- `--n-starts` is now a CLI override (default from preset: 1/1/2/3 for
  smoke/quick/full/paper).
- Run locally: `python full3D_colab_run.py --preset paper --budget 15 --n-starts 2`.
  The paper preset took ~20 min on a 4-core laptop (calibration 19.5 s per
  64-scenario archive evaluation).
- On Colab the same preset calibrates at ~14.9 s/eval on a 2-core runtime; use
  `colab run --timeout 0 full3D_colab_run.py --preset paper --budget 20 --n-starts 2 --zip`
  (CPU runtime — the workload is sparse LU, not GEMM, so a GPU buys nothing).

**Local CPU rerun after the merge (2026-11-21).** The paper preset was re-run on
the assembled tree on a 4-core laptop (`python full3D_colab_run.py --preset paper
--budget 15 --n-starts 2`, 1014.8 s, 49 evaluations) and reproduced this document
exactly: `q* = [12.5, 12.5, 12.5, 12.5, 0×5]`, test `J = 547,665.31`, reliability
`0.96875`, mass-balance error `9.7e-15`, all six gates true. That is a third
machine and a different BLAS after the 3D and opt branches were merged, so the
agreement is not just the same binary.

**Written up as thesis Chapter 6 (2026-11-21).** `TEX/results_3d.tex` turns this
into prose (model, solver gates, presets, baselines, reproducibility, limits) and
adds one caveat that the run's own gate does not express: the test reliability
0.969 is 124/128, whose Wilson interval is [0.922, 0.988]. The point estimate
clears the 0.94 gate; the interval does not. By the reporting rule established in
`DOCS/Results.md` §2.2 the 3D policy is therefore *not* certified, and doing so
needs ~400 test scenarios (~2 h at 18.6 s/eval on one core).
