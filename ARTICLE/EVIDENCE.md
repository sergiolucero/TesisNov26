# EVIDENCE.md — every number in the article

The article's own gate: **no quantitative sentence in `sections/06-results.tex`
without a row here.** Each row names the artifact, the seed count (or `derived`
/ `N=1`), and the command that regenerates it. If a number is not here, it is
`preliminary` and must be labelled so in the text.

Status keys:
- `derived` — true by construction; seed-independent.
- `multi` — median over N>=5 seeds with IQR.
- `gated` — passes a named gate on a named artifact.
- `certified` — re-evaluated out of sample with a Wilson interval.
- `single` — one seed; must be labelled in prose.

---

## 1D PINN baseline (Table `tab:1d`)

| Claim | Value | Class | Seeds | Artifact | Command |
|---|---:|---|---:|---|---|
| Full-grid median | 1.6132e-2 | multi | 5 | `runs/sweeps/baseline_n5/sweep_summary.json` | `./SCRIPTS/seed_sweep.py --seeds 0-4 --metric metrics.regions.full --outdir runs/sweeps/baseline_n5 -- python3 -u SCRIPTS/solve_1d_pinn.py --hard-bc --eps 1e-3 --adam-steps 10000 --lbfgs-steps 250 --outdir {outdir}` |
| Full-grid IQR | 2.1750e-2 | multi | 5 | same | same |
| Full-grid min / max | 1.1710e-2 / 8.1439e-2 | multi | 5 | same | same |
| Interior median (`tau,xi>0.05`) | 1.4232e-2 | multi | 5 | `runs/sweeps/baseline_n5/run/s*/run_config.json` | read `metrics.regions["tau,xi>0.05"]` from each run |
| Interior IQR | 2.5897e-2 | multi | 5 | same | same |
| Interior min / max | 6.6391e-3 / 8.1617e-2 | multi | 5 | same | same |
| Mean full-grid | 3.1874e-2 | multi | 5 | `sweep_summary.json` | same |
| Spearman rho full vs interior | 1.0 | derived from 5 runs | 5 | run JSONs | compute ranks |
| Worst-seed ratio interior/full | 1.00 | derived from 5 runs | 5 | run JSONs | compute ratios |

## FNO

| Claim | Value | Class | Seeds | Artifact | Command |
|---|---:|---|---:|---|---|
| Original FNO rel L2 | 0.973 | single | 1 | `out/fno10/` | `solve_1d_fno.py` original |
| Constant-input ablation | 0.805 plateau / spatial std ~2e-10 | derived + single | 1 | `out/fno10/` | constant-input test |
| Coord channels, 250 epochs | 1.46e-2 | single | 1 | `out/fno10/` | `solve_1d_fno.py` with coords |
| Coord channels, 120 epochs, median | 3.0225e-2 | multi | 5 | `runs/sweeps/fno_coords_120ep/sweep_summary.json` | `seed_sweep.py` |
| IQR / range | 3.1665e-3 / 2.3466e-2–5.9827e-2 | multi | 5 | same | same |
| Profile-only rel L2 | 7.1e-2 | single | 1 | fixed run | `solve_1d_fno.py` |
| Modes 4/12/24 | 2.79/2.35/2.19 % | single | 1 each | mode-sweep artifacts | `--modes` |
| Resolution 64² / 32² | 4.14 / 6.92 % | single | 1 each | resolution artifacts | `--coarse-factor` |

## 2D stochastic optimization (Tables `tab:2d`, `tab:cert`)

| Claim | Value | Class | N | Artifact | Command |
|---|---:|---|---:|---|---|
| float64 policy q* | `[0,9.72,5.86,0,3.38,0,0,0,0]` | certified | 40 test | `runs/colab_multi/20261118-114831` | `collab_run.py` / `multi_colab.py` |
| float64 J train / test | 433,945 / 434,008 | certified | 40 test | same | same |
| float64 reliability train / test | 0.990 / 0.975 | certified | 40 test | same | same |
| float64 wall clock | 4.14 h | gated | — | same | runtime log |
| float32 policy q* | `[0,9.33,8.16,0,0,0,0,0,0]` | certified | 40 test | `out/certified_20261120-111849` | `collab_run.py` |
| float32 J train / test | 428,592 / 428,579 | certified | 40 test | same | same |
| float32 reliability train / test | 0.990 / 0.975 | certified | 40 test | same | same |
| float32 wall clock | 1.67 h | gated | — | same | runtime log |
| 400-scenario certificate float64 | 1/400, 0.998, Wilson [0.986,1.000] | certified | 400 | `out/cert/f64_M400.json` | `check_tail.py --m-scen 400` |
| 400-scenario certificate float32 | 7/400, 0.983, Wilson [0.964,0.991] | certified | 400 | `out/cert/f32_M400.json` | same |
| Fresh draw (seed 900000) J | 428,509 vs 428,592 | certified | 400 | `out/cert/f32_M400.json` + fresh run | `check_tail.py --seed-base 900000` |
| Fresh draw reliability | 391/400 = 0.978, Wilson [0.958,0.988] | certified | 400 | same | same |
| Fresh draw CVaR5%(G) | +0.616 vs +0.005 | certified | 400 | same | same |
| Fresh draw at M=100 | 4/100 = 0.960, Wilson [0.902,0.984] | certified | 100 | same | same |
| Plume peak median | 35.81 → 0.433 mg/L | multi | 80 test | `out/certified_20261120-111849/figures/contamination.png` | `plot_policy_plume.py` |
| Compliance median | 17.23 → 0.0260 mg/L | multi | 80 test | same | same |
| Compliant scenarios | 1/80 → 78/80 | multi | 80 test | same | same |

## 3D (Table `tab:3d`)

| Claim | Value | Class | N | Artifact | Command |
|---|---:|---|---:|---|---|
| Smoke preset | q*=0, degenerate | gated | 12 test | `out3d_colab/run_smoke/` | `full3D_colab_run.py --preset smoke` |
| Full preset | q*=[12.89,12.7,11.91,12.5,0,0], rel 0.979 | gated | 48 test | `out3d_colab/run_paper/` (full artifacts) | `--preset full` |
| Paper preset q* | `[12.5,12.5,12.5,12.5,0,0,0,0,0]` | gated | 128 test | `out3d_colab/run_paper/` | `--preset paper --budget 15 --n-starts 2` |
| Paper J train / test | 547,605.94 / 547,665.31 | gated | 128 test | same | same |
| Paper reliability | 0.96875 | gated | 128 test | same | same |
| Paper cleanup time | 110.6 d vs 3650 d no pump | gated | 128 test | same | same |
| Paper mass balance | 9.7e-15 | gated | — | same | same |
| Third-machine reproduction | q*, J exact; 1014.8 s / 49 evals | gated | 128 test | local run log | same |

## Cost (Table `tab:cost`)

| Claim | Value | Class | Artifact | Command |
|---|---:|---|---|---|
| Batched compass sweep speedup | 47x at 80x40, T4, M=8 | gated | `DOCS/GPU.md` | `bench_backend.py` |
| Mixed float32 speedup | 2.46x at 160x96 | gated | `DOCS/GPU.md` | `bench_backend.py` |
| A100 vs T4 | no gain at this geometry | gated | `DOCS/GPU.md` | `bench_backend.py` |
| Matvecs per evaluation | 4,022 (40.2/solve) | gated | `DOCS/Conclusions.md` §9 | instrumented `bicgstab_solve` |

## Structural / derived

| Claim | Value | Class | Basis |
|---|---:|---|---|
| Residual floor at corner | 1/eps = 1.0e3 | derived | `s'(0)=1/eps` in hard-BC ansatz |
| Corner pointwise error | 1.0 | derived | IC says 0, reference says 1 |
| KL retained variance | 0.9500 | gated | O1 artifact |
| O0 Thiem / velocity | 6.89e-3 / 1.4e-12 | gated | gate artifact |
| O2 mass balance | 7.1e-15 | gated | gate artifact |
| O3 adjoint rel error | 2.6e-11 | gated | gate artifact |

## Repository totals (Section 3.4)

| Quantity | Value | Command |
|---|---:|---|
| Commits | 69 | `git rev-list --count HEAD` |
| Tracked files | 304 | `git ls-files | wc -l` |
| Python lines | 23,739 | `git ls-files '*.py' | xargs wc -l` |
| Test modules | 22 | `ls tests/test_*.py | wc -l` |
| Test functions | 172 | `grep -h 'def test_' tests/*.py | wc -l` |
| Optimization configs | 8 | `ls configs/opt | wc -l` |
| Runtime records | 33 | `wc -l runs/runtimes.jsonl` |

## Run gates (recorded)

| Claim | Value | Artifact | Command |
|---|---:|---|---|
| Tests passing | 172/172 in 27.5 s (2026-11-21) | this file | `python3 tests/run_tests.py` |
| Build check | 3 chapters, 283 labels, 43 macros, 21 packages; no errors | this file | `python3 SCRIPTS/build_tex.py --check-only` |
| Article build | 23 pages; 0 undefined refs/citations | `ARTICLE/article.pdf` | `make check` in `ARTICLE/` |
