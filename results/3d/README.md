# 3D results (versioned)

Lifted from the gitignored `out3d_colab/` run directories. These JSON files are
the machine-readable record behind `DOCS/3D_attribution.md`,
`DOCS/GPU_runs_Colab.md`, and `DOCS/3D_results.md`.

File meanings:

| file | produced by | contents |
|---|---|---|
| `metrics.json` | `full3D_colab_run.py` | optimization result, gates, train/test/no-pump summaries |
| `policy.json` | `full3D_colab_run.py` | optimized `q*` and well coordinates |
| `run_config.json` | `full3D_colab_run.py` | full grid/physics/cost/sampling configuration |
| `rescore.json` | `SCRIPTS/rescore_paper_candidates.py` | numpy/splu re-score of competing candidate policies |
| `verify_splu.json` | `SCRIPTS/verify_papergpu.py` | numpy/splu re-score of a single `q*` |
| `slices1d.json` | `full3D_colab_run.py --slices1d` (or `SCRIPTS/slices1d.py`) | 1D-slice scoring curves (coordinate/scale/equal/split) |

Run directories:

| dir | source run | what it is |
|---|---|---|
| `paper_numpy/` | `out3d_colab/run_paper_local` | numpy/splu paper preset (32x16x8): 4 wells / 50 m3/d |
| `paper_torch/` | `out3d_colab/run_paper_torch` | torch float32 paper preset (32x16x8): 1 south well / 25 m3/d |
| `paperGPU/` | `out3d_colab/run_paperGPU` | torch float32 paperGPU preset (64x32x16): ~1 well / 35 m3/d |
| `paperGPU_slices/` | `out3d_colab/run_paperGPU_slices` | slices-only re-score of the paperGPU `q*` (12 slices) |
| `paper_m256/` | `out3d_colab/run_paper_m256` | torch float32 paper preset with M_train=256: 1 south well / 26 m3/d, test rel 0.945 |

The PNG figures that accompany `slices1d.json` live in `DOCS/assets/`
(`slices1d_coordinate.png`, `slices1d_total.png`, `slices1d_split.png`).
