# Conclusions.md

Consolidated findings for the groundwater remediation PINN/FNO project.
Everything below is either measured in this repository or derived analytically;
numbers that rest on a single seed are labelled as such.

Scope: Phase 1–4 of `plan.md` for the 1D advection–dispersion equation (ADE)
sandbox, plus the stochastic optimization pipeline. 2D/3D are out of scope
(`2D_plan.md`, `3D_plan.md`).

---

## 1. The one result that changes how everything else should be read

**Run-to-run variance dominates every effect we have tried to measure.**

At *fixed* hyperparameters the Phase 1 hard-BC PINN's full-grid relative L2
error ranges over **1.17e-2 … 8.14e-2 across five seeds — a 7× spread**. The
corner-excluded metric is worse still: **12.3×**.

| metric (hard BC, ε=1e-3) | median | IQR | min | max | spread |
| :--- | ---: | ---: | ---: | ---: | ---: |
| full grid | 1.61e-2 | 2.18e-2 | 1.17e-2 | 8.14e-2 | 7.0× |
| corner-excluded (`ξ,τ>0.05`) | 1.42e-2 | 2.59e-2 | 6.64e-3 | 8.16e-2 | 12.3× |

Consequences, applied throughout this document:

- A **single run is not a result.** Any comparison must be paired over N≥5
  seeds, reported as a median with its spread.
- The right decision rule is: a variant counts only if it beats the baseline
  median **by more than the baseline IQR** on the same seeds.
- The distribution is right-skewed (mean 3.19e-2 is twice the median), so
  median + IQR is used, not mean ± std.

Tooling: `SCRIPTS/seed_sweep.py`; recorded baselines in `runs/sweeps/`.

---

## 2. Phase 1 — the sandbox is correct, but the accuracy target is not met

The PINN converges to the Ogata–Banks reference. The architecture and loss
formulation are sound; the remaining error is concentrated at the entrance
corner, which is a genuine discontinuity of the step-inlet problem (IC says
`u=0`, inlet BC says `u=1`, the reference returns 1 at `(0,0)`).

**Correction to earlier drafts.** They quoted `E_rel = 1.35e-2` and
`E_int = 8.0e-3` and concluded the interior target was met. Those were one
seed. The honest statement:

- Only the **best seed** (6.64e-3) is below the 1e-2 target.
- The **median** interior error (1.42e-2) is **not**.
- Reporting the best run would overstate the method by ~2×.

**The corner is not what drives the spread.** Across seeds the two metrics are
perfectly rank-correlated (Spearman ρ = 1) and their ratio ranges 0.57–1.00 —
on the worst seed, excluding the corner removes essentially no error. The
corner is a contributing artefact of the metric, not the governing one.

---

## 3. Phase 2 — hard BCs work; the residual max is an artefact, not a physics error

Hard enforcement is the right choice, but two recorded conclusions needed
correcting.

**a. `max|R| ≈ 10³` is structural, not physical.** It equals `1/ε` exactly.
The ansatz multiplies both terms by `s(τ)=1-e^{-τ/ε}`, so `u_τ` carries
`s'(τ)→1/ε` as `τ→0`. Verified:

- analytically (`s'(0) = 1/ε`);
- numerically identical at **initialization and after training** (10k Adam +
  250 L-BFGS steps change it not at all);
- identical **across seeds 0, 1, 2** (`|R(0,0)| = 1.0000e+03` for all three);
- measured ratio to `1/ε` = **1.000**.

No optimization removes it. It is the residual-side twin of the corner
discontinuity, and it makes a **global residual max a useless diagnostic here**.
Band means are the meaningful quantity.

**b. There are no interior spikes, but the residual is outlet-biased.** At
`τ=0.5`, mean `|R|` rises monotonically with `ξ` from 1.5e-2 (`ξ<0.2`) to
2.5e-1 (`ξ>0.8`) — 16×; and for `ξ>0.8` it rises with `τ` from 3.0e-3 to
2.6e-1 — 20×. Smooth and monotone, coincident with `u→0` where the hard
constraint pins `u(1,τ)=0`. Not a bug. *These trends are from one seed*; the
`1/ε` result above is seed-independent by derivation.

**Three-seed check (new).** Interior band mean `|R|`: 2.13e-2 (s0), 2.01e-2
(s1), 2.35e-2 (s2) — the band diagnostic is stable to ~±8% across seeds, unlike
the global max. Inlet: 3.98e-1 / 3.72e-1 / 3.49e-1. Corner: 72.6 / 68.2 / 67.9.
So the *structural* claim replicates.

---

## 4. Phase 3 — causal τ-ramping does not work (negative result)

Causal τ-ramped PDE weighting was meant to shrink the corner band without
hurting the interior. **It does not.**

| seed | baseline | τ-ramp | winner |
| ---: | ---: | ---: | :--- |
| 0 | 1.417e-2 | 9.574e-3 | ramp |
| 1 | 3.592e-2 | 4.709e-2 | base |
| 2 | 1.171e-2 | 2.840e-2 | base |
| 3 | 8.144e-2 | 5.256e-2 | ramp |
| 4 | 1.613e-2 | 3.834e-2 | base |

Ramp better on **2/5** seeds; median ramp/base ratio **1.31** (the wrong way);
permutation test on the log-median difference **p = 0.38**. Not adopted.

**This is the project's methodological cautionary tale.** Seed 0 alone showed a
32–45% improvement on *every* region — a single-seed sweep would have
"confirmed" the hypothesis. The effect vanished under repetition. Data:
`runs/phase3_tau_ramp.json`.

---

## 5. Phase 4 — the FNO was structurally incapable of the task; now fixed

The originally recorded result (`out/fno10`, rel L2 **0.973**) was not merely
weak, it was **worse than predicting all zeros** (rel L2 is normalized by the
target norm, so the zeros predictor scores exactly 1.0).

**Root cause (found and proven).** The operator's input was two *spatially
constant* channels `(v, D)` broadcast over the whole grid. `fc0` is a pointwise
lift and every spectral layer multiplies Fourier coefficients, so a constant
input has **only the DC mode** — the output is necessarily constant. Measured:
output spatial std **≈2e-10**, i.e. exactly zero. The model learned the right
*level* per scenario (prediction-mean correlation 1.000) but its **profile-only
relative L2 was 1.000** — it had learned nothing about front shape. 300 epochs
plateaued at loss 0.805 from epoch 50 onward; no amount of training could help,
because the architecture could not represent a front.

**Fix.** Append normalized `(ξ,τ)` coordinate channels (standard FNO practice,
Li et al. 2021), so the spectral layers see real non-zero modes. Ablation flag:
`--no-coords`.

**Result.** With 32 scenarios at 128×128 (the dataset was also 10 scenarios at
50×100, not the 32 at 128×128 previously recorded — regenerated):

| run | test rel L2 | vs mean-field (0.221) |
| :--- | ---: | ---: |
| original (constant input only) | 0.973 | 4.4× worse than zeros |
| constant input, 300 epochs | 0.805 plateau | no learning |
| **coord channels, 250 epochs** | **0.0146** | **15× better** |
| coord channels, profile-only | 0.071 | learns front shape |

The trivial-predictor guard added to `solve_1d_fno.py` now prints both
baselines and warns when a run fails to beat them, so this class of error
cannot silently ship again.

**Phase 4 checks (coord channels, 120 epochs unless noted):**

- *Frequency truncation* — modes 4 / 12 / 24 → 2.79% / 2.35% / 2.19%.
  Monotone with diminishing returns; a smooth advection front needs no high `k`.
- *Resolution invariance* — train 128² (250 ep) → 1.46%; train 64² → 4.14%;
  train 32² → 6.92%, all evaluated at 128². Degrades gracefully, confirming the
  FNO is genuinely resolution-agnostic rather than hardcoded to a grid.
- *Normalization* — per-channel standard-Gaussian, stats from the training split
  only (no test leakage).

---

## 6. Stochastic optimization — the full run is feasible, and the budget was ~10× too large

The full `--stage opt` had been recorded as a >1800 s timeout on 1 vCPU. It
completes; the earlier failure was a **budget** problem, not a feasibility one.

Measured cost: **~5 s per objective evaluation** (9 wells, 80×40 grid, 100
transport steps, 40 train / 80 test scenarios). So `max_evals=1500 ×
n_starts=3` implies **~6 h** — that is what timed out.

Bounded run (`configs/opt/benchmark_bounded.yaml`, `max_evals=200`,
`n_starts=1`, `max_iter=20`), **18.8 min**:

- `q* = [14.83, 0, 0, 0, 0, 0, 0, 0, 0] m³/d` (one active well)
- train `J = 419,162`, test `J = 419,089` (train/test gap 0.02%)
- test reliability **0.975** ≥ 1−α−1%
- cleanup time **91 d**

**Convergence is early.** Over the last 75 evaluations (125→200) the objective
improved 419,433 → 419,169, i.e. **0.06%**. Most of the budget buys nothing.
A budget of `max_evals≈150` with 2–3 restarts captures the solution at a
fraction of the cost — this is the number to use for the DO lane.

Also note `--quick` (4 wells, 60 m³/d) and the full benchmark (9 wells,
135 m³/d) are different problems; their `J` values are not comparable.

### What `q*` actually does to the plume

`q*` does not spread the pumping across the layout: it puts the entire budget on
**1 of 9 wells** — index 0 at `(112.5, 51.2) m`, on the plume centreline, at
**14.8 m³/d**. The other eight are shut in.

Measured on test scenario 0 (`SCRIPTS/plot_policy_plume.py`, reproducible from
`policy.json`):

Averaged over the **80 held-out test scenarios** (not one scenario — the table
is a median with IQR, in this project's house style):

| quantity (80 test scenarios) | all wells off | under `q*` | change |
| :--- | ---: | ---: | ---: |
| peak concentration, median | 35.81 mg/L | 0.433 mg/L | **−98.7%** |
|  | IQR 31.3–40.1 | IQR 0.27–1.03 | IQR −97.4% to −99.3% |
|  | range 20.9–66.7 | range 0.11–12.3 | range −81.5% to −99.7% |
| max at compliance point, median | 17.23 mg/L | 0.0260 mg/L | −99.8% |
| scenarios meeting the 1.0 mg/L limit | **1 / 80** | **78 / 80** | — |

The reduction is robust across realisations: the *worst* scenario still drops
81.5%, and the median is 98.7%. The 78/80 compliant figure is the direct
empirical counterpart of the reported 0.975 test reliability — the two
non-compliant scenarios are the same 2.5% the CVaR formulation allows.

Scenario 0 is representative (34.2 → 0.583 mg/L, −98.3%, and compliant at
0.0168 mg/L), which is why it was chosen for the figure.

The plume is **displaced, not erased**: under pumping the residual mass is
drawn toward the well and the peak sits at `x≈115`, just down-gradient of it,
while the far field is cleaned. That is the expected capture-zone behaviour and
is visible in `plume_compare.png`, which plots both cases with independent
colour scales (a shared scale makes the cleaned panel look empty).

Two presentation defects were fixed while producing this. `plot_plume` drew
every well as an identical dot, so a solution concentrating on one well was
indistinguishable from an even layout — it now annotates each active well with
its rate in m³/d and scales the marker, for both the 2D driver and the
stochastic pipeline. And the comparison panel originally used a shared
colourbar, which hid the contrast; each panel now carries its own scale and
stated range. Six tests in `tests/test_plot_plume.py` cover the annotation, the
all-zero-q path, and short `q` vectors.

Note on randomness: `n_kl` resolves to **547** at 95% target variance, so the
80 scenarios carry genuinely heterogeneous conductivity fields — this is not a
single-aquifer result repeated 80 times. Source strength and regional gradient
are held fixed (`random_source: false`), so the table isolates conductivity
uncertainty only.

`q*` was selected on the 40-scenario training archive and scored on the
80-scenario test archive; it was not tuned on the test set.

---

## 7. What is now trustworthy, and what is not

**Trustworthy (multi-seed or derived):**
- Seed-variance figures and the median/IQR baseline (§1).
- The `1/ε` structural residual floor (§3a) — derived, so seed-independent.
- τ-ramping is ineffective (§4) — paired over 5 seeds.
- The FNO coordinate-channel fix and its magnitude (§5) — the failure mode is
  proven by construction (constant input ⇒ constant output), not inferred.
- Resolution-invariance and mode-truncation trends (§5).
- Opt-stage cost and convergence curve (§6).

**Still single-seed or unverified:**
- The outlet-bias trend magnitudes in §3b.
- FNO mode/resolution numbers were run at 120 epochs (one seed each); they show
  the *trend* correctly but should be medians before appearing in a paper.
- 3D and 2.5D: untouched here; `3D_plan.md` stands, and full 3D still does not
  fit the hub (`NewPlan.md` §8).

**Open:**
- Rebuild `TEX/formulation.pdf` — the `.tex` has been corrected (§2) but no
  LaTeX toolchain exists on the assembly host.
- Apply per-seed medians to the FNO sweep table.
- DO-lane execution of the bounded opt run to exercise the dispatch path.

---

## 8. Methodology adopted as a result

1. **Median of N≥5 seeds with IQR**, never a best run; label single runs
   `preliminary`.
2. **Paired comparisons** on the same seeds; unpaired arm comparisons are
   refused by the tooling.
3. **Always print the trivial baseline** (zeros predictor = 1.0 by definition;
   dataset mean-field) next to any relative metric.
4. **Prefer derived artifacts over empirical ones** where possible: the `1/ε`
   floor and the constant-input⇒constant-output proof are structural, and so
   hold for every seed and every run.
5. **A negative result is a result.** Phase 3 saved the project from adopting a
   loss-shaping trick that does not work, which a single-seed sweep would have
   enshrined.

---

*Sources: `runs/sweeps/baseline_n5/`, `runs/phase3_tau_ramp.json`,
`out/fno10/` (superseded), `configs/opt/benchmark_bounded.yaml` (b508372),
`SCRIPTS/seed_sweep.py`. Per-phase detail in `Actions.md`; per-phase plans in
`plan.md`.*
