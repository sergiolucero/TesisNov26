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
| **coord channels, 250 epochs** | **0.0146** (seed 0) | 15× better |
| coord channels, profile-only | 0.071 | learns front shape |

The trivial-predictor guard added to `solve_1d_fno.py` now prints both
baselines and warns when a run fails to beat them, so this class of error
cannot silently ship again.

**Phase 4 checks (coord channels, 120 epochs unless noted):**

- *Frequency truncation* — modes 4 / 12 / 24 → 2.79% / 2.35% / 2.19% **single
  seed**; five-seed medians below (4.06% / 3.02% / 2.47%) keep the ordering and
  the diminishing returns, at a worse absolute level.
- *Resolution invariance* — train 128² (250 ep) → 1.46%; train 64² → 4.14%;
  train 32² → 6.92%, all evaluated at 128². Degrades gracefully, confirming the
  FNO is genuinely resolution-agnostic rather than hardcoded to a grid.
- *Normalization* — per-channel standard-Gaussian, stats from the training split
  only (no test leakage).

**Five-seed medians (all of them).** Every FNO number above was one draw per
configuration. Sweeping five seeds changes two of them.

| arm (128², 32 scenarios) | median test rel L2 | IQR | per seed (s0…s4) | vs mean-field 0.221 |
| :--- | ---: | ---: | :--- | ---: |
| modes 12, 120 ep | 3.02e-2 | 2.82–3.14e-2 | 2.35, 3.02, 2.82, 3.14, 5.98 e-2 | 7.3× |
| **modes 12, 250 ep** | **1.99e-2** | 1.68–2.14e-2 | 1.46, 1.99, 1.68, 2.14, 4.59 e-2 | **11.1×** |
| modes 4, 120 ep | 4.06e-2 | 3.99–4.44e-2 | 2.79, 4.06, 3.99, 4.44, 7.18 e-2 | 5.4× |
| modes 24, 120 ep | 2.47e-2 | 2.44–2.61e-2 | 2.19, 2.47, 2.61, 2.44, 5.00 e-2 | 8.9× |

`runs/sweeps/fno_coords_120ep/`, `fno_coords_250ep/`, `fno_modes_120ep/`.

Paired on the shared seeds: 250 ep beats 120 ep with median ratio **0.660**,
better on **5/5** seeds; modes 24 beats modes 4 with median ratio **0.66**,
better on **5/5**. Both effects are real and replicated — the seed spread is
~3× within an arm, but it is the same ~3× in every arm, so it cancels in the
paired comparison.

What does *not* survive is the absolute level. **The 1.46% quoted for 250 epochs
is seed 0, the best of the five**; the honest figure is **2.0%**, and "15× better
than mean-field" becomes **11×**. The 120-epoch single-seed 2.35% was likewise a
favourable draw (median 3.02%). This is the third time on this project that a
single-seed FNO/PINN number flattered the method, which is why §8 rule 1 exists.

Still single-seed and therefore still `preliminary`: the resolution-transfer
rows (64² 4.14%, 32² 6.92%), the profile-only 0.071, and the 300-epoch plateau.

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
is visible in `contamination.png`, which plots both cases with independent
colour scales (a shared scale makes the cleaned panel look empty) and adds the
breakthrough curves at the compliance point.

Two presentation defects were fixed while producing this. `plot_plume` drew
every well as an identical dot, so a solution concentrating on one well was
indistinguishable from an even layout — it now annotates each active well with
its rate in m³/d and scales the marker, for both the 2D driver and the
stochastic pipeline. And the comparison panel originally used a shared
colourbar, which hid the contrast; each panel now carries its own scale and
stated range. Eleven tests in `tests/test_plot_plume.py` cover the annotation, the
all-zero-q path, short `q` vectors, and the merged figure (summary numbers,
independent colour scales, the breakthrough curves and the case without them).

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
- NumPy↔torch backend parity (§9) — 28 tests at 1e-12…1e-10 and an identical
  end-to-end policy, so the two implementations are the same model.
- GPU throughput (§9) — measured on a T4 and an A100: 47× per compass sweep at
  `M=8`, ~1.9 s per sweep at `M=40`, and an A100 that does not beat the T4.

**Still single-seed or unverified:**
- The outlet-bias trend magnitudes in §3b.
- FNO mode/resolution numbers were run at 120 epochs (one seed each); they show
  the *trend* correctly but should be medians before appearing in a paper.
- 3D stochastic optimization: validated at three presets (smoke/full/paper) —
  see `3D_results.md`; the paper preset (32×16×8, 64/128 scenarios) gives
  q* on 4 of 9 wells, test reliability 0.969, all gates pass. (2.5D axisymmetric
  bridge is still untouched.)

- The GPU numbers here are one geometry (80×40, 9 wells, `nsteps=100`) on rented
  T4/A100 instances, float32. The `M`-scaling line (`1.52 s + 8.9 ms × M`) is a
  two-point fit on the A100; treat extrapolations beyond `M=40` as provisional.
- A full `--stage opt` on a GPU has since completed, twice, at the larger 160×96
  hydrology (`DOCS/GPU.md` §6-7): 4.14 h in float64 and 1.67 h in mixed float32.
  The float32 policy is the accepted answer (J 428,579) and is certified on 400
  scenarios no run had seen, though its reported CVaR tail does not survive that
  fresh draw (+0.005 in-run against +0.616 out of sample).
- ~~Certify the tail~~ — **done, and it is the wrong statistic to certify**
  (2026-11-21). Five policies re-solved on one common 400-scenario draw
  (`runs/tail_calibration/`, `DOCS/Results.md` §2.2): the M=16 policy that
  reported reliability 1.000 on its own sample **fails** out of sample (0.912,
  CI [0.881, 0.936]), so the audit discriminates. But the accepted policy's
  CVaR-5 % takes three values on three disjoint draws (+0.616, +0.254, +0.334)
  whose spread (0.36) is smaller than their own standard errors (±0.21…±0.47),
  because `CVaR-5 %` averages 20 of the most variable draws at M=400. Reliability
  over the same draws is 0.978/0.975/0.978 and J reproduces to 0.02 %. **Certify
  with reliability plus its Wilson interval; quote CVaR with its standard error
  and never rank policies by it.**

**Open:**
- ~~Rebuild `TEX/formulation.pdf`~~ — **done** (2026-11-21). A user-level TinyTeX
  lives in `~/.TinyTeX` on this host, all five chapter PDFs are rebuilt from the
  current sources, and `SCRIPTS/build_tex.py --compile` assembles the three
  disjoint chapters into a 42-page document with no undefined reference or
  citation.
- The assembled thesis is `TEX/build/tesis.pdf` (gitignored, regenerate with the
  script). As of 2026-11-21 it **is** a thesis rather than a staple: `report`
  class, one `\chapter` per selected file, and three new chapters ---
  `introduction` (the remediation problem, the two numbers that are easy to get
  wrong, contributions), `related_work` (subsurface flow, learned surrogates,
  optimization under uncertainty, SAA, verification practice, tail statistics)
  and `conclusions` (results, the methodological result, five limitations, five
  pieces of future work) --- plus a title page, a dedication and an abstract.
  68 pages, 0 undefined references. On the `doc-latex` branch the master is
  titled *A Stochastic Groundwater Remediation Problem solved 25 years later
  with AI* (subtitle unchanged) and carries a dedication page naming Roger Wets,
  Alejandro Jofré and Terry Rockafellar; both are `--title` / `--dedication`
  arguments of `build_tex.py`, so they are one edit each, not a fork. On that
  branch Chapter 1 also opens with the provenance: the problem as a 2001 doctoral
  thesis in Applied Mathematics at UC Davis, advised by Roger Wets and examined
  with Graham Fogg (hydrogeology) and Janko Gravner (probability); what that
  thesis actually produced, namely a deterministic flow-and-transport forward
  model and no stochastic optimization at all; and what the field of the period
  could and could not support. Chapter 1 then states the thesis's problem as a
  single stochastic program,
  `(P) min E[J] + rho_g [CVaR(G)]^2_+ + rho_h [CVaR(-H)]^2_+` subject to the
  heterogeneous Darcy and transport equations, the chance constraint and the
  well bounds, with a table fixing decisions, uncertainty, state, source,
  compliance, permit and risk. Discretization, the surrogates and the algorithm
  stay in Chapters 3-4, by design: the introduction states the program, it does
  not derive it. Chapter 3 now also carries the learned-surrogate evidence the
  title promises: a new section 3.10 reports the five-seed FNO results (250-epoch
  median 1.99e-2, IQR 1.68-2.14e-2; 11.1x better than the 0.221 mean-field
  predictor, 15.1x only at seed 0), the mode-truncation sweep (4.06/3.02/2.47e-2
  at k_max 4/12/24, paired m24>m4 on 5/5, median ratio 0.66), and the
  constant-operator defect (rel L2 0.973, *worse* than the zero predictor's 1.0,
  output spatial std 2e-10, fixed by appending normalized (xi,tau) coordinate
  channels, with a trivial-predictor warning now in solve_1d_fno.py). Before
  this section the introduction and conclusions quoted those numbers and no
  chapter produced them. `results_3d.tex` is now Chapter 6 ("Three dimensions"), seven
sections: model and partial-penetration wells; the seven solver gates (decisive
one: 3D-vs-2D extrusion agreement at 1.9e-13); the paper preset (`q* = 4 x
12.5 m3/d`, test J 547,665, reliability 0.969, cleanup 110.6 d) with the two
trivial baselines (doing nothing; uniform half capacity, 3% cheaper in J and
failing 55% of the time) and three-machine reproducibility; **eight thousand
cells and the search that breaks** (the size_mid run: test reliability 0.875,
fresh 128-draw certificate 0.930 [0.872, 0.963] INCONCLUSIVE, flat ladder, and
the alpha*M_train = 1.2 mechanism, with the cost of a converged start -- 8-9 h,
not the 6.5 h the run notes first projected); **what the plume movie shows**
(untreated/vs/q* panels plus peak 44.8 -> 6.0 mg/L, cells above the limit 35.7%
-> 6.3%); **the wall is the search, not the solver**, which is where the
progressive-hedging line enters: degenerate on the one-stage model (a 9-dim
consensus block has nothing to decompose, so the implementation is a parity
harness), substantive with two-stage recourse, where VSS and EVPI become
reportable; and the limits. Two caveats are stated up front in the chapter and
are self-inflicted: at M_test = 128 the paper preset's interval [0.922, 0.988]
does not clear the 0.94 gate, and no tail is quoted in 3D at all. What it still is *not*: a complete
monograph. The two half-written chapters (`SolvingAlgorithms`, `plans_2d_3d`)
remain outside the assembly because they restate material that
`optimization.tex` already contains.
- ~~Apply per-seed medians to the FNO sweep table~~ — **done** (2026-11-21):
  five seeds at 120 ep, 250 ep and modes 4/12/24 (§5). The 250-epoch headline
  fell from 1.46% to a median 1.99% (IQR 1.68–2.14e-2); the paired gains
  (250>120, m24>m4) replicate 5/5. Resolution-transfer rows and the 300-epoch
  plateau are still single-seed.
- DO-lane execution of the bounded opt run to exercise the dispatch path.
- Remove the per-step kernel launches (§9): worth ~5× on an A100, more than a
  card upgrade.
- Run one full `--stage opt --backend torch --device cuda` and confirm the
  selected `q*` matches the NumPy policy — the last unfilled link in the chain.

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
6. **A speedup is only a speedup if the answer is the same.** Every backend
   comparison in this repository is gated on a relative objective agreement
   (measured at 1e-12) before any timing is quoted, and the slower backend is
   kept as the reference rather than deleted.
7. **Measure the device you are paying for.** Hardware claims are the easiest
   thing to inherit from a README and the hardest to falsify later, so the code
   ships `bench_backend.py` (a verdict from *your* runtime) instead of a table
   of somebody else's speedups.

---

## 9. Compute backend — the GPU is worth paying for, but the bottleneck it removes is launch overhead, not arithmetic

**Before this work the GPU ran two 1D toys and nothing else.** `solve_1d_pinn.py`
and `solve_1d_fno.py` were the only modules importing torch (a 2→64→64→64→64→1
MLP and a small 1D FNO on ~1.5k points, both launch-bound); `physics/fv2d.py` and
every `stoch/*` module are NumPy + SciPy `splu`, and `SCRIPTS/collab_run.py`
forwarded `--device` to the PINN/FNO stages but never to the `opt` stage. The
single most expensive stage — `run_optimization.py --stage opt`, the one that
timed out at >1800 s in §6 — therefore ran on one CPU core with the accelerator
idle. Paying for GPU hours bought nothing; the honest answer at the time was
"no", and it was a statement about the code, not about the hardware.

**The fix is a second backend for that stage.** `physics/fv2d_torch.py` +
`stoch/saa_torch.py` replace per-scenario `splu` with matrix-free 5-point
stencils and batched Krylov solves (CG for the SPD Darcy problem, BiCGSTAB for
transport) over one `S candidates × M scenarios` axis, keeping the *identical*
implicit-Euler scheme — no operator splitting, so the mass-balance identity and
every gate still hold. The NumPy path is untouched and remains the reference.

### It is the same model

Parity, float64, CPU (`tests/test_fv2d_torch.py`, `tests/test_saa_torch.py`;
88 tests in the suite, all passing):

| quantity | torch vs NumPy |
| :--- | ---: |
| background/unit heads `h0`, `hw` | 1e-12 / 1e-15 |
| transport field, one step and 8-step runs | 4e-12 … 9e-12 |
| objective terms `J`, `G`, `H` | 6.5e-11 / 9.3e-10 / 1.3e-14 |
| `C_final`; per-step mass ledger | 6.2e-12; 5e-11 |
| global mass-balance residual | **5.9e-11** (NumPy: 1.8e-15) |
| `--quick --stage opt`, end to end | **identical**: `q* = [0.1, 0, 0, 9.38]`, train J 378906.45, test J 378822.07 |
| float32 forward on a T4 vs NumPy (`--verify`) | rel_max 4.4e-05 |

The one honest weakening: the mass-balance floor moves from 1e-15 to 1e-11,
because Krylov solves stop on a relative residual rather than an exact
factorisation. The `O2` gate is 1 %, so nothing fails — but 1e-15 is a *NumPy*
property and must not be quoted for the batched path.

### On a CPU the batching is a loss, and that is the control

`SCRIPTS/bench_backend.py`, float64, `benchmark.yaml`, one core:

| geometry | backend | archive | one evaluation | compass sweep (18) |
| :--- | :--- | ---: | ---: | ---: |
| 80×40, 100 steps, M=8 | NumPy `splu` | 0.25 s | 3.85 s | 69 s |
| 80×40, 100 steps, M=8 | torch, CPU | 37.2 s | 10.5 s | **168 s** |
| 30×15, 40 steps, M=8 | NumPy `splu` | 0.05 s | 0.11 s | 2.0 s |
| 30×15, 40 steps, M=8 | torch, CPU | 0.18 s | 1.14 s | 2.0 s |

0.37× the SciPy throughput at the full geometry, 0.10× at the quick one. There
is nothing to parallelise on one core, so a batched call costs ~`S`× one
candidate plus Python-loop overhead `splu` does not have. Two consequences: the
batched path is a GPU vehicle and never a CPU optimisation (batched search
auto-enables only on `cuda`/`mps`), and the CPU run is the control that makes
the GPU comparison meaningful — if the objective had disagreed, the timing
would have been irrelevant. The benchmark checks agreement (|dJ|/J = 1.2e-12)
before quoting a ratio.

### On a T4 the same sweep is 47× faster, and the gap widens with `M`

Measured on a rented **Tesla T4** (`SCRIPTS/new_high_colab.py --launch
--gpu-class turing --accelerator T4 bench forward --preset paper --verify`,
torch 2.14.1+cu130, float32, 80×40, `nsteps=100`, 9 wells):

| candidates `S` (`M=8`) | T4 wall clock | ms per (cand·scen) |
| ---: | ---: | ---: |
| 1 | 1.41 s | 176.7 |
| 18 (one compass sweep) | **1.46 s** | 10.16 |
| 288 | 17.58 s | 7.63 |

Against the NumPy sweep of **69 s** for the same 18 candidates at `M=8`, that is
a **47× speedup** — and the important part is the scaling, not the constant:

- `S=1 → 18` cuts the per-slice cost **17×**: batching *is* the win.
- `S=18 → 288` adds only 1.3× more while VRAM grows 14× (0.08 → 1.11 GB), so a
  compass sweep is the right batch size; there is little left above it.
- At `M=40` the sweep measures **1.86 s**, i.e. 1.3× the `M=8` time for 5× the
  scenarios. **The GPU is nearly flat in `M`; NumPy is linear in `M`.** That is
  where the two-order-of-magnitude gap at the paper's scenario count comes
  from, and it is why the `--quick` sizes (450 cells, `M=8`) show the GPU at
  parity while the real run does not.

Budget, from that measurement (18 evaluations per sweep, 9 wells):

| run | SAA evaluations | sweeps | T4, per-sweep measurement | T4, `ms/slice` extrapolation |
| :--- | ---: | ---: | ---: | ---: |
| bounded (1 start, `max_evals=200`) | 200 | 12 | **~22 s** | 24 min |
| paper (3 starts × `max_evals=1500`) | 4500 | 250 | **~7.8 min** | 9.1 h |

The right-hand column is the figure `GPU.md` originally carried, and it is
wrong by ~20×: it takes a *per-slice* cost derived at `M=8` (where 1.5 s of
fixed overhead is amortised over only 144 slices) and multiplies it by `M=40`,
then slips a per-sweep/per-evaluation factor. Measured directly, the paper run
is minutes. **Against ~21 h for the same budget on one CPU core, that is the
entire justification for the subscription** — and it was invisible from the
code audit alone, which is exactly why the audit's "no" needed a measurement
before anyone acted on it.

### The remaining cost is fixed overhead, not arithmetic — so the next lever is software

Two measurements say the same thing:

- At `S=288` a slice moves 1.58 MB in 7.6 ms ≈ 208 MB/s, against ~320 GB/s of
  HBM. The memory floor is ~5 µs, three orders of magnitude below the measured
  time; float32-vs-float64 and other traffic reductions buy almost nothing.
- An **A100 ties the T4** on one compass sweep (10.85 vs 10.16 ms per slice)
  despite 13× the fp32 matmul throughput. Scaling `M` at fixed `S=18` on the
  A100 gives `wall = 1.52 s + 8.9 ms × M`: at `M=40`, **81 % of the call is
  pure overhead** while the card sits idle.

The cost is the ~100 time steps × ~24 dependent BiCGSTAB iterations, each issued
as a separate short solve. Removing the per-step launches (one batched solve for
the whole time history, or a persistent kernel) is worth **2.3 h → 0.45 h on the
A100 — a 5× win from software against the 0× a better card delivered.** Buy the
cheapest card that fits VRAM; spend the difference on engineering time. One
caveat for `bench_backend.py`: because it derives ms/slice at the configured
`M`, it over-estimates a realistic `M=40` evaluation by ~4×, so its projection
is an upper bound, not a forecast.

**Update (2026-11-21): measured on an A100, and the 5x is 2.8x.**

Two things were done about the paragraph above: the launch count was measured
instead of read off the code, and then the proposed lever was implemented and
timed on an A100-SXM4-40GB (`SCRIPTS/bench_krylov_fusion.py`,
`runs/bench/fusion_a100_*.json`, float32, `S=18`, `nsteps=100`, 2-3 reps).

*The launch count.* Instrumenting `bicgstab_solve` at 80×40 / `nsteps=100` /
`M=8` gives **100 solves and 4022 batched matvecs, 40.2 per solve**: ~4000
dependent kernel launches per evaluation, 1.7× the ~2400 assumed here.

*Implementation.* `physics/fv2d_torch.py` gained a stencil fast path that passes
the five coefficient tensors **as arguments**, so the whole BiCGSTAB iteration is
one pure-tensor function (`_stencil_step`) that `torch.compile` can fuse without
capturing anything -- a capture would recompile per candidate batch and cost more
than it saves. Off by default (`TESIS_FUSE_KRYLOV=1` to enable); the legacy
matvec path is untouched and the suite asserts the two agree bit for bit.

*The measurement.* One compass sweep of 18 candidates:

| geometry (float32, A100) | baseline | check every 8 | fused | fused + check8 |
| :--- | ---: | ---: | ---: | ---: |
| 80×40, M=40 | 2.83 s | 2.97 s (0.95×) | **1.30 s (2.18×)** | 1.29 s (2.20×) |
| 160×96, M=40 | 21.42 s | 22.34 s (0.96×) | **7.73 s (2.77×)** | 7.92 s (2.70×) |

So: **fusing the iteration is worth 2.2-2.8×, not 5×.** The direction of the
argument above is right -- the win is software and it is large -- but at the
geometry the campaign actually runs (160×96) it is 2.8×, and the "2.3 h → 0.45 h"
projection is 1.8× optimistic. Parity holds: the objective moves by 1.3e-07 in
float32 (the same order as the check8 slack) and is bit-identical in float64.

Two by-products worth keeping:

- **The host syncs are not the cost.** Checking the residual every 8 iterations
  instead of every one is *neutral to slightly negative* (0.95-0.96×). The
  bottleneck is kernel launches, not the device-to-host reads, which is the
  opposite of what the "one batched solve per step" framing suggests.
- **The two-point cost model is not reproducible to better than ~15%.** Fitting
  `wall(M)` inside one run predicted 18.2 s for the 160×96, M=40 sweep that had
  just been measured directly at 21.4 s. The `wall = 1.52 s + 8.9 ms × M` line
  above (and therefore the "81% overhead at M=40") came from the same kind of
  two-point fit and should be read as an order of magnitude, not a number. On the
  160×96 run the fused path cost `0.50 s + 153 ms × M` against the baseline's
  `0.46 s + 445 ms × M`: fusion cut the *marginal* cost 2.9×, which is where the
  win is, and left the fixed cost alone.

Practical consequence for the campaign: enabling the flag on a GPU run leaves the
answer unchanged and turns a 1.7 h float32 optimization into ~40 min, for a
one-off ~1-2 min Dynamo compile. It stays opt-in so that every number already in
these docs keeps its stated provenance.

### Decision rule

| what you actually run | verdict |
| :--- | :--- |
| `--models smoke` / `--preset quick` (450 cells, `nsteps=40`, `M=8`) | **No.** The GPU is at parity with the CPU there; a subscription buys nothing. |
| full `--stage opt` at the paper geometry (80×40, `M=40`, thousands of evaluations) | **Yes.** ~8 min on a T4 against ~21 h on one core. |
| the same, on an A100 at this batch size | **No.** It ties the T4 on a compass sweep; the overhead dominates. |
| the same, on an A100 with `TESIS_FUSE_KRYLOV=1` | Re-measure. The T4 parity was a *launch-bound* result; fused, the A100 does a 160×96 sweep in 7.7 s against 21.4 s, so "a better card buys nothing" describes the unfused code only. |
| 160×80 and larger | Yes, and more so: batch further and the fixed cost amortises better. |

`--dtype float32` is the screening mode (field error ~1e-5, mass balance ~1e-6,
`--verify` 4.4e-05): enough to rank policies, not enough to publish. Any policy
selected in float32 must be re-scored in float64 (NumPy) before it appears in a
table.

Solver properties (`O1`–`O4`: exact-adjoint gradient, 1e-15 mass balance,
chance-constraint reformulations) deliberately stay on the NumPy reference: they
certify the *solver*, not the throughput, and porting them would weaken exactly
the numbers they exist to pin down. `SLSQP`/the discrete adjoint is NumPy-only
and raises a clear error under `--backend torch` rather than returning a
gradient. The same stencil machinery already exposes `Stencil5.apply_transpose`
(the exact transpose of the transport operator, checked in
`tests/test_fv2d_torch.py`), so batching the adjoint is a port, not a redesign.

---

*Sources: `runs/sweeps/baseline_n5/`, `runs/phase3_tau_ramp.json`,
`out/fno10/` (superseded), `configs/opt/benchmark_bounded.yaml` (b508372),
`SCRIPTS/seed_sweep.py`, `runs/runtimes.jsonl`. §9 additionally: `DOCS/GPU.md`,
`SCRIPTS/bench_backend.py`, `SCRIPTS/bench_krylov_fusion.py`,
`runs/bench/fusion_a100_80x40.json`, `runs/bench/fusion_a100_160x96.json`,
`tests/test_fv2d_torch.py`, `tests/test_saa_torch.py`,
`tests/test_krylov_fusion.py`, `physics/fv2d_torch.py`, `stoch/saa_torch.py`.
Per-phase detail in `Actions.md`; per-phase plans in `plan.md`.*

