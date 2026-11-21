# Actions Log — Groundwater Remediation Models (PINN & FNO)

Debugging the PINN/FNO groundwater transport models following `plan.md`
(Phases 1–4). This file tracks what has been done, what is known, and what is
next.

## Environment

- Python: `/root/base/bin/python3` (3.12.3) — has `torch 2.15.0+cpu`,
  `numpy 2.5.4`, `scipy 1.18.1`, `matplotlib 3.11.2`.
  - ⚠️ The default `python3` has **no torch**; always use `/root/base/bin/python3`.
- Jupyter Lab entrypoint: `jl.sh` (`jupyter lab --ip=0.0.0.0 --allow-root`).
- Orchestrator: Herdr 0.9.5, installed to `/root/.local/bin/herdr` (stable
  channel); start with `herdr` (TUI) or `herdr server` (headless).
- Working directory: `/root/TESIS`.
- **Run discipline:** every training run is capped at 10 min via `timeout 600`.

## Files

| File | Purpose | Status |
|---|---|---|
| `plan.md` | 4-phase debugging plan (1D sandbox → BC → optimizer → FNO) | Phase 1 ✓, Phase 2 ✓, Phase 3 tau-ramp run (negative) — see below |
| `formulation.tex` / `.pdf` | Math formulation: dimensionless ADE, Ogata–Banks ref, PINN & FNO loss | compiled |
| `SCRIPTS/hydrology.py` | Synthetic 1D ADE data generator (Ogata–Banks 1961) | works, verified |
| `SCRIPTS/solve_1d_pinn.py` | PINN baseline solver (soft/hard BC, Adam + L-BFGS, config/manifests, checkpoint/resume, batching, tau-ramp) | working |
| `out/scenario.npz` | Single scenario (Pe=20, tau_max=0.5, 256×256) | generated |
| `out/dataset.npz` | FNO dataset — **as generated: 10 scenarios, 50×100 grid** (v∈[0.05,0.5], D∈[0.1,2.0]); earlier notes saying 32/128×128 were wrong | generated |
| `out/pinn*/` | PINN runs: soft-BC, hard-BC variants, figures + `model.pt` | see run log |
| `figs/*.png` | Data-generator demo figures | generated |

## What has been done

### Phase 1 — 1D sandbox baseline ✅ (PASSES in the interior)
1. **Physics stripped to 1D.** Dimensionless ADE (homogeneous, isotropic,
   non-reactive): `u_tau + u_xi = (1/Pe) u_xixi`, with `u(ξ,0)=0`,
   `u(0,τ)=1`, `u(1,τ)=0`; Pe=20, tau_max=0.5.
2. **Analytical reference** (`SCRIPTS/hydrology.py::u_reference`/`C_reference`):
   Ogata–Banks, `tau→0` limit guarded. Verified: C∈[0,1], inlet=1, IC=0.
3. **PINN solver** (`SCRIPTS/solve_1d_pinn.py`): MLP (tanh hidden, sigmoid output),
   autodiff residual, Adam (resampled collocation) + L-BFGS polish, relative-L2
   + region-wise error breakdown.
4. **Baseline result (soft BC):** rel L2 = **2.43e-2** (full grid), left
   boundary layer 3.79e-2, interior (τ,ξ>0.1) 1.64e-2.

### Phase 2 — BC verification (hard BC) ✅ (best config)
- Added `HardBCMLP` ansatz that hard-encodes the IC and outlet BC and
  (approximately) the inlet:
  `u(ξ,τ) = (1-ξ)·s(τ) + ξ(1-ξ)·s(τ)·NN(ξ,τ)`, with `s(τ)=1-exp(-τ/eps)`.
- Swept the inlet smoothing width `eps`:

| Config | full rel L2 | interior (τ,ξ>0.05) | note |
|---|---|---|---|
| soft BC (baseline) | 2.43e-2 | 1.59e-2 | left layer 3.79e-2 |
| hard BC, s=τ/(τ+eps), eps=1e-3 | 2.02e-2 | 1.76e-2 | left layer 1.91e-2 |
| **hard BC, s=1-exp(-τ/eps), eps=1e-3** | **1.35e-2** | **8.0e-3** | best; interior < 1e-2 |
| hard BC, s=1-exp(-τ/eps), eps=1e-4 | 5.05e-2 | 3.80e-2 | unstable (steep s near τ=0) |

### Key finding — corner singularity
- The remaining full-grid error (1.35e-2 vs interior 8e-3) is concentrated in a
  thin corner band (τ<0.005, ξ<0.03): max |err|=1.0 at the measure-zero point
  (ξ,τ)=(0,0), and ~0.2 in the band τ≈0.002.
- Cause: the step-inlet problem is **discontinuous at (0,0)** — the IC says
  u=0 while the inlet BC says u=1. The reference (Ogata–Banks) returns 1 there;
  any continuous, IC-honoring ansatz returns 0 there. This is a metric artifact,
  **not a network/loss bug**.
- **Conclusion:** the core architecture and loss formulation are correct — the
  PINN converges to the analytical solution to **~8e-3 everywhere except the
  measure-zero corner** (this is the Phase 1 "yes" branch → proceed).

## Next steps (continue here)

- [x] **Phase 1 wrap-up:** `TEX/formulation.tex` §6 (Verification metric) now
      states a reporting policy — median over N>=5 seeds with IQR, never a
      single run — and §8 (Baseline results) replaced the single-run figures
      (1.35e-2 / 8.0e-3) with the five-seed table: full-grid median **1.61e-2**
      (IQR 2.18e-2, spread 7.0x), corner-excluded median **1.42e-2**
      (IQR 2.59e-2, spread 12.3x). Corrects the earlier claim that the interior
      target was met: only the best seed (6.64e-3) beats 1e-2, the median does
      not. Adds the paired-comparison decision rule and the tau-ramp
      counter-example. Verified: braces and environments balanced, all refs
      resolve, Spearman rho=1 between the two metrics (interior/full ratio
      0.57-1.00, so corner exclusion does not reorder or rescue the runs).
      **Still to do on a TeX machine: `latexmk -pdf TEX/formulation.tex`** —
      no LaTeX toolchain here, so `TEX/formulation.pdf` is stale until rebuilt.
- [x] **Phase 2 residual inspection (re-derived + corrected):** re-ran seed 0
      end-to-end (reproduced E_rel=1.4168e-2 exactly) and regenerated the
      figures into `out/residual/` after they were lost. Trained interior mean
      |R|=2.13e-2 (was 2.08e-2), inlet 3.98e-1, IC band 7.78e-1, corner 7.26e1.
      Two corrections to the recorded conclusion:
      (a) the max |R|=1e3 at (0,0) is **structural, not physical** — it equals
      1/eps exactly, because the ansatz's s(tau)=1-exp(-tau/eps) contributes
      u_tau = s'(tau) -> 1/eps. It is identical at init and after training and
      identical for seeds 0/1/2, so no training can remove it; it is a metric
      artefact like the corner discontinuity, not a residual. `inspect_residual.py`
      now prints the 1/eps reference and the ratio (observed 1.000).
      (b) "no interior spikes" holds, but the interior is **outlet-biased**:
      mean |R| rises 16x with xi (1.5e-2 for xi<0.2 -> 2.5e-1 for xi>0.8 at
      tau=0.5) and 20x with tau near the outlet (3.0e-3 at tau=0.1 ->
      2.6e-1 at tau=0.5). Monotone, smooth, coincident with u->0 at the outlet
      where the hard constraint pins u(1,tau)=0. Not a bug, but a trend the 2D
      work should watch.
- [x] **Portability basics:** added device/dtype, config manifest,
      checkpoint/resume + SIGTERM checkpointing, collocation batching and
      gradient accumulation, tau-ramped PDE weighting flag, Dockerfiles,
      compose file, `.dockerignore`, and `runs/best_hardbc.json`.
- [x] **Phase 3 experiment — tau-ramp: NEGATIVE RESULT.** 5 paired seeds per
      arm (hard-BC `eps=1e-3`, Adam 10k + L-BFGS 250, CPU), recorded in
      `runs/phase3_tau_ramp.json`. Seed 0 alone suggested a large win (full
      1.42e-2 → 9.57e-3, interior 8.3e-3 → 4.6e-3), but it does **not
      replicate**: ramp wins only 2/5 seeds, median ramp/base error ratio 1.31
      (i.e. worse), permutation test p=0.38. The real story is that **seed
      variance dominates**: the baseline alone ranges 1.17e-2 … 8.14e-2 (7×)
      at fixed hyperparameters. Conclusion: tau-ramp does not reliably shrink
      the corner band, and single-seed comparisons in this setup are
      uninformative — any future claim needs a multi-seed median. RAR remains
      untested. See `runs/phase3_tau_ramp.json` for the raw numbers.
- [ ] **Phase 3 optimizer:** extend the Adam→L-BFGS schedule (more L-BFGS on a
      fixed fine grid) and add gradient logging if we keep soft BCs.
- [x] **Phase 3 follow-up (seed-robustness): `SCRIPTS/seed_sweep.py`.** The 7×
      spread at fixed settings was the dominant error source, ahead of
      loss-shaping tricks. The harness runs any command across a seed range,
      pulls one scalar per run out of that run's JSON manifest (`--metric
      metrics.regions.full`), and reports **median / IQR / min-max** plus a
      **paired** arm-vs-arm comparison on shared seeds. It refuses to compare
      unpaired runs and prints the spread next to every median, so a claimed
      improvement can be judged against the noise it must beat. Usage is in the
      script's docstring; 17 unit tests in `tests/test_seed_sweep.py` (suite is
      now 49 tests). **Policy: report median-of-N with N>=5 from now on; a
      single-seed number is `preliminary` and must be labelled so.**
- [ ] **Phase 3 follow-up (seed reduction):** knowing the spread exists is not
      the same as shrinking it. Candidate fixes worth testing *with the
      harness*: seeded collocation annealing (ramp collocation density in τ
      rather than the loss weight), multi-restart pick (train k seeds, keep the
      median-quality model), or an ensemble average at eval time. Each must
      beat the 1.61e-2 median baseline by more than its IQR to count.
- [ ] **Phase 4 FNO — currently broken, not merely untested.** `out/fno10`
      reports rel L2 **0.973** (per-sample 0.803 / 1.143; >1.0 is worse than
      predicting all zeros), on a test set of **2 samples**. The recorded
      dataset facts were also wrong: `out/dataset.npz` holds **10 scenarios**
      (not 32) at **50×100** (not 128×128). Before any GPU time: regenerate
      >=32 scenarios, add a trivial-predictor baseline so "worse than zeros"
      fails fast, and re-verify with `seed_sweep.py`. Also note `--save-model
      True` produced a 3.7 MB file named `True` (now gitignored).

## Run log

| Date | Command / action | Result | Notes |
|---|---|---|---|
| (prior) | generated `out/scenario.npz`, `out/dataset.npz` | ok | Pe=20 sandbox; dataset actually 10 scenarios (not 32 — corrected) |
| (prior) | compiled `formulation.tex` | ok | PDF at `formulation.pdf` |
| (prior) | ran `SCRIPTS/solve_1d_pinn.py` once | figures only, error not logged | superseded |
| today | soft-BC baseline, adam 12k + lbfgs 200 | rel L2 2.43e-2 | `out/pinn/` |
| today | hard-BC (rational s), adam 10k + lbfgs 250 | rel L2 2.02e-2 | `out/pinn_hardbc/` |
| today | hard-BC (exp s, eps=1e-3), adam 10k + lbfgs 250 | rel L2 1.35e-2, interior 8e-3 | **best**, `out/pinn_hardbc_exp/` |
| today | hard-BC (exp s, eps=1e-4), adam 10k + lbfgs 250 | rel L2 5.05e-2 | unstable, `out/pinn_hardbc_eps4/` |
| today | error-location diagnostic on best model | corner band dominates | max err 1.0 @ (0,0) |
| today | residual inspection on best model | Phase 2 closed | trained interior mean \|R\|=2.08e-2; spike confined to corner/inlet |
| today | portability/code hardening | smoke-tested | checkpoint/resume, grad accumulation, tau-ramp, manifests, Docker/compose files |
| today | fleet artifacts from `NewPlan.md` | dry-run + smoke | added `run_colab.sh` (Colab T4 entry point, sentinel `/tmp/tesis-colab.done`) and `SCRIPTS/fleet_bootstrap.sh` (idempotent Tailscale join, `--role hub\|relay\|surface\|colab`, `--dry-run`, `--verify`); `run_optimization.py --quick` re-run green in 11.9 s |
| today | installed Herdr 0.9.5 on the hub | ok | official installer → `/root/.local/bin/herdr`; `herdr server` + `herdr status` smoke-tested; hub role of `SCRIPTS/fleet_bootstrap.sh` now installs it (`--herdr`/`--no-herdr`), docs updated |
| today | fleet tooling round 2 | green | added `SCRIPTS/fleet_status.sh` (human + `--json`, non-zero exit on failed checks), `SCRIPTS/herdr-server.service` + bootstrap `--systemd` (service enabled, lingering on), and `[opt]` stage/iteration progress in `run_optimization.py`/`stoch/optimize.py`; quick opt unchanged (q*=[0.1,0,0,9.38], 12.8 s) |
| today | fleet dashboard (`NewPlan.md` §9) | JSON contract verified, routes checked | new `app.py` route `/fleet` + `templates/fleet.html`: shells out to `fleet_status.sh --json` (15 s timeout) and renders host/tailnet/Herdr/repo/sentinel/disk server-side (no JS, tablet-friendly); `/fleet?role=hub` flags a stopped Herdr server; probe failures (missing script, bad JSON, timeout) render a banner instead of a 500. Linked from `/` and `/plots`; docs in `README.md` + `NewPlan.md` §5/§7/§9. Could not boot Flask here (no flask/jinja on this laptop) — validated the JSON schema, all 22 template field refs, route/`url_for` wiring, and the three error paths against the real script; 32 unit tests still green |
| today | Phase 3 tau-ramp sweep, 5 paired seeds per arm | NEGATIVE, artifact kept | seed 0 looked like a big win (full 1.42e-2→9.57e-3) but it does not replicate: ramp better on 2/5 seeds, median ramp/base ratio 1.31, permutation p=0.38. Baseline alone spans 1.17e-2…8.14e-2 at fixed settings, so seed variance dominates. Raw numbers in `runs/phase3_tau_ramp.json`; conclusion recorded in Next steps |
| today | Phase 2 3-seed residual check | structural claim replicates | interior mean |R| 2.13e-2 / 2.01e-2 / 2.35e-2 for seeds 0/1/2 (±8%), inlet 3.98e-1 / 3.72e-1 / 3.49e-1, corner 72.6 / 68.2 / 67.9. The 1/eps = 1.000e3 corner value is identical on every seed, confirming it is structural. The outlet-bias *magnitudes* remain single-seed |
| today | Phase 4 FNO root-caused and fixed | **0.973 -> 0.0146** | the operator was structurally incapable: input was two spatially constant channels, `fc0` is pointwise and spectral layers multiply Fourier coefficients, so only the DC mode exists and the output is constant by construction (measured output spatial std ~2e-10; profile-only rel L2 = 1.000). Proven with a constant-input test. Fix: append normalized (xi,tau) coordinate channels (`--no-coords` for the ablation). 250 epochs -> test 1.46%, 15x better than the 0.221 mean-field baseline. Regenerated the dataset properly: 32 scenarios at 128x128 (was 10 at 50x100). Added trivial-predictor guards that warn when a run fails to beat all-zeros or mean-field |
| today | Phase 4 mode + resolution checks | trends as expected | modes 4/12/24 -> 2.79%/2.35%/2.19% (monotone, diminishing); coarse-factor 2/4 (train 64/32, eval 128) -> 4.14%/6.92%, so resolution invariance holds (graceful degradation, no hardcoding). 120 epochs each, single seed — trend only, needs medians before publication |
| today | full `--stage opt` completed | feasible, budget was 10x too big | cost is ~5 s/eval, so max_evals=1500 x n_starts=3 implies ~6 h — that is what timed out before. Bounded run (`configs/opt/benchmark_bounded.yaml`, 200 evals/1 start/20 iters) finished in **18.8 min**: q*=[14.83,0,...,0] m3/d, train J 419162 / test J 419089, reliability 0.975, cleanup 91 d. Convergence is early: 0.06% gain over the last 75 evals, so max_evals~150 with 2-3 restarts is the right DO-lane budget |
| today | wrote `Conclusions.md` | consolidated | all four phases plus the stochastic pipeline in one place, every number checked against the stored artifacts, with an explicit "trustworthy vs single-seed vs open" section and the methodology rules the seed-variance finding forces |
| today | live Colab telemetry: `logs`/`/runs` read the VM log | **a live run is no longer "quiet"** | Run `20261121-111857` (1×T4, 160×96×100, M100/40/40) sat at iter 6 while both the driver's `r<k>/run.log` and the dashboard said nothing: `colab exec` only streams a cell's output home when the cell *returns*, and the run cell holds the kernel for the whole run, so the local log is 157 bytes of CLI header for hours. Worse, `logs -f` on such a run sent `!tail` through `colab exec`, which queues behind the run cell — the fix had to avoid the kernel entirely. New shared helper `colab_status.remote_log_text()` pulls a remote file with `colab download` (filesystem API, read-only, ~3 s, writes only to a temp dir); `colab_status.py --live-logs`, `/runs?live_logs=1`, `/runs/log/<kind>/<id>?remote=1` and `multi_colab.py logs --source auto|remote` all use it, so the same parser that already read progress out of a finished log now reports `iter 9/40, 163/500 evals, 32.6%` for a run in flight. Pulls are bounded (30 s, one retry) and only for replicas in `running/quiet/starting/provisioning`; a failed pull degrades to the local view with the reason inline. 8 new tests in `tests/test_colab_status.py`. |
| today | a dying run keeps its artifacts, and one NaN candidate stops being fatal | **both fixed, 14 new tests** | Run `20261121-111857` (2 starts x 250 evals, mixed float32, M=100) completed start 1 (250 evals, iter 15, best 471458) and then died at start 2 iter 7 on `transport Krylov ... worst residual nan` -> `RuntimeError: objective is not finite for candidate(s) [5]`. Two driver defects made that worse than it needed to be. (1) `--zip-results` ran only after a *successful* dispatch, so nothing was uploaded and start 1's `out/` died with the VM (`colab download` then reported "File or directory not found"); it now zips from a `finally` (`zip_results_safely`, which never masks the real error) and `run_models()` is a testable seam. (2) `_finite_or_raise` treats one diverging candidate out of the 18 in a compass sweep as fatal; the new opt-in `--allow-nonfinite` (opt-in, `run_optimization.py` and `collab_run.py`) maps it to `+inf`, which can never pass `fc < best - 1e-12`, so the sweep rejects that move and continues, while an *entire* non-finite sweep still raises because that is the batched solve diverging, not one unlucky candidate. Non-finite candidates are counted (`prob.n_nonfinite`, `metrics['stages']['opt']['nonfinite_candidates']`, warned once) and `multi_start` still raises if every start ends non-finite, so a broken backend cannot be silently tolerated. Tests: `tests/test_optimize.py` (policy, sequential search, real batched sweep with one NaN candidate, whole-sweep fatal, all-starts-fatal) and `tests/test_collab_run.py` (zip on crash, zip failure not masking, no zip without the flag). GPU.md section 6 records that mixed float32 at M=100 is "usually fine, occasionally fatal", not the safe mode its first 400-evaluation run suggested. |
