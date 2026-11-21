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
| today | Colab pipeline monitoring endpoint (`/api/colab`) | 11 new tests green | `app.py` gains `/api/colab` (alias `/runs.json`) returning the same snapshot as `/runs` plus a `pipeline` block whose `state` is one word (`running`/`idle`/`finished`/`failed`/…) and `progress_pct`; `?limit`, `?live=1` and `?run=<id>` supported, `/api/colab/<run-id>` returns one run with replicas/progress/tails. 500 only when the call fails, 404 for unknown run ids, `Cache-Control: no-store`, and the `/runs` footer links it. Reuses `colab_status.collect_runs` (no new inference), so the page and the API can never disagree. Tests in `tests/test_app_colab_api.py` drive the Flask test client over synthetic run dirs; verified live against `runs/colab_multi/20261117-151021`. Docs in `README.md`, `USAGE.md` §5, `Collab.md` §7 |
| today | `hydrology_gpu.yaml` gate suite (O0-O4) | **all green** | 160×96, `nsteps=100`, M40/80, NumPy float64, 429 s total. O0 Thiem 6.89e-3 / velocity 1.4e-12; O1 retained variance 0.9500, n_kl=1529, clamp max 2.9e-3, reproducible; O2 mass balance 7.1e-15, uniform q=5 m³/d gives J=529250 and reliability 1.0 vs no-pump J=317653 / reliability 0.025; O3 adjoint rel. error 2.6e-11; O4 all three formulations select q=5 with reliability 1.0 (feasible region saturated, so O4 is uninformative on this config). Artifact: `out/opt/hydrology_gpu_gates/`. |
| today | hydrology 160×96 GPU opt attempts | **blocked in float32** | Three T4 attempts on `hydrology_gpu.yaml` (2 replicas, seeds 0/1): `20261117-151021` was stopped after 8 min with no remote output; `20261117-152140` reached opt iter 1 (19 evals) then `objective is not finite for candidate(s) [10,12,13,17]`; rerun `20261117-154643` with `--krylov-rtol 1e-4` still failed at candidate [6] (transport report `worst_residual=9.96e-5`, `stalled=True`). Root cause is the float32 batched transport Krylov losing finite residuals for some high/low-rate candidates, not the optimizer. The flow solve itself converged (498–547 CG iterations). `collab_run.py` now forwards `--krylov-rtol`/`--krylov-maxiter`; `multi_colab.py` now persists `outdir` (old plans crashed in the `finally` runtime log, hiding the results). Next: validate float64 or a breakdown-safe BiCGSTAB before another paid run. |
| today | hydrology 160×96 float64 smoke opt | **finished both replicas** | `multi_colab.py` run `20261117-160842`, 2×T4, `--dtype float64`, M16/32, `n_starts=1`, 4 sweeps / 72 evals, 9 min wall. Policy: q* = 22.22 m³/d on 5 wells and 9.72 on 4 (sum 150), test J=912500 (M=32), reliability 1.0, mass balance 1.6e-11, gap≈0; no-pump J=323993/reliability 0, uniform half q J=1.186e6/reliability 1.0. The 4-sweep budget is too short and q* is worse than the feasible uniform q=5 from O2 (J=529250), so this is a *backend/plumbing* result, not a final policy. Both replicas were identical because only `optim.seed` differs and `n_starts=1` consumes no RNG; future multi-replica runs must offset `seed.train` (or use n_starts>1). Artifacts: `out/opt/hydrology_gpu_f64_s{0,1}/out/opt/full_opt/`. |
| today | hydrology 160×96 float64 8-sweep opt (seed 17000/710000) | **finished, but budget-limited** | Run `20261117-163253`, 1×T4, `--dtype float64`, M40/80, `n_starts=1`, `max_iter=8` / 146 evals, 36 min. q*=[22.22, 9.72×8] (sum 100); test J=730000, reliability 1.0, mass 1.9e-11, gap≈0. Baseline no-pump J=328745/reliability 0.0375; uniform half q J=1.186e6/reliability 1.0. The pattern search accepts one well move per sweep, so 8 sweeps only bring eight wells down from 22.2 to 9.7; it needs ~9 sweeps just to reach uniform 9.7 and more to reach the feasible q=5 uniform policy (J=529250 on the O2 seed). This is an evaluation-budget/trajectory limitation, not evidence that q*=730k is optimal. Artifacts: `out/opt/hydrology_gpu_f64_b/`. |
| today | hydrology 160×96 float64 8-sweep opt (seed 7000/700000) | **finished, confirms budget limit** | Run `20261117-165143`, 1×T4, float64, M40/80, 8 sweeps / 146 evals, 39 min. q*=[9.72×4, 22.22, 9.72×4] (sum 100), test J=730000, reliability 1.0, mass 4.0e-11. Same J as the 17000-seed run, with the one un-reduced well at a different location: both runs spend all eight accepted moves reducing eight of nine wells by one step, so the ninth sweep (not run) is the one that would reach uniform 9.72. The two independent seeds therefore agree on the *budget* diagnosis, not on a converged policy. Artifacts: `out/opt/hydrology_gpu_f64_a/`. |
| today | hydrology O2 on the 17000/710000 seed | **uniform q=5 dominates the 8-sweep policy** | Re-ran only O2 on the exact scenario seed used by `20261117-163253` (NumPy float64, 2m08s): uniform q=5 m³/d gives test J=**529250**, reliability **1.0**, mass 7.6e-15; no-pump gives J=328745 / reliability 0.0375. The 8-sweep float64 optimizer reached J=730000, i.e. **38% worse than the simple feasible uniform policy**, because the compass search spent every accepted move lowering one well at a time from 22.2 to 9.7 and stopped before the ninth well. This is the concrete case for a lower/feasible initial point (or an all-well move), not a claim that the optimizer cannot improve. Artifact: `out/opt/hydrology_gpu_o2_b/`. |
| today | hydrology float64 opt warm-started at uniform q=5 | **best policy so far: J=456240** | Run `20261117-173721` (seed 17000/710000, M40/80, 1×T4, float64, 4 sweeps / 67 evals, 19 min) started from `optim.init_fraction=0.1` (q0=5 m³/d per well) instead of 0.5*q_max. q*=[5,5,5,5,0,0,0,5,0], sum 25; test J=**456240.94**, reliability **1.0**, mass 6.3e-11, gap −1.6e-5. That is **14% better than uniform q=5** (529250) and **38% better than the default-init 8-sweep policy** (730000) on the same scenario seed. The log shows the mechanism: every sweep removes one well while step stays 12.5, and J is still improving by ~18k/sweep at sweep 4, so this is a lower bound on what the same budget can reach. New `optim.init_fraction` plumbing in `stoch/optimize.py`/`run_optimization.py`. Artifact: `out/opt/hydrology_gpu_f64_uniform/`. |
| today | hydrology float64 8-sweep uniform-init pair | **cost/reliability frontier; one seed fails the gate** | Runs `20261117-175813` (seed 7000/700000) and `20261117-175818` (seed 17000/710000), both M40/80, float64, q0=5, 8 sweeps / ~119 evals, ~30 min each. Both settle on **3 active wells at 5 m³/d (sum 15)**: seed a q*=[5,0,5,5,0,0,0,0,0], test J=419586, reliability **0.90** (gate `>=0.94` **fails**); seed b q*=[5,5,5,0,0,0,0,0,0], test J=419550, reliability **0.9625** (passes), mass 2.4e-11. Compare 4 sweeps on seed b: 5 wells, J=456241, reliability 1.0. The 3-well policy is ~8% cheaper than the 5-well one but sits on the reliability boundary and is seed-sensitive: M40 is too small to certify the 5% tail, so the SAA optimum can overfit it (train reliability 0.95/0.975 vs test 0.90/0.9625). Do not report the 3-well policy as robust without a larger training set or an explicit reliability margin. Artifacts: `out/opt/hydrology_gpu_f64_u8_{a,b}/`. |
| today | re-score 3-well vs 4-well policies on NumPy | **4 wells is the robust frontier; 3-well gate is a torch artifact** | Re-evaluated the uniform 5 m³/d well-count policies on the **NumPy reference** at M=80, layout seed 0. Test J / reliability: uniform 9 wells 529250/1.0; 5 wells 456241–456306 / 0.975–1.0; **4 wells 437938–437964 / 0.975–1.0**; 3 wells 419279 / **0.90** on seed 17000 and the same 3-well policy is the one the torch run reported as 0.9625. The torch float64 test metrics for the 3-well policy disagree with NumPy (J 419550 vs 419279; reliability 0.9625 vs 0.90), so the `final_test_reliability` gate that passed on torch is **not trustworthy** for this high-pumping case; M4/M8 parity is exact, but some tail scenarios diverge. The 4-well policy is ~4% more expensive than 3 wells and is the cheapest one that stays above the 0.94 gate on both seeds evaluated. **Any 3-well policy must be re-scored on NumPy before reporting.** |
| today | hydrology float64 6-sweep uniform-init (seed 7000/700000) | **optimizer oversteps the robust 4-well point** | Run `20261117-191435`, 1×T4, M40/80, float64, q0=5, 6 sweeps / 95 evals, 25 min. The 5-sweep iterate reported best=437957, i.e. the 4-well point (NumPy J=437964 on this seed); the 6th sweep moved to **3 wells** `[5,0,5,5,0,0,0,0,0]`, test J=419586, reliability **0.90** → `final_test_reliability` gate **fails**. Train reliability is 0.95, so this is SAA tail overfitting, not a solver failure. A 5-sweep run (`max_iter=5`) is the right budget for a *robust* policy; the 6-sweep run is only useful as evidence that the 3-well point is brittle. Artifact: `out/opt/hydrology_gpu_f64_u6_a/`. |
| today | hydrology float64 5-sweep uniform-init (seed 7000/700000) | **robust policy: 4 wells, J=437964, reliability 0.975** | Run `20261117-194522`, 1×T4, M40/80, float64, q0=5, 5 sweeps / 82 evals, 22 min. q*=`[5,5,5,5,0,0,0,0,0]` (4 wells, sum 20); train J=437957 / reliability 1.0; **test J=437964.45 / reliability 0.975**; mass 1.0e-10; all gates pass. Torch and NumPy agree exactly on this policy (NumPy J=437964.45, rel 0.975), unlike the 3-well case. This is the best *robust* policy: it is 4% cheaper than 5 wells (456241) and 16% cheaper than uniform q=5 (529250), while the cheaper 3-well policy (419586) drops to 0.90 test reliability. **Use `max_iter=5` with `init_fraction=0.1` as the hydrology default.** Artifact: `out/opt/hydrology_gpu_f64_u5_a/`. |
| today | live Colab telemetry: `logs`/`/runs` read the VM log | **a live run is no longer "quiet"** | Run `20261121-111857` (1×T4, 160×96×100, M100/40/40) sat at iter 6 while both the driver's `r<k>/run.log` and the dashboard said nothing: `colab exec` only streams a cell's output home when the cell *returns*, and the run cell holds the kernel for the whole run, so the local log is 157 bytes of CLI header for hours. Worse, `logs -f` on such a run sent `!tail` through `colab exec`, which queues behind the run cell — the fix had to avoid the kernel entirely. New shared helper `colab_status.remote_log_text()` pulls a remote file with `colab download` (filesystem API, read-only, ~3 s, writes only to a temp dir); `colab_status.py --live-logs`, `/runs?live_logs=1`, `/runs/log/<kind>/<id>?remote=1` and `multi_colab.py logs --source auto|remote` all use it, so the same parser that already read progress out of a finished log now reports `iter 9/40, 163/500 evals, 32.6%` for a run in flight. Pulls are bounded (30 s, one retry) and only for replicas in `running/quiet/starting/provisioning`; a failed pull degrades to the local view with the reason inline. 8 new tests in `tests/test_colab_status.py`. |
| today | a dying run keeps its artifacts, and one NaN candidate stops being fatal | **both fixed, 14 new tests** | Run `20261121-111857` (2 starts x 250 evals, mixed float32, M=100) completed start 1 (250 evals, iter 15, best 471458) and then died at start 2 iter 7 on `transport Krylov ... worst residual nan` -> `RuntimeError: objective is not finite for candidate(s) [5]`. Two driver defects made that worse than it needed to be. (1) `--zip-results` ran only after a *successful* dispatch, so nothing was uploaded and start 1's `out/` died with the VM (`colab download` then reported "File or directory not found"); it now zips from a `finally` (`zip_results_safely`, which never masks the real error) and `run_models()` is a testable seam. (2) `_finite_or_raise` treats one diverging candidate out of the 18 in a compass sweep as fatal; the new opt-in `--allow-nonfinite` (opt-in, `run_optimization.py` and `collab_run.py`) maps it to `+inf`, which can never pass `fc < best - 1e-12`, so the sweep rejects that move and continues, while an *entire* non-finite sweep still raises because that is the batched solve diverging, not one unlucky candidate. Non-finite candidates are counted (`prob.n_nonfinite`, `metrics['stages']['opt']['nonfinite_candidates']`, warned once) and `multi_start` still raises if every start ends non-finite, so a broken backend cannot be silently tolerated. Tests: `tests/test_optimize.py` (policy, sequential search, real batched sweep with one NaN candidate, whole-sweep fatal, all-starts-fatal) and `tests/test_collab_run.py` (zip on crash, zip failure not masking, no zip without the flag). GPU.md section 6 records that mixed float32 at M=100 is "usually fine, occasionally fatal", not the safe mode its first 400-evaluation run suggested. |
| today | independent re-derivation of the certified 160×96 policy (`check_tail.py`) | **certificate holds, tail is worse than reported** | Ran the verifier properly for the first time, which found two defects in it: `sys.path` never included the repo root (the documented command died with `ModuleNotFoundError: stoch`) and the default scenario set was `cfg.seed.train + i`, i.e. the optimizer's own training fields under a docstring that called them "fresh". `--seed-base` now separates the two questions. Reproducing the train set on NumPy: `J 428592.0` against the run's `428592.1`, reliability `99/100` — the float32/torch pipeline and the NumPy reference agree field by field. Drawing 400 scenarios from seed 900000 (nothing the run saw): `J 428509` (0.02% away), reliability `391/400 = 0.978`, Wilson `[0.958, 0.988]` → **PASS** the 0.94 gate, but `CVaR5%(G) = +0.616` instead of the run's `+0.005`: the tail a run reports is a property of the sample it optimized on. The same draw at M=100 gives `4/100 = 0.960`, Wilson `[0.902, 0.984]`, *inconclusive* — the search's own sample size cannot certify the policy it produced. |
| today | accept the 160×96 policy and stop searching at this geometry | **accepted: float32 q\* (2 wells), J 428,579** | Decision recorded in `Results.md` §2.1. Reasoning: the float64 and float32 runs land 1.2% apart on *different* well subsets (the objective is flat along "which wells carry the duty"), a 400-scenario fresh draw reproduces J to 0.02% and clears the 0.94 gate, and the compass step — not the evaluation budget — was deciding the outcome by iter 15. Further search at seed 0 was shown to be pure re-derivation by the crashed two-start run: `20261121-111857` start 1 reproduced `20261120-111849`'s first 15 iterations bit-for-bit. If the flat direction must be resolved, the next experiment is several replicas with **offset scenario seeds** compared on out-of-sample J, not more evaluations of the same seed. |
| today | do offset-seed replicas separate near-equal 160×96 policies? | **no — measured, hypothesis refuted** | Launched two T4 replicas with disjoint scenario draws (`multi_colab.py --set seed.train=27000/37000`), then measured the question directly with `check_tail.py --m-scen 20` instead of waiting 1.7 h: the accepted policy scores `J 428664.8` on seeds 7000-7019 and `428647.9` on 37000-37019 (0.004% apart; 0.02% at M=400). Both replicas were stopped after confirming their first three compass sweeps matched the accepted run digit for digit — with a seed-insensitive objective and a deterministic start, a differently-seeded replica is the *same* run. The flat direction ("which wells carry the duty") is therefore a property of the cost model, not of scenario sampling, and the 1.2% float32/float64 policy gap comes from numerical noise in fine compass steps. `multi_colab.py` gained `--seed-step` (per-replica `seed.train`/`seed.test` offset, since `--seed-base` only moves `optim.seed`, i.e. well placement) so this comparison is one flag when a geometry does have seed sensitivity. |
| today | 3D paper preset re-run on the assembled tree (laptop CPU) | **reproduces the published result exactly** | After the merge of the 3D and 2D-opt branches, `python full3D_colab_run.py --preset paper --budget 15 --n-starts 2` finished in 1014.8 s / 49 evaluations with `q* = [12.5, 12.5, 12.5, 12.5, 0×5]`, test `J = 547,665.31`, reliability `0.96875`, mass-balance error `9.7e-15` and all six gates true — the same numbers as `DOCS/3D_results.md` on a third machine with a different BLAS. The `--budget 15` clamp reduced the search to 24 evaluations per start, so this is a reproduction of the published answer, not a new search. |
| today | install a LaTeX toolchain and build the assembled thesis | **42 pages, 0 undefined refs/citations** | TinyTeX (TeX Live 2026) in `~/.TinyTeX` — no root on this host, so `apt` was not an option and the installer runs as the user; `build_tex.py --compile` finds it without `PATH` help and now reads the final LaTeX log instead of trusting `latexmk`'s exit code (latexmk exits 0 with undefined references). All five chapter PDFs are rebuilt from the current sources. Four real defects surfaced only under a compiler, none of which the static checks could see: (1) `natbib`/`geometry` were emitted once per chapter, so `\usepackage[round,numbers]{natbib}` clashed with `[round,numbers,sort&compress]` — package options are now unioned, with a warning when two chapters set the same key differently; (2) the preamble merger carried only `\newcommand`, so `\newtheorem{remark}` and `\newcolumntype{Y}` were dropped and the first `remark` environment aborted the build — declarations are now keyed by declared name, whatever the command; (3) a two-line `\title{...\\` leaked its continuation (`\large Mathematical formulation...`) into the preamble as typeset text, giving a spurious "Missing \begin{document}"; (4) five chapters each ship a `thebibliography`, so `\bibitem{ogata1961solution}` appeared three times and every citation of it was "multiply defined" — the blocks are now merged and de-duped by key, and a citation no chapter defines is reported. `tests/test_build_tex.py` pins each of those four. |
| today | quantify the "per-step kernel launches" claim before optimizing it | **4022 matvecs per evaluation, 1.7× the assumed 2400** | Instrumented `bicgstab_solve` and ran one 80×40 / `nsteps=100` / `M=8` evaluation: 100 solves, 4022 batched matvecs, 40.2 per solve — so the launch overhead is larger than `DOCS/Conclusions.md` §9 assumed, not smaller. Recorded there with what the 5× actually needs (fuse the ~10 kernels per iteration, or replace BiCGSTAB with a banded/scan solve — both change the arithmetic, so `--verify` parity is part of the work) and with the exact T4 procedure. The 5× stays unverified: it cannot be measured on this box, and landing an unmeasured speedup is how the float32 mode got its "usually fine, occasionally fatal" reputation. |
| today | give the FNO sweep five seeds instead of one | **the 250-epoch headline fell 1.46% → median 1.99%; the *trends* survived** | `seed_sweep.py --seeds 0-4` on two arms: 250 epochs (`runs/sweeps/fno_coords_250ep`, 64.5 min CPU) and modes 4/24 at 120 epochs (`fno_modes_120ep`). The quoted 1.46% was **seed 0, the best of five** (median 1.99e-2, IQR 1.68–2.14e-2, max 4.59e-2) so "15× better than mean-field" is really 11×. Mode truncation 4/12/24 becomes 4.06 / 3.02 / 2.47 e-2 — same ordering, every single-seed value optimistic. What replicates: doubling epochs 120→250 wins with paired median ratio 0.660 on **5/5** seeds, and 24 modes beat 4 with 0.66 on **5/5** — the ~3× seed spread is common to all arms and cancels when paired, which is why the arms were swept rather than compared across runs. Resolution-transfer rows and the 300-epoch plateau are still single-seed. |
