# Debugging Plan: Groundwater Remediation Models (PINN & FNO)

## Phase 1: Problem Simplification (The 1D Sandbox)
Before tweaking hyperparameters on the full 2D/3D porous media flow, we need to ensure the foundational mathematics and PyTorch graphs are correct.
- [x] **Strip spatial dimensions:** Reduce the governing equation (e.g., advection-dispersion equation) to 1D.
- [x] **Simplify physics:** Temporarily disable reaction terms or heterogeneous permeability fields. Assume a homogeneous, isotropic aquifer.
- [x] **Establish a baseline:** Run the PINN/FNO on this simplified 1D case. Does it converge to the known analytical solution? 
    - *If yes:* The issue is likely in handling heterogeneity or dimensionality. Proceed to Phase 2.
    - *If no:* The issue is in the core network architecture or loss function formulation.

  ✅ **Result:** PINN converges to the Ogata--Banks solution to rel L2 ≈ 8e-3
  in the interior (target ≤ 1e-2). Full-grid error 1.35e-2 is dominated by the
  measure-zero corner singularity at (ξ,τ)=(0,0), a metric artifact, not a bug.
  See `Actions.md`.

## Phase 2: Physics & Boundary Condition (BC) Verification
PINNs often struggle because the physics loss and data/BC loss gradients fight each other.
- [x] **Check BC strictness:** Are Dirichlet/Neumann boundary conditions enforced as soft constraints (in the loss function) or hard constraints (architecturally)? Try hard-coding the Dirichlet boundaries if they are currently soft.
- [x] **Scale the PDE variables:** Groundwater flow equations often deal with vastly different scales (e.g., hydraulic conductivity vs. concentration). Non-dimensionalize your inputs (scale everything to [0, 1] or [-1, 1]).
- [x] **Inspect the residual:** Plot the PDE residual across the spatial domain at initialization. Are there massive spikes at the boundaries or injection wells? 

  ✅ **Result (re-derived, seed 0, reproducible):** trained interior mean
  $|R|=2.13\times10^{-2}$, inlet band $3.98\times10^{-1}$, IC band
  $7.78\times10^{-1}$, corner $7.26\times10^{1}$.
  See `out/residual/residual_trained.png`.

  ⚠️ **Correction — the max $|R|=10^3$ is structural, not physical.** It is
  exactly $1/\varepsilon$, because the hard-BC ansatz multiplies by
  $s(\tau)=1-e^{-\tau/\varepsilon}$, so $u_\tau$ carries
  $s'(\tau)\to1/\varepsilon$ as $\tau\to0$. It is therefore the same at
  initialization and after training, identical for every seed (verified for
  seeds 0/1/2: $|R(0,0)|=1.0000\times10^{3}$ for all), and it cannot be
  trained away — it is a property of the ansatz, not a residual. It is a
  metric artefact of the same family as the corner discontinuity of
  `formulation.tex` §6, and it is why the *full-grid* residual is not a useful
  training diagnostic here; the band means above are.
  `inspect_residual.py` now prints the $1/\varepsilon$ reference next to the
  measured max and the ratio (observed $=1.000$) so the two are not confused.

  ⚠️ **Correction — "no interior spikes" needs qualifying.** There is no
  spike, but the interior residual is strongly *outlet-biased*. At fixed
  $\tau_{\max}=0.5$ the mean $|R|$ rises monotonically with $\xi$: $1.5\times10^{-2}$
  ($\xi<0.2$) to $2.5\times10^{-1}$ ($\xi>0.8$), a 16× increase; and at
  $\xi>0.8$ it rises monotonically with $\tau$, $3.0\times10^{-3}$ at
  $\tau=0.1$ to $2.6\times10^{-1}$ at $\tau=0.5$. The outlet region ($\xi>0.8$,
  $\tau>0.05$) has mean $|R|=3.4\times10^{-2}$, 2.2× the plume half
  ($\xi<0.5$, $1.5\times10^{-2}$). The trend is smooth, not a spike, and
  coincides with $u\to0$ at the outlet: the hard constraint pins $u(1,\tau)=0$
  while the reference decays smoothly, so the residual per unit $u$ is largest
  where $u$ is smallest. Worth watching in the 2D extension, where the outlet
  boundary is a larger share of the domain.

## Phase 3: Loss Landscape & Optimizer Tuning
If the losses are unbalanced, the network will ignore the physics to satisfy the boundary data, or vice versa.
- [ ] **Implement dynamic loss weighting:** Use a method like Neural Tangent Kernel (NTK) weighting or Learning Rate Annealing to dynamically balance the PDE loss ($\mathcal{L}_{PDE}$) and the Boundary loss ($\mathcal{L}_{BC}$).
  - Progress: added optional causal/τ-ramped PDE weighting (`--tau-ramp`, `--ramp-tau`, `--ramp-power`) for hard-BC corner-band experiments; not yet benchmarked.
- [ ] **Switch optimizers sequentially:** 
    - Start with Adam for the first 10,000-20,000 epochs to get into the right basin of attraction.
    - Switch to L-BFGS (with a smaller learning rate) for fine-tuning. L-BFGS is often critical for PINN convergence but fails if started too early.
- [ ] **Check for gradient pathologies:** Log the gradient histograms in TensorBoard. Are the gradients for the PDE loss vanishing compared to the data loss?

## Phase 4: FNO-Specific Checks (If applicable)
If the Fourier Neural Operator is the bottleneck, the issue is usually spectral.
- [ ] **Check frequency truncation:** How many Fourier modes ($k_{max}$) are you keeping? Try increasing the modes to capture sharper gradients (e.g., a sharp contaminant plume front).
- [ ] **Verify resolution invariance:** Train on a coarse grid and evaluate on a fine grid. If it fails, your grid formulation or lifting/projection layers might be hardcoded to specific dimensions.
- [ ] **Data normalization:** Ensure the input function space (e.g., initial permeability map) and output space (e.g., final concentration) are normalized using standard Gaussian normalization per channel.