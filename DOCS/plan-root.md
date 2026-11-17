# Debugging Plan: Groundwater Remediation Models (PINN & FNO)

## Phase 1: Problem Simplification (The 1D Sandbox)
Before tweaking hyperparameters on the full 2D/3D porous media flow, we need to ensure the foundational mathematics and PyTorch graphs are correct.
- [ ] **Strip spatial dimensions:** Reduce the governing equation (e.g., advection-dispersion equation) to 1D.
- [ ] **Simplify physics:** Temporarily disable reaction terms or heterogeneous permeability fields. Assume a homogeneous, isotropic aquifer.
- [ ] **Establish a baseline:** Run the PINN/FNO on this simplified 1D case. Does it converge to the known analytical solution? 
    - *If yes:* The issue is likely in handling heterogeneity or dimensionality. Proceed to Phase 2.
    - *If no:* The issue is in the core network architecture or loss function formulation.

## Phase 2: Physics & Boundary Condition (BC) Verification
PINNs often struggle because the physics loss and data/BC loss gradients fight each other.
- [ ] **Check BC strictness:** Are Dirichlet/Neumann boundary conditions enforced as soft constraints (in the loss function) or hard constraints (architecturally)? Try hard-coding the Dirichlet boundaries if they are currently soft.
- [ ] **Scale the PDE variables:** Groundwater flow equations often deal with vastly different scales (e.g., hydraulic conductivity vs. concentration). Non-dimensionalize your inputs (scale everything to [0, 1] or [-1, 1]).
- [ ] **Inspect the residual:** Plot the PDE residual across the spatial domain at initialization. Are there massive spikes at the boundaries or injection wells? 

## Phase 3: Loss Landscape & Optimizer Tuning
If the losses are unbalanced, the network will ignore the physics to satisfy the boundary data, or vice versa.
- [ ] **Implement dynamic loss weighting:** Use a method like Neural Tangent Kernel (NTK) weighting or Learning Rate Annealing to dynamically balance the PDE loss ($\mathcal{L}_{PDE}$) and the Boundary loss ($\mathcal{L}_{BC}$).
- [ ] **Switch optimizers sequentially:** 
    - Start with Adam for the first 10,000-20,000 epochs to get into the right basin of attraction.
    - Switch to L-BFGS (with a smaller learning rate) for fine-tuning. L-BFGS is often critical for PINN convergence but fails if started too early.
- [ ] **Check for gradient pathologies:** Log the gradient histograms in TensorBoard. Are the gradients for the PDE loss vanishing compared to the data loss?

## Phase 4: FNO-Specific Checks (If applicable)
If the Fourier Neural Operator is the bottleneck, the issue is usually spectral.
- [ ] **Check frequency truncation:** How many Fourier modes ($k_{max}$) are you keeping? Try increasing the modes to capture sharper gradients (e.g., a sharp contaminant plume front).
- [ ] **Verify resolution invariance:** Train on a coarse grid and evaluate on a fine grid. If it fails, your grid formulation or lifting/projection layers might be hardcoded to specific dimensions.
- [ ] **Data normalization:** Ensure the input function space (e.g., initial permeability map) and output space (e.g., final concentration) are normalized using standard Gaussian normalization per channel.