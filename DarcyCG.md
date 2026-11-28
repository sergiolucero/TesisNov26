# DarcyCG — Darcy flow, the seven-point stencil, and batched conjugate gradients

**What this documents.** The steady-flow solve shared by `physics/fv3d.py`
(NumPy reference) and the batched torch backend inlined in
`full3D_colab_run.py`: the Darcy equation, its cell-centred finite-volume
discretization into a symmetric seven-point system, and the preconditioned
conjugate-gradient (CG) solver the torch path runs in place of the reference's
sparse LU factorization.

## 1. Darcy flow

Saturated single-phase flow obeys **Darcy's law** [1][2], relating the specific
discharge (Darcy flux) $\mathbf{q}\,[\mathrm{m\,d^{-1}}]$ to the gradient of the
hydraulic head $h\,[\mathrm{m}]$,

$$
\mathbf{q} = -\,K\,\nabla h .
$$

$K$ is the hydraulic-conductivity tensor. Here it is diagonal and grid-aligned,

$$
K = \mathrm{diag}(K_h,\, K_h,\, K_z), \qquad K_z = k_v\, K_h ,
$$

with a vertical anisotropy ratio $k_v$ (`kv_ratio`, default $0.1$); $K_h$ is the
log-normal field produced by the Karhunen–Loève expansion.

Mass conservation for an incompressible fluid with no internal source gives the
steady-state equation

$$
\nabla\cdot(K\,\nabla h) = 0 .
$$

Boundary conditions are prescribed heads on the two $x$ faces (a regional
gradient) and no-flow on the $y$ and $z$ faces:

$$
h(0,y,z) = h_{\mathrm{left}}, \qquad h(L_x,y,z) = h_{\mathrm{right}},
\qquad
\partial_y h\big|_{y=0,L_y} = \partial_z h\big|_{z=0,L_z} = 0 ,
$$

with $h_{\mathrm{left}} = \mathrm{grad}\cdot L_x\cdot \xi$ per scenario
($\xi$ is the random gradient scale; $h_{\mathrm{right}}=0$). Wells do **not**
enter the base solve; they enter by linear superposition (§7).

## 2. Cell-centred finite-volume discretization (TPFA)

The grid is Cartesian with cell spacings $\Delta x,\Delta y,\Delta z$ and one
unknown head per cell. Fluxes are evaluated with the **two-point flux
approximation** (TPFA), which is consistent for a $K$-orthogonal grid — the case
here, since $K$ is diagonal and cell-wise constant [7][8].

For the $x$-face between cells $(i,j,k)$ and $(i+1,j,k)$ the face conductance is

$$
C^x_{i+\frac12,j,k}
= H\!\big(K_{i,j,k},\, K_{i+1,j,k}\big)\,\frac{\Delta y\,\Delta z}{\Delta x},
\qquad
H(a,b) = \frac{2ab}{a+b},
$$

where $H$ is the **harmonic mean** of the two cell conductivities (series
resistance of the two half-cells). The $y$ and $z$ faces are identical, the
$z$-faces using $K_z$:

$$
C^y_{i,j+\frac12,k}
= H\!\big(K_{i,j,k},\, K_{i,j+1,k}\big)\,\frac{\Delta x\,\Delta z}{\Delta y},
\qquad
C^z_{i,j,k+\frac12}
= H\!\big(K^z_{i,j,k},\, K^z_{i,j,k+1}\big)\,\frac{\Delta x\,\Delta y}{\Delta z}.
$$

The Dirichlet $x$ boundaries carry half-cell conductances (factor $2$, since the
boundary node sits at distance $\Delta x/2$),

$$
C^L_{j,k} = 2\,K_{0,j,k}\,\frac{\Delta y\,\Delta z}{\Delta x},
\qquad
C^R_{j,k} = 2\,K_{nx-1,j,k}\,\frac{\Delta y\,\Delta z}{\Delta x}.
$$

Conservation in cell $(i,j,k)$ states that the net flux out of the cell
vanishes,

$$
\sum_{f \in \partial\Omega_{ijk}} C_f\,(h_{ijk} - h_{\mathrm{neighbor}}) = 0 .
$$

## 3. The symmetric seven-point stencil

Expanding the conservation sum gives the seven-point operator applied to
$h = (h_{ijk})$, written matrix-free as (the `Stencil7.apply` form)

$$
\begin{aligned}
(Ah)_{i,j,k} =&\ D_{i,j,k}\,h_{i,j,k}
  - C^x_{i-\frac12}\,h_{i-1,j,k} - C^x_{i+\frac12}\,h_{i+1,j,k} \\
  &- C^y_{j-\frac12}\,h_{i,j-1,k} - C^y_{j+\frac12}\,h_{i,j+1,k} \\
  &- C^z_{k-\frac12}\,h_{i,j,k-1} - C^z_{k+\frac12}\,h_{i,j,k+1},
\end{aligned}
$$

with diagonal

$$
D_{i,j,k}
= C^x_{i-\frac12} + C^x_{i+\frac12}
+ C^y_{j-\frac12} + C^y_{j+\frac12}
+ C^z_{k-\frac12} + C^z_{k+\frac12}
+ C^L_{j,k} + C^R_{j,k}
$$

and right-hand side

$$
b_{i,j,k}
= C^L_{j,k}\,h_{\mathrm{left}}\,\mathbf{1}[i=0]
+ C^R_{j,k}\,h_{\mathrm{right}}\,\mathbf{1}[i=n_x-1].
$$

`Stencil7` stores exactly these seven coefficient arrays — the diagonal and the
six interior face couplings — so the matvec $Ah$ costs one pointwise multiply
plus six shifted multiplies and never materializes the matrix.

## 4. The system $Ah = b$ is symmetric positive definite

The assembled matrix is

$$
A_{ii} = D_i, \qquad A_{ij} = -\,C_f \ \ (i,j\ \text{share face } f),
\qquad A_{ij} = A_{ji}.
$$

Every row satisfies $D_i = \sum_{f}|A_{ij}| + C^L_{j,k} + C^R_{j,k}$, i.e. weak
diagonal dominance with strict dominance wherever a Dirichlet face touches the
cell. $A$ is therefore a symmetric **M**-matrix and, with at least one Dirichlet
boundary, symmetric positive definite (SPD). CG is the natural solver [5][6];
the NumPy reference instead factors $A$ once with `scipy.sparse.linalg.splu`.

## 5. Preconditioned conjugate gradients

CG solves $Ax=b$ for SPD $A$ by minimizing the $A$-norm of the error over a
growing Krylov space, without forming $A$ explicitly [3][4][5][6]. The
implementation (`cg3d`) uses **Jacobi (diagonal) preconditioning**,
$M = \mathrm{diag}(A)$ [9]:

$$
\begin{aligned}
&x_0 = 0,\quad r_0 = b - A x_0,\quad z_0 = M^{-1} r_0,\quad p_0 = z_0,\quad
\rho_0 = \langle r_0, z_0\rangle; \\
&\text{for } k = 0,1,2,\dots\ \text{until } \tfrac{\|r_k\|}{\|b\|} \le \mathrm{rtol}: \\
&\qquad \alpha_k = \frac{\rho_k}{\langle p_k, A p_k\rangle}, \\
&\qquad x_{k+1} = x_k + \alpha_k p_k, \\
&\qquad r_{k+1} = r_k - \alpha_k A p_k, \\
&\qquad z_{k+1} = M^{-1} r_{k+1}, \qquad \rho_{k+1} = \langle r_{k+1}, z_{k+1}\rangle, \\
&\qquad \beta_k = \frac{\rho_{k+1}}{\rho_k}, \qquad p_{k+1} = z_{k+1} + \beta_k p_k .
\end{aligned}
$$

The Jacobi step $z = M^{-1}r$ is the elementwise divide $z = r / \mathrm{diag}$,
and every inner product is a per-system contraction over the spatial indices.

## 6. Batched CG over scenarios

All state is a batch tensor of shape $(B, n_z, n_y, n_x)$: $B$ independent
systems advance through **the same** iteration loop. Inner products reduce over
the three spatial axes only, so $\alpha,\beta,\rho$ are per-system vectors of
length $B$; each system therefore follows its own recurrence, while a single
kernel launch performs $B$ matvecs at once — this is what amortizes GPU launch
latency (the principal win of the torch path). The loop exits only when
**every** system satisfies the tolerance:

$$
\max_{b=1..B}\ \frac{\|r^{(b)}\|}{\|b^{(b)}\|} \le \mathrm{rtol}.
$$

For the flow solve the batch is $B = M\,(n_w+1)$: $M$ conductivity scenarios,
each solved for the background right-hand side and for $n_w$ unit-rate well
responses (§7).

## 7. Well responses by linear superposition

Because flow is linear in the source term, the head for a pumping schedule
$\mathbf{q}\in\mathbb{R}^{n_w}$ is

$$
h(\mathbf{q}) = h_0 + \sum_{j=1}^{n_w} q_j\, h^w_j,
\qquad
A h_0 = b,\qquad A h^w_j = -\,w_j ,
$$

where $w_j$ is the (unit-integral) Gaussian footprint of well $j$. The torch
solver stacks the $n_w+1$ right-hand sides and recovers $h_0$ and all $h^w_j$
from one batched CG. Once heads are known, the face-centred seepage velocity
(Darcy flux divided by porosity $\phi$) is

$$
v_x = -\,\frac{K}{\phi}\,\partial_x h,
\qquad
v_y = -\,\frac{K}{\phi}\,\partial_y h,
\qquad
v_z = -\,\frac{K_z}{\phi}\,\partial_z h,
$$

discretized on faces with the harmonic face conductivity; the $x$ Dirichlet
faces use the half-cell distance $\Delta x/2$, and $y,z$ boundary faces are
zero by the no-flow condition.

## 8. Convergence, tolerances, and the stall guard

- **Tolerance.** `rtol = 1e-11` in float64 and `rtol = 1e-6` in float32
  (`torch_default_rtol`); the solve exits when the worst normalized residual
  over the batch reaches `rtol`.
- **Report flag.** `converged = residual <= max(rtol, 10·rtol)`, i.e. the
  declared-flag is deliberately looser than the strict loop-exit test so a
  solve that stopped on the stall guard is not reported as converged.
- **Stall guard.** If the worst residual fails to improve for
  `STALL_WINDOW = 60` consecutive iterations, the solve is abandoned and marked
  `stalled` rather than burning the full `maxiter` budget.

## References

[1] H. Darcy. *Les fontaines publiques de la ville de Dijon.* Dalmont, Paris, 1856.

[2] J. Bear. *Dynamics of Fluids in Porous Media.* American Elsevier, New York, 1972.

[3] M. R. Hestenes and E. Stiefel. Methods of conjugate gradients for solving
linear systems. *Journal of Research of the National Bureau of Standards*
49(6):409–436, 1952.

[4] J. R. Shewchuk. *An Introduction to the Conjugate Gradient Method Without the
Agonizing Pain.* Carnegie Mellon University, 1994.

[5] Y. Saad. *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM,
Philadelphia, 2003.

[6] G. H. Golub and C. F. Van Loan. *Matrix Computations*, 4th ed. Johns Hopkins
University Press, Baltimore, 2013.

[7] K. Aziz and A. Settari. *Petroleum Reservoir Simulation.* Applied Science
Publishers, London, 1979.

[8] R. Eymard, T. Gallouët, and R. Herbin. Finite volume methods. In *Handbook of
Numerical Analysis*, vol. 7, pp. 713–1020. Elsevier, 2000.

[9] M. Benzi. Preconditioning techniques for large linear systems: a survey.
*Journal of Computational Physics* 182(2):418–477, 2002.
