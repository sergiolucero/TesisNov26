# Contributions to the broader scientific community

This project is a study of **stochastic optimization in a PDE-constrained
setting**: designing a pump-and-treat groundwater remediation policy under
hydraulic-conductivity uncertainty (a log-normal field represented by a
Karhunen–Loève expansion [8]), subject to a CVaR chance constraint [3] on the
compliance-point concentration. The immediate object of study is an aquifer, but
the phenomena uncovered are not specific to aquifers. This note distils the
effort into transferable lessons for the community that works on optimization
under uncertainty with expensive, physics-based forward models.

The structure is: what we learned about the *optimization* (Section 1), what we
learned about the *computation* (Section 2), and where the open questions point
(Section 3).

---

## 1. Methodological contributions: the reliability of a stochastic optimum

### 1.1 A small SAA sample prices the tail with almost no draws — and the optimizer exploits it

A CVaR constraint at level $\alpha$ [3] is estimated from $\alpha M$ scenarios
in a sample-average approximation (SAA) [4,5]. In
this work, a `size_mid` run used $M_{\text{train}}=24$, giving **1.2 draws** in
the 5% tail; even the headline `paper` preset at $M=64$ gives only **3.2
draws**. The optimizer therefore prices the violation it is buying from
approximately one scenario, and every scenario slightly worse than that one is
free. The result is a policy that is optimal *for its sample* and unsafe *for
the aquifer*: reliability $0.875$ out of sample against a $0.95$ target.

The general lesson is that **CVaR/SAA constraints are not binding in the way the
formulation intends once $\alpha M$ is small**, regardless of how smooth or
well-posed the forward model is. This is a property of the sample, not the PDE.
Any PDE-constrained SAA study should report $\alpha M$ explicitly and treat a
result certified at $\alpha M < 5$ as provisional.

### 1.2 Multimodality is real, and which basin you find depends on the search

At the *same* grid, with *identical* random-field realizations and wells, a
serial pattern search and a batched pattern search landed in **two different,
genuine local minima**:

| solution | train objective | train reliability | test objective | test reliability |
|---|---:|---:|---:|---:|
| four wells, 50 m³/d | 547 606 | 1.000 | 557 559 | 0.969 |
| one well, 25 m³/d | 455 747 | 0.984 | 639 489 | 0.914 |

The cheaper-in-sample solution is the one that overfits. The batched search
minimizes the training objective *more* aggressively and finds the fragile
basin; the serial search stops in the robust one. This is a general warning:
**for nonconvex stochastic objectives, "better" optimization (lower in-sample
cost) can mean *worse* generalization**, and a single multi-start run is not a
certificate. Out-of-sample re-scoring is not optional.

### 1.3 Separate numerical error from modelling error with an independent reference

When the two optima above diverged, the first suspicion was float32 arithmetic
in the GPU backend. The decisive test re-scored both policies on an independent
float64 reference (a sparse-LU solver with no shared code). The reference agreed
with the float32 GPU backend to $\sim 10^{-6}$ in objective and reproduced the
documented baseline to the cent — so the divergence was *not* a precision bug.

This is a reusable protocol for the community: **before attributing a surprise
in a mixed-precision or accelerated pipeline to numerics, re-score the solution
on an independent high-precision reference.** A single re-scoring table
separates "the solver lied" from "the problem is genuinely multimodal / the
sample is genuinely too small". We found the latter.

### 1.4 Reliability is a binomial statistic, not a real number

Reliability (fraction of compliant scenarios) moves by $1/M$ per scenario. A
difference of one scenario out of 128 is a *sign flip*, not a measurement. A
point estimate of $0.969$ on 128 scenarios has a Wilson interval [7]
$[0.922, 0.988]$, which does **not** clear a $0.94$ gate. The lesson: **chance
constraints certified from small out-of-sample sets must be reported with
intervals**, and a gate that only clears as a point estimate should be labelled
"encouraging", not "certified".

---

## 2. Computational contributions: where the wall actually is

### 2.1 Batch the *search*, not just the solve

The naive hope — "put the PDE solve on a GPU" — is often wrong for
single-realization forward models. The 3D sparse-LU solve here is
fill-in-bound ($t \propto n^{1.9}$) and a GPU buys nothing on it. The real
speedup came from **batching the derivative-free search**: $2n_{\text{wells}}$
compass candidates $\times M$ scenarios into one matrix-free Krylov solve. The
bottleneck is then *batch size / latency amortization*, not FLOPs. This is a
portable pattern for any derivative-free or multi-scenario PDE optimization:
**treat the search as a batched linear algebra problem, and size the batch to
amortize fixed solver overhead**, rather than refining the single solve.

### 2.2 A cost taxonomy: solver-bound vs search-bound

The same method passes through two regimes. At small-to-mid grids the forward
solve is the cost. At fine grids the *search* becomes the binding constraint:
a "certified" mid-size 3D policy needs $M_{\text{train}}\approx 100$ scenarios
and $\sim$300 evaluations per start, i.e. **~18 hours on one CPU** — the first
point at which the optimizer, not the PDE, is the wall. Naming which regime one
is in (solve-bound vs search-bound) before choosing hardware or an algorithm is
a transferable piece of engineering judgment: it decides whether to invest in a
faster solver, more scenarios, or a cheaper evaluation.

### 2.3 Mixed precision is safe for the objective, delicate for the constraint

float32 Krylov solves settle at a residual floor of $\sim 10^{-5}$ (an order
looser than float64), yet the *objective* still agrees with the float64
reference to $\sim 10^{-4}$ relative — well inside any engineering tolerance.
The place where float32 *matters* is the **reliability gate**, which flips by a
single scenario. Guidance for the community: **mixed precision is fine for
optimization; certify feasibility/constraint quantities in high precision** (or
on a reference solver), because those are boundary statistics.

---

## 3. Where this points

- **Adaptive / two-stage decisions.** A here-and-now design cannot adapt; a
  two-stage recourse formulation (re-tune rates after observing the plume)
  quantifies the *value of the stochastic solution* (VSS) and the *value of
  perfect information* (EVPI) [6]. This project implemented the two-stage model
  and its progressive-hedging solver [1,2] as a verified parity harness; the benchmark-scale
  VSS is the natural next experiment, and it is the quantity operators actually
  care about.

- **Larger samples and variance reduction.** Every overfitting failure above
  traces to $\alpha M$ being small. The fix is more scenarios or
  variance-reduced tail estimators [5]; either is a change to the *estimation*, not
  the optimizer. Our own experiment confirms the direction and its limit: quadrupling
  $M_{\text{train}}$ from 64 to 256 (αM from 3.2 to 12.8 tail draws) lifts the
  fragile one-well policy from test reliability 0.914 to 0.945 — past the gate — but
  does *not* switch the optimizer to the robust 4-well basin (0.969); more data makes
  a cheap design honest, not robust.

- **Verification gates as a public standard.** The project runs every result
  through explicit gates (mass balance to $10^{-9}$, reproducibility, 2D/3D
  agreement, an independent re-score). Making such gates a published, reusable
  standard — not an appendix — would raise the reproducibility bar for
  PDE-constrained stochastic optimization as a whole.

---

**Summary.** The effort's contribution to the community is less "a faster
solver" and more a set of *disciplines* for a field where the expensive part is
trust, not compute: report $\alpha M$, re-score the optimum on an independent
reference before blaming numerics, quote intervals for reliability, separate
solve-bound from search-bound regimes, and batch the search rather than the
solve. Those disciplines are what turned a confusing "the GPU found a different
answer" into a clean, generalizable result about SAA overfitting.

---

## References

1. R. T. Rockafellar and R. J.-B. Wets. *Scenarios and policy aggregation in
   optimization under uncertainty.* Mathematics of Operations Research,
   16(1):119–147, 1991. — Progressive hedging algorithm.
2. J.-P. Watson and D. L. Woodruff. *Progressive hedging innovations for a class
   of stochastic mixed-integer resource allocation problems.* Computational
   Management Science, 8(4):355–370, 2011.
3. R. T. Rockafellar and S. Uryasev. *Optimization of conditional value-at-risk.*
   Journal of Risk, 2(3):21–41, 2000. — CVaR (Rockafellar–Uryasev).
4. A. Shapiro, D. Dentcheva, and A. Ruszczyński. *Lectures on Stochastic
   Programming: Modeling and Theory.* SIAM, Philadelphia, 2009.
5. A. J. Kleywegt, A. Shapiro, and T. Homem-de-Mello. *The sample average
   approximation method for stochastic discrete optimization.* SIAM Journal on
   Optimization, 12(2):479–502, 2002.
6. J. R. Birge and F. Louveaux. *Introduction to Stochastic Programming.*
   Springer, New York, 2nd edition, 2011.
7. E. B. Wilson. *Probable inference, the law of succession, and statistical
   inference.* Journal of the American Statistical Association, 22(158):209–212,
   1927.
8. R. G. Ghanem and P. D. Spanos. *Stochastic Finite Elements: A Spectral
   Approach.* Springer-Verlag, New York, 1991.
