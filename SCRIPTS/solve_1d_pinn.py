"""
solve_1d_pinn.py
================

Phase 1 baseline: solve the dimensionless 1D advection--dispersion equation with
a physics-informed neural network (PINN) and verify against the Ogata--Banks
analytical solution.

    du/dtau + du/dxi = (1/Pe) d2u/dxi2,   (xi,tau) in (0,1)x(0,tau_max]
    u(xi,0) = 0,  u(0,tau) = 1,  u(1,tau) = 0

The network is an MLP (tanh hidden layers, sigmoid output so u in (0,1)).
Derivatives are computed with torch.autograd.  Adam is run first with collocation
points re-sampled every step, then a short L-BFGS polish.

Run:
    python solve_1d_pinn.py --pe 20 --tau-max 0.5 --adam-steps 12000

Author: TESIS (groundwater remediation PINN/FNO debugging)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hydrology import u_reference


def pick_device(prefer=None):
    """Resolve --device. 'auto' -> cuda > mps > cpu; None -> cpu."""
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


torch.set_default_dtype(torch.float32)


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------
class MLP(nn.Module):
    """Fully-connected network mapping (xi, tau) -> u in (0,1)."""

    def __init__(self, hidden: int = 64, layers: int = 4, act=nn.Tanh):
        super().__init__()
        dims = [2] + [hidden] * layers + [1]
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                blocks.append(act())
        self.net = nn.Sequential(*blocks)

    def forward(self, xi: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        x = torch.cat([xi, tau], dim=-1)
        return torch.sigmoid(self.net(x))


class HardBCMLP(nn.Module):
    """Hard-constrained ansatz (Phase 2):

        u(xi, tau) = (1 - xi) * s(tau) + xi * (1 - xi) * s(tau) * NN(xi, tau),
        s(tau) = 1 - exp(-tau / eps).

    This satisfies the IC  u(xi, 0) = 0  and the outlet  u(1, tau) = 0
    exactly, and the inlet  u(0, tau) = s(tau) -> 1  exponentially fast in
    tau (s(5*eps) ~ 0.993), matching the step inlet to within the smoothing
    width eps. The corner (xi,tau)=(0,0) is a genuine discontinuity of the
    step-inlet problem (IC says 0, inlet BC says 1), so an exact simultaneous
    match is impossible with a continuous ansatz.
    """

    def __init__(self, hidden: int = 64, layers: int = 4, act=nn.Tanh,
                 eps: float = 1e-3):
        super().__init__()
        dims = [2] + [hidden] * layers + [1]
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                blocks.append(act())
        self.net = nn.Sequential(*blocks)
        self.eps = eps

    def forward(self, xi: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        s = 1.0 - torch.exp(-tau / self.eps)
        inner = self.net(torch.cat([xi, tau], dim=-1))
        return (1.0 - xi) * s + xi * (1.0 - xi) * s * inner


# --------------------------------------------------------------------------
# Physics residual
# --------------------------------------------------------------------------
def pde_residual(u, xi, tau, Pe):
    """R = u_tau + u_xi - (1/Pe) u_xixi (exact via autodiff)."""
    u_xi, u_tau = torch.autograd.grad(
        u, (xi, tau),
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )
    u_xixi = torch.autograd.grad(
        u_xi, xi,
        grad_outputs=torch.ones_like(u_xi),
        create_graph=True,
    )[0]
    return u_tau + u_xi - (1.0 / Pe) * u_xixi


def losses(model, xi_f, tau_f, xi_ic, tau_ic, tau_bc, Pe, hard_bc=False,
           pde_weight=None):
    """Return (L_pde, L_ic, L_bc) for one collocation batch.

    With hard_bc the ansatz enforces IC/BC exactly, so only the PDE loss is
    nonzero and the IC/BC forward passes are skipped.

    ``pde_weight`` (optional tensor, broadcastable to the residual) applies a
    causal / time-ramped weighting to the squared residual, e.g. to down-weight
    the corner band near tau=0.
    """
    # Interior (PDE residual)
    u_f = model(xi_f, tau_f)
    r = pde_residual(u_f, xi_f, tau_f, Pe)
    r2 = r ** 2
    if pde_weight is not None:
        r2 = r2 * pde_weight
    L_pde = r2.mean()

    if hard_bc:
        zero = torch.zeros((), device=u_f.device)
        return L_pde, zero, zero

    # Initial condition u(xi, 0) = 0
    u_ic = model(xi_ic, tau_ic)
    L_ic = (u_ic ** 2).mean()

    # Boundary conditions u(0, tau) = 1, u(1, tau) = 0
    u_left = model(torch.zeros_like(tau_bc), tau_bc)
    u_right = model(torch.ones_like(tau_bc), tau_bc)
    L_bc = ((u_left - 1.0) ** 2).mean() + (u_right ** 2).mean()

    return L_pde, L_ic, L_bc


def tau_ramp_weight(tau: torch.Tensor, args):
    """Causal time-ramped PDE weight: 0 at tau=0, -> 1 for tau >> ramp_tau.

    w(tau) = 1 - exp(-(tau / ramp_tau) ** ramp_power). Returns None when the
    --tau-ramp option is disabled.
    """
    if not getattr(args, "tau_ramp", False):
        return None
    return 1.0 - torch.exp(-(tau / args.ramp_tau) ** args.ramp_power)


def sample_batch(N_f, N_ic, N_bc, tau_max, device="cpu"):
    """Fresh collocation points (requires_grad on the interior inputs)."""
    xi_f = torch.rand(N_f, 1, device=device)
    tau_f = torch.rand(N_f, 1, device=device) * tau_max
    xi_f.requires_grad_(True)
    tau_f.requires_grad_(True)

    xi_ic = torch.rand(N_ic, 1, device=device)
    tau_ic = torch.zeros(N_ic, 1, device=device)

    tau_bc = torch.rand(N_bc, 1, device=device) * tau_max
    return xi_f, tau_f, xi_ic, tau_ic, tau_bc


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def rel_l2_error(model, Pe, tau_max, nx=256, nt=256, device="cpu"):
    """Relative L2 error vs the Ogata--Banks reference on a fixed grid."""
    dtype = torch.get_default_dtype()
    xi = torch.linspace(0.0, 1.0, nx, device=device, dtype=dtype)
    tau = torch.linspace(0.0, tau_max, nt, device=device, dtype=dtype)
    Xi, Tau = torch.meshgrid(xi, tau, indexing="xy")  # (nt, nx)
    with torch.no_grad():
        u_pred = model(Xi.reshape(-1, 1), Tau.reshape(-1, 1)).reshape(nt, nx)
        u_pred = u_pred.detach().cpu().numpy()
    u_ref = u_reference(Xi.detach().cpu().numpy(), Tau.detach().cpu().numpy(), Pe)
    err = float(np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref))
    return err, u_pred, u_ref, xi.detach().cpu().numpy(), tau.detach().cpu().numpy()


def region_errors(model, Pe, tau_max, nx=256, nt=256, device="cpu"):
    """Relative L2 error on the full grid and on the interior away from the
    entrance corner (tau > tau_c, xi > xi_c), to localize the mismatch."""
    dtype = torch.get_default_dtype()
    xi = np.linspace(0.0, 1.0, nx)
    tau = np.linspace(0.0, tau_max, nt)
    Xi, Tau = np.meshgrid(xi, tau, indexing="xy")
    with torch.no_grad():
        xi_t = torch.tensor(Xi.ravel(), dtype=dtype, device=device).unsqueeze(1)
        tau_t = torch.tensor(Tau.ravel(), dtype=dtype, device=device).unsqueeze(1)
        u_pred = model(xi_t, tau_t).reshape(nt, nx).detach().cpu().numpy()
    u_ref = u_reference(Xi, Tau, Pe)
    diff = np.abs(u_pred - u_ref)
    denom_full = np.linalg.norm(u_ref)
    res = {"full": float(np.linalg.norm(u_pred - u_ref) / denom_full)}
    for cut in (0.005, 0.02, 0.05, 0.10):
        mask = (Tau > cut) & (Xi > cut)
        res[f"tau,xi>{cut}"] = float(
            np.linalg.norm(diff[mask]) / np.linalg.norm(u_ref[mask]))
    # error near the left boundary layer only
    mask_bl = (Tau > 0.0) & (Xi < 0.05)
    res["left_layer"] = float(
        np.linalg.norm(diff[mask_bl]) / np.linalg.norm(u_ref[mask_bl]))
    return res


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(getattr(args, "device", "auto"))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if getattr(args, "dtype", "float32") == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    if args.hard_bc:
        model = HardBCMLP(hidden=args.hidden, layers=args.layers, eps=args.eps)
    else:
        model = MLP(hidden=args.hidden, layers=args.layers)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[pinn] {'hard-BC' if args.hard_bc else 'soft-BC'} params: "
          f"{n_params}, Pe = {args.pe}, tau_max = {args.tau_max}, "
          f"device = {device}, dtype = {torch.get_default_dtype()}")

    # Adam phase (optionally resumed, with gradient accumulation)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {"adam": [], "pde": [], "ic": [], "bc": []}
    start_step = 1
    ckpt_path = args.ckpt_path or os.path.join(args.outdir, "checkpoint.pt")

    if args.resume:
        if not os.path.exists(args.resume):
            raise SystemExit(f"[pinn] --resume checkpoint not found: {args.resume}")
        completed, history = load_checkpoint(args.resume, model, optim, device)
        start_step = completed + 1
        print(f"[pinn] resumed {args.resume}: adam step {completed}, "
              f"continuing from {start_step}")

    # Effective collocation batch = batch_size * grad_accum.
    accum = max(1, int(args.grad_accum))
    batch = (args.batch_size if args.batch_size and args.batch_size > 0
             else max(1, args.N_f // accum))
    if args.batch_size is not None or accum > 1:
        print(f"[pinn] collocation batching: {accum} x {batch} "
              f"(effective {accum * batch} points/step)")

    # SIGTERM (e.g. from `timeout`) -> save a resumable checkpoint before exit.
    ctx = {"model": model, "optim": optim, "step": start_step - 1,
           "history": history}

    def _sigterm(signum, frame):
        print(f"[pinn] SIGTERM at adam step {ctx['step']}: "
              f"saving checkpoint -> {ckpt_path}", flush=True)
        save_checkpoint(ctx["model"], ctx["optim"], ctx["step"],
                        ctx["history"], args, ckpt_path)
        raise SystemExit(143)

    old_handler = signal.signal(signal.SIGTERM, _sigterm)
    t0 = time.time()
    try:
        for step in range(start_step, args.adam_steps + 1):
            optim.zero_grad()
            loss_acc = Lpde_acc = Lic_acc = Lbc_acc = 0.0
            for _ in range(accum):
                xi_f, tau_f, xi_ic, tau_ic, tau_bc = sample_batch(
                    batch, args.N_ic, args.N_bc, args.tau_max, device=device)
                L_pde, L_ic, L_bc = losses(
                    model, xi_f, tau_f, xi_ic, tau_ic, tau_bc,
                    args.pe, args.hard_bc,
                    pde_weight=tau_ramp_weight(tau_f, args))
                loss = args.w_pde * L_pde + args.w_ic * L_ic + args.w_bc * L_bc
                (loss / accum).backward()
                loss_acc += float(loss.detach())
                Lpde_acc += float(L_pde.detach())
                Lic_acc += float(L_ic.detach())
                Lbc_acc += float(L_bc.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            history["adam"].append(loss_acc / accum)
            history["pde"].append(Lpde_acc / accum)
            history["ic"].append(Lic_acc / accum)
            history["bc"].append(Lbc_acc / accum)
            ctx["step"] = step

            if args.ckpt_every and step % args.ckpt_every == 0:
                save_checkpoint(model, optim, step, history, args, ckpt_path)

            if step % 2000 == 0 or step == start_step:
                print(f"[pinn] adam {step:6d}/{args.adam_steps} | "
                      f"loss {loss_acc/accum:.3e} | pde {Lpde_acc/accum:.3e} | "
                      f"ic {Lic_acc/accum:.3e} | bc {Lbc_acc/accum:.3e} | "
                      f"{time.time()-t0:.1f}s")

        # End-of-Adam checkpoint (resume point before the L-BFGS polish).
        save_checkpoint(model, optim, args.adam_steps, history, args, ckpt_path)

        # L-BFGS polish on a fixed collocation grid
        if args.lbfgs_steps > 0:
            xi_f, tau_f, xi_ic, tau_ic, tau_bc = sample_batch(
                args.N_f_fix, args.N_ic, args.N_bc, args.tau_max, device=device)

            def closure():
                optim_lbfgs.zero_grad()
                L_pde, L_ic, L_bc = losses(
                    model, xi_f, tau_f, xi_ic, tau_ic, tau_bc,
                    args.pe, args.hard_bc,
                    pde_weight=tau_ramp_weight(tau_f, args))
                loss = args.w_pde * L_pde + args.w_ic * L_ic + args.w_bc * L_bc
                loss.backward()
                return loss

            optim_lbfgs = torch.optim.LBFGS(
                model.parameters(), lr=args.lbfgs_lr,
                max_iter=args.lbfgs_steps,
                history_size=50,
                line_search_fn="strong_wolfe",
            )
            try:
                optim_lbfgs.step(closure)
                L_pde, L_ic, L_bc = losses(
                    model, xi_f, tau_f, xi_ic, tau_ic, tau_bc,
                    args.pe, args.hard_bc,
                    pde_weight=tau_ramp_weight(tau_f, args))
                print(f"[pinn] lbfgs done | pde {L_pde.item():.3e} | "
                      f"ic {L_ic.item():.3e} | bc {L_bc.item():.3e}")
            except Exception as exc:  # L-BFGS can fail on NaN line search
                print(f"[pinn] lbfgs skipped ({exc})")
    finally:
        signal.signal(signal.SIGTERM, old_handler)

    err, u_pred, u_ref, xi, tau = rel_l2_error(
        model, args.pe, args.tau_max, nx=args.nx_test, nt=args.nt_test,
        device=device)
    region = region_errors(model, args.pe, args.tau_max,
                           nx=args.nx_test, nt=args.nt_test, device=device)
    print(f"[pinn] relative L2 error = {err:.4e}")
    for k, v in region.items():
        print(f"[pinn]   region L2 [{k}] = {v:.4e}")
    if args.outdir:
        save_manifest(args, err, region, device, args.outdir)
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        print(f"[pinn] model saved -> {args.save_model}")

    return model, history, (err, u_pred, u_ref, xi, tau)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_results(history, eval_out, args, outdir):
    os.makedirs(outdir, exist_ok=True)
    err, u_pred, u_ref, xi, tau = eval_out

    # Loss history
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(history["adam"], label="total (Adam)")
    ax.semilogy(history["pde"], alpha=0.7, label=r"$\mathcal{L}_{PDE}$")
    ax.semilogy(history["ic"], alpha=0.7, label=r"$\mathcal{L}_{IC}$")
    ax.semilogy(history["bc"], alpha=0.7, label=r"$\mathcal{L}_{BC}$")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "loss.png"), dpi=150)
    plt.close(fig)

    # Prediction vs reference vs absolute error
    Xi, Tau = np.meshgrid(xi, tau, indexing="xy")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    im0 = axes[0].pcolormesh(Xi, Tau, u_pred, shading="nearest", cmap="viridis")
    axes[0].set_title(r"PINN $u_\theta$")
    im1 = axes[1].pcolormesh(Xi, Tau, u_ref, shading="nearest", cmap="viridis")
    axes[1].set_title(r"Ogata--Banks $u_{ref}$")
    im2 = axes[2].pcolormesh(Xi, Tau, np.abs(u_pred - u_ref),
                             shading="nearest", cmap="magma")
    axes[2].set_title("|error|")
    for ax, im in zip(axes, (im0, im1, im2)):
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\tau$")
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"Pe = {args.pe}, rel. L2 error = {err:.2e}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "comparison.png"), dpi=150)
    plt.close(fig)

    # Profiles
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in np.linspace(0, len(tau) - 1, 6).astype(int):
        ax.plot(xi, u_pred[n], lw=2,
                label=rf"$\tau={tau[n]:.2f}$")
        ax.plot(xi, u_ref[n], "k--", lw=1)
    ax.plot([], [], "k--", label="reference")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$u$")
    ax.set_title("PINN (solid) vs reference (dashed)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "profiles.png"), dpi=150)
    plt.close(fig)
    print(f"[pinn] figures written to {outdir}/")


# --------------------------------------------------------------------------
# Config & manifest helpers
# --------------------------------------------------------------------------
def load_config(path: str) -> dict:
    """Load a run config from JSON (always) or YAML (if PyYAML is present)."""
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
                return yaml.safe_load(f) or {}
            except ImportError as exc:
                raise SystemExit(
                    "[pinn] --config with .yaml needs PyYAML "
                    "(pip install pyyaml); or pass a .json config"
                ) from exc
        return json.load(f)


def git_sha() -> str:
    """Short git SHA of the working tree, or 'unknown' if not a repo."""
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def save_manifest(args, err, region, device, outdir):
    """Persist the effective config + final metrics next to the outputs."""
    os.makedirs(outdir, exist_ok=True)
    payload = {
        "config": {k: v for k, v in vars(args).items() if k != "config"},
        "device": str(device),
        "dtype": str(torch.get_default_dtype()),
        "git_sha": git_sha(),
        "metrics": {"rel_l2_full": err, "regions": region},
    }
    path = os.path.join(outdir, "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[pinn] manifest -> {path}")


def save_checkpoint(model, optim, step, history, args, path):
    """Save a *resumable* run checkpoint (model + Adam optim + step + history)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "step": int(step),
        "history": {k: list(v) for k, v in history.items()},
        "config": {k: v for k, v in vars(args).items()},
        "dtype": str(torch.get_default_dtype()),
        "git_sha": git_sha(),
    }
    torch.save(payload, path)
    print(f"[pinn] checkpoint -> {path}")


def load_checkpoint(path, model, optim, device):
    """Restore model + Adam optimizer from a checkpoint; return (step, history).

    The completed Adam step is returned; the caller resumes from ``step + 1``.
    Optimizer state tensors are moved to ``device`` so checkpoints are portable
    between CPU and GPU runs.
    """
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"])
    optim.load_state_dict(ck["optim"])
    # Adam state (exp_avg, exp_avg_sq) lives outside the parameter tensors.
    for group in optim.param_groups:
        for p in group["params"]:
            state = optim.state.get(p)
            if not state:
                continue
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    history = {"adam": [], "pde": [], "ic": [], "bc": []}
    history.update({k: list(v) for k, v in (ck.get("history") or {}).items()})
    return int(ck.get("step", 0)), history


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="PINN solver for the 1D ADE")
    ap.add_argument("--pe", type=float, default=20.0)
    ap.add_argument("--tau-max", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hard-bc", action="store_true",
                    help="use hard-constrained ansatz (exact IC/outlet BC)")
    ap.add_argument("--eps", type=float, default=1e-3,
                    help="inlet smoothing width for hard BC, s(tau)=1-exp(-tau/eps)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--adam-steps", type=int, default=12000)
    ap.add_argument("--lbfgs-steps", type=int, default=200)
    ap.add_argument("--lbfgs-lr", type=float, default=0.5)
    ap.add_argument("--N-f", type=int, default=1500)
    ap.add_argument("--N-ic", type=int, default=150)
    ap.add_argument("--N-bc", type=int, default=150)
    ap.add_argument("--N-f-fix", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="collocation sub-batch size (None = N-f // grad-accum)")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="gradient-accumulation sub-batches per Adam step")
    ap.add_argument("--resume", default=None,
                    help="resume Adam from a checkpoint written by --ckpt-every/SIGTERM")
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="save a resumable checkpoint every N Adam steps (0 = off)")
    ap.add_argument("--ckpt-path", default=None,
                    help="checkpoint path (default: <outdir>/checkpoint.pt)")
    ap.add_argument("--w-pde", type=float, default=1.0)
    ap.add_argument("--w-ic", type=float, default=10.0)
    ap.add_argument("--w-bc", type=float, default=10.0)
    ap.add_argument("--tau-ramp", action="store_true",
                    help="causal tau-ramped PDE weighting (down-weight the corner)")
    ap.add_argument("--ramp-tau", type=float, default=0.05,
                    help="characteristic tau scale for --tau-ramp")
    ap.add_argument("--ramp-power", type=float, default=1.0,
                    help="exponent of the --tau-ramp weight")
    ap.add_argument("--nx-test", type=int, default=256)
    ap.add_argument("--nt-test", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    help="compute device: auto|cuda|mps|cpu")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float64"],
                    help="model + tensor precision")
    ap.add_argument("--config", default=None,
                    help="JSON or YAML run-config file (explicit CLI flags win)")
    ap.add_argument("--outdir", default="out/pinn")
    ap.add_argument("--save-model", default="out/pinn/model.pt")

    # Record which flags were explicitly given on the command line; those win
    # over values loaded from --config.
    provided = set()
    for tok in sys.argv[1:]:
        if tok.startswith("--"):
            provided.add(tok[2:].split("=", 1)[0].replace("-", "_"))

    args = ap.parse_args()
    if args.config:
        cfg = load_config(args.config)
        cfg.pop("config", None)
        for key, value in cfg.items():
            if key in provided:
                continue
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print(f"[pinn] warning: unknown config key ignored: {key}")

    model, history, eval_out = train(args)
    plot_results(history, eval_out, args, args.outdir)


if __name__ == "__main__":
    main()
