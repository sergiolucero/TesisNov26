"""
solve_1d_fno.py
===============

Phase 4: train a 2D Fourier neural operator (FNO) on the 32-scenario 1D ADE
dataset (`out/dataset.npz`) and run the spectral / resolution / normalization
checks from `plan.md` Phase 4:

  * frequency truncation  -- controlled by --modes1/--modes2 (k_max)
  * resolution invariance -- train on a coarsened grid (--coarse-factor) and
                             evaluate on the native (finer) grid
  * data normalization    -- per-channel standard-Gaussian normalization of the
                             input parameters (v, D) and the output field u
                             (disable with --no-normalize for ablation)

The input to the operator is the *constant* parameter function a(xi, tau) =
(v_norm, D_norm) broadcast over the (xi, tau) grid; the output is the
dimensionless concentration field u(xi, tau) = C/C0. The FNO is resolution
agnostic (it parameterizes the kernel in Fourier space), so it can be trained
on a coarse grid and evaluated on a fine one.

Run:
    python solve_1d_fno.py --epochs 300 --modes1 12 --modes2 12

Author: TESIS (groundwater remediation PINN/FNO debugging)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hydrology import load_scenarios


def pick_device(prefer=None):
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


torch.set_default_dtype(torch.float32)


# --------------------------------------------------------------------------
# FNO building blocks (2D)
# --------------------------------------------------------------------------
class SpectralConv2d(nn.Module):
    """2D spectral convolution: parameterize the kernel directly in Fourier
    space and truncate to the ``modes1 x modes2`` lowest frequencies."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        b, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x)  # (b, c, h, w//2 + 1)
        out_ft = torch.zeros(b, self.out_channels, h, w // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, h)
        m2 = min(self.modes2, w // 2 + 1)
        # positive frequencies
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2],
            self.weights1[:, :, :m1, :m2])
        # negative frequencies (wrap-around)
        out_ft[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2],
            self.weights2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(h, w))


class FNO2d(nn.Module):
    """2D FNO (Li et al., 2021): lift -> K spectral layers -> project."""

    def __init__(self, modes1, modes2, width, in_channels=2, out_channels=1,
                 layers=4):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, width)  # lifting (pointwise)

        self.spe = nn.ModuleList()
        self.skip = nn.ModuleList()
        for _ in range(layers):
            self.spe.append(SpectralConv2d(width, width, modes1, modes2))
            self.skip.append(nn.Conv2d(width, width, 1))

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):  # x: (B, C, H, W)
        x = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for spe, skip in zip(self.spe, self.skip):
            x = F.gelu(spe(x) + skip(x))
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)  # (B, out, H, W)


# --------------------------------------------------------------------------
# Data loading & normalization
# --------------------------------------------------------------------------
def load_dataset(path):
    """Return (params (N,2)=[v,D], fields (N,nt,nx)=u) and grid info."""
    scenarios = load_scenarios(path)
    n = len(scenarios)
    params = np.stack([[s.v, s.D] for s in scenarios]).astype(np.float64)
    fields = np.stack([s.C / s.C0 for s in scenarios]).astype(np.float32)
    nt, nx = scenarios[0].C.shape
    return params, fields, nt, nx


def normalize(x, mean, std):
    return (x - mean) / std


def denormalize(x, mean, std):
    return x * std + mean


def build_input(params_norm, nt, nx, device):
    """Constant parameter function broadcast over the (nt, nx) grid."""
    n = params_norm.shape[0]
    x = np.empty((n, 2, nt, nx), dtype=np.float32)
    for i in range(n):
        x[i, 0] = params_norm[i, 0]
        x[i, 1] = params_norm[i, 1]
    return torch.tensor(x, device=device)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def rel_l2(pred, target, per_sample=False):
    """Relative L2 error; pred/target are (N, 1, H, W) numpy arrays."""
    n = pred.shape[0]
    errs = np.array([
        np.linalg.norm(pred[i].ravel() - target[i].ravel())
        / np.linalg.norm(target[i].ravel())
        for i in range(n)])
    return errs if per_sample else float(errs.mean())


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_fno(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    params, fields, nt_full, nx_full = load_dataset(args.dataset)
    n = params.shape[0]
    print(f"[fno] dataset: {n} scenarios, native grid {nt_full}x{nx_full}, "
          f"v in [{params[:, 0].min():.3f}, {params[:, 0].max():.3f}], "
          f"D in [{params[:, 1].min():.3f}, {params[:, 1].max():.3f}]")

    # Train/test split
    idx = np.random.permutation(n)
    n_train = max(1, int(round(n * args.train_frac)))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    print(f"[fno] split: {len(train_idx)} train / {len(test_idx)} test")

    # Resolution invariance: coarsen the training grid by average-pooling.
    factor = max(1, int(args.coarse_factor))
    fields_t = torch.tensor(fields)
    if factor > 1:
        nt, nx = nt_full // factor, nx_full // factor
        fields_t = F.avg_pool2d(fields_t.unsqueeze(1), kernel_size=factor) \
                    .squeeze(1)  # (N, nt, nx)
        print(f"[fno] training at coarse grid {nt}x{nx} "
              f"(factor {factor}; native {nt_full}x{nx_full})")
    else:
        nt, nx = nt_full, nx_full

    # Per-channel standard-Gaussian normalization (input + output).
    # Stats are computed on the *training* split only (never leak test data).
    p_mean = params[train_idx].mean(axis=0)
    p_std = params[train_idx].std(axis=0)
    p_std = np.where(p_std == 0, 1.0, p_std)
    params_norm = normalize(params, p_mean, p_std)

    u_mean = float(fields_t[train_idx].mean())
    u_std = float(fields_t[train_idx].std())
    u_std = u_std if u_std > 0 else 1.0
    if args.no_normalize:
        u_mean, u_std = 0.0, 1.0
    print(f"[fno] input norm  v: mean={p_mean[0]:.3f} std={p_std[0]:.3f} | "
          f"D: mean={p_mean[1]:.3f} std={p_std[1]:.3f}")
    print(f"[fno] output norm u: mean={u_mean:.3f} std={u_std:.3f}"
          + (" (disabled)" if args.no_normalize else ""))

    # Build tensors (coarse grid for training, native grid for evaluation).
    fields_norm = normalize(fields_t, u_mean, u_std).unsqueeze(1)  # (N,1,nt,nx)
    fields_norm = fields_norm.to(device)

    x_train = build_input(params_norm, nt, nx, device)[train_idx]
    y_train = fields_norm[train_idx]

    # Native-resolution inputs/targets for the resolution-invariance check.
    x_eval = build_input(params_norm, nt_full, nx_full, device)
    y_eval_native = torch.tensor(
        normalize(fields, u_mean, u_std), device=device).unsqueeze(1)

    model = FNO2d(modes1=args.modes1, modes2=args.modes2, width=args.width,
                  in_channels=2, out_channels=1, layers=args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[fno] model: modes=({args.modes1},{args.modes2}), width={args.width}, "
          f"layers={args.layers}, params={n_params}, device={device}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    t0 = time.time()
    bsz = max(1, args.batch_size)

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(train_idx))
        epoch_loss = 0.0
        for start in range(0, len(train_idx), bsz):
            b = perm[start:start + bsz]
            xb, yb = x_train[b], y_train[b]
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss.detach()) * len(b)
        epoch_loss /= len(train_idx)
        history.append(epoch_loss)
        if epoch % 50 == 0 or epoch == 1:
            print(f"[fno] epoch {epoch:4d}/{args.epochs} | "
                  f"loss {epoch_loss:.3e} | {time.time()-t0:.1f}s")

    # Evaluate on the test split.
    model.eval()
    with torch.no_grad():
        pred_norm = model(x_eval)
        pred_native = denormalize(pred_norm, u_mean, u_std).cpu().numpy()
    fields_native = fields  # already u = C/C0

    errs_train = rel_l2(pred_native[train_idx], fields_native[train_idx],
                        per_sample=True)
    errs_test = rel_l2(pred_native[test_idx], fields_native[test_idx],
                       per_sample=True)
    print(f"[fno] rel-L2 train (native eval) = {errs_train.mean():.4e} "
          f"(per-sample [{errs_train.min():.2e}, {errs_train.max():.2e}])")
    print(f"[fno] rel-L2 test  (native eval) = {errs_test.mean():.4e} "
          f"(per-sample [{errs_test.min():.2e}, {errs_test.max():.2e}])")
    print(f"[fno] resolution check: trained "
          f"{'coarse ' + str(nt) + 'x' + str(nx) if factor > 1 else 'native'}, "
          f"evaluated at {nt_full}x{nx_full}")

    if args.outdir:
        save_manifest(args, errs_test.mean(), errs_test, device,
                      p_mean, p_std, u_mean, u_std, args.outdir)
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        print(f"[fno] model saved -> {args.save_model}")
    if args.outdir:
        plot_fno(args, model, x_eval, fields_native, test_idx, u_mean, u_std,
                 history, args.outdir)

    return model, errs_test.mean()


def save_manifest(args, err, errs, device, p_mean, p_std, u_mean, u_std, outdir):
    os.makedirs(outdir, exist_ok=True)
    payload = {
        "config": {k: v for k, v in vars(args).items() if k != "config"},
        "device": str(device),
        "dtype": str(torch.get_default_dtype()),
        "normalization": {
            "v_mean": float(p_mean[0]), "v_std": float(p_std[0]),
            "D_mean": float(p_mean[1]), "D_std": float(p_std[1]),
            "u_mean": float(u_mean), "u_std": float(u_std),
        },
        "metrics": {
            "rel_l2_test_mean": float(err),
            "rel_l2_test_per_sample": [float(e) for e in errs],
        },
    }
    path = os.path.join(outdir, "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[fno] manifest -> {path}")


def plot_fno(args, model, x_eval, fields_native, test_idx, u_mean, u_std,
             history, outdir):
    os.makedirs(outdir, exist_ok=True)
    # Loss curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (normalized)")
    ax.set_title("FNO training loss")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fno_loss.png"), dpi=150)
    plt.close(fig)

    # One test prediction vs truth
    model.eval()
    i = int(test_idx[0])
    with torch.no_grad():
        pred = denormalize(model(x_eval[i:i + 1]), u_mean, u_std)[0, 0] \
            .cpu().numpy()
    truth = fields_native[i]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    im0 = axes[0].imshow(truth.T, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_title("truth $u$")
    im1 = axes[1].imshow(pred.T, origin="lower", aspect="auto", cmap="viridis")
    axes[1].set_title("FNO $u$")
    im2 = axes[2].imshow(np.abs(pred - truth).T, origin="lower", aspect="auto",
                         cmap="magma")
    axes[2].set_title("|error|")
    for ax, im in zip(axes, (im0, im1, im2)):
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"FNO test sample {i}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fno_prediction.png"), dpi=150)
    plt.close(fig)
    print(f"[fno] figures written to {outdir}/")


def main():
    ap = argparse.ArgumentParser(description="FNO for the 1D ADE (Phase 4)")
    ap.add_argument("--dataset", default="out/dataset.npz")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--modes1", type=int, default=12)
    ap.add_argument("--modes2", type=int, default=12)
    ap.add_argument("--width", type=int, default=20)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--coarse-factor", type=int, default=1,
                    help="train on a grid coarsened by this factor (resolution test)")
    ap.add_argument("--no-normalize", action="store_true",
                    help="disable per-channel Gaussian normalization (ablation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="out/fno")
    ap.add_argument("--save-model", default="out/fno/model.pt")
    args = ap.parse_args()

    train_fno(args)


if __name__ == "__main__":
    main()
