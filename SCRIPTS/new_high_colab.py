#!/usr/bin/env python3
"""new_high_colab.py — run the 2D TESIS simulations on the *high-end* Colab GPUs.

Why a second Colab runner
-------------------------
``SCRIPTS/collab_run.py`` sizes every stage for a T4 and never reaches the
batched GPU backend: its ``opt`` stage calls ``run_optimization.py`` without
``--backend torch``, so the scenario archive stays on the SciPy/``splu`` path
even on a rented A100.  On the cards this file targets -- the **v5** Ampere
(A100 40/80 GB, ``colab new --gpu A100``) and the **v6** Hopper/Blackwell
(H100 80 GB or G4 = RTX PRO 6000, ``--gpu H100|G4``) -- that leaves most of the
machine idle, and the run the paper actually needs (3 starts x 1500 evaluations
at M_train=40, ``RUNTIMES.txt``: >1800 s on 8 CPU cores) is still a day.

The card class is *measured*, not assumed: name, compute capability, SM count
and VRAM come from ``torch.cuda.get_device_properties`` and are mapped to

    sm_70..75        turing   T4 (16 GB)                     Colab GPU: T4
    sm_80            v5       A100 (40/80 GB)                Colab GPU: A100
    sm_86/sm_89      ada      RTX A*/L4/L40S (16-48 GB)      Colab GPU: L4
    sm_90+           v6       H100 / Blackwell (G4)          Colab GPU: H100, G4

(``--assume`` forces the class, e.g. to print a plan for a card you have not
rented yet.  Colab's TPU ``v5e``/``v6e`` are *not* used here: this is the GPU
path.  ``DOCS/Collab.md`` is the notebook-level guide and ``DOCS/GPU.md`` records
the torch-backend measurements; ``SCRIPTS/multi_colab.py`` is the multi-VM
driver that ships the repo to a session, and ``SCRIPTS/bench_backend.py`` answers
the neighbouring question *"is the torch backend ahead on this runtime?"*.
This script assumes it is and asks *"how do I fill the card?"*: the ``bench``
stage measures throughput and peak memory per batch entry, and the resulting
bytes/slice is what sizes ``M_train`` for the ``opt`` stage.

What "optimized for v5/v6" means here
-------------------------------------
Three facts drive every number below, and all three are different from what a
T4-era tuning would do:

1. **The PDE kernels are bandwidth bound, not FLOP bound.**  One transport step
   is a batched BiCGSTAB sweep over ``S x M`` fields of ``nx*ny`` cells.  BF16
   and TF32 buy nothing for the transport solve (and bf16 loses the Krylov
   tolerance outright), so the levers are (a) **float32 instead of float64** --
   half the traffic, and ``default_rtol`` loosens to 1e-6 to match -- and
   (b) **batching**: the whole compass set of the pattern search is evaluated in
   one kernel launch, and every extra scenario in the archive is nearly free
   once the batch is resident.  TF32 is enabled anyway because the FNO/PINN
   stages *are* GEMM bound.
2. **VRAM, not FLOPs, caps the batch.**  Per-(candidate, scenario) slice costs
   about ``nx*ny*itemsize`` times a dozen BiCGSTAB vectors, so the scenario
   count that fits is measured by :func:`stage_bench` (peak allocated memory per
   slice) rather than assumed, and the ``opt`` stage's ``M_train`` is clamped to
   what the card can actually hold.
3. **The grid stays at the paper value.**  Resolution changes the physics, so
   the 80x40 grid and ``nsteps`` are only scaled by the ``push`` preset
   (a deliberate resolution study) -- the hardware buys *more scenarios and a
   bigger evaluation budget*, which is what tightens the SAA estimate.

Stages
------
    bench    throughput + peak-memory probe on the card; writes bench.json and
             the recommendation the other stages then use
    forward  deterministic source plume on the card (batched transport), with
             ``--verify`` comparing against the NumPy/``splu`` reference
    opt      SAA optimization through ``run_optimization.py --backend torch``
             on a config retuned for this card (written next to the artifacts)
    pinn    1D PINN via ``collab_run.py`` with card-scaled steps/sizes
    fno     1D FNO (+ its dataset) via ``collab_run.py`` with card-scaled sizes
    all     bench, forward, opt, pinn

Examples
--------
    # on a rented v6 card (or any GPU box): what does this machine do?
    python SCRIPTS/new_high_colab.py bench forward --verify

    # the paper-scale optimization, retuned for an A100
    python SCRIPTS/new_high_colab.py opt --assume v5 --preset hi --zip

    # reproduce the published numbers exactly (no retuning at all)
    python SCRIPTS/new_high_colab.py all --preset paper

    # plumbing check before spending an hour of compute units
    python SCRIPTS/new_high_colab.py bench forward --preset smoke --nx 24 --ny 12

    # rent the VM from the laptop, run, and pull the artifacts back
    python SCRIPTS/new_high_colab.py --launch --gpu-class v6 --accelerator H100 \
        bench forward opt --zip
    python SCRIPTS/new_high_colab.py --launch --gpu-class v5 --dry-run bench
    python SCRIPTS/new_high_colab.py --launch --keep --session tesis-high   # attach later

``--launch`` rents one VM (``colab new``), uploads the repo bundle, runs the
stages in the foreground of a ``%%bash`` cell, downloads the artifacts zip and
releases the VM.  The ``opt`` stage is the long one: ``--max-seconds`` (default
6 h) caps the remote cell, and ``--keep`` leaves the session up so
``colab url -s <session>`` can re-attach after a disconnect.

Exit status: 0 when every requested stage succeeded, 1 when a stage failed,
2 on a usage/preflight error.  Every run appends one record to
``runs/runtimes.jsonl`` (read it with ``SCRIPTS/runtimes_report.py``).

Gotchas inherited from the ``colab`` CLI (verified against google-colab-cli
0.8.0, see ``SCRIPTS/multi_colab.py`` for the long version)
-----------------------------------------------------------
* ``colab exec --timeout`` defaults to 30 s *without output*; every remote call
  here passes ``--timeout 0`` and keeps its own wall-clock cap.
* ``colab exec -f file`` sends the file as one IPython cell, so the two cells
  written by ``--launch`` start with the ``%%bash`` magic -- which is also why
  ``--env`` cannot be used (the cells ``export`` instead).
* a failing shell command inside a ``%%bash`` cell does **not** make the CLI
  return nonzero, so each cell echoes a ``[high] ... exit=<code>`` marker that
  this script parses out of the streamed log.
* ``colab run --gpu`` supports ``T4|L4|G4|H100|A100``; ``--high-mem`` is
  ignored for L4/TPU and needs a Pro/Pro+ entitlement.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent          # SCRIPTS/
ROOT = HERE.parent                              # repository root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "configs" / "opt" / "benchmark.yaml"
DEFAULT_OUT = ROOT / "out" / "high"
RUNTIME_LOG = ROOT / "runs" / "runtimes.jsonl"

# BiCGSTAB keeps ~10-12 full field-sized vectors alive per (candidate, scenario)
# slice. Only used when a bench.json measurement is not available yet.
VEC_FACTOR = 12

STAGE_CHOICES = ("bench", "forward", "opt", "pinn", "fno", "all")
PRESET_CHOICES = ("smoke", "paper", "hi", "push")


# ---------------------------------------------------------------------------
# Card classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CardSpec:
    """What a card class is allowed to spend, per stage.

    ``m_train``/``m_test``/``max_evals`` are *targets*: :func:`derive_tuning`
    clamps them to what the measured VRAM budget can hold.  ``paper`` is
    unaffected by all of this; it re-reads the YAML.
    """

    key: str
    label: str
    colab: tuple[str, ...]          # `colab new --gpu` acceptances
    vram_frac: float                # share of VRAM the transport batch may use
    m_train: int
    m_test: int
    max_evals: int                  # per start
    krylov_maxiter: int
    pinn: dict[str, int]
    fno: dict[str, int]
    bf16: bool


SPECS: dict[str, CardSpec] = {
    "v6": CardSpec(
        key="v6",
        label="v6 (Hopper/Blackwell: H100 80GB, G4/RTX PRO 6000)",
        colab=("H100", "G4"),
        vram_frac=0.80,
        m_train=128, m_test=256, max_evals=4000, krylov_maxiter=6000,
        pinn={"adam": 30000, "lbfgs": 600, "nf": 4000, "nside": 400},
        fno={"epochs": 400, "modes": 24, "width": 64, "batch": 16},
        bf16=True,
    ),
    "v5": CardSpec(
        key="v5",
        label="v5 (Ampere: A100 40/80GB)",
        colab=("A100",),
        vram_frac=0.80,
        m_train=96, m_test=192, max_evals=3000, krylov_maxiter=4000,
        pinn={"adam": 24000, "lbfgs": 400, "nf": 3000, "nside": 300},
        fno={"epochs": 300, "modes": 20, "width": 48, "batch": 12},
        bf16=True,
    ),
    "ada": CardSpec(
        key="ada",
        label="ada (L4/L40S/RTX A*: 16-48GB)",
        colab=("L4",),
        vram_frac=0.75,
        m_train=64, m_test=128, max_evals=2000, krylov_maxiter=4000,
        pinn={"adam": 16000, "lbfgs": 300, "nf": 2000, "nside": 200},
        fno={"epochs": 200, "modes": 16, "width": 32, "batch": 8},
        bf16=True,
    ),
    "turing": CardSpec(
        key="turing",
        label="turing/older (T4 16GB — collab_run.py's target)",
        colab=("T4",),
        vram_frac=0.70,
        m_train=40, m_test=80, max_evals=1500, krylov_maxiter=4000,
        pinn={"adam": 12000, "lbfgs": 200, "nf": 1500, "nside": 150},
        fno={"epochs": 150, "modes": 12, "width": 20, "batch": 8},
        bf16=False,
    ),
    "cpu": CardSpec(
        key="cpu",
        label="cpu (no accelerator: torch CPU backend, for plumbing only)",
        colab=(),
        vram_frac=0.0,
        m_train=40, m_test=80, max_evals=200, krylov_maxiter=2000,
        pinn={"adam": 2000, "lbfgs": 0, "nf": 500, "nside": 64},
        fno={"epochs": 20, "modes": 8, "width": 16, "batch": 4},
        bf16=False,
    ),
}


def classify(major: int, minor: int) -> str:
    """Compute capability -> card class (see the module docstring's table)."""
    if major >= 9:                       # Hopper sm_90, Blackwell sm_100/120
        return "v6"
    if major == 8:
        return "v5" if minor == 0 else "ada"    # A100 vs A*/L4
    return "turing"                      # Volta/Turing/Pascal: the T4 tier


def profile_card(device: str) -> dict[str, Any]:
    """Inspect the accelerator this process can see (no allocation)."""
    try:
        import torch
    except Exception as exc:                                   # pragma: no cover
        return {"available": False, "cuda": False, "name": "no-torch",
                "class": "cpu", "vram_gb": 0.0, "cc": [0, 0], "sms": 0,
                "reason": str(exc)}

    card: dict[str, Any] = {"available": True, "torch": torch.__version__,
                            "cuda": bool(torch.cuda.is_available()),
                            "device": device}
    if not card["cuda"] or str(device).startswith("cpu"):
        card.update({"name": "cpu", "class": "cpu", "vram_gb": 0.0,
                     "cc": [0, 0], "sms": 0, "cuda": False, "index": 0})
        return card

    index = 0
    if ":" in str(device):
        index = int(str(device).split(":", 1)[1])
    props = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info(index)
    card.update({
        "name": props.name,
        "class": classify(props.major, props.minor),
        "cc": [int(props.major), int(props.minor)],
        "sms": int(props.multi_processor_count),
        "vram_gb": round(total / 1e9, 2),
        "free_gb": round(free / 1e9, 2),
        "index": index,
        "cuda_version": torch.version.cuda,
    })
    return card


def setup_torch(card: dict[str, Any], dtype: str) -> Any:
    """Apply the runtime flags the card class wants, then return ``torch``.

    The allocator knob must be set before the first CUDA allocation, which is
    why it lives here as well as in :func:`main` (module import stays cheap so
    ``--help``/``--plan`` work on a machine without torch).
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    if card.get("cuda"):
        # The PDE solve is bandwidth bound, but FNO/PINN are GEMM bound: TF32
        # only ever helps the latter, and only in float32.
        torch.backends.cuda.matmul.allow_tf32 = dtype == "float32"
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = card["class"] in {"v5", "v6", "ada"}
        try:
            torch.set_float32_matmul_precision("high" if dtype == "float32"
                                               else "highest")
        except Exception:                                      # pragma: no cover
            pass
    return torch


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
@dataclass
class Tuning:
    """The resolved numbers every stage then honours."""

    card_class: str
    device: str
    dtype: str
    krylov_rtol: float
    krylov_maxiter: int
    nx: int
    ny: int
    nsteps: int
    m_train: int
    m_test: int
    n_starts: int
    max_evals: int
    budget_gb: float
    bytes_per_slice: float
    max_batch: int
    compass: int
    notes: list[str]


def load_cfg(args: argparse.Namespace) -> dict:
    """Effective config for this run (``smoke`` implies ``apply_quick``)."""
    from stoch.config import load_config

    return load_config(args.config, quick=args.preset == "smoke")


def _bench_measurement(out: Path) -> dict[str, Any] | None:
    """Reuse the last bench.json of this output directory, if it exists."""
    path = out / "bench.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                          # pragma: no cover
        return None
    return data if data.get("bytes_per_slice") else None


def derive_tuning(cfg: dict, spec: CardSpec, card: dict[str, Any],
                  args: argparse.Namespace, out: Path) -> Tuning:
    """Turn (card, preset, overrides, last bench) into one Tuning."""
    dtype = args.dtype
    itemsize = 4 if dtype == "float32" else 8
    notes: list[str] = []

    # -- grid / time ----------------------------------------------------
    nx, ny = int(cfg["domain"]["nx"]), int(cfg["domain"]["ny"])
    nsteps = int(cfg["time"]["nsteps"])
    if args.preset == "push":
        nx, ny = nx * 2, ny * 2
        nsteps = int(round(nsteps * 1.5))
        notes.append("preset=push: grid and nsteps doubled/raised (resolution study, "
                     "not comparable with the paper numbers)")

    # -- scenarios and evaluation budget --------------------------------
    if args.preset in {"paper", "smoke"}:
        m_train = int(cfg["saa"]["M_train"])
        m_test = int(cfg["saa"]["M_test"])
        max_evals = int(cfg["optim"].get("max_evals", 1500))
        notes.append(f"preset={args.preset}: scenarios/budget taken from the config")
    else:
        m_train, m_test = spec.m_train, spec.m_test
        max_evals = spec.max_evals
        notes.append(f"preset={args.preset}: card-class scenario counts "
                     f"(M_train={m_train}, M_test={m_test}, max_evals={max_evals})")

    # -- memory budget --------------------------------------------------
    compass = max(2 * int(cfg["wells"]["n"]), 1)      # the pattern-search sweep
    analytic = nx * ny * itemsize * VEC_FACTOR
    measured = _bench_measurement(out)
    per_slice = float(analytic)
    if measured:
        scale = (nx * ny) / max(int(measured.get("nx", nx)) * int(measured.get("ny", ny)), 1)
        bench_itemsize = 4 if measured.get("dtype", dtype) == "float32" else 8
        scale *= itemsize / bench_itemsize
        per_slice = max(per_slice, float(measured["bytes_per_slice"]) * scale)
        notes.append(f"bytes/slice from bench.json ({measured['bytes_per_slice']:.0f} B, "
                     f"scaled x{scale:.2f})")

    if args.budget_gb is not None:
        budget_bytes = float(args.budget_gb) * 1e9
    elif card.get("cuda"):
        budget_bytes = spec.vram_frac * float(card.get("vram_gb") or 0.0) * 1e9
    else:
        budget_bytes = 4e9                       # CPU: a soft, irrelevant cap
    max_batch = int(budget_bytes / per_slice) if per_slice > 0 else 0

    if max_batch > 0:
        fit = max_batch // compass
        if fit < m_train:
            notes.append(f"clamped M_train {m_train} -> {fit} "
                         f"(only {max_batch} slices fit in {budget_bytes / 1e9:.1f} GB)")
            m_train = max(4, fit)
        if max_batch // max(m_test, 1) < 4:
            m_test = max(8, min(m_test, max_batch // 4))
            notes.append(f"clamped M_test -> {m_test}")

    # -- explicit overrides win -----------------------------------------
    if args.nx is not None:
        nx = int(args.nx)
    if args.ny is not None:
        ny = int(args.ny)
    if args.nsteps is not None:
        nsteps = int(args.nsteps)
    if args.m_train is not None:
        m_train = int(args.m_train)
    if args.m_test is not None:
        m_test = int(args.m_test)
    if args.max_evals is not None:
        max_evals = int(args.max_evals)
    n_starts = int(args.n_starts if args.n_starts is not None
                   else cfg["optim"].get("n_starts", 1))

    return Tuning(
        card_class=spec.key,
        device=args.device,
        dtype=dtype,
        krylov_rtol=float(args.krylov_rtol) if args.krylov_rtol
        else (1e-11 if dtype == "float64" else 1e-6),
        krylov_maxiter=int(args.krylov_maxiter or spec.krylov_maxiter),
        nx=nx, ny=ny, nsteps=nsteps,
        m_train=m_train, m_test=m_test, n_starts=n_starts, max_evals=max_evals,
        budget_gb=round(budget_bytes / 1e9, 2),
        bytes_per_slice=round(per_slice, 1),
        max_batch=max_batch,
        compass=compass,
        notes=notes,
    )


def apply_tuning(cfg: dict, tune: Tuning) -> dict:
    """Write the tuned numbers back into the config the stages consume."""
    cfg.setdefault("domain", {}).update({"nx": tune.nx, "ny": tune.ny})
    cfg.setdefault("time", {})["nsteps"] = tune.nsteps
    cfg.setdefault("saa", {}).update({"M_train": tune.m_train, "M_test": tune.m_test})
    cfg.setdefault("optim", {}).update({"max_evals": tune.max_evals,
                                        "n_starts": tune.n_starts})
    return cfg


def report_tuning(card: dict[str, Any], spec: CardSpec, tune: Tuning) -> None:
    log(f"card: {card.get('name', '?')}  "
        f"sm_{card.get('cc', [0, 0])[0]}{card.get('cc', [0, 0])[1]}  "
        f"{card.get('vram_gb', 0):.1f} GB  {card.get('sms', 0)} SMs  "
        f"-> class={tune.card_class} ({spec.label})")
    if card.get("cuda") is False and card.get("available"):
        log("warning: torch reports no CUDA device; the stages will run on CPU "
            "(fine for --preset smoke plumbing checks only)")
    log(f"tuning: dtype={tune.dtype} krylov_rtol={tune.krylov_rtol:g} "
        f"maxiter={tune.krylov_maxiter}")
    log(f"grid: {tune.nx}x{tune.ny} ({tune.nx * tune.ny} cells), nsteps={tune.nsteps}")
    log(f"scenarios: M_train={tune.m_train} M_test={tune.m_test}; "
        f"SAA budget {tune.n_starts} start(s) x {tune.max_evals} evals")
    log(f"batch: {tune.bytes_per_slice / 1e3:.1f} kB per (candidate,scenario), "
        f"{tune.budget_gb:.1f} GB budget -> {tune.max_batch} slices; "
        f"compass S={tune.compass}")
    for note in tune.notes:
        log(f"  note: {note}")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[high] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def die(msg: str, code: int = 2) -> None:
    print(f"[high] error: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def sync(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_stream(cmd: Sequence[str], logfile: Path | None = None) -> int:
    """Run a subprocess from the repo root, mirroring output into `logfile`."""
    log(f"$ {shlex.join(str(c) for c in cmd)}")
    fh = None
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logfile.open("a", encoding="utf-8")
        fh.write(f"\n### {shlex.join(str(c) for c in cmd)}\n")
    try:
        proc = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if fh is not None:
                fh.write(line)
                fh.flush()
        return proc.wait()
    finally:
        if fh is not None:
            fh.close()


def save_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return path


def normalize_stages(items: Sequence[str]) -> list[str]:
    if not items:
        return ["bench", "forward"]
    out: list[str] = []
    for item in items:
        for stage in (["bench", "forward", "opt", "pinn"] if item == "all" else [item]):
            if stage not in out:
                out.append(stage)
    return out


# ---------------------------------------------------------------------------
# Stage: bench — what does this card actually do?
# ---------------------------------------------------------------------------
def _gemm_probe(torch: Any, card: dict[str, Any], n: int = 2048,
                iters: int = 3) -> dict[str, float]:
    """Rough matmul rate for the training stages (TF32 fp32 and bf16)."""
    if not card.get("cuda"):
        return {}
    rates: dict[str, float] = {}
    device = f"cuda:{int(card.get('index', 0))}"
    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        try:
            a = torch.randn(n, n, dtype=dtype, device=device)
            b = torch.randn(n, n, dtype=dtype, device=device)
            for _ in range(2):
                a @ b
            sync(torch)
            t0 = time.perf_counter()
            for _ in range(iters):
                a @ b
            sync(torch)
            elapsed = time.perf_counter() - t0
            rates[name] = round(2.0 * n ** 3 * iters / elapsed / 1e12, 2)
            del a, b
            torch.cuda.empty_cache()
        except Exception as exc:                               # pragma: no cover
            rates[name] = float("nan")
            log(f"gemm probe ({name}) failed: {exc}")
    return rates


def stage_bench(args: argparse.Namespace, tune: Tuning, card: dict[str, Any],
                cfg: dict, out: Path) -> dict:
    """Measure transport throughput and peak memory per batch entry."""
    import numpy as np

    from run_optimization import base_aquifer, make_cost, make_forward, make_grid, make_kl
    from stoch.random_fields import ScenarioGenerator
    from stoch.saa_torch import TorchArchive
    from stoch.wells import place_wells

    torch = setup_torch(card, tune.dtype)
    grid = make_grid(cfg)
    aquifer = base_aquifer(cfg, grid)
    fwd = make_forward(cfg)
    cost = make_cost(cfg)
    kl = make_kl(cfg, grid)
    gen = ScenarioGenerator(grid, kl,
                            random_source=bool(cfg["random_field"]["random_source"]),
                            random_gradient=bool(cfg["random_field"]["random_gradient"]))

    # The layout only has to be *a* valid one here: this stage measures the
    # transport/objective kernel, not the well design (stage_forward does that).
    mask = np.ones((grid.ny, grid.nx), dtype=bool)
    layout = place_wells(grid, mask, int(cfg["wells"]["n"]),
                         method=cfg["wells"]["method"],
                         seed=int(cfg["optim"]["seed"]))
    nw = layout.n_wells

    m0 = max(1, min(8, tune.m_train))
    seeds = [int(cfg["seed"]["train"]) + i for i in range(m0)]
    scenarios = [gen.sample(s) for s in seeds]
    log(f"bench: grid {grid.nx}x{grid.ny}, nsteps={fwd.nsteps}, "
        f"{m0} scenarios, {nw} wells, dtype={tune.dtype}")

    t0 = time.perf_counter()
    archive = TorchArchive(grid, aquifer, layout.wells, scenarios, fwd,
                           device=tune.device, dtype=tune.dtype,
                           maxiter=tune.krylov_maxiter, label="bench")
    log(f"bench: archive resident in {time.perf_counter() - t0:.1f}s "
        f"({archive.describe().get('gpu', 'cpu')})")

    base_alloc = (torch.cuda.memory_allocated(int(card.get("index", 0)))
                  if card.get("cuda") else 0)
    q_scale = 0.25 * float(cfg["wells"]["q_max"])
    batches = [int(b) for b in args.bench_batches if int(b) > 0]
    if not card.get("cuda"):
        # A CPU transport solve is ~1 s per candidate; keep the sweep short.
        batches = [b for b in batches if b <= 4] or [1]
    if tune.max_batch > 0:
        cap = max(1, tune.max_batch // m0)
        batches = [b for b in batches if b <= cap] or [1]
    batches = sorted(set(batches))

    # One warm-up call so cuBLAS/kernel autotuning is not charged to the measurement.
    archive.evaluate_many(np.full((2, nw), q_scale), cost)
    sync(torch)

    rows: list[dict[str, Any]] = []
    for S in batches:
        Q = np.full((S, nw), q_scale)
        sync(torch)
        if card.get("cuda"):
            torch.cuda.reset_peak_memory_stats(int(card.get("index", 0)))
        t0 = time.perf_counter()
        batch = archive.evaluate_many(Q, cost)
        sync(torch)
        elapsed = time.perf_counter() - t0
        peak = (torch.cuda.max_memory_allocated(int(card.get("index", 0))) - base_alloc
                if card.get("cuda") else 0)
        # .cpu() is required: bench runs on the card, and float() on a CUDA
        # tensor raises "can't convert cuda:0 device type tensor to numpy".
        # Caught by a real L4 run -- the CPU smoke test could not reach it.
        J = batch.J.detach().float().cpu()
        row = {
            "candidates": S,
            "scenarios": m0,
            "slices": S * m0,
            "seconds": round(elapsed, 3),
            "s_per_candidate": round(elapsed / (S * m0), 5),
            "cells_per_s": int(round(S * m0 * fwd.nsteps * grid.n / max(elapsed, 1e-9))),
            "peak_gb": round(peak / 1e9, 3),
            "bytes_per_slice": int(peak / (S * m0)) if peak else 0,
            "J_mean": float(J.mean()),
            "J_finite": bool(torch.isfinite(J).all()),
        }
        rows.append(row)
        log(f"  S={S:<4} M={m0}  {elapsed:7.2f}s  "
            f"{row['s_per_candidate'] * 1e3:7.2f} ms/(cand·scen)  "
            f"{row['cells_per_s'] / 1e6:6.1f} Mcell·step/s  "
            f"peak={row['peak_gb']:.2f} GB")
        del batch

    measured = max((r["bytes_per_slice"] for r in rows), default=0)
    gemm = _gemm_probe(torch, card)

    payload = {
        "card": card,
        "card_class": tune.card_class,
        "dtype": tune.dtype,
        "nx": grid.nx, "ny": grid.ny, "nsteps": fwd.nsteps,
        "m_per_batch": m0, "wells": nw, "compass": tune.compass,
        "rows": rows,
        "gemm_tflops": gemm,
        "bytes_per_slice": measured or int(tune.bytes_per_slice),
        "budget_gb": tune.budget_gb,
        "recommended_m_train": max(4, min(tune.m_train,
                                          tune.max_batch // max(tune.compass, 1))),
        "analytic_bytes_per_slice": int(tune.nx * tune.ny *
                                         (4 if tune.dtype == "float32" else 8)
                                         * VEC_FACTOR),
    }
    save_json(out / "bench.json", payload)
    best = max(rows, key=lambda r: r["cells_per_s"]) if rows else None
    if best:
        log(f"bench: best throughput {best['cells_per_s'] / 1e6:.1f} Mcell·step/s at "
            f"S={best['candidates']}; {best['bytes_per_slice'] / 1e3:.1f} kB/slice measured")
    if gemm:
        log("bench: matmul " + " ".join(f"{k}={v:.1f} TFLOP/s" for k, v in gemm.items()))
    return payload


# ---------------------------------------------------------------------------
# Stage: forward — the deterministic plume, on the card
# ---------------------------------------------------------------------------
def stage_forward(args: argparse.Namespace, tune: Tuning, card: dict[str, Any],
                  cfg: dict, out: Path) -> dict:
    """Batched source plume + optional parity check against the NumPy reference."""
    import numpy as np

    from physics.fv2d_torch import (
        FlowModelTorch,
        flow_responses_torch,
        resolve_device,
        torch_dtype,
        transport_batch,
    )
    from run_optimization import (
        base_aquifer,
        design_plume,
        make_forward,
        make_grid,
        make_kl,
    )
    from stoch.random_fields import ScenarioGenerator

    torch = setup_torch(card, tune.dtype)
    grid = make_grid(cfg)
    fwd = make_forward(cfg)
    kl = make_kl(cfg, grid)
    gen = ScenarioGenerator(grid, kl,
                            random_source=bool(cfg["random_field"]["random_source"]),
                            random_gradient=bool(cfg["random_field"]["random_gradient"]))
    aquifer = base_aquifer(cfg, grid)

    log("forward: designing the well layout from the mean plume (NumPy reference)")
    _, mask, layout = design_plume(cfg, grid, kl, gen, fwd)
    nw = layout.n_wells
    n_scen = max(1, min(int(args.n_scen or 4), tune.m_train))
    scenarios = [gen.sample(int(cfg["seed"]["train"]) + i) for i in range(n_scen)]
    log(f"forward: {nw} wells, {n_scen} scenarios, grid {grid.nx}x{grid.ny}, "
        f"nsteps={fwd.nsteps}")

    dtype = torch_dtype(tune.dtype)
    device = resolve_device(tune.device)
    K = np.stack([np.asarray(s.K, dtype=float) for s in scenarios])
    h_left = np.array([fwd.grad * grid.lx * s.gradient_scale for s in scenarios])
    flow = FlowModelTorch(grid, K, phi=aquifer.phi, b=aquifer.b,
                          h_left=h_left, h_right=0.0, dtype=dtype,
                          device=device, rtol=tune.krylov_rtol,
                          maxiter=tune.krylov_maxiter)
    resp = flow_responses_torch(flow, layout.wells, fwd.eps_well)

    q = np.full((1, nw), float(args.q or 0.0))
    source_idx = grid.index(*fwd.source_xy)
    record = [source_idx, *grid.indices([p[0] for p in fwd.compliance_xy],
                                        [p[1] for p in fwd.compliance_xy])]

    t0 = time.perf_counter()
    res = transport_batch(
        grid, aquifer, flow, resp, q, fwd.dt, fwd.nsteps,
        source_idx=source_idx,
        source_rate=fwd.source_q * fwd.source_c,
        source_steps=fwd.source_steps,
        record_cells=record,
        rtol=tune.krylov_rtol,
        maxiter=tune.krylov_maxiter,
    )
    sync(torch)
    elapsed = time.perf_counter() - t0

    mass = res.mass
    bal = (mass["total"][-1] - mass["total"][0]
           - (mass["source"][-1] - mass["well"][-1]
              - mass["outflow"][-1] - mass["decay"][-1]))
    denom = torch.maximum(torch.maximum(mass["source"][-1].abs(),
                                        mass["total"][0].abs()),
                          torch.full_like(mass["total"][0], 1e-30))
    mass_err = float((bal / denom).abs().max())
    C_final = res.C_final.detach().cpu().numpy()            # (S, M, n)
    C_rec = res.C_record.detach().cpu().numpy()             # (nsteps, S, M, nrec)

    summary: dict[str, Any] = {
        "card": card.get("name"),
        "card_class": tune.card_class,
        "device": str(res.C_final.device),
        "dtype": tune.dtype,
        "grid": [grid.nx, grid.ny],
        "nsteps": fwd.nsteps,
        "scenarios": n_scen,
        "wells": nw,
        "q": q.reshape(-1).tolist(),
        "seconds": round(elapsed, 3),
        "mass_balance_error": mass_err,
        "krylov_iterations": res.report.iterations,
        "krylov_worst_residual": float(res.report.worst_residual),
        "max_C_final": float(C_final.max()),
        "plume_area": float((C_final[0].max(axis=0).reshape(grid.ny, grid.nx)
                             >= float(cfg["wells"]["plume_threshold"])).sum()
                            * grid.dx * grid.dy),
        "compliance_C_max": float(C_rec[:, 0, :, 1:].max()) if C_rec.shape[-1] > 1 else None,
    }

    if args.verify:
        summary["verify"] = _verify_against_numpy(cfg, grid, fwd, layout,
                                                  scenarios[0], q, record,
                                                  C_final[0, 0], tune)

    np.savez_compressed(
        out / "forward.npz",
        C_final=C_final.astype(np.float32),
        C_compliance=C_rec.astype(np.float32),
        wells=layout.wells,
        q=q,
        **{f"mass_{k}": v.detach().cpu().numpy().astype(np.float64)
           for k, v in mass.items()},
    )
    save_json(out / "forward.json", summary)
    log(f"forward: {elapsed:.1f}s, mass-balance {mass_err:.2e}, "
        f"max C={summary['max_C_final']:.3g}, plume {summary['plume_area']:.0f} m2")
    if args.verify:
        v = summary["verify"]
        log(f"forward: parity vs NumPy reference rel_max={v['rel_max']:.2e} "
            f"({'PASS' if v['rel_max'] < v['tolerance'] else 'FAIL'})")
        if v["rel_max"] >= v["tolerance"]:
            raise SystemExit(1)
    return summary


def _verify_against_numpy(cfg, grid, fwd, layout, scenario, q, record,
                          C_gpu, tune: Tuning) -> dict:
    """One-scenario NumPy/``splu`` reference solve, compared with the GPU field."""
    import numpy as np

    from physics.fv2d import FlowModel, flow_responses, transport_run
    from run_optimization import base_aquifer

    aq = base_aquifer(cfg, grid, np.asarray(scenario.K, dtype=float))
    flow = FlowModel(grid, aq, h_left=fwd.grad * grid.lx * scenario.gradient_scale,
                     h_right=0.0)
    resp = flow_responses(flow, layout.wells, fwd.eps_well)
    res = transport_run(grid, aq, flow, resp, np.asarray(q).reshape(-1),
                        fwd.dt, fwd.nsteps,
                        source_idx=grid.index(*fwd.source_xy),
                        source_rate=fwd.source_q * fwd.source_c * scenario.source_scale,
                        source_steps=fwd.source_steps,
                        record_cells=np.asarray(record, dtype=int))
    ref = np.asarray(res.C_final, dtype=float).reshape(-1)
    got = np.asarray(C_gpu, dtype=float).reshape(-1)
    scale = max(float(np.abs(ref).max()), 1e-30)
    return {
        "rel_max": float(np.abs(got - ref).max() / scale),
        "tolerance": 5e-4 if tune.dtype == "float32" else 1e-8,
        "reference": "physics.fv2d (splu, float64)",
        "dtype": tune.dtype,
    }


# ---------------------------------------------------------------------------
# Stage: opt — SAA through the batched torch backend
# ---------------------------------------------------------------------------
def stage_opt(args: argparse.Namespace, tune: Tuning, card: dict[str, Any],
              cfg: dict, out: Path) -> dict:
    """Run ``run_optimization.py --backend torch`` on the retuned config."""
    import yaml

    opt_dir = out / "opt"
    cfg_path = out / "config_opt.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log(f"opt: config for this card -> {cfg_path}")

    cmd = [
        sys.executable, str(HERE / "run_optimization.py"),
        "--config", str(cfg_path),
        "--stage", args.opt_stage,
        "--backend", "torch",
        "--device", tune.device,
        "--dtype", tune.dtype,
        "--krylov-maxiter", str(tune.krylov_maxiter),
        "--krylov-rtol", f"{tune.krylov_rtol:g}",
        "--batched-search",
        "--out", str(opt_dir),
        "--runtime-log", str(args.runtime_log or ""),
    ]
    rc = run_stream(cmd, logfile=out.parent / "logs" / f"opt_{out.name}.log")
    metrics_path = opt_dir / "metrics.json"
    summary: dict[str, Any] = {"exit": rc, "metrics": str(metrics_path)}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        stage = metrics.get("stages", {}).get("opt", {})
        n_evals = int(stage.get("n_evaluations", 0) or 0)
        summary.update({
            "backend": metrics.get("backend"),
            "wall_clock_s": metrics.get("wall_clock_s"),
            "n_evaluations": n_evals,
            "s_per_eval": round(float(stage.get("opt_wall_clock_s", 0.0))
                                / max(n_evals, 1), 3),
            "q_star": stage.get("q_star"),
            "active_wells": stage.get("active_wells"),
            "total_pumping": stage.get("total_pumping"),
            "train_mean_J": (stage.get("train") or {}).get("mean_J"),
            "test_mean_J": (stage.get("test") or {}).get("mean_J"),
            "test_reliability": (stage.get("test") or {}).get("reliability"),
            "gates": metrics.get("gates"),
        })
        if n_evals:
            log(f"opt: {n_evals} evaluations, {summary['s_per_eval']:.3f} s/eval, "
                f"test J={summary['test_mean_J']}")
    if rc != 0:
        raise SystemExit(1)
    return summary


# ---------------------------------------------------------------------------
# Stages: pinn / fno — delegate to collab_run.py with card-scaled sizes
# ---------------------------------------------------------------------------
def _collab_run(args: argparse.Namespace, tune: Tuning, collab_argv: list[str],
                out: Path, tag: str) -> dict:
    cmd = [sys.executable, str(HERE / "collab_run.py"),
           "--device", tune.device, "--out-root", str(out),
           "--no-quick-opt"] + collab_argv
    rc = run_stream(cmd, logfile=out.parent / "logs" / f"{tag}_{out.name}.log")
    return {"exit": rc, "out_root": str(out)}


def stage_pinn(args: argparse.Namespace, tune: Tuning, card: dict[str, Any],
               out: Path) -> dict:
    spec = SPECS[tune.card_class]
    log(f"pinn: adam={spec.pinn['adam']} lbfgs={spec.pinn['lbfgs']} "
        f"N_f={spec.pinn['nf']} N_side={spec.pinn['nside']}")
    argv = [
        "--models", "pinn", "--preset", "full",
        "--pinn-steps", str(spec.pinn["adam"]),
        "--lbfgs-steps", str(spec.pinn["lbfgs"]),
        "--ckpt-every", str(args.ckpt_every),
    ]
    if args.hard_bc:
        argv.append("--hard-bc")
    return _collab_run(args, tune, argv, out, "pinn")


def stage_fno(args: argparse.Namespace, tune: Tuning, card: dict[str, Any],
              out: Path) -> dict:
    spec = SPECS[tune.card_class]
    log(f"fno: epochs={spec.fno['epochs']} modes={spec.fno['modes']} "
        f"width={spec.fno['width']}")
    argv = [
        "--models", "dataset", "fno", "--preset", "full",
        "--fno-epochs", str(spec.fno["epochs"]),
        "--fno-modes", str(spec.fno["modes"]),
        "--fno-width", str(spec.fno["width"]),
    ]
    return _collab_run(args, tune, argv, out, "fno")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def zip_results(out: Path, target: Path | None = None) -> Path:
    target = target or (out.parent / f"{out.name}_artifacts.zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    count = 0
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out.rglob("*")):
            if not path.is_file() or path == target:
                continue
            zf.write(path, Path(out.name) / path.relative_to(out))
            count += 1
    log(f"wrote {target} ({count} files, {target.stat().st_size / 1e6:.2f} MB)")
    return target


def write_report(args: argparse.Namespace, card: dict[str, Any], spec: CardSpec,
                 tune: Tuning, stages: dict[str, Any], out: Path,
                 t_start: float) -> Path:
    payload = {
        "script": Path(__file__).name,
        "preset": args.preset,
        "stages_requested": list(args.stages),
        "card": card,
        "card_spec": {"key": spec.key, "label": spec.label},
        "tuning": asdict(tune),
        "results": stages,
        "wall_clock_s": round(time.time() - t_start, 3),
    }
    path = save_json(out / "report.json", payload)
    what = ", ".join(stages) or "no stage"
    log(f"report: {path} ({what})")
    return path


def log_run(args: argparse.Namespace, tune: Tuning, card: dict[str, Any],
            stages: dict[str, Any], seconds: float, status: str) -> None:
    if not args.runtime_log:
        return
    from stoch.runtimes import log_runtime

    opt = stages.get("opt") or {}
    log_runtime(
        args.runtime_log, __file__, seconds,
        stage=",".join(stages) or "none",
        preset=args.preset,
        card_class=tune.card_class,
        gpu=card.get("name"),
        device=str(tune.device),
        dtype=tune.dtype,
        grid=[tune.nx, tune.ny],
        nsteps=tune.nsteps,
        m_train=tune.m_train,
        max_evals=tune.max_evals,
        s_per_eval=opt.get("s_per_eval"),
        test_mean_J=opt.get("test_mean_J"),
        outdir=str(Path(args.out).resolve()),
        status=status,
    )


# ---------------------------------------------------------------------------
# --launch: rent a VM, ship the repo, run this script there
# ---------------------------------------------------------------------------
MARK_PROVISION = "[high] provision exit="
MARK_RUN = "[high] run exit="
BUNDLE_SKIP_DIRS = {".git", "__pycache__", "out", "runs", "figs", "ZIPS",
                    "True", ".venv", ".ipynb_checkpoints", "node_modules",
                    ".mypy_cache", ".pytest_cache"}
BUNDLE_SKIP_SUFFIXES = {".pyc", ".pyo", ".zip", ".log"}
TORCH_LINE = re.compile(r"^\s*(torch|torchvision|torchaudio|nvidia-|triton)\b", re.I)


def bundle_repo(dest: Path) -> Path:
    """Zip the working tree (no artifacts) plus a torch-free requirements file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    count = 0
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(ROOT.rglob("*")):
            rel = path.relative_to(ROOT)
            if any(part in BUNDLE_SKIP_DIRS for part in rel.parts):
                continue
            if not path.is_file() or path.suffix in BUNDLE_SKIP_SUFFIXES:
                continue
            if path.stat().st_size > 8 * 1024 * 1024:
                continue
            zf.write(path, rel)
            count += 1
        req = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        kept = [ln for ln in req if not TORCH_LINE.match(ln)]
        zf.writestr("requirements_colab.txt",
                    "# generated by new_high_colab.py: torch lines stripped so the\n"
                    "# CUDA-enabled Colab wheel is never replaced by a CPU build\n"
                    + "\n".join(kept) + "\n")
    log(f"bundled {count} files -> {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def render_provision(remote_root: str, remote_zip: str) -> str:
    return "\n".join([
        "%%bash",
        "set -euo pipefail",
        "trap 'echo \"" + MARK_PROVISION + "$?\"' EXIT",
        f"mkdir -p {remote_root} && cd {remote_root}",
        "if [ ! -d repo ]; then",
        f"  python3 -c \"import zipfile; zipfile.ZipFile('{remote_zip}').extractall('repo')\"",
        "fi",
        "cd repo",
        # Colab already ships numpy/scipy/matplotlib/PyYAML/Flask. A pip
        # network hiccup must not kill the run: try once, and if it fails only
        # warn -- the import check below is the real gate.
        "python3 -m pip install -q --retries 3 --timeout 60 "
        "--upgrade-strategy only-if-needed -r requirements_colab.txt"
        " || echo '[high] warning: pip install failed (network?); relying on "
        "the preinstalled Colab packages'",
        "python3 -c \"import numpy, scipy, yaml; print('[high] deps ok')\"",
        "python3 -c \"import torch; print('[high] torch', torch.__version__, "
        "'cuda', torch.cuda.is_available())\" || echo '[high] torch missing'",
        "nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader "
        "2>/dev/null || echo '[high] no nvidia-smi (CPU runtime)'",
        "nproc",
        "echo '[high] provision done'",
    ]) + "\n"


def render_run(remote_root: str, remote_argv: list[str]) -> str:
    inner = "python3 -u SCRIPTS/new_high_colab.py " + shlex.join(remote_argv)
    return "\n".join([
        "%%bash",
        "set -euo pipefail",
        f"if [ ! -d {remote_root}/repo ]; then echo '[high] FATAL: repo missing'; exit 3; fi",
        f"cd {remote_root}/repo",
        "export PYTHONUNBUFFERED=1 MPLBACKEND=Agg",
        # Propagate the traceback switch so a remote failure is debuggable.
        "export HIGH_DEBUG=" + ("1" if os.environ.get("HIGH_DEBUG") else ""),
        f"trap 'echo \"{MARK_RUN}$?\"' EXIT",
        f"{inner} 2>&1 | tee {remote_root}/run.log",
    ]) + "\n"


def marker_exit(logfile: Path, marker: str) -> int | None:
    if not logfile.exists():
        return None
    for line in reversed(logfile.read_text(encoding="utf-8",
                                           errors="replace").splitlines()):
        if marker in line:
            found = re.search(r"-?\d+", line.split(marker, 1)[1])
            if found:
                return int(found.group(0))
    return None


def cli_stream(argv: list[str], logfile: Path, timeout: float | None,
               dry_run: bool) -> int:
    """Run one `colab` call, mirroring its output into `logfile`.

    `timeout` is a hard wall-clock cap on the CLI process (`timeout(1)`
    semantics: killed => 124), so a wedged websocket cannot hang the driver.
    """
    log(f"$ {shlex.join(argv)}")
    if dry_run:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("a", encoding="utf-8") as fh:
            fh.write(f"# dry-run: {shlex.join(argv)}\n")
        return 0
    logfile.parent.mkdir(parents=True, exist_ok=True)
    import threading

    with logfile.open("a", encoding="utf-8") as fh:
        fh.write(f"\n### {shlex.join(argv)}\n")
        fh.flush()
        proc = subprocess.Popen(argv, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, bufsize=1)
        assert proc.stdout is not None
        killer = None
        if timeout:
            killer = threading.Timer(max(1.0, timeout), proc.kill)
            killer.daemon = True
            killer.start()
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                fh.write(line)
                fh.flush()
            rc = proc.wait()
        finally:
            if killer is not None:
                killer.cancel()
        return rc


def launch(args: argparse.Namespace) -> int:
    """Provision a Colab VM, upload the repo, run the requested stages there."""
    key = args.gpu_class or "v6"
    if key not in SPECS or key == "cpu":
        die(f"--gpu-class must be one of v5, v6, ada, turing (got {key!r})")
    accelerator = args.accelerator or SPECS[key].colab[0]
    remote_root = f"/content/high_{time.strftime('%Y%m%d%H%M%S')}"
    run_id = remote_root.rsplit("_", 1)[1]
    session = args.session or f"tesis-high-{run_id[-6:]}"
    out = Path(args.out).expanduser()
    rdir = (out if out.is_absolute() else (ROOT / out).resolve()) / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    logfile = rdir / "cli.log"

    bundle = bundle_repo(rdir / "bundle.zip")
    remote_zip = f"{remote_root}.zip"
    stages = normalize_stages(args.stages)

    remote_argv = stages + [
        "--preset", args.preset,
        "--dtype", args.dtype,
        "--device", "cuda",
        "--assume", key,
        "--out", f"{remote_root}/out",
        "--zip",
    ]
    for flag, value in (("--nx", args.nx), ("--ny", args.ny),
                        ("--nsteps", args.nsteps), ("--m-train", args.m_train),
                        ("--m-test", args.m_test), ("--max-evals", args.max_evals),
                        ("--budget-gb", args.budget_gb)):
        if value is not None:
            remote_argv += [flag, str(value)]
    if args.verify:
        remote_argv.append("--verify")
    if args.hard_bc and "pinn" in stages:
        remote_argv.append("--hard-bc")
    if args.zip_path:
        zip_name = Path(args.zip_path).name
        remote_argv += ["--zip-path", zip_name]
    else:
        zip_name = "artifacts.zip"
    if not args.runtime_log:
        remote_argv += ["--runtime-log", ""]

    (rdir / "provision.py").write_text(render_provision(remote_root, remote_zip),
                                       encoding="utf-8")
    (rdir / "run.py").write_text(render_run(remote_root, remote_argv),
                                 encoding="utf-8")
    artifacts = f"{remote_root}/out/{zip_name}"

    binary = shutil.which(args.colab)
    if binary is None and not args.dry_run:
        die(f"`{args.colab}` not found on PATH. `pip install google-colab-cli`, "
            "then authenticate once with `colab auth` (or pass --auth adc).")
    binary = binary or args.colab

    def cli(*parts: str) -> list[str]:
        argv = [binary] + (["--auth", args.colab_auth] if args.colab_auth else [])
        return argv + [str(p) for p in parts]

    plan = [
        # --high-mem only where Colab actually offers a bigger shape (A100 80GB
        # instead of 40GB, T4 with more host RAM); it is ignored elsewhere.
        ("new", cli("new", "-s", session, "--gpu", accelerator,
                    *(["--high-mem"] if accelerator in {"A100", "T4"} else [])),
         900),
        ("upload", cli("upload", "-s", session, str(bundle), remote_zip), 3600),
        ("provision", cli("exec", "-s", session, "--timeout", "0", "-f",
                          str(rdir / "provision.py")), 3600),
        ("run", cli("exec", "-s", session, "--timeout", "0", "-f",
                    str(rdir / "run.py")), args.max_seconds),
        ("download", cli("download", "-s", session, artifacts,
                         str(rdir / "artifacts.zip")), 3600),
    ]
    if not args.keep:
        plan.append(("stop", cli("stop", "-s", session), 600))

    log(f"session={session} accelerator={accelerator} stages={','.join(stages)}")
    log(f"local run dir: {rdir}")
    for name, argv, timeout in plan:
        rc = cli_stream(argv, logfile, timeout, args.dry_run)
        if name == "run":
            status = marker_exit(logfile, MARK_RUN)
            if status not in (None, 0):
                log(f"run cell exited {status}")
                # Stop the VM here too. Returning early used to leak a running
                # session (and its billing) whenever the remote cell failed --
                # observed with two orphaned L4 sessions after a bench error.
                if not args.keep:
                    cli_stream(cli("stop", "-s", session), logfile, 600,
                               args.dry_run)
                return 1
        if rc != 0 and name != "download":
            log(f"{name} failed with {rc}; see {logfile}")
            if not args.keep and name in {"provision", "run"}:
                cli_stream(cli("stop", "-s", session), logfile, 600, args.dry_run)
            return 1
        if rc != 0:
            log(f"warning: {name} failed with {rc} (artifacts may be missing)")
    unpacked = rdir / "artifacts.zip"
    if unpacked.exists() and not args.dry_run:
        log(f"artifacts: {unpacked} ({unpacked.stat().st_size / 1e6:.2f} MB)")
    if args.keep:
        log(f"session {session} kept alive: `colab url -s {session}` to attach, "
            f"`colab stop -s {session}` to release")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="new_high_colab.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="stages: " + ", ".join(STAGE_CHOICES),
    )
    ap.add_argument("stages", nargs="*", default=[], metavar="STAGE",
                    choices=list(STAGE_CHOICES) + [],
                    help="bench, forward, opt, pinn, fno or all "
                         "(default: bench forward)")

    g = ap.add_argument_group("what to run")
    g.add_argument("--preset", choices=PRESET_CHOICES, default="hi",
                   help="smoke=quick plumbing check; paper=config values "
                        "(published numbers); hi=card-class scenarios/budget "
                        "(default); push=hi plus a 2x finer grid and 1.5x nsteps")
    g.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="optimization config to start from")
    g.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    g.add_argument("--device", default="auto",
                   help="auto|cuda|cuda:N|cpu (passed to the torch scripts)")
    g.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                   help="torch working precision (float32 halves the memory "
                        "traffic; the Krylov tolerance follows)")
    g.add_argument("--assume", choices=list(SPECS), default=None,
                   help="force the card class instead of detecting it "
                        "(planning on a machine without the card)")
    g.add_argument("--opt-stage", default="opt",
                   choices=["all", "gates", "o0", "o1", "o2", "o3", "o4", "opt"],
                   help="stage handed to run_optimization.py by the opt stage")

    g = ap.add_argument_group("sizing overrides (win over the preset)")
    g.add_argument("--nx", type=int, default=None)
    g.add_argument("--ny", type=int, default=None)
    g.add_argument("--nsteps", type=int, default=None)
    g.add_argument("--m-train", type=int, default=None)
    g.add_argument("--m-test", type=int, default=None)
    g.add_argument("--max-evals", type=int, default=None)
    g.add_argument("--n-starts", type=int, default=None)
    g.add_argument("--n-scen", type=int, default=None,
                   help="scenarios used by the forward stage (default 4)")
    g.add_argument("--q", type=float, default=0.0,
                   help="pumping rate per well for the forward stage (default 0)")
    g.add_argument("--budget-gb", type=float, default=None,
                   help="VRAM the transport batch may use (default: a fraction "
                        "of the card's memory)")
    g.add_argument("--bench-batches", default="1,18,72,288",
                   help="comma-separated candidate counts for the bench sweep")
    g.add_argument("--verify", action="store_true",
                   help="forward: also solve one scenario with the NumPy/"
                        "splu reference and report the relative difference")
    g.add_argument("--hard-bc", dest="hard_bc", action="store_true", default=True,
                   help="pinn: hard-boundary-condition ansatz (default on)")
    g.add_argument("--no-hard-bc", dest="hard_bc", action="store_false")
    g.add_argument("--ckpt-every", type=int, default=2000,
                   help="pinn: checkpoint every N Adam steps (a rented VM can "
                        "disconnect; 0 disables)")
    g.add_argument("--krylov-rtol", type=float, default=None)
    g.add_argument("--krylov-maxiter", type=int, default=None)

    g = ap.add_argument_group("output")
    g.add_argument("--zip", dest="zip", action="store_true",
                   help="zip the output directory when the run finishes")
    g.add_argument("--zip-path", default=None, help="explicit zip target")
    g.add_argument("--plan", action="store_true",
                   help="print the resolved tuning and exit without computing")
    g.add_argument("--runtime-log", default=str(RUNTIME_LOG),
                   help="append one JSON record per run ('' disables)")

    g = ap.add_argument_group("--launch (rent a VM through the colab CLI)")
    g.add_argument("--launch", action="store_true",
                   help="provision a Colab session, ship the repo and run the "
                        "stages there")
    g.add_argument("--gpu-class", choices=["v5", "v6", "ada", "turing"],
                   default=None, help="card class to rent (default v6)")
    g.add_argument("--accelerator", default=None,
                   help="Colab accelerator name: T4|L4|G4|H100|A100 "
                        "(default: the first one the class supports)")
    g.add_argument("--session", default=None, help="Colab session name")
    g.add_argument("--keep", action="store_true",
                   help="leave the VM running so `colab exec -s NAME` can attach")
    g.add_argument("--max-seconds", type=float, default=6 * 3600.0,
                   help="hard cap on the remote run cell")
    g.add_argument("--colab", default="colab", help="colab CLI binary")
    g.add_argument("--colab-auth", default=None, choices=[None, "oauth2", "adc"],
                   help="global --auth value for the colab CLI")
    g.add_argument("--dry-run", action="store_true",
                   help="print the colab command plan without running it")

    args = ap.parse_args(argv)
    if args.launch and args.plan:
        ap.error("--launch and --plan are mutually exclusive")
    args.bench_batches = [int(x) for x in
                          str(args.bench_batches).replace(",", " ").split()]
    args.stages = normalize_stages(args.stages)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    t_start = time.time()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parse_args(argv)

    if args.launch:
        return launch(args)

    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out.parent / "logs").mkdir(parents=True, exist_ok=True)

    card = profile_card(args.device)
    if args.assume:
        if card.get("available") and card.get("cuda") and card["class"] != args.assume:
            log(f"warning: detected class {card['class']} ({card.get('name')}) but "
                f"--assume {args.assume} wins; the tuning is for the assumed card")
        card["class"] = args.assume
        card["assumed"] = True
    spec = SPECS[card.get("class", "cpu")]
    if args.device in {"auto", ""} and spec.key == "cpu" and not card.get("cuda"):
        # torch.device("auto") is not a device string; resolve it here since the
        # transport code takes a concrete device.
        args.device = "cpu"

    cfg = load_cfg(args)
    tune = derive_tuning(cfg, spec, card, args, out)
    cfg = apply_tuning(cfg, tune)
    report_tuning(card, spec, tune)
    save_json(out / "tuning.json", {"card": card, "spec": asdict(spec),
                                    "tuning": asdict(tune)})
    if args.plan:
        log("--plan: stopping before any compute")
        return 0

    dispatch = {
        "bench": lambda: stage_bench(args, tune, card, cfg, out),
        "forward": lambda: stage_forward(args, tune, card, cfg, out),
        "opt": lambda: stage_opt(args, tune, card, cfg, out),
        "pinn": lambda: stage_pinn(args, tune, card, out),
        "fno": lambda: stage_fno(args, tune, card, out),
    }
    results: dict[str, Any] = {}
    failed: list[str] = []
    for stage in args.stages:
        log(f"=== stage {stage} ===")
        t0 = time.time()
        try:
            results[stage] = dispatch[stage]()
            results[stage]["seconds"] = round(time.time() - t0, 3)
        except SystemExit as exc:                     # a stage asked to stop
            failed.append(stage)
            results[stage] = {"error": "stage reported failure",
                              "code": exc.code,
                              "seconds": round(time.time() - t0, 3)}
            log(f"stage {stage} failed ({exc.code})")
            break
        except Exception as exc:                      # noqa: BLE001 - report and stop
            failed.append(stage)
            results[stage] = {"error": f"{type(exc).__name__}: {exc}",
                              "seconds": round(time.time() - t0, 3)}
            log(f"stage {stage} raised: {type(exc).__name__}: {exc}")
            # A one-line message is not enough to debug a remote run: the
            # traceback is the only clue you get back from a rented VM.
            if os.environ.get("HIGH_DEBUG"):
                import traceback
                log("traceback:\n" + traceback.format_exc())
            break
        log(f"stage {stage} done in {results[stage]['seconds']:.1f}s")

    write_report(args, card, spec, tune, results, out, t_start)
    if args.zip:
        target = Path(args.zip_path).expanduser() if args.zip_path else None
        if target is not None and not target.is_absolute():
            target = (out / target).resolve()
        # keep the zip inside the output dir so --launch can download one path
        zip_results(out, target or (out / "artifacts.zip"))
    log_run(args, tune, card, results, time.time() - t_start,
            status="failed" if failed else "ok")
    if failed:
        log(f"finished with failures: {', '.join(failed)}")
        return 1
    log(f"finished in {time.time() - t_start:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
