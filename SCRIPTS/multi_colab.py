#!/usr/bin/env python3
"""multi_colab.py — fan large TESIS runs onto Colab VMs through the `colab` CLI.

Why this exists
---------------
The optimization stage is CPU-bound numpy/scipy and one objective evaluation
costs ``M_train`` full transport solves. Measured on an 8-core x86 box
(``Archive.evaluate`` at the paper grid):

    80x40, nsteps=100, M_train=40   ->  15-26 s per objective evaluation

`solve_saa` budgets ``n_starts * max_evals`` evaluations (``max_evals`` is
*per start*), so the shipped `configs/opt/benchmark.yaml` (3 starts x 1500
evals) is a 19-33 hour job -- `RUNTIMES.txt` already records it as
``run_optimization.py --stage opt (full benchmark)  >1800s TIMEOUT``. A Colab
notebook cell is the wrong place for that: one disconnect loses everything,
because the optimizer writes no intermediate checkpoints.

So: run it on Colab VMs from the terminal instead, one replica per VM, and
bring the artifacts back to disk. This script does the whole lifecycle --
bundle the repo, provision a runtime, stream the run's log home, download the
results zip, release the VM -- for N replicas in parallel.

What it does
------------
1. resolves the *effective* config locally with the project's own
   `stoch.config.load_config` (+ `apply_quick` when the preset asks for it),
   then applies `--nx/--ny/--nt` and every `--set a.b=c` override, and writes
   the merged YAML that the remote run will use. The remote therefore runs
   exactly the configuration this script estimated -- `--no-quick-opt` is
   passed to `collab_run.py` so `--quick` cannot silently re-shrink the grid
   behind your back (that is how a "large run" quietly becomes a 30x15 smoke
   test).
2. estimates the wall clock from the measured cost model and refuses to launch
   runs above `--max-hours` unless you insist.
3. zips the repo (tests/docs/out excluded), uploads it, unpacks it on the VM,
   and installs requirements with the **torch lines stripped** so Colab's
   CUDA-enabled torch is never replaced by a CPU wheel.
4. provisions one `colab` session per replica (`--gpu T4|L4|A100|none`),
   executes the run in the foreground of a `%%bash` cell so the kernel stays
   active (activity-based liveness -- a backgrounded job can be reclaimed),
   streams the remote log into ``<outdir>/<run-id>/r<k>/run.log``, then
   downloads the results zip and stops the VM.
5. persists ``plan.json`` so `status`/`logs`/`collect`/`stop` can act on a run
   later, including after the driving shell died.

Usage
-----
    # predicted cost first, no network calls at all
    ./SCRIPTS/multi_colab.py launch --gpu T4 --nx 80 --ny 40 \
        --set optim.max_evals=250 --set saa.M_train=40 --estimate --dry-run

    # one VM, the biggest run that fits in half a day
    ./SCRIPTS/multi_colab.py launch --gpu T4 --nx 80 --ny 40 \
        --opt-stage opt --set optim.max_evals=250 --set optim.n_starts=1 \
        --set saa.M_train=40 --max-hours 6

    # three seeds on three VMs (compare train.mean_J across r0/r1/r2)
    ./SCRIPTS/multi_colab.py launch --gpu L4 --replicas 3 --nx 80 --ny 40 \
        --opt-stage opt --set optim.max_evals=250 --set optim.n_starts=1

    # large PINN instead (GPU actually matters here)
    ./SCRIPTS/multi_colab.py launch --models pinn --preset full --gpu A100 \
        --pinn-steps 40000 --hard-bc --replicas 2

    ./SCRIPTS/multi_colab.py status         # sessions + last known state
    ./SCRIPTS/multi_colab.py logs -f        # tail every replica's log
    ./SCRIPTS/multi_colab.py collect        # (re)download artifacts
    ./SCRIPTS/multi_colab.py stop           # release the VMs

The `colab` CLI is the console script of the **google-colab-cli** package
(``pip install google-colab-cli``; verified against v0.8.0 on this machine:
`colab new/run/exec/upload/download/log/stop/sessions/install/status/usage`).
Note that ``colab_client_notes.txt`` in this repo says
``pip install google-colab-client`` -- that package name does not exist on PyPI,
``google-colab-cli`` is the one that installs the `colab` command.

Authentication defaults to **ADC** (Application Default Credentials): every
CLI call here passes ``--auth adc``, which is what ``COLAB_AUTH`` (or ``--auth``)
overrides. Set that up once with ``gcloud auth application-default login`` (or
point ``GOOGLE_APPLICATION_CREDENTIALS`` at a service-account key); the older
installed-app flow is still reachable via ``--auth oauth2`` after ``colab auth``.
ADC needs no keyring, so it also works non-interactively on the fleet VMs.

Three details of that CLI this script has to work around, all verified against
its source:

* ``colab exec --timeout`` defaults to **30 s without output**, which a long
  transport solve easily exceeds -- every exec here passes ``--timeout 0``.
* ``exec -f file.py`` sends the file as **one IPython cell**, so the cells
  below start with the ``%%bash`` magic. Because of that ``--env KEY=VALUE``
  cannot be used (it prepends Python lines, and a magic must be the first line
  of a cell); the cells ``export`` instead.
* a failing shell script inside a ``%%bash`` cell does **not** make the CLI
  return nonzero, so each cell echoes a ``[multi] ... exit=<code>`` marker and
  the driver reads the run's real status out of the streamed log.

Exit status: 0 when every replica finished and its artifacts arrived,
1 when a replica failed, 2 on a usage/preflight error.
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
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stoch.runtimes import log_runtime  # noqa: E402  (needs ROOT on sys.path)

DEFAULT_OUTDIR = ROOT / "runs" / "colab_multi"
REMOTE_ROOT = "/content/tesis_{run_id}"
REMOTE_ZIP = "/content/tesis_{run_id}.zip"
DEFAULT_CONFIG = ROOT / "configs" / "opt" / "benchmark.yaml"

# Seconds per objective evaluation per (grid cell x time step x scenario).
# Measured with Archive.evaluate on an 8-core x86 box: 80x40x100x40 scenarios
# -> 15-26 s. Well count also enters (2 evals/well/iteration in the pattern
# search) so treat the result as +/-50% and pass --eval-seconds to override.
EVAL_SECONDS_PER_CELL_STEP = 1.5e-6

BUNDLE_SKIP_DIRS = {
    ".git", "__pycache__", "out", "runs", "figs", "ZIPS", "True", ".venv",
    ".ipynb_checkpoints", "node_modules", ".mypy_cache", ".pytest_cache",
}
BUNDLE_SKIP_SUFFIXES = {".pyc", ".pyo", ".zip", ".log"}
TORCH_LINE = re.compile(r"^\s*(torch|torchvision|torchaudio|nvidia-|triton)\b", re.I)

_print_lock = threading.Lock()


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def log(msg: str) -> None:
    with _print_lock:
        print(f"[multi] {msg}", flush=True)


def log_replica(tag: str, msg: str) -> None:
    with _print_lock:
        print(f"[{tag}] {msg}", flush=True)


def die(msg: str, code: int = 2) -> None:
    print(f"[multi] error: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


# Every `colab` invocation authenticates through ADC unless the caller says
# otherwise (--auth oauth2|none) or exports COLAB_AUTH.
DEFAULT_AUTH = os.environ.get("COLAB_AUTH", "adc")


# --------------------------------------------------------------------------
# The `colab` CLI
# --------------------------------------------------------------------------
class ColabCLI:
    """Thin wrapper that builds (and optionally runs) `colab` invocations.

    Global options go *before* the subcommand, which is what the click-based
    CLI accepts everywhere.
    """

    def __init__(self, binary: str, auth: str | None = None,
                 dry_run: bool = False, verbose: bool = False) -> None:
        self.binary = binary
        self.auth = auth
        self.dry_run = dry_run
        self.verbose = verbose

    def argv(self, *parts: str) -> list[str]:
        cmd = [self.binary]
        if self.auth and self.auth != "none":
            cmd += ["--auth", self.auth]
        cmd += [str(p) for p in parts]
        return cmd

    def check(self) -> None:
        """Fail early with an actionable message when the CLI is missing."""
        if self.dry_run:
            return
        found = shutil.which(self.binary)
        if found is None:
            die(
                f"`{self.binary}` not found on PATH. Install the Colab CLI with\n"
                "    pip install google-colab-cli\n"
                "then provide ADC with `gcloud auth application-default login`\n"
                "(ADC is the default; --auth oauth2 uses `colab auth` instead,\n"
                "--auth none lets the CLI pick its own default).\n"
                "(colab_client_notes.txt mentions `google-colab-client`; that\n"
                " package name does not exist on PyPI.)"
            )
        self.binary = found

    def run(self, *parts: str, capture: bool = False,
            stdin: str | None = None) -> subprocess.CompletedProcess[str]:
        argv = self.argv(*parts)
        if self.verbose or self.dry_run:
            log(f"$ {shlex.join(argv)}")
        if self.dry_run:
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.run(
            argv, text=True, input=stdin,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            check=False,
        )

    def stream(self, tag: str, *parts: str, logfile: Path,
               timeout: float | None = None) -> int:
        """Run a CLI call and mirror its output into `logfile` line by line.

        `timeout` is a hard wall-clock cap on the CLI process (not on silence),
        so a wedged `colab exec` can never block the driver forever -- it is
        killed and reported as 124, the same convention `timeout(1)` uses.
        """
        argv = self.argv(*parts)
        log_replica(tag, f"$ {shlex.join(argv)}")
        if self.dry_run:
            with logfile.open("a", encoding="utf-8") as fh:
                fh.write(f"# dry-run: {shlex.join(argv)}\n")
            return 0
        with logfile.open("a", encoding="utf-8") as fh:
            fh.write(f"\n### {shlex.join(argv)}\n")
            fh.flush()
            proc = subprocess.Popen(
                argv, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, bufsize=1,
            )
            assert proc.stdout is not None
            timed_out = False
            deadline = time.monotonic() + timeout if timeout else None
            watcher = threading.Timer(
                max(1.0, timeout), proc.kill) if timeout else None
            if watcher is not None:
                watcher.daemon = True
                watcher.start()
            try:
                for line in proc.stdout:
                    fh.write(line)
                    fh.flush()
                    log_replica(tag, line.rstrip("\n"))
            finally:
                rc = proc.wait()
                if watcher is not None:
                    watcher.cancel()
                timed_out = deadline is not None and time.monotonic() >= deadline
            if timed_out:
                note = f"### killed after {timeout:.0f}s (CLI timeout)\n"
                fh.write(note)
                log_replica(tag, note.strip())
                return 124
            return rc

    def stream_retry(self, tag: str, *parts: str, logfile: Path,
                     timeout: float | None = None, marker: str | None = None,
                     attempts: int = 2, pause: float = 20.0) -> int:
        """`stream` with retries for the flaky kernel-start failure path.

        A freshly assigned Colab VM sometimes drops the websocket while the
        kernel boots ("Connection was lost"), which no amount of waiting
        locally prevents.  Both cells are idempotent (guarded unzip, pip
        re-install, deterministic outputs), so re-sending them is safe; a
        cell that did produce its marker is never retried.
        """
        rc = 1
        for attempt in range(1, max(1, attempts) + 1):
            rc = self.stream(tag, *parts, logfile=logfile, timeout=timeout)
            if rc == 0:
                return 0
            if marker and marker_exit(logfile, marker) is not None:
                return rc
            if attempt < attempts and not self.dry_run:
                log_replica(tag, f"retrying ({attempt}/{attempts - 1}) "
                                  f"after {pause:.0f}s -- rc={rc}")
                time.sleep(pause)
        return rc


# --------------------------------------------------------------------------
# Repo bundle
# --------------------------------------------------------------------------
# Accelerator variants `colab new --gpu` accepts (v0.8.0), plus the CPU spellings.
GPU_CHOICES = {"T4", "L4", "G4", "H100", "A100"}
CPU_CHOICES = {"", "NONE", "CPU"}


def normalize_gpu(gpu: str) -> str:
    """Map --gpu onto what `colab new --gpu` accepts; '' means a CPU runtime."""
    value = (gpu or "").strip()
    if value.upper() in CPU_CHOICES:
        return ""
    if value.upper() in GPU_CHOICES:
        return value.upper()
    die(f"--gpu {gpu!r} is not a Colab accelerator. Pick one of "
        f"{', '.join(sorted(GPU_CHOICES))}, or none/cpu for a CPU runtime.")
    return ""  # unreachable


def bundle_repo(dest_zip: Path) -> Path:
    """Zip the working tree, skipping artifacts that must not travel."""
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    if dest_zip.exists():
        dest_zip.unlink()
    count = 0
    with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(ROOT.rglob("*")):
            rel = path.relative_to(ROOT)
            if any(part in BUNDLE_SKIP_DIRS for part in rel.parts):
                continue
            if not path.is_file():
                continue
            if path.suffix in BUNDLE_SKIP_SUFFIXES or path.name.endswith(".zip"):
                continue
            if path.stat().st_size > 8 * 1024 * 1024:
                continue
            zf.write(path, rel)
            count += 1
    log(f"bundled {count} files -> {dest_zip} "
        f"({dest_zip.stat().st_size / 1e6:.1f} MB)")
    return dest_zip


def safe_requirements(dest: Path) -> Path:
    """Copy requirements.txt without torch lines (Colab ships a CUDA torch)."""
    src = ROOT / "requirements.txt"
    lines = src.read_text(encoding="utf-8").splitlines()
    kept = [ln for ln in lines if not TORCH_LINE.match(ln)]
    dest.write_text(
        "# generated by SCRIPTS/multi_colab.py: torch/torchvision lines stripped\n"
        + "\n".join(kept) + "\n",
        encoding="utf-8",
    )
    log(f"remote requirements -> {dest.name} "
        f"({len(kept)} lines, torch lines removed)")
    return dest


# --------------------------------------------------------------------------
# Config resolution + cost estimate
# --------------------------------------------------------------------------
def parse_sets(items: list[str]) -> dict[str, Any]:
    """Turn ``a.b.c=value`` into nested dict overrides (YAML-parsed value)."""
    import yaml

    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            die(f"--set wants KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip()
        if not key:
            die(f"--set wants KEY=VALUE, got {item!r}")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        cur = out
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
    return out


def deep_update(base: dict, upd: dict) -> dict:
    for key, value in upd.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def resolve_config(args: argparse.Namespace, overrides: dict[str, Any]) -> dict:
    """The effective config the remote run_optimization.py will use."""
    from stoch.config import load_config

    quick = args.preset in {"smoke", "quick"} or args.quick_opt
    cfg = load_config(args.opt_config, quick=quick)
    if args.nx is not None:
        cfg.setdefault("domain", {})["nx"] = int(args.nx)
    if args.ny is not None:
        cfg.setdefault("domain", {})["ny"] = int(args.ny)
    if args.nt is not None:
        cfg.setdefault("time", {})["nsteps"] = int(args.nt)
    deep_update(cfg, overrides)
    return cfg


def quick_clobbered(args: argparse.Namespace, overrides: dict[str, Any]) -> list[str]:
    """Keys apply_quick() would overwrite. Only relevant without --no-quick-opt."""
    if not (args.preset in {"smoke", "quick"} or args.quick_opt):
        return []
    watched = {("domain", "nx"), ("domain", "ny"), ("wells", "n"),
               ("time", "nsteps"), ("saa", "M_train"), ("saa", "M_test"),
               ("optim", "max_iter"), ("optim", "max_evals"), ("optim", "n_starts")}
    hits = []
    for section, key in sorted(watched):
        if key in overrides.get(section, {}):
            hits.append(f"{section}.{key}")
    return hits


def estimate_opt(cfg: dict, eval_seconds: float | None) -> dict[str, float]:
    """Worst-case wall clock for the opt stage, from the measured cost model."""
    d, t, s, o = cfg["domain"], cfg["time"], cfg["saa"], cfg["optim"]
    nx, ny = int(d["nx"]), int(d["ny"])
    nsteps = int(t["nsteps"])
    m_train = int(s["M_train"])
    n_starts = int(o.get("n_starts", 1))
    max_evals = int(o.get("max_evals", 2000))
    if eval_seconds is None:
        eval_seconds = EVAL_SECONDS_PER_CELL_STEP * nx * ny * nsteps * m_train
    evals = n_starts * max_evals
    return {
        "nx": nx, "ny": ny, "nsteps": nsteps, "m_train": m_train,
        "n_starts": n_starts, "max_evals": max_evals,
        "seconds_per_eval": eval_seconds,
        "evals": evals,
        "hours": eval_seconds * evals / 3600.0,
    }


def report_estimate(est: dict[str, float]) -> None:
    log("cost estimate (opt stage, worst case = full eval budget):")
    log(f"  grid {est['nx']:.0f}x{est['ny']:.0f}, nsteps={est['nsteps']:.0f}, "
        f"M_train={est['m_train']:.0f}")
    log(f"  {est['seconds_per_eval']:.1f} s per objective evaluation")
    log(f"  budget {est['n_starts']:.0f} start(s) x {est['max_evals']:.0f} evals "
        f"= {est['evals']:.0f} evals")
    log(f"  -> ~{est['hours']:.1f} h per replica  (+ ~1 min per-eval model error; "
        f"pattern search usually stops early)")


def estimate_guard(est: dict[str, float], max_hours: float,
                   argv0: str) -> None:
    if max_hours <= 0 or est["hours"] <= max_hours:
        return
    die(
        f"estimate {est['hours']:.1f} h per replica exceeds --max-hours "
        f"{max_hours:.1f}. A Colab VM is not a safe place for that.\n"
        "  cheap levers, in order of effect:\n"
        "    --set optim.max_evals=250      (per-start budget; 1500 -> 250 = 6x)\n"
        "    --set optim.method=slsqp       (adjoint: ~1 gradient/iter, not 18 evals)\n"
        "    --set saa.M_train=20           (scenarios per eval; linear)\n"
        "    --ny 20 --set time.nsteps=50   (grid x steps; roughly linear)\n"
        f"  or raise the ceiling: --max-hours {est['hours']:.1f}, or split with "
        "--replicas.\n"
        f"  (re-run with --estimate to see the new number: {argv0} launch ... --estimate)",
        code=2,
    )


# --------------------------------------------------------------------------
# Remote cells
# --------------------------------------------------------------------------
def collab_args(args: argparse.Namespace, remote_cfg: str | None,
                extra: list[str]) -> list[str]:
    """The `collab_run.py` argv executed on the VM."""
    remote_root = REMOTE_ROOT.format(run_id=args.run_id)
    cmd = [
        "--models", *args.models,
        "--preset", args.preset,
        "--out-root", f"{remote_root}/out",
        "--zip-results",
        "--results-zip", f"{remote_root}/artifacts.zip",
    ]
    if remote_cfg:
        cmd += ["--opt-config", remote_cfg]
    cmd += ["--no-quick-opt", "--opt-stage", args.opt_stage]
    for flag, value in (("--nx", args.nx), ("--ny", args.ny), ("--nt", args.nt),
                        ("--pinn-steps", args.pinn_steps),
                        ("--fno-epochs", args.fno_epochs)):
        if value is not None:
            cmd += [flag, str(value)]
    if "pinn" in args.models and args.ckpt_every:
        cmd += ["--ckpt-every", str(args.ckpt_every)]
    if args.hard_bc:
        cmd.append("--hard-bc")
    cmd += extra
    return cmd


MARKER_PROVISION = "[multi] provision exit="
MARKER_RUN = "[multi] run exit="
MARKER_DETACHED = "[multi] detached; job started"
MARKER_START = "[multi] start exit="


def render_provision(args: argparse.Namespace, req_name: str | None) -> str:
    remote_root = REMOTE_ROOT.format(run_id=args.run_id)
    remote_zip = REMOTE_ZIP.format(run_id=args.run_id)
    lines = [
        "%%bash",
        "set -euo pipefail",
        f"trap 'echo \"{MARKER_PROVISION}$?\"' EXIT",
        f"mkdir -p {remote_root}",
        f"cd {remote_root}",
        "if [ ! -d repo ]; then",
        f"  python3 -c \"import zipfile; "
        f"zipfile.ZipFile('{remote_zip}').extractall('repo')\"",
        "fi",
        "cd repo",
    ]
    if args.install == "none":
        lines.append("echo '[multi] install=none; skipping pip'")
    else:
        req = req_name or "requirements.txt"
        lines.append(f"python3 -m pip install -q --upgrade-strategy only-if-needed -r {req}")
        lines.append("python3 -c \"import numpy, scipy, yaml; print('[multi] deps ok')\"")
    lines += [
        "python3 -c \"import torch; print('[multi] torch', torch.__version__, "
        "'cuda', torch.cuda.is_available())\" || echo '[multi] torch missing'",
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader "
        "2>/dev/null || echo '[multi] no nvidia-smi (CPU runtime)'",
        "nproc",
        "echo '[multi] provision done'",
    ]
    return "\n".join(lines) + "\n"


def render_run(args: argparse.Namespace, run_argv: list[str],
               detached: bool, remote_cfg: str | None = None) -> str:
    remote_root = REMOTE_ROOT.format(run_id=args.run_id)
    inner = f"python3 -u SCRIPTS/collab_run.py {shlex.join(run_argv)}"
    guard = []
    if remote_cfg:
        # The config travels inside the repo zip; if it is missing the run
        # would silently use stoch.config defaults at the wrong grid.
        guard = [
            f"if [ ! -f {shlex.quote(remote_cfg)} ]; then",
            f"  echo '[multi] FATAL: {remote_cfg} is missing (bad upload?)'",
            "  exit 3",
            "fi",
        ]
    if detached:
        # Liveness risk: the kernel goes idle, so the backend may reclaim the
        # VM. Fine for sub-hour jobs; use the foreground default otherwise.
        background = (
            f"nohup bash -c '{inner} > {remote_root}/run.log 2>&1; "
            f"echo $? > {remote_root}/exit.code' >/dev/null 2>&1 &"
        )
        return "\n".join([
            "%%bash",
            "set -euo pipefail",
            f"cd {remote_root}/repo",
            "export PYTHONUNBUFFERED=1 MPLBACKEND=Agg",
            f"trap 'echo \"{MARKER_START}$?\"' EXIT",
            *guard,
            background,
            f"echo '{MARKER_DETACHED}'",
        ]) + "\n"
    # A `%%bash` cell runs as a single IPython cell: a nonzero script exit does
    # not make `colab exec` fail, so the exit status is echoed as a marker that
    # the driver parses out of the streamed log.
    return "\n".join([
        "%%bash",
        f"cd {remote_root}/repo",
        "export PYTHONUNBUFFERED=1 MPLBACKEND=Agg",
        "set -o pipefail",
        f"trap 'echo \"{MARKER_RUN}$?\"' EXIT",
        *guard,
        f"{inner} 2>&1 | tee {remote_root}/run.log",
    ]) + "\n"


# --------------------------------------------------------------------------
# Plan + orchestration
# --------------------------------------------------------------------------
def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def plan_path(outdir: Path, run_id: str) -> Path:
    return outdir / run_id / "plan.json"


def load_plan(outdir: Path, run_id: str | None) -> tuple[Path, dict]:
    if run_id is None:
        candidates = sorted(p for p in outdir.glob("*/plan.json"))
        if not candidates:
            die(f"no runs under {outdir}; launch one first")
        path = candidates[-1]
    else:
        path = plan_path(outdir, run_id)
        if not path.exists():
            die(f"no plan at {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def session_name(run_id: str, replica: int) -> str:
    """Short, shell-safe Colab session name (< 30 chars)."""
    compact = re.sub(r"[^A-Za-z0-9]", "", run_id)[-8:] or "run"
    return f"tesis-{compact}-r{replica}"


def human_size(path: Path) -> str:
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size / 1e6:.1f} MB"


def marker_exit(logfile: Path, marker: str) -> int | None:
    """Last ``<marker>N`` echoed by a remote cell, or None if it never ran.

    The cell echoes its own exit status because a `%%bash` cell is a normal
    IPython cell as far as `colab exec` is concerned: a failing shell script
    does not make the CLI return nonzero.
    """
    if not logfile.exists():
        return None
    for line in reversed(logfile.read_text(encoding="utf-8",
                                          errors="replace").splitlines()):
        if marker in line:
            status = re.search(r"-?\d+", line.split(marker, 1)[1])
            if status:
                return int(status.group(0))
    return None


def launch_one(cli: ColabCLI, plan: dict, replica: dict,
               keep: bool, reuse: bool, detached: bool,
               skip_provision: bool, timeouts: dict | None = None,
               attempts: int = 2) -> dict:
    tag = f"r{replica['index']}"
    session = replica["session"]
    rdir = Path(replica["local_dir"])
    logfile = rdir / "cli.log"
    remote_root = REMOTE_ROOT.format(run_id=plan["run_id"])
    remote_zip = REMOTE_ZIP.format(run_id=plan["run_id"])
    to = timeouts or {}
    state: dict[str, Any] = {"index": replica["index"], "session": session,
                             "state": "pending", "started": time.time()}
    try:
        if not reuse:
            gpu_args = ["--gpu", plan["gpu"]] if plan["gpu"] else []
            if plan.get("high_mem"):
                gpu_args.append("--high-mem")
            rc = cli.stream(tag, "new", "-s", session, *gpu_args, logfile=logfile,
                            timeout=to.get("new"))
            if rc != 0:
                state["state"] = "provision-failed"
                return state
        state["state"] = "provisioned"

        rc = cli.stream_retry(tag, "upload", "-s", session, plan["bundle"],
                              remote_zip, logfile=logfile,
                              timeout=to.get("upload"), attempts=attempts)
        if rc != 0:
            state["state"] = "upload-failed"
            return state

        if not skip_provision:
            cli.stream_retry(tag, "exec", "-s", session, "--timeout", "0", "-f",
                             str(rdir / "provision.py"), logfile=logfile,
                             timeout=to.get("provision"), marker=MARKER_PROVISION,
                             attempts=attempts)
            setup_exit = marker_exit(logfile, MARKER_PROVISION)
            if (setup_exit if setup_exit is not None else 1) != 0:
                state["state"] = "setup-failed"
                return state
        state["state"] = "running"

        # --timeout 0: `colab exec` gives up after 30 s *without output* by
        # default, and a single opt evaluation can be silent for minutes.
        #
        # The run cell's own output *is* the run log (collab_run.py streams
        # every iteration there), so keep it separate from the CLI chit-chat.
        rc = cli.stream_retry(tag, "exec", "-s", session, "--timeout", "0", "-f",
                              str(rdir / "run.py"), logfile=rdir / "run.log",
                              timeout=to.get("run"), marker=MARKER_RUN,
                              attempts=attempts)
        if detached:
            started = marker_exit(rdir / "run.log", MARKER_START)
            state["state"] = "detached" if (started in (None, 0)) else "start-failed"
            state["vm"] = "running"
            return state
        run_exit = marker_exit(rdir / "run.log", MARKER_RUN)
        if run_exit is None:
            run_exit = rc
        state["exit"] = run_exit
        state["state"] = "finished" if run_exit == 0 else "run-failed"

        if not plan.get("detached"):
            dl = rdir / "artifacts.zip"
            rc = cli.stream(tag, "download", "-s", session,
                            f"{remote_root}/artifacts.zip", str(dl),
                            logfile=logfile, timeout=to.get("download"))
            state["artifacts"] = str(dl) if rc == 0 and dl.exists() else None
            if rc != 0:
                state["artifacts"] = None
                log_replica(tag, "artifacts.zip not available (run failed early?)")

        cli.stream(tag, "log", "-s", session, "-o", str(rdir / "history.md"),
                   logfile=logfile, timeout=to.get("log"))
        return state
    except KeyboardInterrupt:
        state["state"] = "interrupted"
        return state
    finally:
        # The VM is released here, not after the happy path only: a failed
        # provision or a crashed run used to return early and leak the session
        # (it kept billing and showed up in `colab sessions`).
        state.setdefault("elapsed_h", (time.time() - state["started"]) / 3600.0)
        if keep:
            state["vm"] = "kept"
        elif state["state"] in {"pending", "provision-failed"}:
            state["vm"] = "not-created"
        elif detached and state["state"] == "detached":
            state["vm"] = "running"
        else:
            stop_rc = cli.stream(tag, "stop", "-s", session, logfile=logfile,
                                 timeout=to.get("stop"))
            state["vm"] = "stopped" if stop_rc == 0 else "stop-failed"
        # One line per replica: where it ran, how long, and what it produced.
        est = plan.get("estimate") or {}
        pcfg = plan.get("config") or {}
        log_runtime(
            Path(plan["outdir"]) / "runtimes.jsonl", "multi_colab.py", 
            state["elapsed_h"] * 3600.0,
            stage=plan.get("opt_stage"), preset=plan.get("preset"),
            gpu=plan.get("gpu"), replica=state["index"], session=session,
            models=" ".join(plan.get("models") or []),
            grid=[est.get("nx"), est.get("ny")], nsteps=est.get("nsteps"),
            m_train=est.get("m_train"),
            method=(pcfg.get("optim") or {}).get("method"),
            n_starts=(pcfg.get("optim") or {}).get("n_starts"),
            max_evals=(pcfg.get("optim") or {}).get("max_evals"),
            state=state["state"], vm=state.get("vm"), exit=state.get("exit"),
            status="ok" if state["state"] == "finished" else state["state"],
        )


def cmd_launch(args: argparse.Namespace, extra: list[str]) -> int:
    args.gpu = normalize_gpu(args.gpu)
    overrides = parse_sets(args.set)
    cfg = resolve_config(args, overrides)

    clobbered = quick_clobbered(args, overrides)
    if clobbered:
        log(f"warning: --quick (preset {args.preset}) overrides {', '.join(clobbered)}"
            " on the VM; the merged config below wins only because "
            "--no-quick-opt is passed. Double-check with --estimate.")

    needs_opt = "opt" in args.models
    est = estimate_opt(cfg, args.eval_seconds) if needs_opt else None
    if est:
        report_estimate(est)
    else:
        log("no cost model for this model set (only the opt stage is modelled "
            "from measurements); nothing to estimate")
    if args.estimate:
        log("stopping after the estimate (--estimate)")
        return 0
    if est and args.models == ["opt"] and args.gpu:
        log(f"note: the opt stage is CPU-bound, so --gpu {args.gpu} buys no speed-up "
            "(use --gpu none for a CPU runtime and save compute units)")
    if est:
        estimate_guard(est, args.max_hours, sys.argv[0])

    args.run_id = args.run_id or new_run_id()
    outdir = Path(args.outdir).resolve()
    rundir = outdir / args.run_id
    rundir.mkdir(parents=True, exist_ok=True)
    log(f"run id {args.run_id} -> {rundir}")

    remote_root = REMOTE_ROOT.format(run_id=args.run_id)

    # Per-replica configs come first: they have to travel *inside* the bundle,
    # because `colab exec -f` only transmits the cell file and nothing else.
    import yaml

    replicas = []
    for k in range(args.replicas):
        rdir = rundir / f"r{k}"
        rdir.mkdir(parents=True, exist_ok=True)
        replica_cfg = deep_update(
            json.loads(json.dumps(cfg)),
            {"optim": {"seed": args.seed_base + k}},
        )
        cfg_name = f"config_r{k}.yaml"
        cfg_local = rdir / cfg_name
        cfg_local.write_text(yaml.safe_dump(replica_cfg, sort_keys=False),
                             encoding="utf-8")
        replicas.append({
            "index": k,
            "session": session_name(args.run_id, k),
            "local_dir": str(rdir),
            "config": str(cfg_local),
            "config_name": cfg_name,
            "remote_config": f"{remote_root}/repo/{cfg_name}",
            "seed": args.seed_base + k,
        })

    bundle = bundle_repo(rundir / "repo.zip")
    extras = [Path(rep["config"]) for rep in replicas]
    req_name = None
    if args.install == "safe":
        req = safe_requirements(rundir / "requirements_remote.txt")
        req_name = req.name
        extras.append(req)
    elif args.install == "requirements":
        req_name = "requirements.txt"
    if extras:
        with zipfile.ZipFile(bundle, "a", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in extras:
                zf.write(path, path.name)
        log(f"added to the bundle: {', '.join(p.name for p in extras)}")

    for rep in replicas:
        rdir = Path(rep["local_dir"])
        run_argv = collab_args(args, rep["remote_config"] if needs_opt else None,
                               extra)
        rep["run_argv"] = run_argv
        (rdir / "run_argv.json").write_text(json.dumps(run_argv, indent=2) + "\n",
                                            encoding="utf-8")
        (rdir / "provision.py").write_text(render_provision(args, req_name),
                                          encoding="utf-8")
        (rdir / "run.py").write_text(
            render_run(args, run_argv, args.detach,
                       rep["remote_config"] if needs_opt else None),
            encoding="utf-8")

    plan = {
        "run_id": args.run_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": args.models,
        "preset": args.preset,
        "gpu": args.gpu,
        "high_mem": args.high_mem,
        "install": args.install,
        "detached": args.detach,
        "bundle": str(bundle),
        "estimate": est,
        "config": cfg,
        "replicas": replicas,
        "colab_bin": args.colab_bin,
        "auth": args.auth,
    }
    path = plan_path(outdir, args.run_id)
    path.write_text(json.dumps(plan, indent=2, sort_keys=False) + "\n",
                    encoding="utf-8")
    log(f"plan -> {path}")

    cli = ColabCLI(args.colab_bin, auth=args.auth, dry_run=args.dry_run,
                   verbose=args.verbose)
    cli.check()
    if args.dry_run:
        log("dry run: no VM is provisioned, nothing is uploaded")

    log(f"launching {len(replicas)} replica(s) on GPU={args.gpu or 'none'}, "
        f"install={args.install}")
    timeouts = {
        "new": args.cli_timeout,
        "upload": args.cli_timeout,
        "provision": args.provision_timeout,
        "run": args.run_timeout,
        "download": args.cli_timeout,
        "log": args.cli_timeout,
        "stop": args.cli_timeout,
    }
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.concurrency,
                                                   len(replicas)))) as pool:
        futures = [
            pool.submit(launch_one, cli, plan, rep, args.keep, args.reuse,
                        args.detach, args.skip_provision, timeouts,
                        args.exec_retries)
            for rep in replicas
        ]
        for fut in futures:
            results.append(fut.result())

    plan["results"] = results
    path.write_text(json.dumps(plan, indent=2, sort_keys=False) + "\n",
                    encoding="utf-8")

    print()
    log("summary")
    failures = 0
    for res in sorted(results, key=lambda r: r["index"]):
        rdir = Path(replicas[res["index"]]["local_dir"])
        art = rdir / "artifacts.zip"
        ok = res.get("state") in {"finished", "detached"}
        failures += 0 if ok else 1
        size = human_size(art) if art.exists() else "none"
        log(f"  r{res['index']} {res['session']}: {res.get('state')} "
            f"vm={res.get('vm', 'unknown')} "
            f"({res.get('elapsed_h', 0.0):.2f} h) artifacts={size}")
        if not ok:
            log(f"       log: {rdir / 'run.log'}")
    if failures:
        log(f"{failures} replica(s) did not finish; see {rundir}")
        return 1
    return 0


# --------------------------------------------------------------------------
# status / logs / collect / stop
# --------------------------------------------------------------------------
def cmd_status(args: argparse.Namespace) -> int:
    path, plan = load_plan(Path(args.outdir).resolve(), args.run_id)
    log(f"run {plan['run_id']} ({path})  created {plan['created']}")
    log(f"  models={','.join(plan['models'])} preset={plan['preset']} "
        f"gpu={plan['gpu'] or 'none'} install={plan['install']}")
    if plan.get("estimate"):
        e = plan["estimate"]
        log(f"  estimate ~{e['hours']:.1f} h/replica "
            f"({e['evals']:.0f} evals x {e['seconds_per_eval']:.1f} s)")
    cli = ColabCLI(plan.get("colab_bin", "colab"), auth=plan.get("auth"),
                   dry_run=args.dry_run, verbose=args.verbose)
    listed = ""
    if not args.dry_run:
        proc = cli.run("sessions", capture=True)
        listed = (proc.stdout or "") + (proc.stderr or "")
    for res in plan.get("results", []):
        rdir = Path(plan["replicas"][res["index"]]["local_dir"])
        art = rdir / "artifacts.zip"
        log(f"  r{res['index']} {res['session']}: {res.get('state')} "
            f"vm={res.get('vm', 'unknown')} "
            f"log={rdir / 'run.log'} artifacts={'yes' if art.exists() else 'no'}")
    if args.dry_run:
        log("$ " + shlex.join(cli.argv("sessions")))
    elif listed.strip():
        log("colab sessions:")
        for line in listed.strip().splitlines():
            log(f"  {line}")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    _, plan = load_plan(Path(args.outdir).resolve(), args.run_id)
    replicas = plan["replicas"]
    if args.replica is not None:
        replicas = [r for r in replicas if r["index"] == args.replica]
        if not replicas:
            die(f"no replica {args.replica}")
    paths = [Path(r["local_dir"]) / "run.log" for r in replicas]

    if plan.get("detached"):
        # Nothing was streamed home: ask the VM for the tail of its run.log.
        cli = ColabCLI(plan.get("colab_bin", "colab"), auth=plan.get("auth"),
                       dry_run=args.dry_run, verbose=args.verbose)
        remote_root = REMOTE_ROOT.format(run_id=plan["run_id"])
        t_start = time.time()
        while True:
            for rep in replicas:
                tag = f"r{rep['index']}"
                tail = cli.run("exec", "-s", rep["session"], capture=True,
                               stdin=f"!tail -n {args.lines} {remote_root}/run.log; "
                                     f"cat {remote_root}/exit.code 2>/dev/null\n")
                for line in ((tail.stdout or "") + (tail.stderr or "")).splitlines():
                    log_replica(tag, line)
            if not args.follow:
                return 0
            if args.follow_seconds and time.time() - t_start > args.follow_seconds:
                return 0
            time.sleep(args.interval)

    if not args.follow:
        for rep, path_ in zip(replicas, paths):
            log(f"{path_} ({'exists' if path_.exists() else 'missing'})")
            if path_.exists():
                tail = path_.read_text(encoding="utf-8",
                                       errors="replace").splitlines()[-args.lines:]
                for line in tail:
                    log_replica(f"r{rep['index']}", line)
        return 0

    # Follow: print only what is new in each replica log since the last poll.
    log(f"following {', '.join(str(p) for p in paths)} "
        f"(interval {args.interval}s, ctrl-c to stop)")
    offsets = {p: 0 for p in paths}
    t_start = time.time()
    try:
        while True:
            for rep, path_ in zip(replicas, paths):
                if not path_.exists():
                    continue
                with path_.open("rb") as fh:
                    fh.seek(offsets[path_])
                    chunk = fh.read().decode("utf-8", errors="replace")
                    offsets[path_] = fh.tell()
                for line in chunk.splitlines():
                    log_replica(f"r{rep['index']}", line)
            if args.follow_seconds and time.time() - t_start > args.follow_seconds:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log("stopped following")
        return 0


def cmd_collect(args: argparse.Namespace) -> int:
    _, plan = load_plan(Path(args.outdir).resolve(), args.run_id)
    cli = ColabCLI(plan.get("colab_bin", "colab"), auth=plan.get("auth"),
                   dry_run=args.dry_run, verbose=args.verbose)
    remote_root = REMOTE_ROOT.format(run_id=plan["run_id"])
    failures = 0
    for rep in plan["replicas"]:
        rdir = Path(rep["local_dir"])
        target = rdir / "artifacts.zip"
        rc = cli.stream(f"r{rep['index']}", "download", "-s", rep["session"],
                        f"{remote_root}/artifacts.zip", str(target),
                        logfile=rdir / "cli.log")
        ok = rc == 0 and target.exists()
        failures += 0 if ok else 1
        log(f"r{rep['index']} {rep['session']}: "
            f"{'collected' if ok else 'no artifacts'} -> {target}")
        cli.stream(f"r{rep['index']}", "log", "-s", rep["session"],
                   "-o", str(rdir / "history.md"), logfile=rdir / "cli.log")
    return 0 if failures == 0 else 1


def cmd_stop(args: argparse.Namespace) -> int:
    _, plan = load_plan(Path(args.outdir).resolve(), args.run_id)
    cli = ColabCLI(plan.get("colab_bin", "colab"), auth=plan.get("auth"),
                   dry_run=args.dry_run, verbose=args.verbose)
    for rep in plan["replicas"]:
        rdir = Path(rep["local_dir"])
        rc = cli.stream(f"r{rep['index']}", "stop", "-s", rep["session"],
                        logfile=rdir / "cli.log")
        log(f"r{rep['index']} {rep['session']}: stop rc={rc}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--outdir", default=str(DEFAULT_OUTDIR),
                       help="local root for plans, logs and artifacts "
                            "(default: runs/colab_multi)")
        p.add_argument("--run-id", default=None,
                       help="run id to act on (default: the newest run)")
        p.add_argument("--dry-run", action="store_true",
                       help="print the colab CLI calls instead of running them")
        p.add_argument("--verbose", action="store_true",
                       help="echo every CLI call")
        p.add_argument("--colab-bin", default="colab",
                       help="the colab CLI executable (default: colab)")
        p.add_argument("--auth", default=DEFAULT_AUTH,
                       help="colab global --auth value: adc (default, from "
                            "$COLAB_AUTH) | oauth2 | none")

    la = sub.add_parser("launch", help="plan, provision and run the replicas")
    common(la)
    la.add_argument("--models", nargs="+", default=["opt"],
                    choices=["smoke", "all", "dataset", "pinn", "fno", "both",
                             "opt", "tests"],
                    help="models to run on the VM (default: opt)")
    la.add_argument("--preset", choices=["smoke", "quick", "full"],
                    default="full", help="collab_run preset (default: full)")
    la.add_argument("--opt-stage", default="opt",
                    choices=["all", "gates", "o0", "o1", "o2", "o3", "o4", "opt"],
                    help="optimization stage (default: opt; use gates first)")
    la.add_argument("--opt-config", default=str(DEFAULT_CONFIG),
                    help="base optimization config to merge (default: %(default)s)")
    la.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted config override, repeatable "
                         "(e.g. --set optim.max_evals=250)")
    la.add_argument("--nx", type=int, default=None, help="domain.nx override")
    la.add_argument("--ny", type=int, default=None, help="domain.ny override")
    la.add_argument("--nt", type=int, default=None, help="time.nsteps override")
    la.add_argument("--pinn-steps", type=int, default=None)
    la.add_argument("--fno-epochs", type=int, default=None)
    la.add_argument("--ckpt-every", type=int, default=0,
                    help="PINN checkpoint frequency passed through")
    la.add_argument("--hard-bc", action="store_true")
    la.add_argument("--quick-opt", action="store_true",
                    help="bake --quick settings in (small grid, few scenarios)")
    la.add_argument("--replicas", type=int, default=1,
                    help="number of VMs / independent seeds (default: 1)")
    la.add_argument("--seed-base", type=int, default=0,
                    help="replica k gets optim.seed = seed_base + k")
    la.add_argument("--gpu", default="T4",
                    help="T4 | L4 | G4 | H100 | A100 | none (none = CPU runtime; "
                         "the opt stage is CPU-bound so a GPU buys nothing)")
    la.add_argument("--high-mem", action="store_true",
                    help="request a high-RAM machine shape (needs Colab Pro)")
    la.add_argument("--install", choices=["safe", "requirements", "none"],
                    default="safe",
                    help="remote pip install: safe = requirements minus torch "
                         "(default), requirements = the file as written, none")
    la.add_argument("--detach", action="store_true",
                    help="background the job on the VM and return (liveness risk: "
                         "idle kernels may be reclaimed; use for short jobs)")
    la.add_argument("--keep", action="store_true",
                    help="leave the VMs running after the run")
    la.add_argument("--reuse", action="store_true",
                    help="skip `colab new` (attach to existing sessions)")
    la.add_argument("--skip-provision", action="store_true",
                    help="skip the unpack+install cell (repo already in place)")
    la.add_argument("--concurrency", type=int, default=8,
                    help="how many VMs to drive at once (default: 8)")
    la.add_argument("--estimate", action="store_true",
                    help="print the cost estimate and stop")
    la.add_argument("--eval-seconds", type=float, default=None,
                    help="measured seconds per objective evaluation on your "
                         "hardware (overrides the built-in cost model)")
    la.add_argument("--cli-timeout", type=float, default=900.0,
                    help="hard cap in seconds for a short CLI call (new, "
                         "upload, download, log, stop); 0 disables it "
                         "(default: 900)")
    la.add_argument("--provision-timeout", type=float, default=1800.0,
                    help="hard cap for the provision cell, i.e. unzip + pip "
                         "(default: 1800)")
    la.add_argument("--run-timeout", type=float, default=6 * 3600.0,
                    help="hard cap for the run cell; a wedged run is killed "
                         "and the VM released (default: 21600 = 6 h; 0 = none)")
    la.add_argument("--exec-retries", type=int, default=2,
                    help="attempts per remote cell; a fresh Colab VM sometimes "
                         "drops the kernel websocket on first contact "
                         "(default: 2, i.e. one retry)")
    la.add_argument("--max-hours", type=float, default=4.0,
                    help="refuse to launch an estimated run longer than this "
                         "(0 disables the guard; default: 4)")

    st = sub.add_parser("status", help="show the plan and live colab sessions")
    common(st)

    lo = sub.add_parser("logs", help="print (or follow) the replica logs")
    common(lo)
    lo.add_argument("--replica", type=int, default=None)
    lo.add_argument("--lines", type=int, default=40)
    lo.add_argument("-f", "--follow", action="store_true")
    lo.add_argument("--follow-seconds", type=float, default=0.0,
                    help="stop following after N seconds (0 = forever)")
    lo.add_argument("--interval", type=float, default=5.0)

    co = sub.add_parser("collect", help="download artifacts from the VMs")
    common(co)

    sp = sub.add_parser("stop", help="stop the VM sessions")
    common(sp)

    return ap


def main() -> int:
    ap = build_parser()
    args, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]  # separator for the forwarded collab_run.py args
    if extra and args.command == "launch":
        # Everything unknown is forwarded verbatim to collab_run.py on the VM.
        log(f"forwarding to collab_run.py: {shlex.join(extra)}")
    elif extra:
        die(f"unexpected arguments: {shlex.join(extra)}")

    if args.command == "launch":
        if args.replicas < 1:
            die("--replicas must be >= 1")
        return cmd_launch(args, extra)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "logs":
        return cmd_logs(args)
    if args.command == "collect":
        return cmd_collect(args)
    if args.command == "stop":
        return cmd_stop(args)
    ap.error("unknown command")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[multi] interrupted", file=sys.stderr)
        sys.exit(130)
