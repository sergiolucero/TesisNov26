#!/usr/bin/env python3
"""Drive a 3D GPU run on Colab: bundle, launch detached, poll, collect, stop.

`colab_runner.py` runs a script with `colab run`, which is fine for short jobs but
wrong for a multi-hour 3D run: if the client disconnects the remote process dies
with it, and there is no session left to fetch results from.  This driver uses the
pattern that survived every failure mode seen on this project:

    launch   create the VM, upload a flat bundle, extract, kick with nohup
    status   download the remote run.log and show the tail
    collect  download the result directory / zip
    stop     terminate the VM (never leave one billing)

Everything works through the REST file API where possible (`colab download`),
because the exec channel is the flaky one -- it can die for hours while the VM is
alive and `colab download` still works.

Typical use::

    export COLAB_AUTH=oauth2          # 'adc' raises DefaultCredentialsError
    python3 SCRIPTS/colab3d.py launch --preset size_mid --gpu T4 \\
        --m-train 100 --max-evals 300 --budget 75 --dtype float64
    python3 SCRIPTS/colab3d.py status --run 20261122-101500
    python3 SCRIPTS/colab3d.py collect --run 20261122-101500
    python3 SCRIPTS/colab3d.py stop --run 20261122-101500
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_RUNS = ROOT / "runs" / "colab3d"
REMOTE_ROOT = "/content/tesis3d"
EXCLUDE_DIRS = {".git", "runs", "out", "out3d_colab", "__pycache__", "figdata",
                "figures", "TEX", "data", "site", "assets", "templates"}
EXCLUDE_SUFFIX = {".pyc", ".npz", ".png", ".pdf", ".zip", ".mp4", ".gif", ".pt"}


def log(msg: str) -> None:
    print(f"[3dgpu] {msg}", flush=True)


def colab_bin() -> str:
    for cand in (os.environ.get("COLAB_BIN"), shutil.which("colab"),
                 str(Path.home() / ".local" / "bin" / "colab")):
        if cand and Path(cand).exists():
            return cand
    sys.exit("colab CLI not found (pip install google-colab-cli)")


def auth_args() -> list[str]:
    """oauth2, not adc: `adc` now raises DefaultCredentialsError on this account."""
    return ["--auth", os.environ.get("COLAB_AUTH", "oauth2")]


def exec_cell(session: str, code: str, timeout: int = 600, wait: int = 240):
    """`colab exec` takes a FILE, not inline code, hence the temp file."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code)
        path = fh.name
    try:
        return run_cli(["exec", "-s", session, "-f", path, "--timeout", str(wait)],
                       timeout=timeout)
    finally:
        Path(path).unlink(missing_ok=True)


def run_cli(args: list[str], timeout: int | None = 900, check: bool = False):
    cmd = [colab_bin(), *auth_args(), *args]
    log("$ " + " ".join(shlex.quote(c) for c in cmd))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"! timeout after {timeout}s")
        return None
    out = (res.stdout or "") + (res.stderr or "")
    if check and res.returncode != 0:
        raise SystemExit(f"command failed (rc={res.returncode}):\n{out[-3000:]}")
    return res.returncode, out


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #
def build_bundle(path: Path) -> int:
    n = 0
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in ROOT.rglob("*"):
            rel = item.relative_to(ROOT)
            if any(part in EXCLUDE_DIRS for part in rel.parts[:-1]):
                continue
            if rel.parts and rel.parts[0] in EXCLUDE_DIRS:
                continue
            if item.is_dir() or item.suffix in EXCLUDE_SUFFIX:
                continue
            zf.write(item, rel.as_posix())
            n += 1
    return n


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_launch(args) -> None:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    session = args.session or f"tesis3d-{stamp[-6:]}"
    local = LOCAL_RUNS / stamp
    local.mkdir(parents=True, exist_ok=True)
    remote = f"{REMOTE_ROOT}/{stamp}"

    # Refuse to launch while somebody else's GPU is up: Colab allows two.
    rc_out = run_cli(["sessions"])
    if rc_out and "No active sessions" not in rc_out[1]:
        log("! an active Colab session exists:")
        log(rc_out[1].strip()[:800])
        if not args.force:
            sys.exit("refusing to launch (use --force to override; two is the cap)")

    bundle = local / "repo.zip"
    n = build_bundle(bundle)
    log(f"bundle {n} files, {bundle.stat().st_size / 1e6:.1f} MB -> {bundle}")

    remote_argv = [
        "python3", "-u", "full3D_colab_run.py",
        "--preset", args.preset,
        "--backend", "torch",
        "--device", "cuda",
        "--dtype", args.dtype,
        "--out", f"out3d_colab/{args.name}",
        "--budget", str(args.budget),
        "--zip",
    ]
    for flag, value in (("--m-train", args.m_train), ("--m-test", args.m_test),
                        ("--max-evals", args.max_evals),
                        ("--n-starts", args.n_starts),
                        ("--krylov-rtol", args.krylov_rtol),
                        ("--krylov-maxiter", args.krylov_maxiter)):
        if value is not None:
            remote_argv += [flag, str(value)]
    remote_argv += list(args.extra or [])

    log(f"launching session {session} gpu={args.gpu} high_mem={args.high_mem}")
    run_cli(["new", "-s", session, "--gpu", args.gpu]
            + (["--high-mem"] if args.high_mem else []),
            timeout=args.cli_timeout, check=True)

    # `colab upload` cannot create intermediate directories (it answers 500),
    # so the remote tree has to exist first.
    exec_cell(session, "import os\n"
                       f"os.makedirs({remote!r}, exist_ok=True)\n"
                       f"print('remote ready', os.path.isdir({remote!r}))",
              timeout=args.cli_timeout, wait=120)

    run_cli(["upload", "-s", session, str(bundle), f"{remote}/repo.zip"],
            timeout=args.cli_timeout, check=True)

    extract = (
        "import zipfile, pathlib\n"
        f"p = pathlib.Path({remote!r} + '/repo'); p.mkdir(parents=True, exist_ok=True)\n"
        f"zipfile.ZipFile({remote!r} + '/repo.zip').extractall(p)\n"
        "src = (p / 'full3D_colab_run.py').read_text()\n"
        # bundle self-check: a patched working tree proves nothing about the VM
        "print('files', len(list(p.rglob('*.py'))), "
        "'hardened', 'STALL_WINDOW' in src and 'finite_or_raise3d' in src)\n"
        "import torch\n"
        "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\n"
    )
    rc = exec_cell(session, extract, timeout=args.cli_timeout, wait=240)
    if rc:
        log(rc[1].strip()[-400:])

    # Kick detached: the exec channel will not survive a multi-hour cell, and a
    # disconnect must not kill the run.
    kick = (
        "import subprocess\n"
        f"argv = {remote_argv!r}\n"
        f"logf = open({remote!r} + '/run.log', 'w')\n"
        f"p = subprocess.Popen(argv, cwd={remote!r} + '/repo', stdout=logf,\n"
        "                     stderr=subprocess.STDOUT, start_new_session=True)\n"
        "print('kicked pid', p.pid)\n"
    )
    rc = exec_cell(session, kick, timeout=300, wait=120)
    if rc:
        log(rc[1].strip()[-300:])

    plan = {"run": stamp, "session": session, "local": str(local),
            "remote": remote, "remote_argv": remote_argv, "gpu": args.gpu,
            "dtype": args.dtype, "budget_min": args.budget,
            "launched": dt.datetime.now().isoformat(timespec="seconds")}
    (local / "plan.json").write_text(json.dumps(plan, indent=2))
    log(f"launched. poll with:  python3 SCRIPTS/colab3d.py status --run {stamp}")
    log(f"results land in {remote}/repo/out3d_colab/{args.name}")


def _plan(run: str) -> dict:
    p = LOCAL_RUNS / run / "plan.json"
    if not p.exists():
        sys.exit(f"no plan.json for run {run} (looked in {p})")
    return json.loads(p.read_text())


def cmd_status(args) -> None:
    plan = _plan(args.run)
    out = LOCAL_RUNS / args.run / "run.log"
    rc = run_cli(["download", "-s", plan["session"],
                  f"{plan['remote']}/run.log", str(out)], timeout=300)
    if not rc or rc[0] != 0:
        log("download failed (VM stopped, or the log does not exist yet)")
        if out.exists():
            log(f"showing the previous copy from {out} (mtime "
                f"{dt.datetime.fromtimestamp(out.stat().st_mtime)})")
        else:
            return
    lines = out.read_text(errors="replace").splitlines()
    log(f"{len(lines)} lines in run.log")
    for line in lines[-args.tail:]:
        print(line)


def cmd_collect(args) -> None:
    plan = _plan(args.run)
    local = Path(plan["local"])
    # The run dir name is in the remote argv (--out <path>).
    argv = plan["remote_argv"]
    out_idx = argv.index("--out") + 1
    remote_run_dir = f"{plan['remote']}/repo/{argv[out_idx]}"
    got = []
    for name in ("metrics.json", "policy.json", "run_config.json",
                 "colab3d_results.zip"):
        dest = local / name
        rc = run_cli(["download", "-s", plan["session"],
                      f"{remote_run_dir}/{name}", str(dest)], timeout=900)
        if rc and rc[0] == 0 and dest.exists() and dest.stat().st_size > 0:
            got.append((name, dest.stat().st_size))
    log(f"collected {[n for n, _ in got] or 'nothing'}")
    if any(n == "colab3d_results.zip" for n, _ in got):
        with zipfile.ZipFile(local / "colab3d_results.zip") as zf:
            zf.extractall(local / "artifacts")
        log(f"unpacked -> {local / 'artifacts'}")


def cmd_stop(args) -> None:
    plan = _plan(args.run)
    run_cli(["stop", "-s", plan["session"]], timeout=300)
    log(f"stop issued for {plan['session']}; verify with: colab sessions")


def cmd_sessions(args) -> None:
    rc = run_cli(["sessions"])
    if rc:
        print(rc[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("launch", help="create the VM and start the run")
    lp.add_argument("--preset", default="size_mid")
    lp.add_argument("--gpu", default="T4", help="T4/L4/G4/A100/H100")
    lp.add_argument("--high-mem", action="store_true")
    lp.add_argument("--session", default=None)
    lp.add_argument("--name", default=None,
                    help="remote out dir name (default: preset + timestamp)")
    lp.add_argument("--budget", type=float, default=75.0,
                    help="optimizer wall-clock budget in minutes")
    lp.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    lp.add_argument("--m-train", type=int, default=None)
    lp.add_argument("--m-test", type=int, default=None)
    lp.add_argument("--max-evals", type=int, default=None)
    lp.add_argument("--n-starts", type=int, default=None)
    lp.add_argument("--krylov-rtol", type=float, default=None)
    lp.add_argument("--krylov-maxiter", type=int, default=None)
    lp.add_argument("--cli-timeout", type=int, default=900)
    lp.add_argument("--force", action="store_true",
                    help="launch even if another session appears active")
    lp.add_argument("extra", nargs="*", help="extra args forwarded to the 3D runner")
    lp.set_defaults(func=cmd_launch)

    sp = sub.add_parser("status", help="download and tail the remote run.log")
    sp.add_argument("--run", required=True)
    sp.add_argument("--tail", type=int, default=25)
    sp.set_defaults(func=cmd_status)

    cp = sub.add_parser("collect", help="download metrics/policy/zip")
    cp.add_argument("--run", required=True)
    cp.set_defaults(func=cmd_collect)

    tp = sub.add_parser("stop", help="terminate the VM")
    tp.add_argument("--run", required=True)
    tp.set_defaults(func=cmd_stop)

    qp = sub.add_parser("sessions", help="list active Colab sessions")
    qp.set_defaults(func=cmd_sessions)

    args = ap.parse_args()
    if args.cmd == "launch" and not args.name:
        args.name = f"{args.preset}_{dt.datetime.now().strftime('%m%d-%H%M')}"
    args.func(args)


if __name__ == "__main__":
    main()
