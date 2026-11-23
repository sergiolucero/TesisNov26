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
def build_chain(remote: str, remote_argv: list, certify: int) -> str:
    """Source of the on-VM chain: run the search, then optionally certify.

    Prefer certify=0. The VM is reclaimed after ~40-50 min (four launches, four
    reclaims, none explained by the workload), so anything chained onto the end
    of the search is gambling with the artifact that already exists. A 400-field
    certificate costs ~13 min on 8 local cores and needs no GPU.

    Kept as a pure function so it can be parsed and inspected before a paid
    launch: the chain is the one piece of the driver that only ever executes on
    the VM, where a typo costs a rented GPU-hour.
    """
    out_idx = remote_argv.index("--out") + 1
    run_out = remote_argv[out_idx]
    lines = [
        "import subprocess",
        f"R = {remote!r}",
        "def stage(argv, logname):",
        "    with open(R + '/' + logname, 'w') as fh:",
        "        p = subprocess.Popen(argv, cwd=R + '/repo', stdout=fh,",
        "                             stderr=subprocess.STDOUT,",
        "                             start_new_session=True)",
        "        return p.wait()",
        "def main():",
        f"    argv = {remote_argv!r}",
        "    rc = stage(argv, 'run.log')",
        "    print('opt rc', rc, flush=True)",
    ]
    if certify and certify > 0:
        lines += [
            f"    if rc == 0 and {int(certify)} > 0:",
            f"        c = {int(certify)}",
            f"        run_out = {run_out!r}",
            "        cert = ['python3', '-u', 'SCRIPTS/check_tail3d.py',",
            "                '--run', run_out, '--m-scen', str(c), '--ladder',",
            "                '--json', run_out + '/cert/fresh_%d.json' % c]",
            "        print('cert rc', stage(cert, 'cert.log'), flush=True)",
        ]
    lines += ["    print('chain done', flush=True)", "main()", ""]
    return "\n".join(lines)


def kick_cell(remote: str, chain: str) -> str:
    """Cell that writes the chain to the VM and starts it detached.

    Detached on purpose: a cell that blocks for 30 minutes dies with the
    websocket, and the exec channel disconnects routinely.
    """
    return (
        "import subprocess\n"
        f"src = {chain!r}\n"
        f"open({remote!r} + '/chain.py', 'w').write(src)\n"
        f"logf = open({remote!r} + '/chain.log', 'w')\n"
        "p = subprocess.Popen(['python3', '-u', "
        f"{remote!r} + '/chain.py'], cwd={remote!r},\n"
        "                     stdout=logf, stderr=subprocess.STDOUT,\n"
        "                     start_new_session=True)\n"
        "print('kicked chain pid', p.pid)\n"
    )


def cmd_launch(args) -> None:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    session = args.session or f"tesis3d-{stamp[-6:]}"
    local = LOCAL_RUNS / stamp
    local.mkdir(parents=True, exist_ok=True)
    remote = f"{REMOTE_ROOT}/{stamp}"

    # Refuse to launch while somebody else's GPU is up: Colab allows two.
    # Not checked for a dry run -- it provisions nothing, so it must be usable
    # precisely when the cap is full.
    rc_out = None if args.dry_run else run_cli(["sessions"])
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
                        ("--krylov-maxiter", args.krylov_maxiter),
                        ("--seed", args.seed)):
        if value is not None:
            remote_argv += [flag, str(value)]
    remote_argv += list(args.extra or [])

    chain = build_chain(remote, remote_argv, int(args.certify or 0))
    kick = kick_cell(remote, chain)

    if args.dry_run:
        # A paid launch deserves a look before the meter starts.
        print("session:", session, "| gpu:", args.gpu, "| remote:", remote)
        print("argv:", " ".join(remote_argv))
        print("--- chain.py that would run on the VM ---")
        print(chain)
        print("dry run: no VM is provisioned")
        return

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
    cert_files = []
    if args.certify and args.certify > 0:
        for ext in ("json", "log"):
            cert_files.append(f"cert/fresh_{args.certify}.{ext}")
    for name in ("metrics.json", "policy.json", "run_config.json",
                 "colab3d_results.zip", *cert_files):
        dest = local / Path(name).name if "/" in name else local / name
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
    lp.add_argument("--seed", type=int, default=None,
                    help="optim.seed: with n_starts=1 this alone changes the "
                         "search trajectory (start k=0 is deterministic)")
    lp.add_argument("--certify", type=int, default=0,
                    help="after the search, run check_tail3d.py on this many "
                         "FRESH scenarios on the VM. Usually the wrong trade: "
                         "the VM lives ~40-50 min and the certificate needs no "
                         "GPU -- collect the artifacts, then certify locally")
    lp.add_argument("--cli-timeout", type=int, default=900)
    lp.add_argument("--dry-run", action="store_true",
                    help="print the remote command and the chain, create nothing")
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
    cp.add_argument("--certify", type=int, default=0,
                    help="also fetch cert/fresh_<N>.json produced by --certify")
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
