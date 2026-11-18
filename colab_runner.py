#!/usr/bin/env python3
"""Run any repo script on a Google Colab runtime and report what it produced.

This wraps the ``colab`` CLI from ``google-colab-client``::

    pip install google-colab-client
    pip install keyring
    colab sessions --auth oauth2          # one-time OAuth setup

Once authenticated you can hand it any script in this repository::

    python colab_runner.py solve_1d_pinn.py --gpu T4 -- --hard-bc --adam-steps 300
    python colab_runner.py hydrology.py --gpu L4 -- --nx 128 --nt 128
    python colab_runner.py collab_run.py --gpu A100 -- --models smoke --zip-results
    python colab_runner.py run_optimization.py --gpu T4 -- --stage gates --quick

The runner snapshots the repository file tree before and after ``colab run``
and prints exactly which files were generated or modified (so you know what
"the relevant results" are), writes a machine-readable manifest, and keeps a
copy of the remote stdout/stderr log.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Directories we never treat as "results" produced by the remote run.
DEFAULT_EXCLUDE = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".ipynb_checkpoints",
    ".idea",
    ".vscode",
}

# Runner bookkeeping lives under here and is never reported as remote output.
STATE_DIR = ROOT / "out" / "colab_client"


def log(msg: str) -> None:
    print(f"[colab_runner] {msg}", flush=True)


def find_colab() -> list[str]:
    """Locate the ``colab`` executable (or honor a COLAB_BIN command string)."""
    override = os.environ.get("COLAB_BIN")
    if override:
        return shlex.split(override)
    found = shutil.which("colab")
    if not found:
        sys.exit(
            "ERROR: 'colab' not found on PATH.\n"
            "Install and authenticate the Colab client first:\n"
            "    pip install google-colab-client\n"
            "    pip install keyring\n"
            "    colab sessions --auth oauth2\n"
            "Or set COLAB_BIN='/path/to/colab'."
        )
    return [found]


def snapshot(root: Path, exclude: set[str]) -> dict[str, tuple[int, int]]:
    """Map repo-relative path -> (size, mtime_ns) for every non-excluded file."""
    state: dict[str, tuple[int, int]] = {}
    for path in root.rglob("*"):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        if any(part in exclude for part in parts):
            continue
        if not path.is_file():
            continue
        try:
            st = path.stat()
            state[rel.as_posix()] = (st.st_size, st.st_mtime_ns)
        except OSError:
            continue
    return state


def diff(before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]) -> list[dict]:
    """Return new/changed/removed entries between two snapshots."""
    out: list[dict] = []
    for rel in sorted(set(after) - set(before)):
        size, _mtime = after[rel]
        out.append({"path": rel, "status": "new", "size_bytes": size})
    for rel in sorted(set(before) - set(after)):
        size, _mtime = before[rel]
        out.append({"path": rel, "status": "removed", "size_bytes": size})
    for rel in sorted(set(before) & set(after)):
        if before[rel] != after[rel]:
            out.append(
                {
                    "path": rel,
                    "status": "modified",
                    "size_bytes": after[rel][0],
                    "size_bytes_before": before[rel][0],
                }
            )
    return out


def build_command(colab: list[str], args: argparse.Namespace) -> list[str]:
    cmd = list(colab) + ["run"]
    if args.gpu:
        cmd += ["--gpu", args.gpu]
    cmd.append(args.script)
    cmd += args.script_args
    return cmd


def tee_run(cmd: list[str], log_path: Path) -> int:
    """Run ``cmd``, streaming stdout/stderr to both the terminal and ``log_path``."""
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"command: {' '.join(cmd)}\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        proc.wait()
        fh.write(f"\n[colab_runner] exit code: {proc.returncode}\n")
    return proc.returncode


def report(generated: list[dict], log_path: Path, manifest_path: Path) -> None:
    new = [g for g in generated if g["status"] == "new"]
    modified = [g for g in generated if g["status"] == "modified"]
    removed = [g for g in generated if g["status"] == "removed"]

    print()
    log("=" * 70)
    log(f"generated/removed files: {len(new)} new, {len(modified)} modified, {len(removed)} removed")
    log(f"run log:      {log_path}")
    log(f"manifest:     {manifest_path}")

    if new:
        print("\n  New files:")
        for g in new:
            print(f"    + {g['path']}  ({g['size_bytes']} B)")
    if modified:
        print("\n  Modified files:")
        for g in modified:
            print(f"    ~ {g['path']}  ({g['size_bytes_before']} -> {g['size_bytes']} B)")
    if removed:
        print("\n  Removed files:")
        for g in removed:
            print(f"    - {g['path']}")
    if not generated:
        print("\n  (no file changes detected)")
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # Everything after ``--`` is forwarded verbatim to the remote script;
    # everything before it belongs to this runner.
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        sep = argv.index("--")
        runner_argv, script_args = argv[:sep], argv[sep + 1 :]
    else:
        runner_argv, script_args = argv, []

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("script", help="Repo script to run on Colab (e.g. solve_1d_pinn.py).")
    ap.add_argument("--gpu", default=os.environ.get("COLAB_GPU", "T4"),
                    help="Colab accelerator: T4, L4, A100, V100, CPU, none, ... (default: T4)")
    ap.add_argument("--outdir", type=Path, default=STATE_DIR,
                    help="Runner bookkeeping dir for logs/manifests (default: out/colab_client)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Extra directory names to exclude from the generated-file report.")
    ap.add_argument("--no-manifest", action="store_true", help="Skip writing the JSON manifest.")
    args = ap.parse_args(runner_argv)
    args.script_args = script_args
    return args


def main() -> None:
    args = parse_args()
    colab = find_colab()

    script_path = Path(args.script)
    if not script_path.exists():
        sys.exit(f"ERROR: script not found: {args.script}")
    if script_path.resolve().parent == ROOT:
        rel_script = script_path.name
    else:
        rel_script = args.script

    args.outdir.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = args.outdir / "logs" / f"{Path(rel_script).stem}-{timestamp}.log"
    manifest_path = args.outdir / "manifests" / f"{Path(rel_script).stem}-{timestamp}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    exclude = DEFAULT_EXCLUDE | set(args.exclude) | {args.outdir.resolve().name}

    cmd = build_command(colab, args)
    log(f"$ {' '.join(cmd)}")
    log(f"snapshotting file tree before run (excluding {sorted(exclude)})")
    before = snapshot(ROOT, exclude)

    log("running on Colab (this may take a while)...")
    rc = tee_run(cmd, log_path)

    log("snapshotting file tree after run")
    after = snapshot(ROOT, exclude)

    generated = diff(before, after)

    manifest = {
        "timestamp": timestamp,
        "command": cmd,
        "exit_code": rc,
        "script": rel_script,
        "files": generated,
    }
    if not args.no_manifest:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report(generated, log_path, manifest_path)
    if rc != 0:
        log(f"WARNING: colab run exited with code {rc}")
        sys.exit(rc)


if __name__ == "__main__":
    main()
