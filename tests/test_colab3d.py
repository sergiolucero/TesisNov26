"""Tests for the Colab 3D driver's pure parts (no VM, no network).

The driver is the one piece of this repo that spends money when it is wrong, and
the piece it spends it on -- the chain that runs on the VM -- is the only code
that never executes locally.  So the two things tested hardest here are the two
that a typo would silently destroy: the bundle contents (a bundle missing the
hardening is a paid run of the bug you thought you fixed) and the chain source
(it has to parse, and it has to certify what it found).
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import zipfile
from pathlib import Path

import unittest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "colab3d_under_test", ROOT / "SCRIPTS" / "colab3d.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["colab3d_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


c3d = _load()

REMOTE = "/content/tesis3d/20260101-000000"
ARGV = ["python3", "-u", "full3D_colab_run.py", "--preset", "size_mid",
        "--backend", "torch", "--dtype", "float64",
        "--out", "out3d_colab/run_x", "--budget", "45", "--zip",
        "--m-train", "100", "--max-evals", "200"]


class TestChain(unittest.TestCase):
    def test_chain_is_valid_python(self):
        src = c3d.build_chain(REMOTE, ARGV, 0)
        ast.parse(src)  # raises if the generated source is broken

    def test_chain_runs_the_requested_argv_verbatim(self):
        src = c3d.build_chain(REMOTE, ARGV, 0)
        self.assertIn(repr(ARGV), src)
        self.assertIn("stage(argv, 'run.log')", src)

    def test_chain_certifies_only_when_asked(self):
        plain = c3d.build_chain(REMOTE, ARGV, 0)
        cert = c3d.build_chain(REMOTE, ARGV, 400)
        self.assertNotIn("check_tail3d", plain)
        self.assertIn("check_tail3d", cert)
        self.assertIn("--m-scen", cert)
        self.assertIn("fresh_%d.json", cert)
        ast.parse(cert)

    def test_chain_certificate_targets_the_runs_own_directory(self):
        src = c3d.build_chain(REMOTE, ARGV, 128)
        self.assertIn("out3d_colab/run_x", src)
        self.assertIn("'--run', run_out", src)

    def test_chain_certificate_runs_only_after_a_successful_search(self):
        src = c3d.build_chain(REMOTE, ARGV, 400)
        # if the search failed there is nothing to certify
        self.assertIn("if rc == 0", src)

    def test_chain_records_its_own_exit(self):
        src = c3d.build_chain(REMOTE, ARGV, 0)
        self.assertIn("chain done", src)


class TestKickCell(unittest.TestCase):
    def test_kick_cell_is_valid_python_and_detaches(self):
        chain = c3d.build_chain(REMOTE, ARGV, 0)
        cell = c3d.kick_cell(REMOTE, chain)
        ast.parse(cell)
        self.assertIn("start_new_session=True", cell)
        # writes the chain to the VM first: the cell itself must stay short
        self.assertIn("/chain.py", cell)

    def test_kick_cell_round_trips_the_chain_source(self):
        chain = c3d.build_chain(REMOTE, ARGV, 128)
        cell = c3d.kick_cell(REMOTE, chain)
        # the chain is embedded as a repr, so re-parsing it recovers the source
        tree = ast.parse(cell)
        assigned = [n for n in tree.body
                    if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", None) == "src" for t in n.targets)]
        self.assertEqual(len(assigned), 1)
        recovered = ast.literal_eval(assigned[0].value)
        self.assertEqual(recovered, chain)


class TestBundle(unittest.TestCase):
    def test_bundle_excludes_heavy_directories(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "repo.zip"
            n = c3d.build_bundle(dest)
            self.assertGreater(n, 20)
            names = set(zipfile.ZipFile(dest).namelist())

        for heavy in ("runs/", "out/", "out3d_colab/", ".git/"):
            self.assertFalse([x for x in names if x.startswith(heavy)],
                             f"{heavy} leaked into the bundle")
        self.assertFalse([x for x in names if x.endswith((".pyc", ".npz", ".pdf"))])

    def test_bundle_carries_the_pieces_the_vm_needs(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "repo.zip"
            c3d.build_bundle(dest)
            names = set(zipfile.ZipFile(dest).namelist())
        # the 3D runner, the certifier it chains into, and the movie module
        self.assertIn("full3D_colab_run.py", names)
        self.assertIn("SCRIPTS/check_tail3d.py", names)
        self.assertIn("SCRIPTS/plume_movie3d.py", names)

    def test_bundle_contains_the_hardening_it_is_supposed_to(self):
        """A patched working tree proves nothing about what the VM executes."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "repo.zip"
            c3d.build_bundle(dest)
            with zipfile.ZipFile(dest) as zf:
                src = zf.read("full3D_colab_run.py").decode()
        for marker in ("STALL_WINDOW", "finite_or_raise3d", "stalled",
                       "torch_default_rtol"):
            self.assertIn(marker, src)


class TestPlan(unittest.TestCase):
    def test_missing_plan_is_a_clean_refusal(self):
        with self.assertRaises(SystemExit):
            c3d._plan("19000101-000000")

    def test_plan_round_trips(self):
        run = "19000101-000001"
        d = c3d.LOCAL_RUNS / run
        d.mkdir(parents=True, exist_ok=True)
        try:
            (d / "plan.json").write_text(
                '{"run": "%s", "session": "s", "remote": "/content/x"}' % run)
            plan = c3d._plan(run)
            self.assertEqual(plan["session"], "s")
        finally:
            (d / "plan.json").unlink(missing_ok=True)
            d.rmdir()


if __name__ == "__main__":
    unittest.main()
