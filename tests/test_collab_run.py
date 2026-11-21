"""``--zip-results`` must survive the run it is archiving.

A two-hour Colab run crashed after completing one of its two starts; because the
zip was written only after a *successful* dispatch, nothing was uploaded and the
partial ``out/`` died with the terminated session.  These tests pin the new
contract: the artifacts are written even when the stage raises, and a zip
failure never replaces the error that caused it.
"""
from __future__ import annotations

import argparse
import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "SCRIPTS"))

import collab_run  # noqa: E402  (needs SCRIPTS/ on sys.path)


def _args(tmp: Path, zip_results: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        models=["opt"], preset="full", out_root=tmp / "out",
        results_zip=tmp / "artifacts.zip", zip_results=zip_results,
        device="cpu", gpu=None, nx=None, ny=None, nt=None,
    )


class TestZipResults(unittest.TestCase):
    def test_zip_carries_the_out_tree(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            args = _args(tmp)
            (args.out_root / "opt" / "full_opt").mkdir(parents=True)
            (args.out_root / "opt" / "full_opt" / "metrics.json").write_text("{}")
            collab_run.zip_results(args)
            with zipfile.ZipFile(args.results_zip) as zf:
                self.assertIn("out/opt/full_opt/metrics.json", zf.namelist())

    def test_zip_of_an_empty_out_root_still_writes_a_readable_archive(self):
        # the repo's figs/ is bundled too, so "empty" means "no out/ entries":
        # the point is that a run that produced nothing yields a valid zip the
        # driver can still download.
        with TemporaryDirectory() as td:
            args = _args(Path(td))
            collab_run.zip_results(args)
            self.assertTrue(zipfile.is_zipfile(args.results_zip))
            names = zipfile.ZipFile(args.results_zip).namelist()
            self.assertFalse([n for n in names if n.startswith("out/")])

    def test_zip_failure_is_reported_not_raised(self):
        with TemporaryDirectory() as td:
            args = _args(Path(td))
            # a file where the parent directory should be: mkdir raises
            blocked = Path(td) / "blocked"
            blocked.write_text("")
            args.results_zip = blocked / "artifacts.zip"
            collab_run.zip_results_safely(args)   # must not raise


class TestCrashingStageStillUploads(unittest.TestCase):
    def test_artifacts_are_written_when_the_stage_raises(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            args = _args(tmp)
            (args.out_root / "opt").mkdir(parents=True)
            (args.out_root / "opt" / "partial.json").write_text('{"start": 1}')

            def boom(_args):
                raise RuntimeError("objective is not finite (Krylov went NaN)")

            with mock.patch.dict(collab_run.DISPATCH, {"opt": boom}), \
                    mock.patch.object(collab_run, "RUNTIME_LOG", tmp / "rt.jsonl"):
                with self.assertRaises(RuntimeError):
                    collab_run.run_models(args)

            self.assertTrue(args.results_zip.is_file(),
                            "a crashing run must still upload what it computed")
            with zipfile.ZipFile(args.results_zip) as zf:
                self.assertIn("out/opt/partial.json", zf.namelist())

    def test_a_clean_run_zips_once_at_the_end(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            args = _args(tmp)

            def ok(_args):
                (_args.out_root / "opt").mkdir(parents=True, exist_ok=True)
                (_args.out_root / "opt" / "metrics.json").write_text("{}")

            with mock.patch.dict(collab_run.DISPATCH, {"opt": ok}), \
                    mock.patch.object(collab_run, "RUNTIME_LOG", tmp / "rt.jsonl"):
                collab_run.run_models(args)

            with zipfile.ZipFile(args.results_zip) as zf:
                self.assertIn("out/opt/metrics.json", zf.namelist())

    def test_no_zip_without_the_flag(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            args = _args(tmp, zip_results=False)

            def boom(_args):
                raise RuntimeError("boom")

            with mock.patch.dict(collab_run.DISPATCH, {"opt": boom}), \
                    mock.patch.object(collab_run, "RUNTIME_LOG", tmp / "rt.jsonl"):
                with self.assertRaises(RuntimeError):
                    collab_run.run_models(args)

            self.assertFalse(args.results_zip.exists())


if __name__ == "__main__":
    unittest.main()
