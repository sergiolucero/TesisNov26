"""Tests for SCRIPTS/colab_status.py (the /runs page data source).

The page is only useful if it can tell a finished run from a running one from
a crashed one without asking Colab.  These tests pin that inference down with
synthetic run directories: an explicit result wins, then an exit marker, then
the last CLI stage plus the age of the newest log line.
"""

import importlib.util
import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "colab_status", ROOT / "SCRIPTS" / "colab_status.py")
colab_status = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(colab_status)


CONFIG = {
    "domain": {"nx": 80, "ny": 40},
    "time": {"nsteps": 100},
    "optim": {"n_starts": 1, "max_iter": 20, "max_evals": 50},
}


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class StatusTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.multi = self.root / "runs" / "colab_multi"
        self.high = self.root / "out" / "high"
        self.multi.mkdir(parents=True)
        self.high.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _multi(self, run_id="20260101-120000", **plan_overrides):
        run_dir = self.multi / run_id
        rdir = run_dir / "r0"
        rdir.mkdir(parents=True)
        plan = {
            "run_id": run_id,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "models": ["opt"],
            "preset": "full",
            "gpu": "T4",
            "config": CONFIG,
            "estimate": {"hours": 1.5},
            "replicas": [{
                "index": 0,
                "session": "tesis-test-r0",
                "local_dir": str(rdir),
            }],
        }
        plan.update(plan_overrides)
        _write(run_dir / "plan.json",
               json.dumps(plan, indent=2) + "\n")
        return run_dir, rdir

    def _snapshot(self, **kwargs):
        return colab_status.collect_runs(
            root=self.root, limit=kwargs.pop("limit", 10),
            multi_dir=self.multi, high_dir=self.high, **kwargs)


class TestMultiRuns(StatusTestCase):
    def test_finished_result_wins_and_reports_progress(self):
        run_dir, rdir = self._multi(results=[{
            "index": 0, "session": "tesis-test-r0", "state": "finished",
            "exit": 0, "vm": "stopped", "elapsed_h": 0.5,
        }])
        _write(rdir / "run.log",
               "[opt] 10:00:00 opt start 1/1\n"
               "[opt] 10:00:05   iter   3 evals=   7 best=1.0\n"
               "[multi] run exit=0\n")
        (rdir / "artifacts.zip").write_bytes(b"PK\x03\x04test")

        snap = self._snapshot()
        run = snap["runs"][0]
        self.assertEqual(run["state"], "finished")
        self.assertFalse(run["is_current"])
        self.assertEqual(run["elapsed_text"], "30m")
        self.assertEqual(run["replicas"][0]["progress"]["evals"], 7)
        self.assertEqual(run["replicas"][0]["progress"]["max_evals"], 50)
        self.assertEqual(run["replicas"][0]["progress"]["pct"], 14.0)
        self.assertTrue(run["replicas"][0]["artifacts"]["present"])
        self.assertEqual(run["work"], "opt · full · 80×40×100")

    def test_running_run_log_beats_a_stale_cli_provision_stage(self):
        # The run cell's `### ... exec ... run.py` marker goes to run.log, so
        # the last cli.log stage can still read "provision" after the run has
        # started.  run.log is the evidence that wins.
        run_dir, rdir = self._multi()
        _write(rdir / "cli.log",
               "### colab exec -s tesis-test-r0 --timeout 0 -f "
               f"{rdir / 'provision.py'}\n[multi] provision exit=0\n")
        _write(rdir / "run.log",
               f"### colab exec -s tesis-test-r0 --timeout 0 -f "
               f"{rdir / 'run.py'}\n"
               "[opt] 10:00:00 stage opt: method=pattern\n"
               "[opt] 10:00:05   iter   2 evals=  12 best=1.0\n")

        snap = self._snapshot()
        run = snap["runs"][0]
        self.assertEqual(run["state"], "running")
        self.assertTrue(run["is_current"])
        self.assertTrue(snap["has_current"])
        self.assertEqual(snap["counts"]["running"], 1)
        self.assertEqual(run["replicas"][0]["progress"]["pct"], 24.0)

    def test_failure_in_run_log_is_detected_even_when_cli_says_provision(self):
        run_dir, rdir = self._multi()
        _write(rdir / "cli.log",
               "### colab exec -s tesis-test-r0 --timeout 0 -f "
               f"{rdir / 'provision.py'}\n[multi] provision exit=0\n")
        _write(rdir / "run.log",
               f"### colab exec -s tesis-test-r0 --timeout 0 -f "
               f"{rdir / 'run.py'}\n"
               "Traceback (most recent call last):\n"
               'RuntimeError: Connection was lost.\n')
        snap = self._snapshot()
        run = snap["runs"][0]
        self.assertEqual(run["state"], "failed")
        self.assertIn("Connection was lost", run["error"])

    def test_running_log_that_has_gone_quiet_is_not_called_running(self):
        run_dir, rdir = self._multi()
        _write(rdir / "cli.log",
               "### colab exec -s tesis-test-r0 --timeout 0 -f "
               f"{rdir / 'provision.py'}\n[multi] provision exit=0\n")
        log = _write(rdir / "run.log", "[opt] 10:00:00 still here\n")
        old = time.time() - colab_status.QUIET_SECONDS - 60
        for path in (rdir / "run.log", rdir / "cli.log", log):
            path.touch()
            os.utime(path, (old, old))

        snap = self._snapshot()
        self.assertEqual(snap["runs"][0]["state"], "quiet")

    def test_failed_result_maps_to_failed(self):
        run_dir, rdir = self._multi(results=[{
            "index": 0, "state": "setup-failed", "vm": "stopped",
        }])
        _write(rdir / "cli.log", "### colab exec -f provision.py\n"
                                 "ModuleNotFoundError: no yaml\n")
        snap = self._snapshot()
        run = snap["runs"][0]
        self.assertEqual(run["state"], "failed")
        self.assertEqual(run["tone"], "fail")
        self.assertTrue(run["error"])

    def test_dry_run_plan_is_not_reported_as_live_work(self):
        run_dir, rdir = self._multi()
        _write(rdir / "cli.log",
               "# dry-run: colab new -s tesis-test-r0 --gpu T4\n"
               "# dry-run: colab stop -s tesis-test-r0\n")
        snap = self._snapshot()
        run = snap["runs"][0]
        self.assertEqual(run["state"], "dry-run")
        self.assertEqual(run["elapsed_text"], "—")


class TestHighRuns(StatusTestCase):
    def _high(self, run_id="20260101120000", run_exit=None,
              connection_lost=False, dry_run=False):
        rdir = self.high / run_id
        rdir.mkdir(parents=True)
        _write(rdir / "run.py",
               "%%bash\npython3 -u SCRIPTS/new_high_colab.py opt "
               "--preset paper --nx 80 --ny 40 --zip\n")
        lines = [f"### /usr/bin/colab new -s tesis-a100 --gpu A100 --high-mem",
                 "[colab] Session READY.",
                 f"### /usr/bin/colab exec -s tesis-a100 --timeout 0 -f "
                 f"{rdir / 'run.py'}"]
        if run_exit is not None:
            lines.append(f"[high] run exit={run_exit}")
        if connection_lost:
            lines += ["Traceback (most recent call last):",
                      'RuntimeError: Connection was lost.']
        if dry_run:
            lines = ["# dry-run: " + line[4:] for line in lines]
        _write(rdir / "cli.log", "\n".join(lines) + "\n")
        if run_exit == 0:
            (rdir / "artifacts.zip").write_bytes(b"PK\x03\x04test")
        return rdir

    def test_finished_high_run(self):
        rdir = self._high(run_exit=0)
        snap = self._snapshot()
        run = snap["runs"][0]
        self.assertEqual(run["kind"], "high")
        self.assertEqual(run["state"], "finished")
        self.assertEqual(run["gpu"], "A100")
        self.assertEqual(run["replicas"][0]["session"], "tesis-a100")
        self.assertIn("paper", run["work"])
        self.assertTrue(run["replicas"][0]["artifacts"]["present"])

    def test_connection_loss_is_a_failure_not_a_running_run(self):
        self._high(connection_lost=True)
        snap = self._snapshot()
        run = snap["runs"][0]
        self.assertEqual(run["state"], "failed")
        self.assertIn("Connection was lost", run["error"])

    def test_high_dry_run(self):
        self._high(dry_run=True)
        snap = self._snapshot()
        self.assertEqual(snap["runs"][0]["state"], "dry-run")
        self.assertEqual(snap["runs"][0]["elapsed_text"], "—")


class TestLogAccess(StatusTestCase):
    def test_log_text_returns_tail_for_a_known_run(self):
        run_dir, rdir = self._multi(results=[{
            "index": 0, "state": "finished", "exit": 0, "vm": "stopped",
        }])
        _write(rdir / "run.log", "line one\nline two\n")
        found = colab_status.log_text("multi", run_dir.name, 0,
                                      root=self.root)
        self.assertIsNotNone(found)
        text, rel = found
        self.assertIn("line two", text)
        self.assertTrue(rel.endswith("run.log"))

    def test_unknown_run_and_bad_kind_are_refused(self):
        self.assertIsNone(colab_status.log_text("multi", "../etc", 0,
                                                root=self.root))
        self.assertIsNone(colab_status.log_text("nope", "x", 0,
                                                root=self.root))

    def test_missing_colab_binary_is_reported_without_raising(self):
        result = colab_status.live_sessions("definitely-not-a-colab-binary",
                                            timeout=0.2)
        self.assertFalse(result["ok"])
        self.assertIn("not on PATH", result["error"])

    def test_log_text_remote_serves_the_vm_log_for_a_live_run(self):
        run_dir, rdir = self._multi()
        _write(rdir / "run.log",
               "### colab exec -s tesis-test-r0 --timeout 0 -f run.py\n")
        remote = ("[opt] 10:00:00 opt start 1/1\n"
                  "[opt] 10:00:05   iter   4 evals=  19 best=1.0\n")
        with mock.patch.object(colab_status, "remote_log_text",
                               return_value=(remote, "")) as pull:
            found = colab_status.log_text("multi", run_dir.name, 0,
                                          root=self.root, remote=True)
        self.assertIsNotNone(found)
        text, rel = found
        pull.assert_called_once()
        self.assertIn("iter   4", text)
        self.assertIn("(remote)", rel)

    def test_log_text_remote_falls_back_to_the_local_copy(self):
        run_dir, rdir = self._multi()
        _write(rdir / "run.log",
               "### colab exec -s tesis-test-r0 --timeout 0 -f run.py\n")
        with mock.patch.object(colab_status, "remote_log_text",
                               return_value=(None, "session is gone")):
            found = colab_status.log_text("multi", run_dir.name, 0,
                                          root=self.root, remote=True)
        self.assertIsNotNone(found)
        text, rel = found
        self.assertIn("remote log unavailable", text)
        self.assertIn("session is gone", text)
        self.assertTrue(rel.endswith("run.log"))

    def test_remote_log_text_reports_a_missing_binary(self):
        text, err = colab_status.remote_log_text(
            "tesis-test-r0", "/content/tesis-test/run.log",
            binary="definitely-not-a-colab-binary")
        self.assertIsNone(text)
        self.assertIn("not on PATH", err)


class TestRemoteLogs(StatusTestCase):
    def _frozen_run(self):
        """A live run as it looks locally: only the streaming exec's header."""
        run_dir, rdir = self._multi()
        _write(rdir / "cli.log",
               "### colab exec -s tesis-test-r0 --timeout 0 -f "
               f"{rdir / 'provision.py'}\n[multi] provision exit=0\n")
        _write(rdir / "run.log",
               "### colab exec -s tesis-test-r0 --timeout 0 -f "
               f"{rdir / 'run.py'}\n")
        return run_dir, rdir

    def test_live_logs_turn_a_frozen_run_into_real_progress(self):
        run_dir, rdir = self._frozen_run()
        remote = ("[collab_run] === running opt (full) ===\n"
                  "[opt] 10:00:00 opt start 1/1\n"
                  "[opt] 10:00:05   iter   6 evals= 109 best=8.2e+05\n")
        with mock.patch.object(colab_status, "remote_log_text",
                               return_value=(remote, "")):
            snap = self._snapshot(live_logs=True)
        rep = snap["runs"][0]["replicas"][0]
        self.assertEqual(rep["log_source"], "remote")
        self.assertEqual(rep["state"], "running")
        self.assertEqual(rep["progress"]["iteration"], 6)
        self.assertEqual(rep["progress"]["evals"], 109)
        self.assertEqual(rep["progress"]["pct"], 100.0)   # 109 of 50 -> clamped
        self.assertEqual(snap["runs"][0]["state"], "running")
        self.assertTrue(snap["live_logs"])

    def test_without_live_logs_the_frozen_run_is_only_reporting_the_header(self):
        run_dir, rdir = self._frozen_run()
        rep = self._snapshot()["runs"][0]["replicas"][0]
        self.assertEqual(rep["log_source"], "local")
        self.assertIsNone(rep["progress"]["iteration"])

    def test_a_failed_pull_keeps_the_local_view(self):
        run_dir, rdir = self._frozen_run()
        with mock.patch.object(colab_status, "remote_log_text",
                               return_value=(None, "session lost")):
            rep = self._snapshot(live_logs=True)["runs"][0]["replicas"][0]
        self.assertEqual(rep["log_source"], "local")

    def test_finished_runs_are_not_pulled(self):
        run_dir, rdir = self._multi(results=[{
            "index": 0, "state": "finished", "exit": 0, "elapsed_h": 0.5,
        }])
        _write(rdir / "run.log", "[opt] 10:00:00   iter 1 evals= 3\n"
                                 "[multi] run exit=0\n")
        with mock.patch.object(colab_status, "remote_log_text") as pull:
            self._snapshot(live_logs=True)
        pull.assert_not_called()

    def test_remote_root_prefers_remote_config_and_falls_back(self):
        plan_rep = {"remote_config": "/content/tesis-abc/repo/config_r0.yaml"}
        self.assertEqual(colab_status._remote_root(plan_rep, "abc"),
                         "/content/tesis-abc")
        self.assertEqual(colab_status._remote_root({}, "20260101-120000"),
                         "/content/tesis_20260101-120000")


class TestProgress(unittest.TestCase):
    def test_iteration_progress(self):
        text = ("[opt] 10:00:00 opt start 1/2\n"
                "[opt] 10:00:05   iter  10 evals=  25 best=1.0\n")
        progress = colab_status._progress(text, CONFIG)
        self.assertEqual(progress["iteration"], 10)
        self.assertEqual(progress["evals"], 25)
        self.assertEqual(progress["max_evals"], 100)
        self.assertEqual(progress["pct"], 25.0)

    def test_start_only_progress(self):
        progress = colab_status._progress("opt start 2/4\n", CONFIG)
        self.assertEqual(progress["pct"], 25.0)
        self.assertIn("start 2/4", progress["text"])


if __name__ == "__main__":
    unittest.main()
