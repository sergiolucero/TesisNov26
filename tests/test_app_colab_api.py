"""Tests for the JSON monitoring endpoint in SCRIPTS/app.py (`/api/colab`).

The HTML page (/runs) is for a human; this endpoint is what a cron job or a
tailnet probe polls, so the contract that matters is: one word answers "is it
running", a non-200 only when the *call* failed, and no caching in between.

The tests drive the Flask test client over real synthetic run directories
(`runs/colab_multi/<id>/plan.json` + replica logs) collected by
`colab_status.collect_runs`, so the inference under test is the production
one, not a hand-built snapshot.
"""

import importlib.util
import json
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "SCRIPTS"

try:
    import flask  # noqa: F401
    HAS_FLASK = True
except ImportError:                                   # the tablet/hub may lack it
    HAS_FLASK = False

CONFIG = {
    "domain": {"nx": 80, "ny": 40},
    "time": {"nsteps": 100},
    "optim": {"n_starts": 1, "max_iter": 20, "max_evals": 50},
}


def _load_app():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "tesis_app_under_test", SCRIPTS / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAS_FLASK, "flask is not installed")
class ColabApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_module = _load_app()

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.multi = self.root / "runs" / "colab_multi"
        self.high = self.root / "out" / "high"
        self.multi.mkdir(parents=True)
        self.high.mkdir(parents=True)

        self.app_module.app.config["TESTING"] = True
        self.client = self.app_module.app.test_client()
        # Point the page's data source at the temp tree; the code path
        # (collect_runs -> _pipeline -> jsonify) stays the real one.
        orig_collect = self.app_module.collect_runs

        def temp_tree_collect(**kwargs):
            return orig_collect(root=self.root, multi_dir=self.multi,
                                high_dir=self.high, **kwargs)

        patcher = mock.patch.object(self.app_module, "collect_runs",
                                    temp_tree_collect)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    # -- fixtures ---------------------------------------------------------
    def _write(self, path: Path, text: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def _multi(self, run_id="20260101-120000", results=None, replicas=1,
               log="", plan_overrides=None, started=None):
        run_dir = self.multi / run_id
        reps = []
        for index in range(replicas):
            rdir = run_dir / f"r{index}"
            rdir.mkdir(parents=True)
            # Progress and exit markers are read from run.log; stage markers
            # from cli.log -- write both so either inference path is covered.
            self._write(rdir / "run.log", log)
            self._write(rdir / "cli.log", log)
            reps.append({"index": index, "session": f"tesis-{run_id}-r{index}",
                         "local_dir": str(rdir)})
        plan = {
            "run_id": run_id,
            "created": started or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "models": ["opt"],
            "preset": "full",
            "gpu": "T4",
            "config": CONFIG,
            "estimate": {"hours": 1.5},
            "replicas": reps,
            "results": results or [],
        }
        plan.update(plan_overrides or {})
        self._write(run_dir / "plan.json", json.dumps(plan, indent=2) + "\n")
        return run_dir

    def _get(self, url):
        resp = self.client.get(url)
        return resp, resp.get_json()

    # -- tests ------------------------------------------------------------
    def test_running_pipeline_is_announced_as_running(self):
        self._multi(run_id="20260101-120000",
                    log="### provision exec\nopt start 1/1\niter 3 evals=9\n")
        resp, body = self._get("/api/colab")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["pipeline"]["state"], "running")
        self.assertEqual(body["pipeline"]["active"], 1)
        self.assertEqual(body["pipeline"]["run"]["run_id"], "20260101-120000")
        self.assertEqual(body["pipeline"]["progress_pct"], 18.0)
        self.assertEqual(resp.headers["Cache-Control"], "no-store")

    def test_finished_run_reports_finished_and_no_active_run(self):
        self._multi(log="### provision exec\n[multi] run exit=0\n")
        _, body = self._get("/api/colab")
        self.assertEqual(body["pipeline"]["state"], "finished")
        self.assertEqual(body["pipeline"]["active"], 0)
        self.assertEqual(body["counts"]["finished"], 1)

    def test_failed_run_reports_failed_with_its_failure_line(self):
        self._multi(log="### provision exec\n[multi] run exit=1\n")
        _, body = self._get("/api/colab")
        self.assertEqual(body["pipeline"]["state"], "failed")
        self.assertEqual(body["counts"]["failed"], 1)
        self.assertEqual(body["pipeline"]["run"]["error"], "exit 1")

    def test_no_runs_is_idle_not_an_error(self):
        resp, body = self._get("/api/colab")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(body["pipeline"]["state"], "idle")
        self.assertEqual(body["pipeline"]["active"], 0)
        self.assertIsNone(body["pipeline"]["run"])
        self.assertEqual(body["runs"], [])

    def test_limit_and_run_filter(self):
        self._multi(run_id="20260101-120000", started="2026-01-01T12:00:00",
                    log="[multi] run exit=0\n")
        self._multi(run_id="20260102-120000", started="2026-01-02T12:00:00",
                    log="opt start 1/1\niter 1 evals=5\n")
        _, body = self._get("/api/colab?limit=1")
        self.assertEqual(body["shown"], 1)
        self.assertEqual(body["runs"][0]["run_id"], "20260102-120000")

        _, body = self._get("/api/colab?run=20260101-120000")
        self.assertEqual(body["shown"], 1)
        self.assertEqual(body["runs"][0]["run_id"], "20260101-120000")
        self.assertEqual(body["pipeline"]["state"], "finished")

    def test_unknown_run_filter_is_404(self):
        resp, body = self._get("/api/colab?run=does-not-exist")
        self.assertEqual(resp.status_code, 404)
        self.assertFalse(body["ok"])
        self.assertIn("unknown run", body["error"])

    def test_single_run_detail_and_404_for_unknown(self):
        self._multi(run_id="20260103-120000",
                    log="opt start 1/1\niter 2 evals=7\n")
        resp, body = self._get("/api/colab/20260103-120000")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(body["ok"])
        self.assertIsNone(body["error"])       # API-level error stays null
        self.assertEqual(body["run"]["run_id"], "20260103-120000")
        self.assertEqual(len(body["run"]["replicas"]), 1)
        self.assertIn("iter 2", body["run"]["replicas"][0]["last_line"])

        resp, body = self._get("/api/colab/nope")
        self.assertEqual(resp.status_code, 404)
        self.assertFalse(body["ok"])

    def test_runs_json_alias_matches_api(self):
        self._multi(log="[multi] run exit=0\n")
        alias_resp, alias_body = self._get("/runs.json")
        api_resp, api_body = self._get("/api/colab")
        self.assertEqual(alias_resp.status_code, api_resp.status_code)
        # Compare the stable contract fields: elapsed/age tick between calls.
        for key in ("state", "active", "progress_pct", "headline"):
            self.assertEqual(alias_body["pipeline"][key],
                             api_body["pipeline"][key], key)
        self.assertEqual([r["run_id"] for r in alias_body["runs"]],
                         [r["run_id"] for r in api_body["runs"]])

    def test_collect_failure_is_a_500_with_json(self):
        def boom(**kwargs):
            raise RuntimeError("run dirs unreadable")
        with mock.patch.object(self.app_module, "collect_runs", boom):
            resp, body = self._get("/api/colab")
        self.assertEqual(resp.status_code, 500)
        self.assertFalse(body["ok"])
        self.assertIn("run dirs unreadable", body["error"])
        self.assertEqual(body["pipeline"]["state"], "error")

    def test_live_flag_is_passed_through(self):
        seen = {}

        def capture(**kwargs):
            seen.update(kwargs)
            return {"generated": "now", "generated_ts": 0.0,
                    "root": str(self.root), "multi_dir": "runs/colab_multi",
                    "high_dir": "out/high", "headline": "No Colab runs found",
                    "subhead": "", "counts": {"running": 0, "finished": 0,
                                              "failed": 0, "other": 0},
                    "total": 0, "shown": 0, "runs": [], "has_current": False,
                    "live": None}
        with mock.patch.object(self.app_module, "collect_runs", capture):
            self.client.get("/api/colab?live=1&limit=3")
        self.assertTrue(seen["live"])
        self.assertEqual(seen["limit"], 3)

    def test_runs_page_links_the_json_endpoint(self):
        self._multi(log="[multi] run exit=0\n")
        resp = self.client.get("/runs")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"/api/colab", resp.data)


if __name__ == "__main__":
    unittest.main()
