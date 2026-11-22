#!/usr/bin/env python3
"""Tests for the 3D plume movie (SCRIPTS/plume_movie3d.py + the solver hook).

The point of the movie module is a promise: *a run always leaves a picture
behind*.  So the tests cover the two ways that promise breaks in practice --
a wrong pixel (the renderer silently lying) and a lost artifact (the renderer
raising and taking the run's evidence with it) -- rather than image aesthetics.

No PDE is solved except in the last class, which builds a 8x6x3 grid with four
timesteps to check that the snapshots the renderer consumes really are the
implicit-Euler iterates.
"""

import importlib.util
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "SCRIPTS"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses resolves annotations through sys.modules[cls.__module__], so the
    # module has to be registered before exec_module or @dataclass blows up.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pm = _load("plume_movie3d")


def _blob(nz=6, ny=12, nx=24, xc=8.0, yc=6.0, kc=3.0, amp=10.0, sigma=1.6):
    """Gaussian concentration blob, shape (nz, ny, nx)."""
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx),
                          indexing="ij")
    return (amp * np.exp(-((i - xc) ** 2 + (j - yc) ** 2) / (2 * sigma ** 2)
                         - (k - kc) ** 2 / (2 * 1.2 ** 2))).astype(np.float32)


def _meta(nx=24, ny=12, nz=6):
    return {"nx": nx, "ny": ny, "nz": nz, "lx": 200.0, "ly": 100.0, "lz": 20.0,
            "creg": 1.0, "total_pumping": 10.0,
            "wells": [[120.0, 50.0, 5.0, 13.0, 7.0]],
            "source": [30.0, 50.0, 10.0], "compliance": [185.0, 50.0, 10.0]}


class TestColourTable(unittest.TestCase):
    def test_lut_is_the_intended_ramp(self):
        """Guards the off-by-one column bug that silently produced viridis.

        That bug read the u-coordinate as the red channel, so red was 0-1
        everywhere and the ramp came out green; every downstream test still
        passed.  These three anchors are unambiguous.
        """
        lut = pm.colour_lut(256)
        r0, g0, b0 = lut[0]
        self.assertLess(int(b0), 40)                       # starts near-black
        r1, g1, b1 = lut[128]
        self.assertGreater(int(r1), 100)                    # mid: red/purple
        self.assertGreater(int(r1), int(b1))
        r2, g2, b2 = lut[255]
        self.assertGreater(int(r2), 200)                    # ends pale yellow
        self.assertGreater(int(g2), int(r2) - 40)
        self.assertGreater(int(b2), 120)

    def test_colourise_is_monotone_in_intensity(self):
        vals = np.array([[0.0, 1.0, 10.0, 100.0]], dtype=np.float32)
        img = pm.colourise(vals, np.zeros_like(vals), vmax=100.0)
        lum = img.astype(float).sum(axis=(0, 2))
        self.assertTrue(np.all(np.diff(lum) > -1.0))

    def test_shared_scale_keeps_a_dimmer_stack_dimmer(self):
        """The same value must map to the same colour in both panels."""
        a = np.full((4, 4), 10.0, np.float32)
        b = np.full((4, 4), 2.0, np.float32)
        zero = np.zeros((4, 4), np.float32)
        ia = pm.colourise(a, zero, vmax=10.0)
        ib = pm.colourise(b, zero, vmax=10.0)
        self.assertGreater(ia.astype(float).sum(), ib.astype(float).sum())


class TestProjection(unittest.TestCase):
    def test_angle_zero_puts_x_to_the_right(self):
        v = _blob(xc=18.0, yc=6.0)
        mip, _ = pm.rotate_mip(v, 0.0, out_w=200)
        col = int(np.argmax(mip.max(axis=0)))
        self.assertGreater(col, 100)      # right half

    def test_blob_on_the_left_stays_left(self):
        mip, _ = pm.rotate_mip(_blob(xc=5.0, yc=6.0), 0.0, out_w=200)
        self.assertLess(int(np.argmax(mip.max(axis=0))), 100)

    def test_rotation_moves_the_projection(self):
        """A quarter turn is a different view of the same object."""
        v = _blob(xc=18.0, yc=6.0)
        a, _ = pm.rotate_mip(v, 0.0, out_w=200)
        b, _ = pm.rotate_mip(v, 90.0, out_w=200)
        ca = int(np.argmax(a.max(axis=0)))
        cb = int(np.argmax(b.max(axis=0)))
        self.assertNotEqual(ca, cb)

    def test_vertical_axis_is_height_not_depth(self):
        """Low-z blob must appear near the bottom of the image, high-z near top."""
        low = pm.rotate_mip(_blob(kc=0.0), 0.0, out_w=120, out_h=48)[0]
        high = pm.rotate_mip(_blob(kc=5.0), 0.0, out_w=120, out_h=48)[0]
        self.assertGreater(int(np.argmax(low.max(axis=1))), 24)
        self.assertLess(int(np.argmax(high.max(axis=1))), 24)

    def test_mip_is_never_below_any_slice(self):
        """A projection cannot be darker than the darkest ray maximum."""
        v = _blob()
        mip, _ = pm.rotate_mip(v, 20.0, out_w=160)
        self.assertGreaterEqual(float(mip.max()), float(v.max()) * 0.5)
        self.assertLessEqual(float(mip.max()), float(v.max()) + 1e-5)

    def test_depth_is_normalised(self):
        _, dep = pm.rotate_mip(_blob(), 15.0, out_w=100)
        self.assertGreaterEqual(float(dep.min()), 0.0)
        self.assertLessEqual(float(dep.max()), 1.0)
        self.assertAlmostEqual(float(dep.min()), 0.0, places=6)
        self.assertAlmostEqual(float(dep.max()), 1.0, places=6)

    def test_bad_rank_raises(self):
        with self.assertRaises(ValueError):
            pm.rotate_mip(np.zeros((4, 4)), 0.0, out_w=64)


class TestCapture(unittest.TestCase):
    def test_capture_arrays_shapes_and_meta(self):
        frames = [_blob() for _ in range(5)]
        st = pm.capture_arrays(np.arange(5) * 10.0, frames, meta=_meta())
        self.assertEqual(st.C.shape, (5, 6, 12, 24))
        self.assertEqual(st.n_frames, 5)
        self.assertAlmostEqual(st.peak, float(np.max(frames)), places=4)
        self.assertEqual(st.jsonable()["n_frames"], 5)

    def test_subsample_preserves_the_mean(self):
        frames = [_blob(amp=3.0) for _ in range(2)]
        full = pm.capture_arrays([0.0, 1.0], frames)
        coarse = pm.capture_arrays([0.0, 1.0], frames, stride_xy=2, stride_z=2)
        self.assertEqual(coarse.C.shape, (2, 3, 6, 12))
        self.assertAlmostEqual(float(full.C.mean()), float(coarse.C.mean()), places=4)

    def test_time_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            pm.capture_arrays([0.0, 1.0], [_blob()])

    def test_bad_rank_raises(self):
        with self.assertRaises(ValueError):
            pm.capture_arrays([0.0], [np.zeros((4, 4))])


class TestAlwaysProducesArtifacts(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile_dir())
        self.stacks = []
        for k, amp in enumerate((10.0, 2.0)):
            frames = [_blob(amp=amp, xc=6.0 + 2.0 * f) for f in range(4)]
            self.stacks.append(pm.capture_arrays(
                np.linspace(0.0, 365.0, 4), frames, meta=_meta()))

    def test_movie_writes_gif_and_fields(self):
        art = pm.make_movie(self.stacks[0], self.stacks[1], self.tmp / "ok")
        self.assertIsNone(art.get("error"))
        self.assertTrue(Path(art["gif"]).exists())
        self.assertTrue(Path(art["fields"]).exists())
        self.assertEqual(art["n_frames"], 4)

    def test_mp4_written_when_ffmpeg_exists(self):
        art = pm.make_movie(self.stacks[0], self.stacks[1], self.tmp / "mp4")
        if shutil.which("ffmpeg"):
            self.assertTrue(art.get("mp4"), art)
            self.assertTrue(Path(art["mp4"]).exists())
        else:                                          # pragma: no cover
            self.assertIsNone(art.get("mp4"))

    def test_encode_falls_back_to_gif_without_ffmpeg(self):
        real = shutil.which

        def no_ffmpeg(_name):
            return None

        shutil.which = no_ffmpeg
        try:
            art = pm.make_movie(self.stacks[0], self.stacks[1], self.tmp / "noff")
        finally:
            shutil.which = real
        self.assertIsNone(art.get("mp4"))
        self.assertTrue(art.get("gif"))
        self.assertIsNone(art.get("error"))

    def test_a_broken_render_still_leaves_evidence(self):
        """The whole contract: no exception, error noted, raw stacks saved."""
        # rank-3 field: crop/rotate cannot cope, so the renderer must fail --
        # and still leave the error note and the raw stacks behind.
        bad = pm.FieldStack(times=np.array([0.0]), C=np.zeros((4, 4), np.float32),
                            meta={})
        art = pm.make_movie(self.stacks[0], bad, self.tmp / "broken")
        self.assertTrue(art.get("error"))
        self.assertTrue((self.tmp / "broken" / "movie3d_error.txt").exists())
        self.assertTrue((self.tmp / "broken" / "fields.npz").exists())

    def test_contact_sheet(self):
        art = pm.make_movie(self.stacks[0], self.stacks[1], self.tmp / "sheet")
        self.assertTrue(art.get("contact"))
        path = Path(art["contact"])
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 1000)

    def test_round_trip_through_npz(self):
        art = pm.make_movie(self.stacks[0], self.stacks[1], self.tmp / "rt")
        a, b = pm.load_stack(Path(art["fields"]))
        self.assertEqual(a.C.shape, self.stacks[0].C.shape)
        np.testing.assert_allclose(a.C, self.stacks[0].C, rtol=1e-6, atol=1e-6)
        self.assertEqual(a.meta.get("creg"), 1.0)


class TestSolverCapture(unittest.TestCase):
    """The snapshots must be the implicit-Euler iterates, not a re-simulation."""

    def setUp(self):
        try:
            import full3D_colab_run as f3d
        except Exception as exc:                        # pragma: no cover
            self.skipTest(f"3D runner unavailable: {exc}")
        self.f3d = f3d

    def test_capture_n_returns_evenly_spaced_iterates(self):
        f3d = self.f3d
        grid = f3d.Grid3D(8, 6, 3, 80.0, 60.0, 12.0)
        aq = f3d.Aquifer3D(K=np.full((3, 6, 8), 10.0), kv_ratio=0.1, phi=0.3,
                           R=1.0, alpha_l=2.0, alpha_th=0.2, alpha_tv=0.02)
        flow = f3d.FlowModel3D(grid, aq, h_left=1e-3 * grid.lx, h_right=0.0)
        wells = np.array([[40.0, 30.0, 8.0, 4.0]])
        resp = f3d.flow_responses_3d(flow, wells, 1.5)
        flow.release_factorization()
        res = f3d.transport_run_3d(
            grid, aq, flow, resp, np.array([5.0]), 5.0, 8,
            source_idx=grid.index(20.0, 30.0, 6.0), source_rate=100.0,
            source_steps=3, capture_n=4)
        self.assertEqual(len(res.C_frames), 4)
        self.assertEqual(res.C_frames[0].shape, (3, 6, 8))
        # last snapshot is the final state, exactly
        np.testing.assert_allclose(res.C_frames[-1].ravel(), res.C_final,
                                   rtol=1e-12, atol=1e-12)
        # captures are increasing in time and the last one is the final step
        self.assertEqual(res.frame_times[-1], 8 * 5.0)
        self.assertTrue(np.all(np.diff(res.frame_times) > 0))

    def test_capture_off_by_default(self):
        f3d = self.f3d
        grid = f3d.Grid3D(6, 4, 2, 60.0, 40.0, 10.0)
        aq = f3d.Aquifer3D(K=np.full((2, 4, 6), 10.0), kv_ratio=0.1, phi=0.3)
        flow = f3d.FlowModel3D(grid, aq, h_left=1e-3 * grid.lx, h_right=0.0)
        resp = f3d.flow_responses_3d(flow, np.array([[30.0, 20.0, 6.0, 4.0]]), 1.5)
        flow.release_factorization()
        res = f3d.transport_run_3d(grid, aq, flow, resp, np.array([1.0]), 5.0, 3,
                                   source_idx=0, source_rate=10.0, source_steps=2)
        self.assertEqual(len(res.C_frames), 0)
        self.assertIsNone(res.frame_times)


def tempfile_dir() -> str:
    import tempfile
    return tempfile.mkdtemp(prefix="movie3d_")


if __name__ == "__main__":
    unittest.main(verbosity=2)
