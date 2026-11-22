#!/usr/bin/env python3
"""Turntable movies of a 3D plume, rendered from the transport solution itself.

Why this module exists
----------------------
A 3D optimization run ends with a number (``J``) and a policy (``q*``).  The
number says whether the policy is cheap, the certificate says whether it is safe,
and neither says what the policy *does*.  This module answers the third question
by capturing the concentration field as it evolves and rendering it as a short
movie: untreated plume beside the plume under ``q*``, same scenario, same colour
scale, orbiting camera.

The design goal is **it always produces a movie**.  Every step has a fallback:

    capture  -> more or fewer snapshots, coarser grid (never fatal)
    render   -> numpy + Pillow only (no matplotlib, no 3D toolkit, no display)
    encode   -> mp4 via ffmpeg, else animated GIF via Pillow
    wrapper  -> any failure still writes the raw stacks and an error note

so the artifact cannot be lost to a missing codec, a headless session, an OOM in
a figure, or a rendering bug at frame 37 of 48.

Rendering: rotating maximum-intensity projection
------------------------------------------------
The camera orbits the domain about the vertical axis.  For output pixel
``(u, v)`` the ray runs along ``e2 = (-sin t, cos t)`` and the image plane's
horizontal axis is ``e1 = (cos t, sin t)``::

    x = cx + (u - U/2) du cos t - s sin t
    y = cy + (u - U/2) du sin t + s cos t

Sampling that line through each horizontal slice and taking the maximum over
``z`` gives an image that reads as a volume rather than a smear; the depth at
which the maximum was found is kept so the render can shade by depth (nearer is
brighter).  Cost is O(pixels * nz * depth samples) with no resampling of the
volume, so a 32x16x8 field renders in milliseconds -- which is what makes
"always render the movie" affordable inside an optimization run.

Colour is **not** normalised per frame: one scale, fixed by the untreated peak
over the whole movie, is used for both panels at every time.  Per-frame
autoscaling would draw a decaying plume at constant brightness, which is exactly
the distortion a figure like this must not have.

CLI (re-render from a saved stack; used by hand and by the tests):

    python3 SCRIPTS/plume_movie3d.py --npz out3d_colab/run_paper/movies/fields.npz \
        --out /tmp/movie --fps 12
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

try:  # Pillow ships with matplotlib, so it exists wherever figures work.
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:  # pragma: no cover - bare install
    Image = None          # type: ignore
    ImageDraw = None      # type: ignore
    ImageFont = None      # type: ignore
    _HAS_PIL = False


# ---------------------------------------------------------------------------
# colour: a hand-rolled "inferno-like" lookup table
# ---------------------------------------------------------------------------
# Deliberately not matplotlib: the movie must render where the figure stack is
# absent or broken, and a 256-entry LUT costs nothing.
_LUT_STOPS = np.array([
    [0.00, 3, 3, 12],
    [0.15, 27, 12, 65],
    [0.30, 74, 12, 107],
    [0.45, 120, 28, 109],
    [0.60, 165, 44, 96],
    [0.75, 207, 68, 70],
    [0.88, 237, 121, 48],
    [1.00, 252, 255, 164],
], dtype=float)


def colour_lut(n: int = 256) -> np.ndarray:
    """Return an ``(n, 3)`` uint8 table for normalised intensity."""
    u = np.linspace(0.0, 1.0, n)
    xs = _LUT_STOPS[:, 0]
    out = np.empty((n, 3), dtype=float)
    for c in range(3):
        out[:, c] = np.interp(u, xs, _LUT_STOPS[:, c + 1])
    return np.clip(out, 0, 255).astype(np.uint8)


_LUT = colour_lut(256)


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------
@dataclass
class FieldStack:
    """Time series of concentration fields for one policy on one scenario.

    ``C`` is ``(T, nz, ny, nx)`` float32 and ``times`` has length ``T``.  ``meta``
    carries what the renderer annotates (wells, source, compliance point,
    threshold, pumping, cell sizes) so frames can be regenerated from a saved
    ``fields.npz`` without re-solving anything.
    """

    times: np.ndarray
    C: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.C.shape[0])

    @property
    def peak(self) -> float:
        return float(np.max(self.C)) if self.C.size else 0.0

    def jsonable(self) -> dict:
        return {
            "n_frames": self.n_frames,
            "times": [float(t) for t in self.times],
            "shape": list(self.C.shape),
            "peak": self.peak,
            "meta": _jsonable(self.meta),
        }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [float(v) for v in np.asarray(obj).ravel()]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    return str(obj)


def _stride_mean(C: np.ndarray, axis: int, stride: int) -> np.ndarray:
    """Strided box mean along ``axis`` -- one pass, no scipy."""
    if stride <= 1:
        return C
    n = C.shape[axis]
    usable = (n // stride) * stride
    if usable == 0:
        return C
    head = np.take(C, np.arange(usable), axis=axis)
    shape = list(head.shape)
    shape[axis:axis + 1] = [usable // stride, stride]
    return head.reshape(shape).mean(axis=axis + 1)


def subsample(C: np.ndarray, stride_xy: int = 1, stride_z: int = 1) -> np.ndarray:
    """Coarsen ``(T, nz, ny, nx)`` by averaging cells (rendering cost only)."""
    if stride_z > 1:
        C = _stride_mean(C, 1, stride_z)
    if stride_xy > 1:
        C = _stride_mean(C, 2, stride_xy)
        C = _stride_mean(C, 3, stride_xy)
    return np.ascontiguousarray(np.asarray(C, dtype=np.float32))


def capture_arrays(times: Sequence[float], fields: Iterable[np.ndarray],
                   meta: Optional[dict] = None,
                   stride_xy: int = 1, stride_z: int = 1) -> FieldStack:
    """Build a :class:`FieldStack` from already-solved snapshots.

    Kept independent of the solver so this module imports nothing from it and can
    be unit-tested on synthetic blobs.
    """
    arr = np.asarray(list(fields), dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"fields must be (T, nz, ny, nx); got {arr.shape}")
    if stride_xy > 1 or stride_z > 1:
        arr = subsample(arr, stride_xy=stride_xy, stride_z=stride_z)
    t = np.asarray(list(times), dtype=float)
    if t.size != arr.shape[0]:
        raise ValueError(f"{t.size} times for {arr.shape[0]} frames")
    return FieldStack(times=t, C=arr, meta=dict(meta or {}))


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample of ``img[ny, nx]`` at fractional coords; outside reads 0."""
    ny, nx = img.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    out = np.zeros(x.shape, dtype=np.float32)
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        yy = y0 + dy
        ok_y = (yy >= 0) & (yy < ny)
        if not ok_y.any():
            continue
        yc = np.clip(yy, 0, ny - 1)
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            xx = x0 + dx
            ok = ok_y & (xx >= 0) & (xx < nx)
            if not ok.any():
                continue
            xc = np.clip(xx, 0, nx - 1)
            out[ok] += (wy * wx)[ok] * img[yc[ok], xc[ok]]
    return out


def rotate_mip(volume: np.ndarray, angle_deg: float, out_w: int,
               out_h: Optional[int] = None, n_depth: Optional[int] = None,
               supersample: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Max-intensity projection of ``volume[nz, ny, nx]`` from ``angle_deg``.

    ``angle_deg = 0`` looks along ``-y`` (domain x points right).  The image's
    **vertical axis is z (height)** and the ray direction is the depth axis, which
    is what makes this a projection rather than a pile-up: for every height band
    and every column we take the maximum along the ray and remember how deep it
    was.

    Returns ``(mip, depth)``, both ``(out_h, out_w)``: raw intensities (NOT
    normalised, so the caller can apply one scale across a whole movie) and
    normalised depth in ``[0, 1]`` with 0 nearest.
    """
    v = np.asarray(volume, dtype=np.float32)
    if v.ndim != 3:
        raise ValueError(f"volume must be (nz, ny, nx); got {v.shape}")
    nz, ny, nx = v.shape
    out_h = int(out_h or max(nz, 1))
    ss = max(1, int(supersample))
    W = int(out_w) * ss
    H = int(out_h) * ss
    nd = int(n_depth or max(ny, nx) * 1.5)
    t = math.radians(float(angle_deg))
    ct, st = math.cos(t), math.sin(t)
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    diag = math.hypot(nx, ny)
    u = (np.arange(W) - (W - 1) / 2.0) * (diag / max(W - 1, 1))
    sgrid = np.linspace(-diag / 2.0, diag / 2.0, nd)
    uu, ss_ = np.meshgrid(u, sgrid, indexing="ij")      # (W, nd)
    x = cx + uu * ct - ss_ * st
    y = cy + uu * st + ss_ * ct

    # One MIP per height band: max along the ray, and the depth that achieved it.
    mip_z = np.zeros((nz, W), dtype=np.float32)
    dep_z = np.zeros((nz, W), dtype=np.float32)
    for k in range(nz):
        vals = _bilinear(v[k], x, y)                    # (W, nd)
        j = np.argmax(vals, axis=1)
        mip_z[k] = vals[np.arange(W), j]
        dep_z[k] = sgrid[j]

    # Resample the height axis to the requested pixel count (linear in z).
    if H != nz:
        zo = np.linspace(0, nz - 1, H)
        i0 = np.floor(zo).astype(np.int32)
        i1 = np.minimum(i0 + 1, nz - 1)
        w = (zo - i0)[:, None].astype(np.float32)
        mip = (1 - w) * mip_z[i0] + w * mip_z[i1]
        dep = (1 - w) * dep_z[i0] + w * dep_z[i1]
    else:
        mip, dep = mip_z, dep_z
    if ss > 1:
        mip = mip.reshape(H // ss, ss, W // ss, ss).mean(axis=(1, 3))
        dep = dep.reshape(H // ss, ss, W // ss, ss).mean(axis=(1, 3))
    # Note the flip: slice k=0 is the BOTTOM of the domain, and row 0 of the
    # image is its TOP.  Without this a plume would hang from the ceiling.
    mip = mip[::-1]
    dep = dep[::-1]
    span = float(dep.max() - dep.min())
    dep = (dep - dep.min()) / span if span > 0 else np.zeros_like(dep)
    return np.ascontiguousarray(mip, dtype=np.float32), dep.astype(np.float32)


def colourise(mip: np.ndarray, depth: np.ndarray, vmax: Optional[float] = None,
              near_gain: float = 0.55, floor: float = 0.0,
              log: bool = True) -> np.ndarray:
    """Map ``(mip, depth)`` to an ``(H, W, 3)`` uint8 image on a shared scale.

    ``vmax`` is the *movie-wide* maximum: the only scale that lets a viewer
    compare frames and panels.  ``floor`` is the smallest concentration that
    still shows, and the default mapping is logarithmic because a plume spans
    orders of magnitude -- on a linear scale the injection cell (tens of grams
    per litre by construction, since the source is a mass *rate* into one cell)
    would blacken everything that matters.
    """
    m = np.asarray(mip, dtype=np.float32)
    top = float(vmax if vmax is not None else (m.max() or 1.0))
    lo = max(float(floor), 0.0)
    if log:
        span = math.log10(max(top, 1e-30) / max(lo, 1e-30)) if lo > 0 else \
            math.log10(max(top, 1e-30) + 1.0)
        if lo > 0:
            rel = np.log10(np.maximum(m, lo) / lo) / max(span, 1e-9)
        else:                                             # additive log fallback
            rel = np.log10(1.0 + np.maximum(m, 0.0)) / max(span, 1e-9)
        v = np.clip(rel, 0.0, 1.0)
    else:
        v = np.power(np.clip((m - lo) / max(top - lo, 1e-30), 0.0, 1.0), 0.55)
    d = np.clip(np.asarray(depth, dtype=np.float32), 0.0, 1.0)
    shade = 1.0 - near_gain * d
    idx = np.clip((v * 255).astype(np.int32), 0, 255)
    rgb = _LUT[idx].astype(np.float32) * (shade * v ** 0.35)[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _font(size: int):
    if not _HAS_PIL:
        return None
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf",
                 "Helvetica.ttc", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def project_xy(xw: float, yw: float, meta: dict, angle_deg: float,
               panel_w: int) -> Optional[float]:
    """World ``(x, y)`` -> horizontal pixel offset inside a projection panel.

    Uses the same cell-space mapping as :func:`rotate_mip`, so markers land on
    the volume instead of near it.
    """
    nx = float(meta.get("nx", 1))
    ny = float(meta.get("ny", 1))
    lx = float(meta.get("lx", nx))
    ly = float(meta.get("ly", ny))
    diag = math.hypot(nx, ny)
    t = math.radians(float(angle_deg))
    ct, st = math.cos(t), math.sin(t)
    dx = (xw - lx / 2.0) / (lx / nx)          # world -> cell units
    dy = (yw - ly / 2.0) / (ly / ny)
    u_cell = dx * ct + dy * st
    if diag <= 0:
        return None
    return panel_w / 2.0 + u_cell / diag * panel_w


def draw_triad(dr, x0: float, y0: float, angle_deg: float, size: float = 26.0,
               fonts=None) -> None:
    """Corner axis indicator that turns with the camera.

    Without it a rotating projection is ambiguous: the viewer cannot tell plume
    motion from camera motion.  A triad that rotates with the view resolves that
    in one glance.
    """
    t = math.radians(float(angle_deg))
    ct, st = math.cos(t), math.sin(t)
    axes = (("x", (ct, st), (235, 120, 120)),
            ("y", (-st, ct), (120, 200, 235)),
            ("z", (0.0, -1.0), (150, 235, 150)))
    for name, (hx, hy), col in axes:
        dr.line([(x0, y0), (x0 + hx * size, y0 + hy * size)], fill=col, width=2)
        if fonts is not None:
            dr.text((x0 + hx * size * 1.25 - 3, y0 + hy * size * 1.25 - 6), name,
                    font=fonts, fill=col)


def draw_outline(dr, meta: dict, angle_deg: float, x0: int, panel_w: int,
                 head: int, panel_h: int, colour=(62, 62, 76)) -> None:
    """Projected footprint of the domain, drawn at the top and bottom edges.

    A projection alone gives no sense of scale or extent; the box does, and it
    costs four line segments.
    """
    lx = float(meta.get("lx", 0.0))
    ly = float(meta.get("ly", 0.0))
    if lx <= 0 or ly <= 0:
        return
    us = []
    for (xw, yw) in ((0.0, 0.0), (lx, 0.0), (lx, ly), (0.0, ly)):
        u = project_xy(xw, yw, meta, angle_deg, panel_w)
        if u is None:
            return
        us.append(u)
    for y in (head + 2, head + panel_h - 3):
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            ua, ub = us[a], us[b]
            if 0 <= ua < panel_w and 0 <= ub < panel_w:
                dr.line([(x0 + ua, y), (x0 + ub, y)], fill=colour, width=1)
    for u in us:                                     # vertical corner posts
        if 0 <= u < panel_w:
            dr.line([(x0 + u, head + 2), (x0 + u, head + panel_h - 3)],
                    fill=colour, width=1)


def render_frame(vol_none: np.ndarray, vol_opt: np.ndarray, angle_deg: float,
                 vmax: float, panel_w: int = 460, meta: Optional[dict] = None,
                 t_days: float = 0.0, t_max: float = 1.0,
                 trace_none: Optional[np.ndarray] = None,
                 trace_opt: Optional[np.ndarray] = None,
                 title: str = "", floor: float = 0.0,
                 log: bool = False) -> "Image.Image":
    """One movie frame: untreated | under q*, with landmarks and a trace strip."""
    meta = dict(meta or {})
    if not _HAS_PIL:                                   # pragma: no cover
        raise RuntimeError("Pillow is required to render frames")
    panel_h = max(150, int(panel_w * 0.52))
    pad, head, foot = 18, 66, 116
    W = pad * 3 + panel_w * 2
    H = head + panel_h + foot
    img = Image.new("RGB", (W, H), (10, 10, 16))
    dr = ImageDraw.Draw(img)
    f_big, f_mid, f_small = _font(20), _font(15), _font(12)

    lz = float(meta.get("lz", 0.0)) or 1.0

    def z_to_y(zw: float) -> float:
        """Physical height -> pixel row inside the panel (z up)."""
        return head + panel_h - 3 - (float(zw) / lz) * (panel_h - 6)

    def panel(arr: np.ndarray, x0: int, label: str, sub: str) -> None:
        mip, dep = rotate_mip(arr, angle_deg, out_w=panel_w, out_h=panel_h)
        img.paste(Image.fromarray(colourise(mip, dep, vmax=vmax, floor=floor,
                                            log=log)),
                  (x0, head))
        draw_outline(dr, meta, angle_deg, x0, panel_w, head, panel_h)
        dr.rectangle([x0, head, x0 + panel_w - 1, head + panel_h - 1],
                     outline=(70, 70, 84))
        dr.text((x0 + 2, head - 21), label, font=f_mid, fill=(235, 235, 240))
        if sub:
            dr.text((x0 + 2, head + panel_h + 6), sub, font=f_small,
                    fill=(170, 170, 185))
        # wells: the screen interval, not the whole column.  4th element of a
        # well row is z_top, 5th is z_bot when the caller supplies them.
        for row in meta.get("wells", []):
            wx, wy, wq = float(row[0]), float(row[1]), float(row[2])
            ztop = float(row[3]) if len(row) > 3 else lz
            zbot = float(row[4]) if len(row) > 4 else 0.0
            u = project_xy(wx, wy, meta, angle_deg, panel_w)
            if u is None or not (0 <= u < panel_w):
                continue
            on = wq > 1e-6
            col = (90, 220, 160) if on else (105, 105, 118)
            dr.line([(x0 + u, z_to_y(ztop)), (x0 + u, z_to_y(zbot))],
                    fill=col, width=2 if on else 1)
            dr.ellipse([x0 + u - 2, z_to_y(ztop) - 2, x0 + u + 2, z_to_y(ztop) + 2],
                       fill=col)
            if on:
                dr.text((x0 + min(max(u + 3, 0), panel_w - 30), z_to_y(ztop) - 13),
                        f"{wq:.1f}", font=f_small, fill=col)
        for key, col, txt in (("source", (120, 200, 255), "src"),
                              ("compliance", (255, 90, 90), "reg")):
            p = meta.get(key)
            if not p:
                continue
            u = project_xy(p[0], p[1], meta, angle_deg, panel_w)
            if u is None or not (0 <= u < panel_w):
                continue
            zy = z_to_y(p[2] if len(p) > 2 else 0.0)
            dr.line([(x0 + u, head + 2), (x0 + u, head + panel_h - 3)],
                    fill=col, width=1)
            dr.text((x0 + min(max(u + 3, 0), panel_w - 26), zy - 6), txt,
                    font=f_small, fill=col)
        draw_triad(dr, x0 + 20, head + panel_h - 28, angle_deg, size=22,
                   fonts=f_small)

    panel(vol_none, pad, "untreated", "no pumping")
    panel(vol_opt, pad * 2 + panel_w, "under q*",
          "total %s m3/d" % _fmt(meta.get("total_pumping", 0.0)))

    if title:
        dr.text((pad, 12), title, font=f_big, fill=(240, 240, 245))
    # bottom-right of the header band: the panel labels own the left side, and a
    # caption that collides with a panel label is unreadable at 460 px.
    label = str(meta.get("scenario_label", ""))
    if label:
        tw = dr.textlength(label, font=f_small) if hasattr(dr, "textlength") else 130
        dr.text((W - pad - tw, 50), label, font=f_small, fill=(165, 165, 180))
    dr.text((W // 2 - 70, 14), f"t = {t_days:6.1f} d", font=f_big,
            fill=(250, 250, 255))

    bar_w, bar_h = 130, 10
    bx, by = W - pad - bar_w, 22
    ramp = (np.logspace(math.log10(max(floor, vmax * 1e-4)), math.log10(vmax),
                        bar_w) if (log and floor > 0)
            else np.linspace(0.0, vmax, bar_w))
    grad = colourise(ramp[None, :].repeat(bar_h, 0).astype(np.float32),
                     np.zeros((bar_h, bar_w), np.float32), vmax=vmax,
                     floor=floor, log=log)
    img.paste(Image.fromarray(grad), (bx, by))
    dr.text((bx, by + bar_h + 3), f"{vmax:.3g} mg/L" + (" (log)" if log else ""),
                font=f_small,
            fill=(170, 170, 185))

    if trace_none is not None and trace_opt is not None and len(trace_none):
        gx0 = pad
        gy0 = head + panel_h + 46
        gw, gh = W - 2 * pad - 30, max(24, foot - 72)
        dr.rectangle([gx0, gy0, gx0 + gw, gy0 + gh], outline=(60, 60, 72))
        creg = float(meta.get("creg", 1.0))
        top = max(float(np.max(trace_none)), float(np.max(trace_opt)), creg) * 1.05
        for arr, col in ((trace_none, (120, 200, 255)),
                         (trace_opt, (90, 220, 160))):
            xs = gx0 + np.linspace(0, gw, len(arr))
            ys = gy0 + gh - np.clip(np.asarray(arr, float) / top, 0, 1) * gh
            dr.line(list(zip(xs.tolist(), ys.tolist())), fill=col, width=1)
        yreg = gy0 + gh - min(creg / top, 1.0) * gh
        dr.line([(gx0, yreg), (gx0 + gw, yreg)], fill=(255, 120, 120), width=1)
        xcur = gx0 + gw * min(max(t_days / max(t_max, 1e-9), 0.0), 1.0)
        dr.line([(xcur, gy0), (xcur, gy0 + gh)], fill=(255, 255, 255), width=1)
        dr.text((gx0, gy0 - 14),
                "compliance-point concentration (mg/L); red = %.2g limit" % creg,
                font=f_small, fill=(160, 160, 175))
        dr.text((gx0 + gw - 150, gy0 - 14), "untreated", font=f_small,
                fill=(120, 200, 255))
        dr.text((gx0 + gw - 70, gy0 - 14), "under q*", font=f_small,
                fill=(90, 220, 160))
    return img


def _fmt(v: Any) -> str:
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "0"


# ---------------------------------------------------------------------------
# encoding, with the fallback chain that makes "always" true
# ---------------------------------------------------------------------------
def encode(frames: Sequence["Image.Image"], mp4: Path, gif: Path,
           fps: float = 12.0) -> dict:
    """Write ``mp4`` (ffmpeg) and ``gif`` (Pillow) independently; report which."""
    out: dict = {"mp4": None, "gif": None, "fps": float(fps),
                 "n_frames": len(frames)}
    if not frames:
        return out
    ff = shutil.which("ffmpeg")
    if ff:
        try:
            mp4.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory() as td:
                for i, im in enumerate(frames):
                    im.save(Path(td) / f"f{i:05d}.png")
                # yuv420p needs even dimensions; pad rather than crop so the
                # frame we rendered is the frame we get.
                cmd = [ff, "-y", "-loglevel", "error", "-framerate", str(fps),
                       "-i", str(Path(td) / "f%05d.png"),
                       "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                       "-movflags", "+faststart", str(mp4)]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode == 0 and mp4.exists():
                    out["mp4"] = str(mp4)
                else:
                    out["mp4_error"] = (proc.stderr or "").strip()[:400]
        except Exception as exc:                       # pragma: no cover
            out["mp4_error"] = f"{type(exc).__name__}: {exc}"
    if _HAS_PIL:
        try:
            gif.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(gif, save_all=True, append_images=list(frames[1:]),
                           duration=int(round(1000.0 / max(fps, 1e-6))),
                           loop=0, optimize=True)
            if gif.exists():
                out["gif"] = str(gif)
        except Exception as exc:                       # pragma: no cover
            out["gif_error"] = f"{type(exc).__name__}: {exc}"
    return out


def contact_sheet(frames: Sequence["Image.Image"], path: Path, ncols: int = 4,
                  scale: float = 0.42, n_pick: int = 8) -> Optional[str]:
    """One PNG grid of evenly spaced frames -- the version that survives LaTeX."""
    if not frames or not _HAS_PIL:
        return None
    try:
        picks = _pick_evenly(list(range(len(frames))), min(n_pick, len(frames)))
        thumbs = []
        for i in picks:
            im = frames[i]
            thumbs.append(im.resize(
                (max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                Image.LANCZOS))
        tw, th = thumbs[0].size
        rows = math.ceil(len(thumbs) / ncols)
        sheet = Image.new("RGB", (tw * ncols, th * rows), (10, 10, 16))
        for n, im in enumerate(thumbs):
            sheet.paste(im, ((n % ncols) * tw, (n // ncols) * th))
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path)
        return str(path)
    except Exception:                                  # pragma: no cover
        return None


def _pick_evenly(items: Sequence[int], k: int) -> list:
    if k >= len(items):
        return list(items)
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[int(i)] for i in idx]


# ---------------------------------------------------------------------------
# the guarantee: one call that always leaves artifacts behind
# ---------------------------------------------------------------------------
def make_movie(stack_none: FieldStack, stack_opt: FieldStack, outdir: Path,
               fps: float = 12.0, spin_deg: float = 40.0,
               panel_w: int = 460, title: str = "",
               save_raw: bool = True, log_scale: bool = False) -> dict:
    """Render, encode, and report -- without ever raising.

    Both stacks must be the same scenario on the same grid.  The colour scale is
    fixed by the larger of the two peaks, so pumping is visible as a change in
    brightness rather than as an autoscale artefact.

    On failure the raw stacks and ``movie3d_error.txt`` are still written, which
    is the whole point: a run that spent an hour on a GPU must not come back
    without evidence.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    art: dict = {"outdir": str(outdir)}
    t0 = time.time()
    try:
        T = min(stack_none.n_frames, stack_opt.n_frames)
        if T <= 0:
            raise ValueError("no frames captured")
        times = np.asarray(stack_opt.times[:T], dtype=float)
        # Robust top of the scale: the 99.5th percentile over every frame of both
        # panels.  The raw peak is the injection cell (a mass rate dumped into
        # one cell), which is one or two orders above the plume and would flatten
        # the thing we came to look at.
        both = np.concatenate([stack_none.C[:T].ravel(), stack_opt.C[:T].ravel()])
        positive = both[both > 0]
        vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
        vmax = max(vmax, 1e-12)
        creg = float(stack_none.meta.get("creg", 1.0))
        # One decade below the regulatory limit is the darkest thing that shows:
        # the plume edge is what a reader is looking for, not the numerical halo.
        floor = max(creg * 0.1, vmax * 1e-3) if log_scale else 0.0
        meta = dict(stack_none.meta)
        meta.update({k: v for k, v in stack_opt.meta.items()
                     if k in ("total_pumping", "scenario_label")})
        trace_none = _trace(stack_none, T)
        trace_opt = _trace(stack_opt, T)

        frames = []
        for f in range(T):
            angle = spin_deg * (f / max(T - 1, 1))
            frames.append(render_frame(
                stack_none.C[f], stack_opt.C[f], angle, vmax,
                panel_w=panel_w, meta=meta, t_days=float(times[f]),
                t_max=float(times[-1]), trace_none=trace_none,
                trace_opt=trace_opt, title=title, floor=floor,
                log=log_scale))
        art["n_frames"] = len(frames)
        art.update(encode(frames, outdir / "plume3d.mp4",
                          outdir / "plume3d.gif", fps=fps))
        sheet = contact_sheet(frames, outdir / "plume3d_contact.png")
        if sheet:
            art["contact"] = sheet
    except Exception as exc:                           # the point of the module
        art["error"] = f"{type(exc).__name__}: {exc}"
        try:
            (outdir / "movie3d_error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        except Exception:
            pass
    finally:
        if save_raw:
            try:
                np.savez_compressed(
                    outdir / "fields.npz",
                    C_none=stack_none.C, C_opt=stack_opt.C,
                    times=np.asarray(stack_none.times, dtype=float),
                    meta=json.dumps(_jsonable(stack_opt.meta)))
                art["fields"] = str(outdir / "fields.npz")
            except Exception as exc:
                art["fields_error"] = f"{type(exc).__name__}: {exc}"
        art["render_s"] = round(time.time() - t0, 3)
        try:
            (outdir / "movie3d_manifest.json").write_text(
                json.dumps(art, indent=2), encoding="utf-8")
        except Exception:
            pass
    return art


def _trace(stack: FieldStack, T: int) -> np.ndarray:
    """Compliance-point concentration per frame, from ``meta`` or the field peak."""
    tr = stack.meta.get("trace")
    if tr is not None and len(tr) >= T:
        return np.asarray(tr[:T], dtype=float)
    if stack.C.size == 0:
        return np.zeros(T)
    return np.max(stack.C[:T].reshape(T, -1), axis=1)


def stack_stats(stack: FieldStack, creg: float | None = None) -> dict:
    """Quantitative companion to the picture: peak / mean / exceedance in time.

    A maximum-intensity projection answers "where", not "how much"; these numbers
    answer "how much", so a caption never has to over-read the image.
    """
    C = np.asarray(stack.C, dtype=np.float64)
    if C.size == 0:
        return {}
    threshold = float(stack.meta.get("creg", 1.0) if creg is None else creg)
    per_frame_peak = C.reshape(C.shape[0], -1).max(axis=1)
    frac_above = (C > threshold).reshape(C.shape[0], -1).mean(axis=1)
    return {
        "peak": float(C.max()),
        "mean": float(C.mean()),
        "p99": float(np.percentile(C, 99.0)),
        "final_peak": float(per_frame_peak[-1]),
        "final_mean": float(C[-1].mean()),
        "final_frac_above_creg": float(frac_above[-1]),
        "peak_over_time": per_frame_peak.tolist(),
        "frac_above_creg_over_time": frac_above.tolist(),
        "creg": threshold,
        "n_frames": int(C.shape[0]),
    }


def load_stack(npz_path: Path) -> tuple[FieldStack, FieldStack]:
    """Rebuild the two stacks saved by :func:`make_movie`; no re-solve needed."""
    with np.load(npz_path, allow_pickle=False) as z:
        cn = z["C_none"]
        co = z["C_opt"]
        times = np.asarray(z["times"], dtype=float)
        meta = json.loads(str(z["meta"]))
    return (FieldStack(times=times, C=np.asarray(cn, np.float32), meta=meta),
            FieldStack(times=times, C=np.asarray(co, np.float32), meta=meta))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Render a plume movie from a saved fields.npz",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--panel-w", type=int, default=460)
    ap.add_argument("--spin", type=float, default=40.0,
                    help="total camera orbit over the movie, degrees")
    ap.add_argument("--frames", type=int, default=0, help="0 = all snapshots")
    ap.add_argument("--log", action="store_true",
                    help="logarithmic colour scale (default: linear, which shows "
                         "the pumping effect far more clearly)")
    args = ap.parse_args(argv)

    a, b = load_stack(Path(args.npz))
    if args.frames > 0:
        k = min(args.frames, a.n_frames, b.n_frames)
        a = FieldStack(a.times[:k], a.C[:k], a.meta)
        b = FieldStack(b.times[:k], b.C[:k], b.meta)
    art = make_movie(a, b, Path(args.out), fps=args.fps, spin_deg=args.spin,
                     panel_w=args.panel_w, log_scale=bool(args.log))
    print(json.dumps(art, indent=2))
    return 0 if not art.get("error") else 1


if __name__ == "__main__":
    sys.exit(_main())
