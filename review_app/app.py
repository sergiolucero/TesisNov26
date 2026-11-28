#!/usr/bin/env python3
"""Local Flask dashboard for the paperGPU Colab run.

Watches the *live* ``colab run`` log (``out3d_colab/run_paperGPU_slices/``)
plus the already-validated run (``out3d_colab/run_paperGPU/``) and the NumPy
``splu`` verification (``verify_splu.json``), and renders a single self-refreshing
page: session status, optimization progress, the last log lines, the numpy-vs-torch
comparison, and the 1D-slice figures once they land.

Run::

    python review_app/app.py            # http://127.0.0.1:5000
    python review_app/app.py --port 5057

No Colab credentials are needed -- the page only reads local files; the optional
"session" card shells out to ``colab sessions`` and degrades gracefully if the
CLI or credentials are absent.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
CURRENT = ROOT / "out3d_colab" / "run_paperGPU_slices"       # the live run
VALIDATED = ROOT / "out3d_colab" / "run_paperGPU"            # the validated run
VERIFY = VALIDATED / "verify_splu.json"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _tail(path: Path, n: int = 60) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-n:]


OPT_RE = re.compile(
    r"\[opt\] it=(\d+) evals=(\d+) step=([\d.eE+-]+) best=([\d.eE+-]+) "
    r"sum_q=([\d.eE+-]+) active=(\d+)"
)
MS_RE = re.compile(r"multi-start (\d+)/(\d+) \(q0 sum=([\d.eE+-]+)")
SLICE_RE = re.compile(r"slices1d: scoring (.+)$")
Q_RE = re.compile(r"q\* = \[([^\]]*)\].*total ([\d.eE+-]+) m3/d, (\d+) active")
J_RE = re.compile(r"train J = ([\d.eE+-]+)\s+test J = ([\d.eE+-]+)")
REL_RE = re.compile(r"test reliability = ([\d.eE+-]+)\s+mean cleanup = ([\d.eE+-]+) d")
DONE_RE = re.compile(r"results -> (\S+)\s+\(([\d.eE+-]+)s\)")


def _phase(lines: list[str]) -> str:
    text = "\n".join(lines)
    if not lines:
        return "waiting (no log yet)"
    if "Creating session" in text and "resources:" not in text:
        return "provisioning VM"
    if "optimization done" in text and "results ->" not in text:
        return "slicing / final eval"
    if "results ->" in text or "slices-only done" in text:
        return "done"
    if "optimize: batched pattern search" in text:
        return "optimizing"
    if "resources:" in text:
        return "building archives"
    if "Creating session" in text or "Session READY" in text:
        return "starting"
    return "starting"


def _parse_log() -> dict:
    lines = _tail(CURRENT / "colab_run.log", 200)
    last_opt = None
    last_ms = None
    slices_seen: list[str] = []
    q_star = total = active = None
    train_j = test_j = reliability = cleanup = None
    done_dir = wall_s = None

    for ln in lines:
        m = OPT_RE.search(ln)
        if m:
            last_opt = {"it": int(m.group(1)), "evals": int(m.group(2)),
                        "step": float(m.group(3)), "best": float(m.group(4)),
                        "sum_q": float(m.group(5)), "active": int(m.group(6))}
        m = MS_RE.search(ln)
        if m:
            last_ms = {"start": int(m.group(1)), "n_starts": int(m.group(2)),
                       "q0_sum": float(m.group(3))}
        m = SLICE_RE.search(ln)
        if m:
            slices_seen.append(m.group(1).strip())
        m = Q_RE.search(ln)
        if m:
            q_star = [float(x) for x in m.group(1).split(",") if x.strip() != ""]
            total = float(m.group(2))
            active = int(m.group(3))
        m = J_RE.search(ln)
        if m:
            train_j = float(m.group(1))
            test_j = float(m.group(2))
        m = REL_RE.search(ln)
        if m:
            reliability = float(m.group(1))
            cleanup = float(m.group(2))
        m = DONE_RE.search(ln)
        if m:
            done_dir = m.group(1)
            wall_s = float(m.group(2))

    return {
        "phase": _phase(lines),
        "log_tail": _tail(CURRENT / "colab_run.log", 24),
        "opt": last_opt,
        "multistart": last_ms,
        "slices_seen": slices_seen,
        "summary": {
            "q_star": q_star, "total_pumping": total, "active_wells": active,
            "train_j": train_j, "test_j": test_j,
            "reliability": reliability, "cleanup_d": cleanup,
            "done_dir": done_dir, "wall_s": wall_s,
        },
        "finished": bool(done_dir),
    }


# cached (slow) colab session listing
_SESS_CACHE = {"ts": 0.0, "data": None}


def _colab_sessions() -> list[dict]:
    now = time.time()
    if now - _SESS_CACHE["ts"] < 60:
        return _SESS_CACHE["data"] or []
    colab = shutil.which("colab")
    out = []
    if colab:
        try:
            r = subprocess.run([colab, "--auth", "oauth2", "sessions"],
                               capture_output=True, text=True, timeout=60,
                               cwd=ROOT)
            for ln in r.stdout.splitlines():
                if "[" in ln and "]" in ln:
                    name = ln.split("]")[0].lstrip("[")
                    rest = ln.split("]", 1)[1]
                    out.append({"name": name, "detail": rest.strip()})
        except (OSError, subprocess.TimeoutExpired):
            out = []
    _SESS_CACHE["ts"] = now
    _SESS_CACHE["data"] = out
    return out


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------

def _compact_opt(m: dict) -> dict:
    o = m.get("opt", {})
    return {
        "f_star": o.get("f_star"),
        "q_star": o.get("q_star"),
        "total_pumping": o.get("total_pumping"),
        "active_wells": o.get("active_wells"),
        "n_evaluations": o.get("n_evaluations"),
        "nit": o.get("nit"),
        "per_eval_s": o.get("per_eval_s"),
        "train": {k: o.get("train", {}).get(k) for k in
                  ("mean_J", "se_J", "reliability", "mean_tau", "cvar_G")},
        "test": {k: o.get("test", {}).get(k) for k in
                 ("mean_J", "se_J", "reliability", "mean_tau", "cvar_G")},
    }


def _verify_rows(ver: dict, val: dict) -> dict | None:
    """numpy (splu) vs torch (float32) side by side, from the two json files."""
    if not ver or not val or "opt" not in val:
        return None
    torch_tr = val["opt"].get("train", {})
    torch_te = val["opt"].get("test", {})
    keys = ("mean_J", "mean_G", "reliability", "mean_H", "mean_tau",
            "J_pump", "J_treat", "J_res", "J_time", "max_mass_balance_error")
    rows: dict[str, list] = {}
    for arch, t in (("train", torch_tr), ("test", torch_te)):
        rows[arch] = [{"key": k, "numpy": ver.get(arch, {}).get(k),
                       "torch": t.get(k)} for k in keys]
    return rows


def _images() -> list[dict]:
    img: list[dict] = []
    for label, base in (("slices", CURRENT / "slices1d"),
                        ("current figures", CURRENT / "figures"),
                        ("validated figures", VALIDATED / "figures"),
                        ("validated slices", VALIDATED / "slices1d")):
        if not base.is_dir():
            continue
        for p in sorted(base.glob("*.png")):
            rel = p.relative_to(ROOT).as_posix()
            img.append({"label": label, "name": p.name, "rel": rel})
    return img


GATE_RELIABILITY = 0.94  # 1 - alpha - 0.01


def _slice_summary_rows(d: dict) -> list[dict]:
    """Reduce a slices1d.json payload to one readable row per slice."""
    q = [float(x) for x in d.get("q", [])]
    rows = []
    for s in d.get("slices", []):
        x = s.get("x", [])
        f = s.get("f", [])
        rel = s.get("reliability", [])
        total = s.get("total", [])
        if s["kind"] == "coordinate":
            w = s["well"]
            iq = min(range(len(x)), key=lambda k: abs(x[k] - q[w]))
            im = min(range(len(f)), key=lambda k: f[k])
            rows.append({
                "kind": f"well {w}",
                "at_q": f"{q[w]:.2f}",
                "f_at_q": f"{f[iq]:.0f}",
                "f_min": f"{f[im]:.0f}",
                "local_min": abs(f[iq] - f[im]) <= 1.0,
                "active": q[w] > 1e-9,
                "note": "active" if q[w] > 1e-9 else "off",
            })
        elif s["kind"] == "scale":
            order = sorted(range(len(total)), key=lambda k: total[k])
            cross = next((total[k] for k in order if rel[k] >= GATE_RELIABILITY), None)
            iq = min(range(len(total)), key=lambda k: abs(total[k] - sum(q)))
            im = min(range(len(f)), key=lambda k: f[k])
            rows.append({
                "kind": "scale (s*q*)",
                "at_q": f"T={sum(q):.1f}",
                "f_at_q": f"{f[iq]:.0f}",
                "f_min": f"{f[im]:.0f}",
                "local_min": abs(f[iq] - f[im]) <= 1.0,
                "active": True,
                "note": f"rel>=0.94 at T~{cross:.1f}" if cross is not None else "",
            })
        elif s["kind"] == "equal":
            im = min(range(len(f)), key=lambda k: f[k])
            iq = min(range(len(x)), key=lambda k: abs(x[k] - sum(q)))
            rows.append({
                "kind": "equal split",
                "at_q": f"T={sum(q):.1f}",
                "f_at_q": f"{f[iq]:.0f}",
                "f_min": f"{f[im]:.0f}",
                "local_min": False,
                "active": True,
                "note": f"best T={x[im]:.1f}",
            })
        elif s["kind"] == "split":
            i, j = s["i"], s["j"]
            share = q[i] / (q[i] + q[j]) if (q[i] + q[j]) > 1e-12 else 0.0
            iq = min(range(len(x)), key=lambda k: abs(x[k] - share))
            im = min(range(len(f)), key=lambda k: f[k])
            rows.append({
                "kind": f"split wells {i},{j}",
                "at_q": f"share={share:.3f}",
                "f_at_q": f"{f[iq]:.0f}",
                "f_min": f"{f[im]:.0f}",
                "local_min": abs(f[iq] - f[im]) <= 1.0,
                "active": True,
                "note": f"min share={x[im]:.3f}",
            })
    return rows


def _slices_data() -> list[dict]:
    """All slices1d.json payloads across known run dirs."""
    out = []
    for label, base in (("paperGPU slices", CURRENT),
                        ("validated", VALIDATED)):
        p = base / "slices1d" / "slices1d.json"
        d = _read_json(p)
        if d:
            out.append({"label": label, "q": d.get("q"),
                        "rows": _slice_summary_rows(d)})
    return out


def _slice_images() -> list[dict]:
    """Every 1D-slice PNG across known run dirs."""
    img = []
    for label, base in (("paperGPU slices", CURRENT),
                        ("validated", VALIDATED)):
        sdir = base / "slices1d"
        if not sdir.is_dir():
            continue
        for p in sorted(sdir.glob("*.png")):
            img.append({"label": label, "name": p.name,
                        "rel": p.relative_to(ROOT).as_posix()})
    return img


def _state() -> dict:
    log = _parse_log()
    cur = _read_json(CURRENT / "metrics.json")
    val = _read_json(VALIDATED / "metrics.json")
    ver = _read_json(VERIFY)
    return {
        "now": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sessions": _colab_sessions(),
        "run": log,
        "current_metrics": _compact_opt(cur) if cur else None,
        "validated_metrics": _compact_opt(val) if val else None,
        "verify": _verify_rows(ver, val),
        "images": _images(),
    }


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="10">
<title>paperGPU review</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#101418;
         color:#e6e8eb; margin:0; padding:24px; }
  h1 { font-size:20px; margin:0 0 4px; }
  h2 { font-size:15px; margin:20px 0 8px; color:#9ecbff; }
  .sub { color:#8b949e; font-size:12px; margin-bottom:16px; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:10px;
          padding:14px 16px; margin-bottom:12px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:12px; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { text-align:right; padding:5px 8px; border-bottom:1px solid #21262d; }
  th:first-child, td:first-child { text-align:left; }
  th { color:#8b949e; font-weight:600; }
  pre { background:#0d1117; border:1px solid #21262d; border-radius:8px;
        padding:10px 12px; overflow:auto; font-size:12px; line-height:1.45; }
  .badge { display:inline-block; padding:2px 9px; border-radius:999px; font-size:11px;
           font-weight:600; }
  .b-run { background:#1f6feb33; color:#58a6ff; }
  .b-done { background:#23863633; color:#3fb950; }
  .b-wait { background:#6e768133; color:#8b949e; }
  img { max-width:100%; border:1px solid #30363d; border-radius:8px; margin:4px 0; }
  .fig { margin-bottom:18px; }
  .fig .cap { font-size:12px; color:#8b949e; margin-top:2px; }
</style>
</head>
<body>
<h1>paperGPU review</h1>
<div class="sub">auto-refresh every 10 s &middot; {{ s.now }} &middot;
  <a href="/slices1d" style="color:#58a6ff">1D slices</a></div>

<div class="card">
  <b>Live run</b> &middot;
  {% set ph = s.run.phase %}
  <span class="badge {{ 'b-done' if ph == 'done' else ('b-run' if ph not in ('waiting (no log yet)',) else 'b-wait') }}">
    {{ ph }}</span>
  {% if s.run.summary.finished %}
    <span class="badge b-done">finished in {{ '%.0f'|format(s.run.summary.wall_s or 0) }} s</span>
  {% endif %}
  <table style="margin-top:10px">
    {% if s.run.multistart %}
    <tr><td>multi-start</td><td>{{ s.run.multistart.start }} / {{ s.run.multistart.n_starts }}
        (q0 sum {{ '%.1f'|format(s.run.multistart.q0_sum) }})</td></tr>
    {% endif %}
    {% if s.run.opt %}
    <tr><td>iteration / evals</td><td>{{ s.run.opt.it }} / {{ s.run.opt.evals }}</td></tr>
    <tr><td>best f</td><td>{{ '%.4g'|format(s.run.opt.best) }}</td></tr>
    <tr><td>sum_q / active</td><td>{{ '%.2f'|format(s.run.opt.sum_q) }} m3/d / {{ s.run.opt.active }}</td></tr>
    <tr><td>step</td><td>{{ '%.4g'|format(s.run.opt.step) }}</td></tr>
    {% endif %}
    {% if s.run.summary.q_star %}
    <tr><td>q*</td><td>{{ s.run.summary.q_star | map('round', 2) | list }}</td></tr>
    <tr><td>total / active</td><td>{{ '%.2f'|format(s.run.summary.total_pumping) }} m3/d / {{ s.run.summary.active_wells }}</td></tr>
    <tr><td>train J / test J</td><td>{{ '%.2f'|format(s.run.summary.train_j) }} / {{ '%.2f'|format(s.run.summary.test_j) }}</td></tr>
    <tr><td>test reliability / cleanup</td><td>{{ '%.3f'|format(s.run.summary.reliability) }} / {{ '%.0f'|format(s.run.summary.cleanup_d) }} d</td></tr>
    {% endif %}
    {% if s.run.slices_seen %}
    <tr><td>slices scored</td><td>{{ s.run.slices_seen|length }} (last: {{ s.run.slices_seen[-1] }})</td></tr>
    {% endif %}
  </table>
</div>

<div class="grid">
  <div class="card">
    <h2>Colab sessions</h2>
    {% if s.sessions %}
      {% for s2 in s.sessions %}<div style="font-size:13px">[{{ s2.name }}] {{ s2.detail }}</div>{% endfor %}
    {% else %}<span style="color:#8b949e">no active session (or CLI unavailable)</span>{% endif %}
  </div>
  <div class="card">
    <h2>log tail</h2>
    <pre>{% for l in s.run.log_tail %}{{ l }}
{% endfor %}</pre>
  </div>
</div>

{% macro optblock(title, m) -%}
<div class="card">
  <h2>{{ title }}</h2>
  {% if not m %}<span style="color:#8b949e">not available yet</span>
  {% else %}
  <table>
    <tr><td>f*</td><td>{{ '%.4g'|format(m.f_star) if m.f_star is not none else '-' }}</td></tr>
    <tr><td>q*</td><td>{{ m.q_star }}</td></tr>
    <tr><td>total / active</td><td>{{ '%.2f'|format(m.total_pumping) if m.total_pumping is not none else '-' }} m3/d / {{ m.active_wells }}</td></tr>
    <tr><td>evaluations / iters</td><td>{{ m.n_evaluations }} / {{ m.nit }}</td></tr>
    <tr><td>per-eval</td><td>{{ '%.3f'|format(m.per_eval_s) if m.per_eval_s is not none else '-' }} s</td></tr>
    <tr><td>train mean_J / reliab</td><td>{{ '%.2f'|format(m.train.mean_J) if m.train.mean_J is not none else '-' }} / {{ '%.3f'|format(m.train.reliability) if m.train.reliability is not none else '-' }}</td></tr>
    <tr><td>test mean_J / reliab</td><td>{{ '%.2f'|format(m.test.mean_J) if m.test.mean_J is not none else '-' }} / {{ '%.3f'|format(m.test.reliability) if m.test.reliability is not none else '-' }}</td></tr>
  </table>
  {% endif %}
</div>
{%- endmacro %}

<h2>Results</h2>
<div class="grid">
  {{ optblock('current slices run', s.current_metrics) }}
  {{ optblock('validated run (reference)', s.validated_metrics) }}
</div>

{% if s.verify %}
<div class="card">
  <h2>NumPy splu verification of q* (float32 torch vs reference)</h2>
  {% for arch in ['train', 'test'] %}
  <b style="font-size:13px">{{ arch }}</b>
  <table style="margin-bottom:8px">
    <tr><th></th><th>numpy</th><th>torch</th></tr>
    {% for row in s.verify[arch] %}
    <tr><td>{{ row.key }}</td>
        <td>{{ '%.6g'|format(row.numpy) if row.numpy is not none else '-' }}</td>
        <td>{{ '%.6g'|format(row.torch) if row.torch is not none else '-' }}</td></tr>
    {% endfor %}
  </table>
  {% endfor %}
</div>
{% endif %}

{% if s.images %}
<h2>Figures</h2>
{% for im in s.images %}
<div class="fig">
  <div class="cap">{{ im.label }} / {{ im.name }}</div>
  <img src="/media/{{ im.rel }}" alt="{{ im.name }}">
</div>
{% endfor %}
{% endif %}

</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML, s=_state())


@app.route("/api/state")
def api_state():
    return jsonify(_state())


@app.route("/media/<path:rel>")
def media(rel: str):
    p = (ROOT / rel).resolve()
    root = ROOT.resolve()
    if not str(p).startswith(str(root)) or not p.is_file():
        abort(404)
    return send_from_directory(str(root), rel)


SLICES_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>paperGPU &middot; 1D slices</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#101418;
         color:#e6e8eb; margin:0; padding:24px; }
  h1 { font-size:20px; margin:0 0 4px; }
  h2 { font-size:15px; margin:20px 0 8px; color:#9ecbff; }
  .sub { color:#8b949e; font-size:12px; margin-bottom:16px; }
  a { color:#58a6ff; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:10px;
          padding:14px 16px; margin-bottom:12px; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { text-align:right; padding:5px 8px; border-bottom:1px solid #21262d; }
  th:first-child, td:first-child { text-align:left; }
  th { color:#8b949e; font-weight:600; }
  .ok { color:#3fb950; }
  .no { color:#f85149; }
  img { max-width:100%; border:1px solid #30363d; border-radius:8px; margin:4px 0; }
  .fig { margin-bottom:20px; }
  .cap { font-size:12px; color:#8b949e; margin-top:2px; }
</style>
</head>
<body>
<h1>1D slices</h1>
<div class="sub"><a href="/">&larr; back to dashboard</a></div>

{% for d in data %}
<div class="card">
  <h2>{{ d.label }}</h2>
  {% if d.q %}
  <div style="font-size:12px;color:#8b949e;margin-bottom:6px">
    q* = [{{ d.q | map('round', 2) | list | join(', ') }}] &middot; total {{ d.q | sum | round(2) }} m3/d
  </div>
  {% endif %}
  <table>
    <tr><th>slice</th><th>at q*</th><th>f @ q*</th><th>f min</th><th>local min</th><th>note</th></tr>
    {% for r in d.rows %}
    <tr>
      <td>{{ r.kind }}</td>
      <td>{{ r.at_q }}</td>
      <td>{{ r.f_at_q }}</td>
      <td>{{ r.f_min }}</td>
      <td>{% if r.local_min %}<span class="ok">yes</span>{% else %}<span class="no">no</span>{% endif %}</td>
      <td style="color:#8b949e">{{ r.note }}</td>
    </tr>
    {% endfor %}
  </table>
</div>
{% endfor %}

{% if images %}
<h2>Figures</h2>
{% for im in images %}
<div class="fig">
  <div class="cap">{{ im.label }} / {{ im.name }}</div>
  <img src="/media/{{ im.rel }}" alt="{{ im.name }}">
</div>
{% endfor %}
{% else %}
<p style="color:#8b949e">no slice figures found yet.</p>
{% endif %}

</body>
</html>
"""


@app.route("/slices1d")
def slices1d():
    return render_template_string(
        SLICES_HTML, data=_slices_data(), images=_slice_images())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    print(f"review_app -> http://{args.host}:{args.port}  (Ctrl+C to stop)")
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
