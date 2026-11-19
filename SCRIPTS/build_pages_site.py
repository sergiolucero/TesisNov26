#!/usr/bin/env python3
"""build_pages_site.py
======================

Generate a fully-static, GitHub-Pages-ready snapshot of every plot in the repo.

The Flask app (``SCRIPTS/app.py``) serves ``/plots`` dynamically, but GitHub
Pages can only serve static files. This script flattens that experience into
a committed ``site/`` directory:

    site/
      index.html        homepage linking into the gallery
      plots.html        gallery of every plot (grouped by directory)
      plot.html         single-plot viewer (prev/next + arrow keys)
      plots.js          manifest: window.PLOTS = [{rel, src, group}, ...]
      plots/<relpath>   the actual image files (copied)

Run from the repo root:

    python SCRIPTS/build_pages_site.py

It reads plot images from ``out/``, ``out3d_colab/``, ``runs/sweeps/`` and
``figs/`` (when present) and copies them under ``site/plots/``, preserving
their repo-relative paths so the URLs stay stable and meaningful.

Usage notes:
  * idempotent -- ``site/`` is wiped and rebuilt on every run.
  * stdlib only -- no third-party dependencies.
  * does NOT touch the Flask app or the source images.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.parse
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repository root
SITE = os.path.join(ROOT, "site")
PLOTS_DIR = os.path.join(SITE, "plots")

# Same file types the Flask /plots gallery treats as browsable.
PLOT_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".pdf"}

# Directories (relative to repo root) scanned for plot images, in priority
# order. Paths under these are copied into site/plots/<same relpath>.
SCAN_DIRS = ["out", "out3d_colab", "runs/sweeps", "figs"]


def iter_plots() -> list[str]:
    """Return sorted, repo-relative POSIX paths for every plot image."""
    found: list[str] = []
    for scan in SCAN_DIRS:
        base = os.path.join(ROOT, scan)
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            for name in files:
                if os.path.splitext(name)[1].lower() not in PLOT_EXTS:
                    continue
                full = os.path.join(root, name)
                rel = os.path.relpath(full, ROOT).replace(os.sep, "/")
                found.append(rel)
    return sorted(found)


def group_plots(rels: list[str]) -> "OrderedDict[str, list[str]]":
    """Group plot paths by their parent directory, preserving sort order."""
    groups: "OrderedDict[str, list[str]]" = OrderedDict()
    for rel in rels:
        key = os.path.dirname(rel) or "."
        groups.setdefault(key, []).append(rel)
    return groups


def copy_plots(rels: list[str]) -> None:
    """Copy every plot into site/plots/, preserving its relpath."""
    if os.path.isdir(PLOTS_DIR):
        shutil.rmtree(PLOTS_DIR)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for rel in rels:
        src = os.path.join(ROOT, rel)
        dst = os.path.join(PLOTS_DIR, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def q(rel: str) -> str:
    """URL-encode a relpath for use in the ?p= query string."""
    return urllib.parse.quote(rel, safe="")


# ---------------------------------------------------------------------------
# Static HTML (same visual language as templates/*.html, but no Jinja2).
# ---------------------------------------------------------------------------

_CSS = """    :root {
      --bg: #f7f8fa;
      --card: #ffffff;
      --border: #e3e6ea;
      --text: #1f2430;
      --muted: #5b6472;
      --accent: #0f6f9f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                   Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
    }
    header {
      background: #10212e;
      color: #fff;
      padding: 1.5rem;
    }
    header h1 { margin: 0 0 .3rem; font-size: 1.5rem; }
    header p { margin: 0; color: #c9d4dd; }
    header a { color: #7fdcff; }
    .wrap { max-width: 1300px; margin: 0 auto; padding: 1.5rem; }
    .group { margin-bottom: 2rem; }
    .group h2 {
      font-size: 1rem;
      color: var(--accent);
      border-bottom: 1px solid var(--border);
      padding-bottom: .4rem;
      margin: 0 0 .9rem;
    }
    .group h2 .count { color: var(--muted); font-weight: 400; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 1rem;
    }
    .card {
      display: block;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      text-decoration: none;
      color: inherit;
      transition: border-color .15s, transform .15s;
    }
    .card:hover { border-color: var(--accent); transform: translateY(-2px); }
    .card img {
      display: block;
      width: 100%;
      height: 150px;
      object-fit: cover;
      object-position: top;
      background: #fff;
      border-bottom: 1px solid var(--border);
    }
    .card .label {
      padding: .55rem .7rem;
      font-size: .85rem;
      word-break: break-all;
    }
    .card .label .ext { color: var(--muted); }
    .empty { color: var(--muted); }
    footer { text-align: center; color: var(--muted); font-size: .85rem;
             padding: 0 1.5rem 2rem; }
    code { background: #eef1f4; padding: .1rem .35rem; border-radius: 4px; }
    a { color: var(--accent); }"""


def render_index(total: int, groups: "OrderedDict[str, list[str]]") -> str:
    links = "\n".join(
        f'      <li><a href="plots.html#{g}"><code>{g}</code> '
        f'({len(files)})</a></li>'
        for g, files in groups.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TESIS — plot archive (static)</title>
  <style>
{_CSS}
    .toc {{ background: var(--card); border: 1px solid var(--border);
           border-radius: 10px; padding: 1.25rem 1.5rem; }}
    .toc ul {{ margin: .6rem 0 0; padding-left: 1.2rem; }}
    .toc li {{ margin: .25rem 0; }}
  </style>
</head>
<body>
  <header>
    <h1>TESIS — plot archive</h1>
    <p>Static, GitHub-Pages-ready snapshot of the <code>/plots</code> viewer.
       {total} plot files across {len(groups)} directories.</p>
  </header>

  <div class="wrap">
    <p><strong><a href="plots.html">Browse all plots →</a></strong></p>
    <div class="toc">
      <div><strong>Contents</strong></div>
      <ul>
{links}
      </ul>
    </div>
  </div>

  <footer>Generated by <code>SCRIPTS/build_pages_site.py</code> — served from
    <code>site/</code>.</footer>
</body>
</html>
"""


def render_plots(groups: "OrderedDict[str, list[str]]", total: int) -> str:
    body: list[str] = []
    if not groups:
        body.append('    <p class="empty">No plots found.</p>')
    for directory, files in groups.items():
        anchor = directory.replace("/", "-").replace(".", "")
        heading = directory if directory != "." else "(root)"
        cards: list[str] = []
        for rel in files:
            src = "plots/" + rel
            ext = os.path.splitext(rel)[1]
            base = os.path.basename(rel)
            stem = base[: -len(ext)] if ext else base
            cards.append(
                f'        <a class="card" href="plot.html?p={q(rel)}">\n'
                f'          <img src="{src}" alt="{rel}" loading="lazy">\n'
                f'          <div class="label">{stem} '
                f'<span class="ext">{ext}</span></div>\n'
                f'        </a>'
            )
        body.append(
            f'    <div class="group" id="{anchor}">\n'
            f'      <h2><code>{directory}/</code> '
            f'<span class="count">({len(files)})</span></h2>\n'
            f'      <div class="grid">\n' + "\n".join(cards) +
            f'\n      </div>\n    </div>'
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TESIS — plot browser (static)</title>
  <style>
{_CSS}
  </style>
</head>
<body>
  <header>
    <h1>Plot browser</h1>
    <p>{total} plots — <a href="index.html">home</a></p>
  </header>

  <div class="wrap">
{chr(10).join(body)}
  </div>

  <footer>Generated by <code>SCRIPTS/build_pages_site.py</code> — served from
    <code>site/</code>.</footer>
</body>
</html>
"""


def render_plot(manifest_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TESIS — plot viewer</title>
  <style>
{_CSS}
    header h1 {{ margin: 0; font-size: 1.1rem; word-break: break-all; }}
    .toolbar {{
      display: flex; flex-wrap: wrap; align-items: center;
      justify-content: space-between; gap: .75rem; margin-bottom: 1rem;
    }}
    .nav {{ display: flex; gap: .5rem; }}
    .nav a, .nav span {{
      padding: .45rem .85rem; border-radius: 7px;
      border: 1px solid var(--border); background: var(--card);
      text-decoration: none; color: var(--accent); font-size: .9rem;
    }}
    .nav span {{ color: var(--muted); background: transparent; }}
    .counter {{ color: var(--muted); font-size: .9rem; }}
    .figure {{
      background: var(--card); border: 1px solid var(--border);
      border-radius: 10px; padding: 1rem; text-align: center;
    }}
    .figure img {{
      max-width: 100%; height: auto;
      border: 1px solid var(--border); border-radius: 6px; background: #fff;
    }}
    .meta {{ margin-top: .8rem; font-size: .85rem; color: var(--muted); }}
    .meta a {{ color: var(--accent); }}
    kbd {{
      background: #eef1f4; border: 1px solid var(--border);
      border-bottom-width: 2px; border-radius: 4px; padding: 0 .3rem;
      font-size: .8rem;
    }}
  </style>
</head>
<body>
  <header>
    <h1><code id="title"></code></h1>
    <p><a href="plots.html">← All plots</a> &nbsp;·&nbsp;
       <a href="index.html">Home</a></p>
  </header>

  <div class="wrap">
    <div class="toolbar">
      <div class="nav">
        <a id="prev" href="#">← Previous</a>
        <a id="next" href="#">Next →</a>
      </div>
      <div class="counter"><span id="counter"></span></div>
    </div>

    <div class="figure">
      <img id="figure" alt="">
      <div class="meta">
        <a id="raw" target="_blank" rel="noopener">Open raw file</a>
        &nbsp;·&nbsp; navigate with <kbd>←</kbd> <kbd>→</kbd>
      </div>
    </div>
  </div>

  <script>
    window.PLOTS = {manifest_json};

    const params = new URLSearchParams(location.search);
    const rel = params.get("p");
    let idx = window.PLOTS.findIndex((p) => p.rel === rel);
    if (idx === -1) idx = 0;

    function render() {{
      const p = window.PLOTS[idx];
      if (!p) return;
      document.title = p.rel + " — TESIS plot viewer";
      document.getElementById("title").textContent = p.rel;
      document.getElementById("figure").src = p.src;
      document.getElementById("figure").alt = p.rel;
      document.getElementById("raw").href = p.src;
      document.getElementById("counter").textContent =
        "Plot " + (idx + 1) + " of " + window.PLOTS.length;
      const prev = idx > 0 ? window.PLOTS[idx - 1] : null;
      const next = idx < window.PLOTS.length - 1 ? window.PLOTS[idx + 1] : null;
      const prevEl = document.getElementById("prev");
      const nextEl = document.getElementById("next");
      if (prev) {{
        prevEl.href = "plot.html?p=" + encodeURIComponent(prev.rel);
        prevEl.style.visibility = "visible";
      }} else prevEl.style.visibility = "hidden";
      if (next) {{
        nextEl.href = "plot.html?p=" + encodeURIComponent(next.rel);
        nextEl.style.visibility = "visible";
      }} else nextEl.style.visibility = "hidden";
    }}

    document.addEventListener("keydown", (event) => {{
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      if (event.key === "ArrowLeft" && idx > 0) {{
        idx -= 1;
        history.replaceState(null, "", "plot.html?p=" +
          encodeURIComponent(window.PLOTS[idx].rel));
        render();
      }}
      if (event.key === "ArrowRight" && idx < window.PLOTS.length - 1) {{
        idx += 1;
        history.replaceState(null, "", "plot.html?p=" +
          encodeURIComponent(window.PLOTS[idx].rel));
        render();
      }}
    }});

    render();
  </script>
</body>
</html>
"""


def main() -> int:
    rels = iter_plots()
    if not rels:
        print("ERROR: no plot images found. Run the pipelines first, or "
              "check SCAN_DIRS.", file=sys.stderr)
        return 1

    groups = group_plots(rels)

    # Fresh build.
    if os.path.isdir(SITE):
        shutil.rmtree(SITE)
    os.makedirs(SITE, exist_ok=True)

    copy_plots(rels)

    manifest = [
        {"rel": rel, "src": "plots/" + rel, "group": os.path.dirname(rel) or "."}
        for rel in rels
    ]

    # plots.js + manifest.json (manifest.json kept as a convenience).
    with open(os.path.join(SITE, "plots.js"), "w", encoding="utf-8") as fh:
        fh.write("window.PLOTS = " + json.dumps(manifest, ensure_ascii=False) + ";\n")
    with open(os.path.join(SITE, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    with open(os.path.join(SITE, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(render_index(len(rels), groups))
    with open(os.path.join(SITE, "plots.html"), "w", encoding="utf-8") as fh:
        fh.write(render_plots(groups, len(rels)))
    with open(os.path.join(SITE, "plot.html"), "w", encoding="utf-8") as fh:
        fh.write(render_plot(json.dumps(manifest, ensure_ascii=False)))

    print(f"Built static site in {os.path.relpath(SITE, ROOT)}/")
    print(f"  {len(rels)} plots, {len(groups)} directories")
    for directory, files in groups.items():
        print(f"  {directory}/  ({len(files)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
