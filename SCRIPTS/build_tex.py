#!/usr/bin/env python3
"""build_tex.py
===============

Assemble the thesis from its standalone chapters, and check it offline.

There is no master document in ``TEX/``: each file is a self-contained ``article``
with its own preamble, and ``results_2d.tex`` is the only chapter-shaped fragment.
That is fine for editing -- every file compiles on its own -- but it means "the
thesis" has never existed as a document, and a reference that resolves inside one
chapter can dangle in the assembled one.

The files are also **not disjoint documents**: ``SolvingAlgorithms.tex`` and
``plans_2d_3d.tex`` restate material that ``optimization.tex`` and
``formulation.tex`` now cover, which shows up immediately as duplicated
``\\label`` targets (46 of them). Assembling all six would print the same
equation twice under one number, so the default selection is the disjoint subset
that tells the current story -- ``formulation``, ``optimization``,
``results_2d`` -- and the rest are treated as standalone documents (two of them
already ship as PDFs). ``--chapters`` overrides that once the editorial question
of what the thesis actually contains has an answer.

This script does the assembly without touching the sources:

1. **Check** (always, no LaTeX needed): balanced braces/environments, ``$`` and
   ``\\[`` pairing per file, every ``\\ref``/``\\eqref``/``\\autoref`` resolving to
   a ``\\label`` in *some* chapter, no duplicate labels, and every ``\\input``/
   ``\\includegraphics`` target present on disk.
2. **Merge the preambles**: union of ``\\usepackage`` in first-seen order, and the
   macro definitions de-duplicated by name.  A name defined twice with *different*
   bodies is reported as a conflict rather than silently resolved, because that is
   a real modelling question (two chapters calling ``\\R`` different things is a
   bug in the writing, not in this script).
3. **Write** ``TEX/build/tesis.tex`` plus ``TEX/build/body/<chapter>.tex`` (the
   bodies with ``\\maketitle``/``\\tableofcontents``/``\\title`` stripped), so the
   build directory is the only place that changes.
4. Optionally ``--compile`` with ``latexmk``/``pdflatex`` if a toolchain exists
   (it does not on the assembly host; the PDF rebuild belongs on do-relay).

Usage
-----
    python3 SCRIPTS/build_tex.py               # check + assemble
    python3 SCRIPTS/build_tex.py --check-only  # check, touch nothing
    python3 SCRIPTS/build_tex.py --compile     # assemble + run pdflatex

Exit status is 1 when the check finds an error, so it can gate a commit.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "TEX"
BUILD = TEX / "build"

# Checked in this order; only DEFAULT_CHAPTERS is assembled.
ALL_CHAPTERS = [
    ("formulation", "Formulation"),
    ("optimization", "Stochastic optimization"),
    ("results_2d", "Numerical results"),
    ("SolvingAlgorithms", "Solving algorithms"),
    ("current_equations_algorithms", "Equations and algorithms under test"),
    ("plans_2d_3d", "Plans"),
]
DEFAULT_CHAPTERS = ["formulation", "optimization", "results_2d"]

MACRO_RE = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand|DeclareMathOperator\*?)"
    r"\s*\{?\s*\\([A-Za-z@]+)\s*\}?\s*"
    r"(?:\[[^\]]*\])?\s*")
LABEL_RE = re.compile(r"\\label\{([^}]*)\}")
REF_RE = re.compile(r"\\(?:ref|eqref|autoref|pageref)\{([^}]*)\}")
INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]*)\}")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}")
ENV_RE = re.compile(r"\\(begin|end)\{([^}]*)\}")
STRIP = re.compile(r"^\s*\\(maketitle|tableofcontents|title|author|date)\b")
COMMENT = re.compile(r"(?<!\\)%.*")


def strip_comments(text: str) -> str:
    return "\n".join(COMMENT.sub("", line) for line in text.splitlines())


def split_document(text: str, name: str) -> tuple[str, str]:
    """Return ``(preamble, body)`` for a standalone ``article`` file."""
    text = strip_comments(text)
    start = text.find("\\begin{document}")
    if start < 0:
        # already a fragment
        return "", text.replace("\\end{document}", "")
    head = text[:start]
    tail = text[start + len("\\begin{document}"):]
    tail = tail.split("\\end{document}")[0]
    body = "\n".join(line for line in tail.splitlines() if not STRIP.match(line))
    return head, body


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def check_balance(name: str, body: str, rep: Report) -> None:
    depth = 0
    line_no = 1
    for i, ch in enumerate(body):
        if ch == "\n":
            line_no += 1
        elif ch == "{" and (i == 0 or body[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or body[i - 1] != "\\"):
            depth -= 1
            if depth < 0:
                rep.error(f"{name}:{line_no}: unbalanced '}}'")
                depth = 0
    if depth:
        rep.error(f"{name}: {depth} unclosed '{{'")

    stack: list[tuple[str, int]] = []
    for m in ENV_RE.finditer(body):
        kind, env = m.group(1), m.group(2)
        where = body[:m.start()].count("\n") + 1
        if kind == "begin":
            stack.append((env, where))
        elif not stack or stack[-1][0] != env:
            opened = f"{stack[-1][0]} at line {stack[-1][1]}" if stack else "nothing"
            rep.error(f"{name}:{where}: \\end{{{env}}} closes {opened}")
            if stack:
                stack.pop()
        else:
            stack.pop()
    for env, where in stack:
        rep.error(f"{name}:{where}: \\begin{{{env}}} never closed")

    if body.count("$") % 2:
        rep.warn(f"{name}: odd number of '$' (check for an unclosed inline formula)")
    # `\\[2pt]` is row spacing in a table, so only unescaped display math counts
    open_disp = len(re.findall(r"(?<!\\)\\\[", body))
    close_disp = len(re.findall(r"(?<!\\)\\\]", body))
    if open_disp != close_disp:
        rep.warn(f"{name}: \\[ and \\] counts differ ({open_disp} vs {close_disp})")


def check_files(name: str, body: str, rep: Report) -> None:
    for target in INPUT_RE.findall(body):
        path = TEX / (target if target.endswith(".tex") else target + ".tex")
        if not path.is_file():
            rep.error(f"{name}: \\input{{{target}}} does not exist")
    for target in GRAPHICS_RE.findall(body):
        path = TEX / target
        if not path.is_file() and not any(
                (TEX / f"{target}{ext}").is_file()
                for ext in (".pdf", ".png", ".jpg", ".jpeg", ".eps")):
            rep.error(f"{name}: \\includegraphics{{{target}}} does not exist")


def collect_macros(head: str) -> dict[str, str]:
    """Map macro name -> the whole definition line (normalised)."""
    found: dict[str, str] = {}
    for line in head.splitlines():
        m = MACRO_RE.search(line)
        if m:
            found.setdefault(m.group(1), " ".join(line.split()))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-only", action="store_true",
                    help="validate and report, write nothing")
    ap.add_argument("--compile", action="store_true",
                    help="run latexmk/pdflatex after assembling (needs a toolchain)")
    ap.add_argument("--chapters", default=",".join(DEFAULT_CHAPTERS),
                    help="comma-separated chapters to assemble, in order "
                         "(default: %(default)s)")
    args = ap.parse_args()

    selected = [c.strip() for c in args.chapters.split(",") if c.strip()]
    unknown = [c for c in selected if c not in dict(ALL_CHAPTERS)]
    if unknown:
        print(f"[tex] unknown chapter(s): {', '.join(unknown)}; "
              f"known: {', '.join(s for s, _ in ALL_CHAPTERS)}", file=sys.stderr)
        return 2

    rep = Report()
    packages: list[str] = []
    macros: dict[str, tuple[str, str]] = {}   # name -> (definition, chapter)
    labels: dict[str, str] = {}
    bodies: list[tuple[str, str]] = []

    for stem, title in ALL_CHAPTERS:
        path = TEX / f"{stem}.tex"
        if not path.is_file():
            rep.error(f"missing chapter TEX/{stem}.tex")
            continue
        head, body = split_document(path.read_text(encoding="utf-8"), stem)
        check_balance(stem, body, rep)
        check_files(stem, body, rep)

        for line in head.splitlines():
            if line.strip().startswith("\\usepackage"):
                entry = line.strip()
                if entry not in packages:
                    packages.append(entry)
        for name, definition in collect_macros(head).items():
            if name in macros and macros[name][0] != definition:
                rep.warn(f"macro \\{name} defined differently in "
                         f"{macros[name][1]} and {stem}; keeping the first")
            else:
                macros.setdefault(name, (definition, stem))

        for label in LABEL_RE.findall(body):
            # duplicates are adjudicated below, once the selection is known
            labels.setdefault(label, stem)

        for ref in REF_RE.findall(body):
            if ref.startswith("#"):
                continue
            if ref not in labels and ref not in LABEL_RE.findall("\n".join(
                    b for _, b in bodies)):
                rep.warn(f"{stem}: \\ref{{{ref}}} is not defined in any chapter "
                         "(fine if the target lives outside TEX/)")
        bodies.append((stem, body))

    # A chapter that repeats another one's labels cannot share a document: the
    # second copy silently renumbers the first.  Report it as a fact about the
    # sources, and only fail when the *selected* chapters conflict.
    by_stem = {stem: body for stem, body in bodies}
    labels_of: dict[str, set[str]] = {}
    for stem, body in bodies:
        labels_of[stem] = set(LABEL_RE.findall(body))
    for i, (stem_a, _t) in enumerate(bodies):
        for stem_b, _t2 in bodies[i + 1:]:
            shared = labels_of[stem_a] & labels_of[stem_b]
            both = stem_a in selected and stem_b in selected
            if shared:
                msg = (f"{stem_a} and {stem_b} share {len(shared)} label(s) "
                       f"(e.g. {sorted(shared)[0]})")
                if both:
                    rep.error(msg + " -- not assemblable together")
                else:
                    rep.warn(msg + "; not selected together, so fine")
    missing = [c for c in selected if c not in by_stem]
    for stem in missing:
        rep.error(f"selected chapter {stem} could not be read")
    bodies = [(stem, body) for stem, body in bodies if stem in selected]

    for warning in rep.warnings:
        print(f"[tex] warning: {warning}")
    for error in rep.errors:
        print(f"[tex] error: {error}", file=sys.stderr)

    print(f"[tex] {len(bodies)} chapters, {len(labels)} labels, "
          f"{len(macros)} macros, {len(packages)} packages")
    if rep.errors:
        print(f"[tex] {len(rep.errors)} error(s); nothing written", file=sys.stderr)
        return 1
    if args.check_only:
        return 0

    if BUILD.exists():
        shutil.rmtree(BUILD)
    (BUILD / "body").mkdir(parents=True)

    for stem, body in bodies:
        (BUILD / "body" / f"{stem}.tex").write_text(
            body.strip() + "\n", encoding="utf-8")

    preamble = ["\\documentclass[11pt]{article}"]
    preamble += packages
    preamble += ["", "% --- shared macros (first definition wins) ---"]
    preamble += [definition for definition, _ in macros.values()]
    preamble += [
        "",
        "\\title{Stochastic optimization of pump-and-treat remediation\\\\"
        "\\large assembled from TEX/*.tex by SCRIPTS/build_tex.py}",
        "\\date{}",
        "",
        "\\begin{document}",
        "\\maketitle",
        "\\tableofcontents",
    ]
    # No wrapper headings: every chapter body already opens with a real
    # \section, so wrapping it in \section* would leave the inner counters at
    # zero (subsections numbered 0.1) and break \ref into those sections.
    for stem, _body in bodies:
        preamble += ["\\clearpage", f"\\input{{body/{stem}}}"]
    preamble += ["\\end{document}", ""]
    (BUILD / "tesis.tex").write_text("\n".join(preamble), encoding="utf-8")
    print(f"[tex] wrote {BUILD.relative_to(ROOT)}/tesis.tex "
          f"({len(bodies)} chapters)")

    if args.compile:
        tool = shutil.which("latexmk") or shutil.which("pdflatex")
        if tool is None:
            print("[tex] no LaTeX toolchain on this host -- assemble only "
                  "(rebuild the PDF on a machine with TeX Live)", file=sys.stderr)
            return 0
        cmd = ([tool, "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                "tesis.tex"] if tool.endswith("latexmk")
               else [tool, "-interaction=nonstopmode", "-halt-on-error", "tesis.tex"])
        return subprocess.call(cmd, cwd=BUILD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
