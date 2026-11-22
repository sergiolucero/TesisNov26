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
import glob
import os
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
    ("introduction", "Introduction"),
    ("related_work", "Background and related work"),
    ("formulation", "Formulation"),
    ("optimization", "Stochastic optimization"),
    ("results_2d", "Numerical results"),
    ("results_3d", "Three dimensions"),
    ("SolvingAlgorithms", "Solving algorithms"),
    ("current_equations_algorithms", "Equations and algorithms under test"),
    ("plans_2d_3d", "Plans"),
    ("conclusions", "Conclusions"),
]
DEFAULT_CHAPTERS = ["introduction", "related_work", "formulation", "optimization",
                    "results_2d", "results_3d", "conclusions"]

# The thesis is one document with one title; the chapters are written as
# standalone articles and carry their own, which are dropped like \maketitle.
THESIS_TITLE = ("A Stochastic Groundwater Remediation Problem solved 25 "
                "years later with AI")
THESIS_AUTHOR = "Sergio Lucero"
THESIS_SUBTITLE = ("Applied to two- and three-dimensional aquifer "
                   "remediation design")
# A dedication page between the title and the abstract. Plain text, or the path
# of a file holding it (--dedication accepts either, like --abstract).
THESIS_DEDICATION = ("To Roger Wets, Alejandro Jofr\\'e and Terry Rockafellar, "
                     "who taught us that a stochastic program is a statement "
                     "about a decision, not about a number.")

# A preamble carries more than \newcommand: theorem environments, column
# types and operator names are declared the same way and are just as fatal to
# drop (a missing \newtheorem{remark} aborts the chapter at its first remark).
DECL_RE = re.compile(
    r"\\(newcommand|renewcommand|providecommand|DeclareMathOperator\*?"
    r"|newtheorem\*?|newtheoremstyle|newcolumntype|DeclareRobustCommand\*?"
    r"|def|let)\s*\{?\s*(\\?[A-Za-z@][A-Za-z0-9@]*)")
SKIP_PREAMBLE_RE = re.compile(
    r"^\s*\\(documentclass|usepackage|title|author|date|maketitle"
    r"|tableofcontents|bibliography|bibliographystyle|begin|end)\b")
LABEL_RE = re.compile(r"\\label\{([^}]*)\}")
REF_RE = re.compile(r"\\(?:ref|eqref|autoref|pageref)\{([^}]*)\}")
INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]*)\}")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}")
ENV_RE = re.compile(r"\\(begin|end)\{([^}]*)\}")
STRIP = re.compile(r"^\s*\\(maketitle|tableofcontents|title|author|date|appendix)\b")
USE_RE = re.compile(r"\\usepackage(?:\[([^\]]*)\])?\{([^}]*)\}")
COMMENT = re.compile(r"(?<!\\)%.*")


def strip_comments(text: str) -> str:
    return "\n".join(COMMENT.sub("", line) for line in text.splitlines())


def drop_title_blocks(head: str) -> str:
    """Remove ``\\title{...}``/``\\author{...}`` including multi-line bodies.

    A chapter title routinely runs over two lines (``\\title{...\\\\`` then
    ``\\large Subtitle}``), and the continuation line is not a declaration --
    left in the merged preamble it is typeset text before ``\\begin{document}``.
    """
    out: list[str] = []
    depth = 0
    for line in head.splitlines():
        if depth == 0 and re.match(r"\s*\\(title|author|date)\b", line):
            depth += line.count("{") - line.count("}")
            continue
        if depth > 0:
            depth += line.count("{") - line.count("}")
            continue
        out.append(line)
    return "\n".join(out)


ABS_BLOCK_RE = re.compile(r"\\begin\{abstract\}.*?\\end\{abstract\}", re.S)


def drop_abstract(body: str) -> str:
    """Remove a chapter's own ``abstract`` environment.

    Standalone articles open with an abstract; a thesis has exactly one, in the
    front matter. Left in, each chapter would print its own "Abstract" heading in
    the middle of the document -- which is how this was noticed.
    """
    return ABS_BLOCK_RE.sub("", body)

# Each selected file becomes a real \chapter, so a body whose first heading
# merely repeats the chapter title ("Introduction", "Background and related
# work") would print it twice and number the section 1.0.1 -- the section counter
# is never incremented at that level. Every other body keeps its \section, which
# then numbers as 1.1, 1.2, ... under the chapter.
HEADING_RE = re.compile(
    r"^(\s*)\\(subsubsection|subsection|section|paragraph)(\*?)(?=[\s\[{])",
    re.M)
OPENER_RE = re.compile(r"^[ \t]*\\(section|chapter)(\*?)\s*\{([^}]*)\}[ \t]*$",
                       re.M)


def _norm_heading(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


PROMOTE = {"subsection": "section", "subsubsection": "subsection"}


def drop_duplicate_opener(body: str, chapter_title: str) -> str:
    """Fold a chapter's own opener into the ``\\chapter`` heading.

    A body that opens with ``\\section{Introduction}`` under
    ``\\chapter{Introduction}`` would print the title twice and number its
    subsections ``1.0.1`` (the section counter is never incremented at that
    level). Dropping the opener therefore has to promote what followed it by one
    level -- otherwise the chapter keeps a phantom section that does not exist.
    Run-in ``\\paragraph`` headings stay put: promoting them to
    ``\\subsubsection`` would turn a run-in lead-in into a display heading.
    """
    m = OPENER_RE.search(body)
    if not m or _norm_heading(m.group(3)) != _norm_heading(chapter_title):
        return body
    rest = body[m.end():]

    def bump(match: re.Match) -> str:
        level = PROMOTE.get(match.group(2))
        return (f"{match.group(1)}\\{level}{match.group(3)}" if level
                else match.group(0))

    return HEADING_RE.sub(bump, rest)


def split_document(text: str, name: str) -> tuple[str, str]:
    """Return ``(preamble, body)`` for a standalone ``article`` file."""
    text = strip_comments(text)
    start = text.find("\\begin{document}")
    if start < 0:
        # already a fragment
        return "", text.replace("\\end{document}", "")
    head = drop_title_blocks(text[:start])
    tail = text[start + len("\\begin{document}"):]
    tail = tail.split("\\end{document}")[0]
    body = "\n".join(line for line in tail.splitlines() if not STRIP.match(line))
    return head, drop_abstract(body)


BIB_BLOCK_RE = re.compile(
    r"\\begin\{thebibliography\}\{[^}]*\}(.*?)\\end\{thebibliography\}",
    re.S)
BIB_ITEM_RE = re.compile(r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}")
CITE_RE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")


def take_bibliography(body: str) -> tuple[str, list[tuple[str, str]]]:
    """Lift every ``thebibliography`` block out of ``body``.

    Each chapter ships its own reference list, and different chapters cite the
    same work under the same key, so an assembled document needs one merged
    list: two ``\\bibitem{ogata1961solution}`` would otherwise make every
    citation of it "multiply defined".
    """
    entries: list[tuple[str, str]] = []

    def _grab(match: re.Match[str]) -> str:
        inner = match.group(1)
        parts = BIB_ITEM_RE.split(inner)
        # split() gives [before, key1, text1, key2, text2, ...]
        for key, text in zip(parts[1::2], parts[2::2]):
            entries.append((key.strip(), f"\\bibitem{{{key.strip()}}}{text.rstrip()}"))
        return ""

    return BIB_BLOCK_RE.sub(_grab, body), entries


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


def collect_decls(head: str) -> dict[tuple[str, str], str]:
    """Map ``(command, declared name)`` -> the whole declaration line.

    Keying on the *declared name* is what lets two chapters define the same
    macro without the second one silently winning (or, worse, both landing in
    the preamble and the later one raising "command already defined").
    """
    found: dict[tuple[str, str], str] = {}
    for line in head.splitlines():
        stripped = " ".join(line.split())
        if not stripped or SKIP_PREAMBLE_RE.match(stripped):
            continue
        m = DECL_RE.search(stripped)
        if m:
            found.setdefault((m.group(1), m.group(2)), stripped)
        else:
            # \setlength, \sisetup, \newenvironment, ... : keep verbatim,
            # de-duplicated by the whole line
            found.setdefault(("raw", stripped), stripped)
    return found


def _find_latex() -> tuple[str, str | None] | None:
    """Locate latexmk/pdflatex, preferring PATH then a user TinyTeX install.

    Returns ``(tool, bindir)``; TinyTeX needs its own directory on the *child's*
    PATH, because pdflatex shells out to sibling binaries (mktexpk, kpsewhich)
    that a bare ``shutil.which`` hit would leave unreachable.
    """
    for name in ("latexmk", "pdflatex"):
        found = shutil.which(name)
        if found:
            return found, None
    for name in ("latexmk", "pdflatex"):
        hits = sorted(glob.glob(str(Path(f"~/.TinyTeX/bin/*/{name}").expanduser())))
        if hits:
            return hits[0], str(Path(hits[0]).parent)
    return None


def report_log(rep: "Report") -> None:
    """Read the last LaTeX pass and complain about anything still dangling.

    ``latexmk`` exits 0 with undefined references, so the return code alone
    would call a broken PDF a success.
    """
    log = BUILD / "tesis.log"
    if not log.is_file():
        return
    text = log.read_text(encoding="utf-8", errors="replace")
    refs = re.findall(r"Reference `([^']*)' on page (\d+) undefined", text)
    cites = re.findall(r"Citation `([^']*)' .*undefined", text)
    multi = re.findall(r"Citation `([^']*)' .*multiply defined", text)
    pages = re.findall(r"Output written on .*?\((\d+) pages", text)
    overfull = len(re.findall(r"Overfull \\hbox", text))
    for label, page in refs:
        rep.error(f"undefined reference {label} (page {page})")
    for key in cites:
        rep.error(f"undefined citation {key}")
    for key in multi:
        rep.warn(f"citation {key} is defined more than once")
    if pages:
        print(f"[tex] compiled tesis.pdf: {pages[-1]} pages, "
              f"{len(refs)} undefined refs, {len(cites)} undefined citations, "
              f"{overfull} overfull boxes")


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
    ap.add_argument("--title", default=THESIS_TITLE, help="document title")
    ap.add_argument("--author", default=THESIS_AUTHOR, help="document author")
    ap.add_argument("--subtitle", default=THESIS_SUBTITLE,
                    help="line under the title ('' to omit)")
    ap.add_argument("--build-dir", default=None,
                    help="where to write tesis.tex/body (default: TEX/build). "
                         "Tests point this at a temp dir so they cannot delete a "
                         "built PDF")
    ap.add_argument("--dedication", default=THESIS_DEDICATION,
                    help="dedication text, or the path of a file holding it "
                         "(\'\' to omit the page)")
    ap.add_argument("--abstract", default=str(TEX / "abstract.tex"),
                    help="fragment emitted as the document abstract "
                         "(default: %(default)s; '' to omit)")
    args = ap.parse_args()

    global BUILD
    if args.build_dir:
        BUILD = Path(args.build_dir)
        if not BUILD.is_absolute():
            BUILD = ROOT / BUILD

    dedication = ""
    if args.dedication:
        dpath = Path(args.dedication)
        if not dpath.is_absolute():
            candidate = ROOT / dpath
            if candidate.exists():
                dpath = candidate
        if dpath.exists() and dpath.is_file():
            dedication = strip_comments(dpath.read_text(encoding="utf-8")).strip()
        else:
            dedication = args.dedication.strip()

    abstract = ""
    if args.abstract:
        apath = Path(args.abstract)
        if not apath.is_absolute():
            apath = ROOT / apath
        if apath.is_file():
            abstract = strip_comments(apath.read_text(encoding="utf-8")).strip()
        else:
            print(f"[tex] no abstract at {args.abstract} -- assembling without one")
    doc_title, doc_author = args.title, args.author
    subtitle_part = (f"\\\\\n\\large\\vspace{{0.3em}}\\normalsize {args.subtitle}"
                     if args.subtitle else "")

    selected = [c.strip() for c in args.chapters.split(",") if c.strip()]
    unknown = [c for c in selected if c not in dict(ALL_CHAPTERS)]
    if unknown:
        print(f"[tex] unknown chapter(s): {', '.join(unknown)}; "
              f"known: {', '.join(s for s, _ in ALL_CHAPTERS)}", file=sys.stderr)
        return 2

    rep = Report()
    packages: list[str] = []
    options: dict[str, list[str]] = {}
    owner: dict[tuple[str, str], str] = {}
    macros: dict[tuple[str, str], tuple[str, str]] = {}  # decl -> (line, chapter)
    labels: dict[str, str] = {}
    bib: dict[str, tuple[str, str]] = {}    # key -> (entry, chapter)
    bodies: list[tuple[str, str]] = []
    pending_refs: list[tuple[str, str]] = []

    for stem, chapter_title in ALL_CHAPTERS:
        path = TEX / f"{stem}.tex"
        if not path.is_file():
            rep.error(f"missing chapter TEX/{stem}.tex")
            continue
        head, body = split_document(path.read_text(encoding="utf-8"), stem)
        check_balance(stem, body, rep)
        check_files(stem, body, rep)

        for line in head.splitlines():
            for opts, names in USE_RE.findall(line.strip()):
                for name in (n.strip() for n in names.split(",")):
                    if not name:
                        continue
                    if name not in packages:
                        packages.append(name)
                        options.setdefault(name, [])
                    for opt in (o.strip() for o in opts.split(",")):
                        if not opt:
                            continue
                        key, _, value = opt.partition("=")
                        prev = {o.partition("=")[0]: o for o in options[name]}
                        if key in prev and prev[key] != opt:
                            rep.warn(
                                f"\\usepackage[{name}]: "
                                f"'{prev[key]}' in {owner.setdefault((name, key), stem)} "
                                f"vs '{opt}' in {stem}; keeping the first")
                            continue
                        if opt not in options[name]:
                            options[name].append(opt)
                    owner.setdefault((name, "pkg"), stem)
        for key, definition in collect_decls(head).items():
            if key in macros and macros[key][0] != definition:
                what = key[1] if key[0] != "raw" else definition[:40]
                rep.warn(f"{key[0]} {what} defined differently in "
                         f"{macros[key][1]} and {stem}; keeping the first")
            else:
                macros.setdefault(key, (definition, stem))

        for label in LABEL_RE.findall(body):
            # duplicates are adjudicated below, once the selection is known
            labels.setdefault(label, stem)

        # \ref targets are checked after every chapter has been read: the front
        # matter points forward into chapters that have not been parsed yet, and
        # an in-loop check called those references undefined on the first run.
        pending_refs.extend((stem, ref) for ref in REF_RE.findall(body))
        body, entries = take_bibliography(body)
        for key, entry in entries:
            if key in bib and bib[key][0] != entry:
                rep.warn(f"bibliography: {key} differs between {bib[key][1]} "
                         f"and {stem}; keeping the first")
            else:
                bib.setdefault(key, (entry, stem))
        bodies.append((stem, body))

    # The assembler itself emits \label{chap:<stem>} after each \chapter, so
    # those are defined even though no body contains them.
    defined = (set(labels)
               | {lab for _, b in bodies for lab in LABEL_RE.findall(b)}
               | {f"chap:{stem}" for stem, _ in bodies})
    for stem, ref in pending_refs:
        if ref.startswith("#") or ref in defined:
            continue
        rep.warn(f"{stem}: \\ref{{{ref}}} is not defined in any chapter "
                 "(fine if the target lives outside TEX/)")

    for stem, body in bodies:
        cited: list[str] = []
        for group in CITE_RE.findall(body):
            cited.extend(k.strip() for k in group.split(",") if k.strip())
        missing = [k for k in cited if k not in bib]
        if missing:
            rep.warn(f"{stem}: cites {', '.join(sorted(set(missing))[:4])}"
                     f"{' ...' if len(missing) > 4 else ''} but no chapter "
                     "defines that reference")

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

    titles = dict(ALL_CHAPTERS)
    for stem, body in bodies:
        body = drop_duplicate_opener(body, titles[stem])
        (BUILD / "body" / f"{stem}.tex").write_text(
            body.strip() + "\n", encoding="utf-8")

    preamble = ["\\documentclass[11pt]{report}"]
    for name in packages:
        opts = options.get(name) or []
        preamble.append(f"\\usepackage[{','.join(opts)}]{{{name}}}"
                        if opts else f"\\usepackage{{{name}}}")
    preamble += ["", "% --- shared declarations (first definition wins) ---"]
    preamble += [definition for definition, _ in macros.values()]
    preamble += [
        "",
        # Bodies live in TEX/build/body/ but their figures are addressed the way
        # the chapters do it, relative to TEX/ (\includegraphics{figs/x.png}).
        # Absolute, because the assembled file is regenerated by this script and
        # a relative path would quietly depend on where it is compiled from.
        f"\\graphicspath{{{{{TEX}/}}{{./}}}}",
        "",
        f"\\title{{{doc_title}{subtitle_part}}}",
        "\\date{\\today}",
        f"\\author{{{doc_author}}}",
        "",
        "\\begin{document}",
        "\\maketitle",
    ]
    if dedication:
        preamble += ["", "\\thispagestyle{empty}", "\\vspace*{0.22\\textheight}",
                     "\\begin{flushright}",
                     "\\begin{minipage}{0.74\\textwidth}\\raggedleft\\itshape",
                     dedication,
                     "\\end{minipage}", "\\end{flushright}", "\\clearpage"]
    if abstract:
        preamble += ["", abstract]
    preamble += ["\\setcounter{tocdepth}{2}", "\\tableofcontents"]
    # No wrapper headings: every chapter body already opens with a real
    # \section, so wrapping it in \section* would leave the inner counters at
    # zero (subsections numbered 0.1) and break \ref into those sections.
    # The chapter anchor belongs to the \chapter line, not to whatever \section
    # happens to come first: a \label after a \section numbers 3.1 and
    # "Chapter~\ref{chap:formulation}" then reads "Chapter 3.1".
    for stem, _body in bodies:
        preamble += [f"\\chapter{{{titles[stem]}}}\\label{{chap:{stem}}}",
                     f"\\input{{body/{stem}}}"]
    if bib:
        preamble += ["", "\\clearpage", "\\begin{thebibliography}{99}"]
        preamble += [entry for entry, _ in bib.values()]
        preamble += ["\\end{thebibliography}"]
    preamble += ["\\end{document}", ""]
    (BUILD / "tesis.tex").write_text("\n".join(preamble), encoding="utf-8")
    shown = BUILD.relative_to(ROOT) if BUILD.is_relative_to(ROOT) else BUILD
    print(f"[tex] wrote {shown}/tesis.tex "
          f"({len(bodies)} chapters)")

    if args.compile:
        found = _find_latex()
        if found is None:
            print("[tex] no LaTeX toolchain on this host -- assemble only "
                  "(rebuild the PDF on a machine with TeX Live)", file=sys.stderr)
            return 0
        # A failed earlier pass leaves an .aux the next run trusts, which turns a
        # content error into a misleading "Missing $ inserted" at \tableofcontents.
        for stale in ("tesis.aux", "tesis.toc", "tesis.out", "tesis.log",
                      "tesis.fls", "tesis.fdb_latexmk"):
            (BUILD / stale).unlink(missing_ok=True)
        tool, bindir = found
        env = None
        if bindir:
            env = dict(os.environ)
            env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
        cmd = ([tool, "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                "tesis.tex"] if tool.endswith("latexmk")
               else [tool, "-interaction=nonstopmode", "-halt-on-error", "tesis.tex"])
        rc = subprocess.call(cmd, cwd=BUILD, env=env)
        report_log(rep)
        for warning in rep.warnings:
            print(f"[tex] warning: {warning}", file=sys.stderr)
        for error in rep.errors:
            print(f"[tex] error: {error}", file=sys.stderr)
        return rc or (1 if rep.errors else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
