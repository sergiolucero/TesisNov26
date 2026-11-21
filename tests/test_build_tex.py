"""Tests for SCRIPTS/build_tex.py (assembling TEX/*.tex into one document).

The tool exists because the chapters are standalone ``article`` files that
happen to share label names and preamble declarations.  The properties worth
pinning are the ones that broke while writing it:

* a two-line ``\\title{...}`` must not leak its second line into the preamble
  (it becomes typeset text before ``\\begin{document}``);
* duplicate ``\\bibitem`` keys from two chapters must collapse to one entry,
  otherwise every citation of the shared work is "multiply defined";
* two chapters that repeat a label cannot be assembled together, while the same
  pair is fine as long as both are not selected;
* the assembled document has no dangling reference.

No LaTeX is required: the checks are static, and the compile path is skipped
when no toolchain is installed.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SCRIPTS"))

import build_tex  # noqa: E402


class TestPreambleExtraction(unittest.TestCase):
    def test_two_line_title_is_dropped(self):
        head = ("\\documentclass{article}\n"
                "\\title{First line\\\\\n"
                "\\large Second line}\n"
                "\\usepackage{amsmath}\n")
        out = build_tex.drop_title_blocks(head)
        self.assertNotIn("Second line", out)
        self.assertIn("\\usepackage{amsmath}", out)

    def test_body_of_title_is_kept(self):
        head = "\\title{T}\n\\author{A}\n\\date{}\n\\pagestyle{empty}\n"
        out = build_tex.drop_title_blocks(head)
        self.assertNotIn("\\title", out)
        self.assertIn("\\pagestyle{empty}", out)

    def test_split_document_strips_the_preamble(self):
        text = ("\\documentclass{article}\n\\usepackage{amsmath}\n"
                "\\newcommand{\\R}{\\mathbb{R}}\n"
                "\\begin{document}\n\\section{One}\nbody\n\\end{document}\n")
        head, body = build_tex.split_document(text, "x")
        self.assertIn("\\newcommand{\\R}", head)
        self.assertNotIn("\\documentclass", body)
        self.assertIn("\\section{One}", body)

    def test_declarations_are_keyed_by_declared_name(self):
        head = ("\\newtheorem{remark}{Remark}\n"
                "\\newcolumntype{Y}{>{\\raggedright}X}\n"
                "\\newcommand{\\R}{\\mathbb{R}}\n")
        decls = build_tex.collect_decls(head)
        self.assertEqual(decls[("newtheorem", "remark")],
                         "\\newtheorem{remark}{Remark}")
        self.assertIn(("newcolumntype", "Y"), decls)
        self.assertIn(("newcommand", "\\R"), decls)


class TestBibliographyMerge(unittest.TestCase):
    def test_duplicate_items_collapse(self):
        body = ("\\begin{thebibliography}{9}\n"
                "\\bibitem{a} First.\n"
                "\\bibitem{b} Second.\n"
                "\\end{thebibliography}\n"
                "tail text\n")
        stripped, entries = build_tex.take_bibliography(body)
        self.assertEqual([k for k, _ in entries], ["a", "b"])
        self.assertNotIn("thebibliography", stripped)
        self.assertIn("tail text", stripped)

    def test_block_is_removed_even_when_not_at_the_end(self):
        body = ("\\begin{thebibliography}{99}\n\\bibitem{x} X.\n"
                "\\end{thebibliography}\n\\section{After}\n")
        stripped, _ = build_tex.take_bibliography(body)
        self.assertIn("\\section{After}", stripped)


class TestChapterLevelAssembly(unittest.TestCase):
    """A stand-alone article is not a thesis chapter until it is demoted.

    Each selected file becomes a real ``\\chapter``. Three things follow and all
    three were bugs first: a body whose opener only restates the chapter title
    must lose it (otherwise the section numbers ``1.0.1``), a chapter's own
    ``abstract`` must not appear mid-document, and its ``\\appendix`` must not
    flip every later chapter into appendix numbering.
    """

    def test_duplicate_opener_is_dropped(self):
        body = "\\section{Introduction}\n\\label{chap:x}\n\\section{Why}\n"
        out = build_tex.drop_duplicate_opener(body, "Introduction")
        self.assertNotIn("\\section{Introduction}", out)
        self.assertIn("\\label{chap:x}", out)      # the label stays
        self.assertIn("\\section{Why}", out)

    def test_a_different_opener_is_kept(self):
        body = "\\section{Governing equation and assumptions}\n"
        out = build_tex.drop_duplicate_opener(body, "Formulation")
        self.assertIn("\\section{Governing equation", out)

    def test_case_spacing_and_trailing_punctuation_do_not_defeat_the_match(self):
        body = "\\section{background  and related WORK:}\n\\section{Next}\n"
        out = build_tex.drop_duplicate_opener(body, "Background and related work")
        self.assertTrue(out.lstrip().startswith("\\section{Next}"))

    def test_chapter_abstract_is_removed(self):
        body = ("\\begin{abstract}\nChapter summary.\n\\end{abstract}\n"
                "\\section{Start}\ntext\n")
        out = build_tex.drop_abstract(body)
        self.assertNotIn("abstract", out)
        self.assertIn("\\section{Start}", out)

    def test_chapter_appendix_marker_is_stripped(self):
        text = ("\\documentclass{article}\n\\begin{document}\n"
                "\\section{Main}\nbody\n\\appendix\n\\section{Extra}\n")
        _head, body = build_tex.split_document(text, "x")
        self.assertNotIn("\\appendix", body)
        self.assertIn("\\section{Extra}", body)     # material is kept

    def test_assembled_master_uses_chapters_and_one_abstract(self):
        tmp = tempfile.mkdtemp(prefix="build_tex_")
        self.addCleanup(shutil.rmtree, tmp, True)
        proc = subprocess.run(
            [sys.executable, str(ROOT / "SCRIPTS" / "build_tex.py"),
             "--chapters", "introduction,conclusions", "--build-dir", str(tmp),
             "--title", "A Test Title", "--author", "A. Author"],
            capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        master = (Path(tmp) / "tesis.tex").read_text()
        self.assertIn("\\documentclass[11pt]{report}", master)
        self.assertIn("\\chapter{Introduction}", master)
        self.assertIn("\\chapter{Conclusions}", master)
        self.assertIn("A Test Title", master)
        self.assertIn("A. Author", master)
        # exactly one abstract, in the front matter, before the first chapter
        self.assertEqual(master.count("\\begin{abstract}"), 1)
        self.assertLess(master.index("\\begin{abstract}"),
                        master.index("\\chapter{Introduction}"))
        for stem in ("introduction", "conclusions"):
            body = (Path(tmp) / "body" / f"{stem}.tex").read_text()
            self.assertNotIn("\\begin{abstract}", body)
            self.assertNotIn("\\appendix", body)


class TestAssembledDocument(unittest.TestCase):
    def test_check_only_passes(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "SCRIPTS" / "build_tex.py"),
             "--check-only"],
            capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("chapters", proc.stdout)

    def test_unknown_chapter_is_refused(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "SCRIPTS" / "build_tex.py"),
             "--check-only", "--chapters", "formulation,nope"],
            capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("unknown chapter", proc.stderr)

    def test_overlapping_selection_is_refused(self):
        # optimization.tex restates SolvingAlgorithms.tex, so selecting both
        # must fail rather than print the same equation under two numbers.
        proc = subprocess.run(
            [sys.executable, str(ROOT / "SCRIPTS" / "build_tex.py"),
             "--check-only", "--chapters", "optimization,SolvingAlgorithms"],
            capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("not assemblable", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main()
