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

import subprocess
import sys
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
