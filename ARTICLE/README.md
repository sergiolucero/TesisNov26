# ARTICLE/ — “Reusable by Construction”

First draft of the article, built from the repository's results and from
material lifted out of `TEX/`.

## Build

```bash
cd ARTICLE
make          # latexmk -pdf
make check    # build + fail on undefined refs/citations
make clean
```

Requirements: a LaTeX toolchain with `latexmk`, `natbib`, `booktabs`,
`tabularx`, `siunitx`, `listings`, `caption`, `microtype`, `tikz`. On this host
TinyTeX lives in `~/.TinyTeX`; `listings` was installed with
`tlmgr install listings` (caption likewise).

The built PDF is `article.pdf` (23 pages at the time of writing). It is
gitignored like the other build products; rebuild it rather than committing it.

## What is here

| Path | Contents |
|---|---|
| `article.tex` | Master: preamble, title, `\input`s, bibliography |
| `sections/` | Body-only section files (no preambles) |
| `figs/` | Selected figures copied from `out/`, `out3d_colab/` |
| `refs.bib` | Bibliography (converted from the `TEX/` `\bibitem`s plus new entries) |
| `EVIDENCE.md` | Every number in §6 with artifact, seed count and command |
| `SOURCE_MAP.md` | Which `TEX/`/`DOCS/` file each section came from, and what was dropped |

## Status

- Skeleton, all sections and appendices drafted.
- `make check` passes: 0 undefined references, 0 undefined citations, 0
  overfull boxes over 10 pt.
- `python3 tests/run_tests.py`: 172/172 pass in 27.5 s.
- `python3 SCRIPTS/build_tex.py --check-only`: 3 chapters, 283 labels, 43
  macros, 21 packages, no errors.

## Still to do

1. Read the draft end to end for argument, not just LaTeX.
2. Add the timeline figure (`figs/timeline.pdf`) and decide whether the
   residual figure earns its space.
3. Replace the FNO “before” panel with a before/after pair from a fixed run.
4. Cut to the chosen venue's length only if required: the draft is ~9,500 words,
   inside the 8,500–10,000 target.
5. Get the author list, model versions and AI-disclosure wording approved.
6. Decide whether to quote a verbatim Pi/DeepSeek session in §3 or §7.

## Rules

- No number in `sections/06-results.tex` without a row in `EVIDENCE.md`.
- `TEX/` is archival: do not edit it to make the article build.
- One preamble, one bibliography, one `\input` per section.
- Article labels are unprefixed because this is a separate document; if the
  article is ever merged into the thesis, prefix both sides with `art:`.
