# SOURCE_MAP.md — where each article section comes from

The article does **not** `\input` the files in `TEX/`. Each `TEX/` file is a
standalone `article` with its own preamble and bibliography, and several of them
overlap (46 duplicate labels in the assembled thesis). `ARTICLE/` lifts the
body text, strips the preamble, and edits it for the new narrative. `TEX/` stays
the archival standalone set.

| Article file | Source | Action | Notes |
|---|---|---|---|
| `sections/00-abstract.tex` | new | written | ~250 words |
| `sections/01-intro.tex` | `DOCS/README.md`, `HUMAN.md`, `GROWTH.md` | new | Opens with the three bad numbers |
| `sections/02-case.tex` | `TEX/formulation.tex` §1–6, `TEX/optimization.tex` §4–6, `TEX/current_equations_algorithms.tex` | lifted + condensed | Equations ADE/Ogata/PINN/hard-BC/Darcy/transport/KL/chance; full derivations remain in `TEX/` |
| `sections/03-method.tex` | `DOCS/Actions.md`, `HUMAN.md`, `SYNCH.md`, git history | new | The six-step loop and memory hierarchy |
| `sections/04-mechanisms.tex` | `DOCS/GROWTH.md`, `DOCS/USAGE.md`, `tests/`, `configs/`, `SCRIPTS/` | new + code excerpts | The five mechanisms; excerpts are real code |
| `sections/05-failures.tex` | `DOCS/Actions.md`, `DOCS/Conclusions.md`, `DOCS/Results.md`, commits | new | Failure catalogue table + narratives |
| `sections/06-results.tex` | `TEX/results_2d.tex` (lifted), `DOCS/3D_results.md` (rewritten), `TEX/formulation.tex` §8–9 (table lifted) | lifted + rewritten | 2D tables nearly verbatim; 1D table from `formulation.tex` |
| `sections/07-discussion.tex` | new | written | Harness vs model, failure modes, protocol, threats |
| `sections/08-conclusion.tex` | new | written | Restates C1–C3 |
| `sections/appendix.tex` | `DOCS/2D_plan.md`, `DOCS/3D_results.md`, `DOCS/USAGE.md`, `TEX/plans_2d_3d.tex` | tables lifted | Gate definitions, repo map, evidence ledger, commands |
| `refs.bib` | `TEX/formulation.tex` bibliography + new entries | converted | `ogata1961solution` and `li2020fourier` were `\bibitem`s |
| `figs/` | `out/figs/`, `out/residual/`, `out/certified_20261120-111849/figures/`, `out3d_colab/figures/`, `out/fno10/` | copied | See `figs/README.md` |

## Deliberate omissions

| Source | Why not lifted |
|---|---|
| `TEX/optimization.tex` §7–13 | 2,200 lines of algorithm variants; the article cites the file and keeps the compact program in §2.2. |
| `TEX/SolvingAlgorithms.tex` | 1,727 lines; the comparison table is appendix material at most. |
| `TEX/plans_2d_3d.tex` §4–6 | PINN/FNO derivations duplicate `formulation.tex`; only the gate tables are lifted. |
| `TEX/build/tesis.pdf` | An assembly, not an article; no introduction or conclusions. |
| `DOCS/SYNCH.md`, `DOCS/NewPlan.md` | Fleet operations; one paragraph in §3, not a section. |

## Label conventions

- Article labels use the `art:` prefix where a collision with `TEX/` is
  possible (`sec:art-...`, `tab:art-...`). Current files use unprefixed labels
  because the article is a separate document; if the two are ever merged, the
  prefix must be applied to both sides.
- Every `\ref`/`\citep` key in `ARTICLE/` is checked by `make check`.

## Material that is *not* yet lifted

| Item | Status |
|---|---|
| A timeline figure from `git log` | to make (`figs/timeline.pdf`) |
| Architecture diagram | currently only the loop TikZ figure |
| FNO before/after panel | `fno_prediction.png` is the original failure; a fixed-run panel is not yet rendered |
| Residual figure | copied but not included (space); add if the venue wants it |
