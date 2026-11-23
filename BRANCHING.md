# Branches: code and prose live apart

Two lines of work share this repository and have different ideas of "verified".

| branch | owns | is verified by |
| :--- | :--- | :--- |
| `code-sims` | `physics/`, `stoch/`, `SCRIPTS/`, `tests/`, `configs/` and the machine-readable evidence (`DOCS/Results.md`, `DOCS/3D_results.md`, `DOCS/GPU.md`, `RUNTIMES.txt`) | `python3 tests/run_tests.py`, and a run that reproduces |
| `doc-latex` | `ARTICLE/`, `TEX/`, `DOCS/Conclusions.md`, `DOCS/FIGURES.md`, `FIGURES/` | `cd ARTICLE && make check` |

`main` fast-forwards to `code-sims`: it means "the latest integrated code".

*(This supersedes the branch paragraph of `DOCS/LANES.md`, which argued for one
branch; everything else in that file — path ownership, provenance rules, no
lane-wide git operations — still stands and is what keeps this survivable.)*

**Merge direction is one-way: `doc-latex` merges `code-sims`, never the reverse.**
Code never needs the prose; the prose always needs the numbers. A one-way merge
cannot go into conflict in both directions at once.

The leak is real and has a rule: a certified number is *produced* by a run and
*quoted* by a paragraph, so it crosses the boundary every time. Evidence files
(`DOCS/Results.md`, `RUNTIMES.txt`) therefore belong to `code-sims` — the commit
that runs the experiment updates the numbers — and `doc-latex` pulls them in when
it needs to cite them. Prose never introduces a number that is not in an evidence
file first.

## Work in a worktree, not by switching branches

    git worktree add ../TesisNov26-sims -b code-sims main

A branch switch carries **uncommitted** changes across with it, so switching the
shared tree from `doc-latex` to `code-sims` would move whatever the other session
has half-written into the wrong branch. A worktree gives each line its own
directory and its own index, and the shared tree stays exactly as it was.

    git worktree list
    /home/sergio/TesisNov26       5d209e9 [doc-latex]
    /home/sergio/TesisNov26-sims  f912ef9 [code-sims]

Remove a worktree with `git worktree remove ../TesisNov26-sims` once it is no
longer needed (or just leave it; it costs a checkout).

## What is *not* in either branch

`out/`, `out3d_colab/` and `runs/` are gitignored, on purpose: they hold
certificates, movies and run logs, which are evidence but not source. Their
numbers are copied into `DOCS/Results.md` / `RUNTIMES.txt` in the same sitting,
because a gitignored certificate is one `rm -rf` away from being unreproducible.
