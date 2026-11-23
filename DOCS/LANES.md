# Two lanes: simulations and the written argument

This repository is worked on by more than one agent session at a time. One lane
runs simulations and produces numbers; the other writes the documentation and the
thesis. They share a single checkout, a single git index and one CPU, so the
coordination that matters is not "who is editing what file" but **how a number
travels from a run to a sentence**.

Read this before starting work in either lane.

## Why the two lanes are not symmetric

| | simulations | documentation / thesis |
|---|---|---|
| bound by | latency — hours of compute, results arrive in bursts | throughput — minutes per result, dozens per hour |
| failure mode | a number that is superseded and never retracted | a number quoted from a live log or a run note |
| output | artifacts (`metrics.json`, `runs/**/*.json`, figures) | prose, tables, PDFs |
| consumes | configs, code, GPU/CPU | committed artifacts **only** |

Because the coupling is one-directional (documentation consumes numbers,
simulations consume nothing), the protocol is built around artifact provenance
rather than locks. A lock protects a file for minutes; provenance protects a claim
for the life of the thesis.

## Path ownership

Write only inside your lane. "Read only" is not a restriction on looking — it is a
restriction on *writing*.

| lane | owns | read-only |
|---|---|---|
| **simulations** | `stoch/`, `physics/`, `SCRIPTS/run_*.py`, `SCRIPTS/colab3d.py`, `SCRIPTS/multi_colab.py`, `configs/`, `out*/`, `runs/`, `RUNTIMES.txt`, `DOCS/SIM_PLAN_L1_L3.md`, `DOCS/RESEARCH_LINES.md`, `DOCS/3D_results.md` | `DOCS/Results.md`, `DOCS/USAGE.md`, `TEX/`, `ARTICLE/` |
| **thesis** | `TEX/` (including `TEX/figs/`), `SCRIPTS/build_tex.py`, `SCRIPTS/make_figures.py`, `SCRIPTS/check_tail*.py`, `tests/test_build_tex.py`, `tests/test_check_tail*.py`, `DOCS/USAGE.md`, `DOCS/Conclusions.md`, `DOCS/Results.md`, `DOCS/FIGURES.md`, `DOCS/ArticlePlan.md`, `ARTICLE/` | `RUNTIMES.txt`, `runs/`, `out*/`, `stoch/`, `physics/` |
| **both, append-only** | `DOCS/Actions.md` — add your own dated row, never edit or reorder another lane's row, commit it immediately | |
| **shared, single owner** | `DOCS/3D_results.md`, `RUNTIMES.txt`: the lane that produced the measurement owns its numbers; the other lane quotes it and does not edit it | |

`DOCS/Actions.md` is append-only on purpose: it is the only file both lanes must
touch, and a one-row-per-change rule keeps the conflict window to seconds.

When a lane needs something from the other and neither agent is running, the
request goes in [`HANDOFF.md`](HANDOFF.md): one entry per ask, answered in place in
the four-field format below. A request that cannot be answered is withdrawn, not
left open, and the entry is deleted in the same commit that lands the number in
prose.

## The three rules

### 1. No lane-wide git operations

```bash
git add -A            # NO: stages the other lane's work in progress
git add .             # NO: same
git commit -a         # NO: same
git stash             # NO: removes the other lane's working tree
git checkout -- .     # NO: discards the other lane's uncommitted edits
git reset --hard      # NO
git rebase            # only when the other lane has committed or is idle
```

Commit with explicit pathspecs, which is what makes two agents in one checkout
safe at all:

```bash
git add TEX/ DOCS/Conclusions.md DOCS/Actions.md
git commit -F - <<'MSG'
...
MSG
```

A pull that needs a clean tree (`git pull --rebase` fails on unstaged edits) is a
lane-crossing operation: coordinate it, or have the other lane commit first.

### 2. A number enters `TEX/` only from a committed artifact

Acceptable sources: a `metrics.json` inside a run directory, a tracked
`runs/**/*.json`, a tracked figure, a line in `RUNTIMES.txt` that names the run it
came from.

Not acceptable sources: a live `run.log` tail, a mid-flight stdout line, a chat
summary of a run, an arithmetic projection nobody wrote down ("78 s per
evaluation, so about 6.5 h").

**Corollary — corrections are announced.** When a measurement is superseded
(or its derived arithmetic turns out to be wrong), the lane that owns it says so
in one line — what was wrong, what is right, which artifact proves it — and the
other lane re-greps for the stale value. Two real examples from this repository:
a per-scenario cost divided by a larger scenario count (0.775 s/scenario was
really 1.06 s, and the 6.5 h projection was really 8–9 h), and a policy reported at
reliability 0.969 without its Wilson interval, which does not clear the 0.94 gate.
Both had already reached prose.

### 3. The thesis lane does not run the full test suite casually

`python3 tests/run_tests.py` now covers both lanes' modules. Run it against the
other lane's half-finished work and you get failures that are not yours and that
you cannot fix. While writing, gate on the targeted tests:

```bash
python3 tests/test_build_tex.py                       # ~0.7 s, no LaTeX needed
python3 SCRIPTS/build_tex.py --compile                # ~40 s
pdftotext TEX/build/tesis.pdf - | head -40            # the real gate
```

and run the full suite at commit time, reading any failure in `test_pha`,
`test_dro`, `test_streaming`, `test_check_tail3d` or `test_fv3d_torch` as the
other lane's. A passing build is not sufficient: it will happily assemble a
duplicated table-of-contents entry or a stale number, which is what the
`pdftotext` check is for.

## Compute etiquette

- The simulation lane sets `OMP/OPENBLAS/MKL_NUM_THREADS=1` on purpose (threaded
  BLAS oversubscribes these transport-bound runs: 5.28 s → 2.02 s per evaluation
  at 160×80). Treat the cores as theirs.
- Documentation work is seconds of LaTeX; it never needs to be scheduled.
- The real collision is memory, not CPU: this box has 8 cores and 15 GB, and two
  `check_tail`-scale studies plus a browser leave about 2 GB free. One
  memory-heavy study at a time, at most three parallel workers.
- Long studies are launched detached (`setsid nohup ... < /dev/null &`) with logs
  under the run's own directory, never in a foreground pane that a later command
  can kill.

## Handoff format

At each milestone the simulation lane emits one message, in this shape:

```
result:      what was measured, in one sentence, with the number
artifact:    path(s) to metrics.json / runs/**/*.json / figure
commit:      sha that contains the artifact (or "uncommitted, final" + reason)
supports:    the one claim this licenses in prose
contradicts: the stale number or claim it replaces, if any
```

The documentation lane pulls at that point; the simulation lane does not push
prose. Everything else — interim progress, failed attempts, half-converged runs —
stays in the log files and does not cross.

## Branches

> **Superseded for the branch layout.** Sergio chose a two-branch split on
> 2026-11-22: `code-sims` for code and the evidence files, `doc-latex` for the
> prose, with `main` fast-forwarded to `code-sims` and a one-way merge
> `doc-latex <- code-sims`. See `BRANCHING.md`. Everything below about *how* to
> commit inside a branch — path-scoped adds, no lane-wide git operations — still
> stands, and it is what made the split survivable.

Stay on **one branch**, with path-scoped commits. Two live branches would cost more
in conflicts over the shared append-only files (`DOCS/Actions.md`, `RUNTIMES.txt`,
`runs/*.jsonl`) than the isolation buys.

Fork a branch only when a refactor will leave `stoch/` or `physics/` broken for
hours, and then use a git worktree at a fixed base rather than a parallel trunk:

```bash
herdr worktree create --branch exp-<name> --base doc-latex --path ~/TesisNov26-<name>
```

Merge it back by rebase once the refactor is green, so the mainline never sees a
red suite it did not cause.

## Quick checklist before you commit

- [ ] every file in the commit is inside your lane, or is your own `Actions.md` row
- [ ] every number in the diff traces to an artifact path you can name
- [ ] no derived arithmetic that a second person cannot reproduce from the artifact
- [ ] `git status --short` shows the other lane's work still dirty and still yours to leave alone
- [ ] the build/test gate for your lane passed (`build_tex.py --compile` + `pdftotext`, or the suite)
