# ArticlePlan.md — “Reusable by Construction”

**Status:** first draft scaffolded in `ARTICLE/` (2026-11-21). `make check`
passes with 0 undefined references/citations; 172/172 tests pass. The draft is
longer than the target and has not had an editorial read.
**Goal:** write an article that uses this repository as a case study for how AI
assistance (mostly the **Pi** coding agent running **DeepSeek** models) took a
research codebase from one-off runs to code that is reusable, verifiable and
portable.
**Working title:** *Reusable by Construction: An AI-Assisted Path from Research
Scripts to a Verified Groundwater Remediation Stack*
**Length target:** 8,500–10,000 words main text, 14–18 pages, 8 figures,
6 tables, 2 appendices. A 4,500-word short version is a later cut, not a
separate draft.

---

## 1. Editorial brief

### 1.1 Audience and venue

Primary audience: computational geoscience / environmental modelling / research
software engineering. They care about *whether the result survives*, not about
the size of the neural network.

Possible venues, in order of fit:

1. *Geoscientific Model Development* (GMD) — methods + code + reproducibility;
   requires a code availability statement and a model description.
2. *Environmental Modelling & Software* — the reusable-stack angle is the point.
3. *Computing in Science & Engineering* — the AI-collaboration angle is the
   point.
4. A long-form technical article / blog + repository, if the peer-review
   timeline is not wanted.

The outline below is venue-neutral. GMD would need a shorter AI section and a
longer model-description appendix; CISE would need the reverse.

### 1.2 One-sentence thesis

> Reusability is not a refactor; it is the accumulation of falsifiable gates,
> written memory, and one reference implementation — and an AI agent made that
> accumulation cheap enough to do on every failure instead of once at the end.

### 1.3 Three claims the article must earn

| # | Claim | Evidence the article must show |
|---|---|---|
| C1 | Reusability is measurable, not asserted. | Same policy reproduces on a third machine; NumPy↔torch parity; fresh-scenario certificates; cross-backend agreement. |
| C2 | The reusable artifacts came from a repeatable loop, not a heroic rewrite. | Each mechanism/test/gate is traced to a specific failure and a commit. |
| C3 | AI’s contribution was verification throughput and memory, not code volume. | Failure catalogue where the agent found/proved/pinned defects; examples where it was confidently wrong and the human/gate caught it. |

### 1.4 Non-goals

- Not a thesis, and not the 42-page `TEX/build/tesis.pdf` assembly. That document
  is an assembly of chapters with no introduction or conclusions; it is source
  material, not the article.
- Not a full derivation of the ADE, the stochastic program, or every algorithm.
  Derivations are compressed and delegated to appendices / cited chapters.
- Not a model benchmark (“Pi vs DeepSeek vs Gemini”). The article is about a
  *workflow*, with model names disclosed for reproducibility.
- Not a claim that AI writes better code. The claim is narrower: it makes
  verification and record-keeping cheap enough to be habitual.

### 1.5 Voice and disclosure

Use “we” for the project, and name the collaborators explicitly when describing
the process: **human (PI/editor)**, **Pi (agent harness: filesystem, shell,
tests, git)**, **DeepSeek (model)**, with Gemini as a tertiary reviewer. A short
“Collaboration note” box in §3 should state this plainly, including that the
human chose the gates, accepted the results, and wrote the final judgements.

---

## 2. Story spine (four acts)

1. **The mess.** Research code as a sequence of runs: a best-seed number in a
   slide, a dataset whose recorded shape was wrong, an FNO that was structurally
   incapable of learning a front, and a “5× speedup” that had never been
   measured. The problem is not missing tests; it is missing *decisions about
   what counts as evidence*.
2. **The discipline.** The loop that produced the repo: read the memory files →
   reproduce the claim → make the smallest change → run the gate → write the
   finding → commit with the finding in the message. AI makes each turn cheap;
   the repo makes each turn durable.
3. **The payoff.** Results that survive re-derivation: the 3D paper preset
   reproduced on a third machine; the 2D policy certified on 400 unseen
   scenarios; the 1D baseline honestly reported as a median with a 7× spread.
4. **The collaboration.** What Pi/DeepSeek did well, where they failed, and the
   human protocol that turned plausible output into evidence.

The narrative tension is **“the number looked fine” vs “the number survived.”**
Every section should return to that.

---

## 3. Detailed outline and word budget

| § | Section | Words | Job | Primary sources |
|---|---|---:|---|---|
| 0 | Abstract | 250 | Thesis, case, three results, one caveat | written last |
| 1 | Introduction | 900 | Research-code problem; why AI changes the economics; contributions | `DOCS/README.md`, `HUMAN.md`, `GROWTH.md` |
| 2 | The scientific setting | 1,300 | Enough physics to make the gates meaningful | `TEX/formulation.tex`, `TEX/optimization.tex`, `TEX/current_equations_algorithms.tex` |
| 3 | Method: the AI-assisted reusability loop | 1,400 | The loop, roles, artifacts-as-memory, evidence | `DOCS/Actions.md`, `SYNCH.md`, git history |
| 4 | Five mechanisms that made the code reusable | 2,200 | The new contribution; before/after for each | `GROWTH.md`, `USAGE.md`, `tests/`, `configs/`, code |
| 5 | Failure catalogue: every defect becomes a guard | 1,700 | Empirical heart; show, don’t boast | `Actions.md`, `Conclusions.md`, `Results.md`, commits |
| 6 | Results that survive re-derivation | 1,300 | Payoff; tables, not prose | `TEX/results_2d.tex`, `3D_results.md`, `Conclusions.md` |
| 7 | Discussion: what AI did and did not do | 1,000 | Strengths, failure modes, human protocol, threats | new + failure catalogue |
| 8 | Conclusion | 350 | Restate C1–C3, one forward-looking sentence | new |
| — | Reproducibility / code availability | 200 | Commands, configs, artifacts, licenses | `README.md`, `USAGE.md` |
| A | Appendix: repository map and gate definitions | 1,000 | Reuse tables, not prose | `USAGE.md`, `2D_plan.md`, `plans_2d_3d.tex` |
| B | Appendix: full result tables and seed sweeps | 800 | Every number with seed count | `runs/`, `out/`, `Conclusions.md` |

Main text ≈ 10,400 words. If the target is 8,500, cut §2 by 400, §4 by 500, and
§5 by 400 — never cut §6’s caveats.

### 3.1 Section-level beats

**§1 Introduction.** Open with the concrete embarrassment: a recorded FNO
relative L2 of 0.973 was not weak, it was worse than predicting zeros; a
dataset’s recorded shape was wrong; the “best” PINN number was the best of five
seeds with a 7× spread. Then the pivot: the fix was not a better model, it was a
better *epistemic process*. State C1–C3 and the roadmap.

**§2 The scientific setting.** Three pages maximum:
- 2.1 1D ADE sandbox, Ogata–Banks reference, PINN hard-BC ansatz, FNO operator;
  the corner singularity and why two metrics are reported.
- 2.2 2D/3D Darcy + transport, wells, log-normal `K`, KL expansion, CVaR chance
  constraint, SAA, pattern search.
- 2.3 What a “result” is in this project: median of N≥5 seeds + IQR; a gate is
  a go/no-go; a certificate is an out-of-sample re-evaluation with a Wilson
  interval.

**§3 Method.** The loop is the unit of analysis. Name its six steps and give one
concrete turn for each. Describe the memory hierarchy: `HUMAN.md` (for the
human), `Actions.md` (run log), `Conclusions.md` (what is trustworthy),
`USAGE.md` (how to run), commit messages (why). Note the rule that the agent
must read the docs before acting, and that a finding is not done until it is
written where the next session will look.

**§4 Five mechanisms.** Each subsection follows the same template: *failure that
demanded it → mechanism → what it caught → cost*. The five:
1. **One reference implementation, many backends.** NumPy is the reference;
   `physics/fv2d_torch.py` and `stoch/saa_torch.py` are the fast path; parity is
   a gate (28 tests at 1e-12…1e-10), not a hope.
2. **Configuration and manifests.** `configs/opt/*.yaml`, `run_config.json`, git
   SHA, seed, device, dtype; a missing `--config` is an error, not a silent
   default. This is what makes “reproduce the run on another machine” one
   command.
3. **Gates, not assertions.** O0–O4 for the 2D stack, G2.x/G3.x for the physics,
   six 3D gates, plus trivial-predictor guards. A gate has a threshold, an
   artifact, and a failure mode.
4. **Statistics as infrastructure.** `seed_sweep.py`: median/IQR, paired
   comparison, refusal to compare unpaired runs. The decision rule (beat the
   median by more than the IQR) is encoded, not remembered.
5. **Artifact-first remote execution.** Colab/fleet runs zip results from a
   `finally`, live logs are pulled from the VM filesystem, `check_tail.py`
   re-scores a policy on fresh scenarios. The run is not the process; the
   artifacts are.

**§5 Failure catalogue.** A table plus 4–6 short narratives. Columns: symptom →
root cause → permanent guard → commit. Minimum entries:
- FNO constant input ⇒ constant output (0.973 → 1.99e-2 median over five
  seeds, 1.46e-2 best seed; proven by construction).
- One diverging Krylov candidate killed a whole run ⇒ `--allow-nonfinite`;
  a dying run lost its artifacts ⇒ zip in `finally` (14 tests).
- `build_tex.py` merged preambles and dropped `\newtheorem`; four compiler-only
  defects, each pinned by a test.
- Seed variance / τ-ramp negative result (the cautionary tale).
- 3-well vs 4-well policy: a torch-passing reliability gate that NumPy
  contradicted (0.9625 vs 0.90).
- “A100 is no faster than a T4” / “4022 matvecs per evaluation”: measuring the
  claim before optimizing it.
- `max|R| = 1/ε` is structural, not physics; the residual max is a useless
  diagnostic here.

**§6 Results.** One table per scale, each with seed count and artifact path:
- 1D: full-grid median 1.61e-2 (IQR 2.18e-2, 7.0× spread), interior 1.42e-2
  (12.3×); FNO 5-seed median 3.02e-2 at 120 epochs.
- 2D: two certified policies 1.2% apart on different well sets; 400-scenario
  certificate 0.998/0.983; fresh draw keeps J to 0.02% but moves CVaR from
  +0.005 to +0.616.
- 3D: paper preset `q* = [12.5×4, 0×5]`, test J 547,665.31, reliability
  0.96875, mass balance 9.7e-15, reproduced on a third machine in 1014.8 s.
- Cost: 47× from batching a compass sweep; 2.46× from mixed float32; A100 not
  faster than T4 at this geometry.

**§7 Discussion.** Separate *agent harness* from *model*:
- What the harness bought: the ability to run the test, read the failure, edit
  the file, rerun, and write the finding — thousands of times, without fatigue.
- What the model bought: cross-language parity work, root-cause hypotheses,
  documentation synthesis, test generation.
- Failure modes observed: confident single-seed claims; plausible but
  structurally wrong fixes; over-broad “cleanups”; metric drift when a script
  is edited; the temptation to report the best run.
- Human protocol: demand a failing test first; require a negative result to be
  recorded; ask “what would falsify this?”; keep the final judgement and the
  gate thresholds human-owned.
- Threats to validity: one project, one team, changing models, no controlled
  comparison; the article is a experience report, not an experiment.

**§8 Conclusion.** Reusability is a property of the *evidence trail*, not of the
directory layout. The AI made the trail cheap; the gates made it trustworthy.

---

## 4. Reusing the TeX sources

### 4.1 What exists

`TEX/` contains six standalone `article` documents plus one chapter-shaped
fragment:

| File | Lines | Role today |
|---|---:|---|
| `formulation.tex` | 425 | 1D ADE, PINN/FNO, verification metric, seed-variance results |
| `optimization.tex` | 2,212 | Full stochastic program: wells, KL, SAA, CVaR, algorithms |
| `plans_2d_3d.tex` | 1,018 | 2D/3D models, reference hierarchy, gates, roadmap |
| `SolvingAlgorithms.tex` | 1,727 | Algorithm selection: NLP, PHA, decomposition, derivative-free |
| `current_equations_algorithms.tex` | 240 | Compact “equations and algorithms under test” note |
| `results_2d.tex` | 126 | 2D certified policies, certificate, plume, cost |
| `build_tex.py` | — | Assembles `formulation` + `optimization` + `results_2d` into 42 pages |

The assembled thesis is **not** the article: it has no introduction, related
work or conclusions, and the chapters overlap (46 duplicate labels). Do not
`\input` the standalone files directly; their preambles collide and their
section numbering is chapter-local.

### 4.2 Reuse rule: lift the body, not the preamble

Create a new `ARTICLE/` tree. Copy the body text of the reusable sections into
article-shaped files, strip the standalone preamble, and edit for the new
narrative. Keep `TEX/` as the archival standalone set. Record every lifted
section in `ARTICLE/SOURCE_MAP.md` so the provenance is auditable.

```
ARTICLE/
  article.tex              # master: preamble, title, \input sections
  refs.bib
  SOURCE_MAP.md            # TEX source -> article section -> edit made
  sections/
    00-abstract.tex
    01-intro.tex
    02-case.tex            # lifted/condensed from formulation + optimization
    03-method.tex          # new
    04-mechanisms.tex      # new, with code excerpts
    05-failures.tex        # new, from Actions/Conclusions/commits
    06-results.tex         # lifted from results_2d.tex + 3D_results.md
    07-discussion.tex      # new
    08-conclusion.tex      # new
    appendix.tex           # gates, repo map, full tables
  figs/                    # selected figures, copied or symlinked
  Makefile                 # latexmk -pdf article.tex
```

### 4.3 Source-to-section map

| Article section | Source | Action | Why |
|---|---|---|---|
| §2.1 1D model | `formulation.tex` §1–6 | **Lift and condense** (~60%) | Already written; only trim the derivation and fix notation. |
| §2.1 metrics | `formulation.tex` §8 | **Lift nearly verbatim** | Median/IQR and corner-exclusion policy are exactly the article’s point. |
| §2.2 stochastic program | `optimization.tex` §4–6 | **Condense to ~2 pages** | Full program goes to Appendix A; keep the decision variables, chance constraint, SAA/CVaR only. |
| §2.2 equations | `current_equations_algorithms.tex` | **Harvest equations** | Compact, implementation-matched forms; avoid a third restatement. |
| §2.3 gates | `plans_2d_3d.tex` §7 | **Lift the gate tables** | G2.x/G3.x tables are already publication-shaped. |
| §3 method | `DOCS/Actions.md`, `HUMAN.md`, `SYNCH.md` | **Rewrite as narrative** | Raw run log; must become a loop description, not a diary. |
| §4 mechanisms | `GROWTH.md`, `USAGE.md`, code, tests | **New writing + excerpts** | The core contribution does not exist in any TeX file yet. |
| §5 failures | `Actions.md`, `Conclusions.md`, `Results.md`, commits | **New writing** | Same: the failure catalogue is the new empirical content. |
| §6.1 1D results | `formulation.tex` §9, `Conclusions.md` §1–5 | **Lift table + rewrite** | Table is reusable; prose must be cut to one page. |
| §6.2 2D results | `results_2d.tex` | **Lift nearly verbatim** | Already chapter-shaped and numbered; only adjust cross-refs. |
| §6.3 3D results | `3D_results.md` | **Rewrite as table + 3 paragraphs** | Markdown table converts directly. |
| §6.4 cost | `results_2d.tex` §cost, `Conclusions.md` §9 | **Condense** | Keep 47×, 2.46×, A100/T4, 4022 matvecs. |
| Appendix A | `2D_plan.md`, `plans_2d_3d.tex`, `USAGE.md` | **Reuse tables** | Gate definitions and script inventory are reference material. |
| Appendix B | `runs/sweeps/`, `out/`, `Conclusions.md` | **Reuse numbers** | Every number gets an artifact path and seed count. |
| — | `SolvingAlgorithms.tex` | **Do not reuse in full** | 1,727 lines; cite one comparison table if the venue wants algorithm justification. |

### 4.4 Mechanical rules

1. **One master, one preamble.** `ARTICLE/article.tex` owns the preamble,
   `\title`, bibliography and `\input`s. Section files are body-only.
2. **No duplicate labels.** Prefix article labels `art:` (e.g.
   `\label{sec:art-mechanisms}`) to avoid collisions with lifted `TEX/` labels.
3. **One bibliography.** `refs.bib` (biblatex or natbib). The lifted chapters
   ship `thebibliography` blocks; convert every `\bibitem` to a BibTeX entry and
   de-duplicate by key (`ogata1961solution`, `li2020fourier`, …).
4. **Figures live in `ARTICLE/figs/`.** Copy the selected PNGs; do not
   `\includegraphics{../out/...}` from the build tree.
5. **Build gate.** `latexmk -pdf -interaction=nonstopmode article.tex` must end
   with 0 undefined references, 0 undefined citations, 0 overfull boxes over
   10 pt. Add `ARTICLE/Makefile` and a `check` target.
6. **Keep `TEX/build/tesis.tex` untouched.** The article is a separate
   deliverable; do not make the thesis build depend on it.

### 4.5 Master skeleton (proposed)

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,siunitx}
\usepackage[round,numbers]{natbib}
\usepackage[colorlinks=true,allcolors=blue]{hyperref}
\newcommand{\R}{\mathbb{R}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Cvar}{\mathrm{CVaR}}
\title{Reusable by Construction:\\ An AI-Assisted Path from Research Scripts
       to a Verified Groundwater Remediation Stack}
\author{S. Lucero\thanks{With the Pi coding agent (DeepSeek models).}}
\date{\today}
\begin{document}
\maketitle
\input{sections/00-abstract}
\input{sections/01-intro}
\input{sections/02-case}
\input{sections/03-method}
\input{sections/04-mechanisms}
\input{sections/05-failures}
\input{sections/06-results}
\input{sections/07-discussion}
\input{sections/08-conclusion}
\bibliographystyle{plainnat}
\bibliography{refs}
\appendix
\input{sections/appendix}
\end{document}
```

---

## 5. Evidence base

### 5.1 What the repo already contains

| Quantity | Value (2026-11-21) | Source |
|---|---:|---|
| Commits | 69 | `git rev-list --count HEAD` |
| Tracked files | 304 | `git ls-files \| wc -l` |
| Python lines | 23,739 | `git ls-files '*.py' \| xargs wc -l` |
| Test files | 22 | `tests/test_*.py` |
| Test functions | 172 | `grep -h "def test_" tests/*.py \| wc -l` |
| Optimization configs | 8 | `configs/opt/` |
| Runtime records | 33 | `runs/runtimes.jsonl` |
| Assembled thesis | 42 pages | `TEX/build/tesis.pdf` |

Do not quote the test count as “172 passing” until the suite has been run on the
article’s tree and the result recorded in Appendix B.

### 5.2 Evidence ledger (build this first)

Create `ARTICLE/EVIDENCE.md` with one row per number that will appear in the
article:

| Claim | Value | Seeds / N | Artifact | Command / commit |
|---|---|---|---|---|
| 1D full-grid median | 1.61e-2 | 5 | `runs/sweeps/baseline_n5` | `seed_sweep.py` |
| 1D interior median | 1.42e-2 | 5 | `runs/sweeps/baseline_n5` | `seed_sweep.py` |
| FNO 120-epoch median | 3.02e-2 | 5 | `runs/sweeps/fno_coords_120ep` | `seed_sweep.py` |
| 2D float32 policy | 428,579 | 40 test | `out/certified_20261120-111849` | `check_tail.py` |
| 2D fresh certificate | 391/400 | 400 | `out/cert/*.json` | `check_tail.py --seed-base 900000` |
| 3D paper preset | 547,665.31 | 128 test | `out3d_colab/run_paper` | `full3D_colab_run.py` |

No sentence in §6 without a row here. The article’s own quality gate is that
every quantitative claim is either multi-seed, derived, or explicitly labelled
single-seed.

### 5.3 Commands to run before drafting

```bash
# 1. Timeline and commit taxonomy
git log --reverse --format='%h|%ad|%s' --date=short > /tmp/git_timeline.txt
git log --format='%h %s' | grep -ciE 'test|fix|gate|verify|parity|certif'

# 2. Test inventory and one clean run
grep -h 'def test_' tests/*.py | sed 's/.*def //;s/(.*//' | sort > /tmp/test_names.txt
python tests/run_tests.py 2>&1 | tail -20

# 3. Build checks the article will cite
python SCRIPTS/build_tex.py --check-only

# 4. Runtime cost table
python SCRIPTS/runtimes_report.py

# 5. Gate artifacts
ls out/opt/hydrology_gpu_gates/ out3d_colab/run_paper/
```

Also collect 3–5 representative Pi/DeepSeek sessions (prompts + diffs) from the
human’s session history. The article needs at least one verbatim exchange: a
wrong hypothesis, the failing test that killed it, and the corrected fix. Without
that, §3 and §7 are assertions.

### 5.4 Commit vignettes to develop

| Commit | Story it tells |
|---|---|
| `ecf9537` | τ-ramp negative result; seed variance dominates |
| `149ce53` | `seed_sweep.py`: turning a statistical lesson into a tool |
| `3807908` | Replacing a best-seed claim with a median in `formulation.tex` |
| `bb95c08` | The residual max is structural (`1/eps`), not physics |
| `947fae8` | `Conclusions.md`: one place for what is trustworthy |
| `1868f33` | Float32 mode bought back, then certified |
| `4e506b1` | Survive a diverging candidate; keep artifacts of a dying run |
| `12dcebf` | 42-page assembly exposed 46 duplicate labels and 4 real defects |
| `3628b5d` | A real LaTeX toolchain found what static checks could not |
| `ff82689` | The 3D paper preset reproduces on a third machine |

---

## 6. Figures and tables

### 6.1 Figures

| # | Figure | Source | Status |
|---|---|---|---|
| 1 | Timeline: commits vs findings vs gates | new TikZ in `sections/03-method.tex` | done |
| 2 | Stack architecture: reference / batched / surrogate / gates / artifacts | new TikZ | to make |
| 3 | The AI-assisted reusability loop | new TikZ in `sections/03-method.tex` | done |
| 4 | 1D PINN vs Ogata–Banks: field + profiles | `out/figs/field.png`, `profiles.png` | exists |
| 5 | Residual maps at init and after training | `out/residual/residual_{init,trained}.png` | exists |
| 6 | FNO failure and fix: constant output vs coordinate channels | `out/fno10/fno_prediction.png` + new fixed run | partial |
| 7 | 2D policy effect on the plume | `out/certified_20261120-111849/figures/contamination.png` | exists |
| 8 | 3D well layout + convergence | `out3d_colab/figures/{well_layout,convergence}.png` | exists |

Combine 4+5 into one two-panel figure if space is tight. Figure 2 and 3 are the
only genuinely new drawings; budget one session for them.

### 6.2 Tables

| # | Table | Source |
|---|---|---|
| 1 | What a result means: metric, seeds, gate, certificate | new, from `Conclusions.md` §7 |
| 2 | 1D baseline: median/IQR/min/max/spread, full vs interior | `formulation.tex` Table |
| 3 | Failure catalogue: symptom → cause → guard → commit | new, from `Actions.md` |
| 4 | 2D certified policies: float64 vs float32 | `results_2d.tex` |
| 5 | Out-of-sample certificates and Wilson intervals | `results_2d.tex` + `out/cert/` |
| 6 | 3D presets: grid, n_kl, scenarios, q*, reliability, J | `3D_results.md` |

---

## 7. The AI-collaboration narrative

### 7.1 The distinction the article must make

Do not write “AI wrote the code.” Write what is true and more interesting:

- **Pi is the harness:** it can read the repo, run the test, see the failure,
  edit the file, rerun, and write the finding. Its advantage is *iteration
  count and memory*, not a superior single suggestion.
- **DeepSeek is the model:** it proposes the hypothesis, the patch, the test, the
  summary. Its failure modes are model failure modes: plausible structure,
  confident wrong claims, metric drift.
- **The human is the PI:** chooses the gates, decides what counts as evidence,
  rejects the tempting number, and owns the final text.

### 7.2 What to show, not claim

- One verbatim session excerpt: a wrong hypothesis, the gate that falsified it,
  the corrected fix.
- One example of the agent doing something a human would skip: 28 parity tests,
  a 400-scenario certificate, a timeline reconstruction from 69 commits.
- One example of the agent being confidently wrong: the “5× speedup” that was
  never measured; the single-seed τ-ramp “win”; the FNO fix that needed a
  constant-input proof before it was accepted.
- One example of documentation as memory: `HUMAN.md`’s gotchas (wrong Python,
  961 MB RAM, exit 143, stdout buffering) preventing a repeated waste of time.

### 7.3 Claims to avoid

- “AI reduced development time by X%” — no controlled baseline exists.
- “The agent wrote N% of the code” — line attribution is not measurable here and
  would invite a pointless debate.
- “The model is better than a human” — the article is about a workflow, not a
  bake-off.
- “The code is reusable because it has tests” — the article’s point is that the
  *gates and the evidence trail* are what make it reusable.

---

## 8. Writing order and milestones

Do not write front-to-back. Write the parts whose evidence is already frozen.

| Pass | Deliverable | Exit criterion |
|---|---|---|
| 0 | `ARTICLE/EVIDENCE.md` + `SOURCE_MAP.md` | Every §6 number has an artifact path and seed count |
| 1 | Skeleton + lifted §2 and §6.2 | `latexmk` builds; 0 undefined refs/cites; no duplicated labels |
| 2 | §5 failure catalogue (table + 4 narratives) | Each row has a commit and a test/gate |
| 3 | §4 five mechanisms | Each has failure → mechanism → caught → cost |
| 4 | §3 method + §7 discussion | Roles and loop are explicit; one verbatim session excerpt |
| 5 | §1 intro + §8 conclusion + abstract | Abstract contains one number, one caveat, one claim |
| 6 | Figures, tables, bibliography, consistency | Every figure referenced; every citation resolved; numbers match ledger |
| 7 | Adversarial review | A reader can answer “what would falsify this?” for C1–C3 |

### 8.1 Article-level gates

Borrow the repo’s own discipline. The article is not done until:

1. **Claim gate:** every quantitative sentence is in `EVIDENCE.md` with N and an
   artifact path.
2. **Caveat gate:** each headline result has its “still single-seed” or
   “inconclusive” caveat within one paragraph.
3. **Reproduction gate:** a reader with the repo can run the command in the
   caption and get the number (or a documented tolerance).
4. **Attribution gate:** the human/agent/model division is stated once, plainly,
   and not overclaimed.
5. **Build gate:** `latexmk` clean; 0 undefined refs/citations; no overfull box
   over 10 pt; figures at ≤ 300 dpi.
6. **Length gate:** main text ≤ 10,000 words; §2 ≤ 1,500; §6 ≤ 1,500.

### 8.2 Suggested session plan

- **Session 1:** evidence ledger + timeline + test inventory; decide venue.
- **Session 2:** extract §2 and §6 from `TEX/`; get a clean build.
- **Session 3:** failure catalogue (the empirical core).
- **Session 4:** five mechanisms; code excerpts.
- **Session 5:** method + discussion; session excerpts.
- **Session 6:** intro/abstract/conclusion; figures; full consistency pass.
- **Session 7:** adversarial read; cut 15%; final build.

---

## 9. Risks and open decisions

| Risk | Mitigation |
|---|---|
| The article reads as a repo changelog | Every subsection answers “what did this make possible?” not “what did we do?” |
| AI section reads as an advertisement | Include the confidently-wrong examples and the human protocol; no productivity percentages. |
| The physics is either too thin or too heavy | §2 is 1,300 words; full derivations in appendix; cite `optimization.tex`/`SolvingAlgorithms.tex`. |
| Numbers drift between article and repo | `EVIDENCE.md` is the single source; regenerate tables from artifacts, never retype. |
| Lifted TeX carries stale claims | Every lifted paragraph is checked against `Conclusions.md` §7 and the artifact; stale single-seed numbers are replaced. |
| Venue mismatch | Decide venue before Session 3; it changes §3/§4 ratio, not the outline. |
| Author/licence/consent for AI disclosure | Decide author list, acknowledgements and licence early; disclose model names and versions. |

**Decisions needed from the human before drafting:**

1. Venue (GMD vs EMS vs CISE vs long-form article).
2. Author list and how to name the AI collaborators (Pi/DeepSeek versions).
3. Whether to include verbatim session excerpts, and which ones.
4. Whether the article is allowed to say the code is “reusable” in the title, or
   whether the claim should be narrower (“toward reusable code”).
5. Target length: 8,500 or 10,000 words.

---

## 10. First concrete actions — done 2026-11-21

1. `ARTICLE/` exists with `article.tex`, `sections/`, `figs/`, `refs.bib`,
   `EVIDENCE.md`, `SOURCE_MAP.md`, `Makefile`, `README.md`.
2. `ARTICLE/EVIDENCE.md` is built from `Conclusions.md`/`Results.md` and the
   artifacts; every §6 number has a row.
3. `formulation.tex` §1–6/§8–9 and `results_2d.tex` are lifted into
   `ARTICLE/sections/02-case.tex` and `06-results.tex`; preambles and
   `thebibliography` blocks are stripped.
4. `python3 tests/run_tests.py` → 172/172 in 27.5 s;
   `python3 SCRIPTS/build_tex.py --check-only` → 3 chapters, 283 labels, 43
   macros, 21 packages, no errors; `make check` in `ARTICLE/` → 22 pages, 0
   undefined refs/citations.
5. The failure catalogue table is drafted in `sections/05-failures.tex`.

**Next:** editorial read, timeline figure, FNO before/after panel, cut to venue
length, author/model disclosure.
