# SYNCH.md — Keeping Every Machine on the Same Commit

**Project:** Stochastic Groundwater Remediation (2D/3D PINNs, FNOs, Monte Carlo / MODFLOW-MT3DMS)
**Status:** proposal — nothing in this document is implemented yet (there is no `.github/` in the repo)
**Companion docs:** `DOCS/NewPlan.md` (fleet topology, §3 repo flow, §6 security, §9 open items), `DOCS/HUMAN.md` (operator gotchas), `RUNTIMES.txt` (what is cheap to verify)
**Question answered here:** can GitHub Actions be the thing that keeps `ubuntu-hub`, `do-relay`, `surface-ctrl`, Colab and any new box on the same state — and does Tailscale belong in that loop?

---

## 0. TL;DR

* **GitHub Actions should own _verification_ and _notification_. It should not own
  _transport_.** A workflow that runs the test suite, builds the TeX, and then
  tells the nodes "main moved to `<sha>`" gives the fleet a single definition of
  "synchronized"; letting CI push file contents to nodes does not.
* **Tailscale is the transport, not the sync engine.** It has no file-sync
  semantics of its own. What it gives is: MagicDNS names that resolve from a
  GitHub runner, ACLs that let one ephemeral CI node reach exactly the servers
  it may reach, and Tailscale SSH so no keys live in Actions secrets.
* **Recommended shape (phases in §7):**
  1. CI gate on every PR/push (tests + LaTeX + gates smoke). *No node contact.*
  2. A pull timer on each node (`systemd` timer / cron) running a
     `SCRIPTS/sync_pull.sh` that does `fetch` + `merge --ff-only`, refuses on a
     dirty tree, and records the SHA. *No inbound reachability needed.*
  3. A self-hosted runner on `ubuntu-hub` so Actions can run hub-only work
     (74 s gates, tailnet probes) without exposing SSH.
  4. Optional: Actions → tailnet push (ephemeral `tailscale/github-action` node)
     to make step 2 immediate instead of "within 10 minutes".
* **The layer that actually breaks silently is not git — it is the Python
  environment** (`requirements.txt` uses `>=`, and torch differs per node) and
  the **ignored artifacts** (`out/`, `figs/`, `ZIPS/`, `*.npz`). Git sync alone
  will not make two machines reproduce each other. See §6.

---

## 1. What "synchronized" has to mean

Five different things are colloquially called "the repo", and they need
different mechanisms. Conflating them is how a fleet ends up with identical
`git log` output and different numbers in `summary.json`.

| Layer | Authority | Mechanism | State today |
| :--- | :--- | :--- | :--- |
| Code + docs + `TEX/*.tex` | `origin/main` on GitHub | `git pull --ff-only` | manual, per machine |
| Compiled `TEX/*.pdf` | committed (`.gitignore` whitelists `!TEX/*.pdf`) | same as code | manual, re-committed by hand |
| Python env | `requirements.txt` (floor pins) + node-specific torch wheel | per-node install | drifts; no lock file |
| Run artifacts (`out/`, `figs/`, `ZIPS/`, `runs/runtimes.jsonl`) | **not** git (`.gitignore`) | archive + `rsync` over tailnet, or CI artifact | ad hoc |
| Service state (`herdr server`, `docker compose`, `app.py`) | each node | systemd / compose | manual |

A machine is "synchronized" when (a) `git rev-parse HEAD` equals the expected
SHA, (b) the working tree is clean, (c) the env hash matches, and (d) the node's
role-appropriate services are up. `SCRIPTS/fleet_status.sh` already reports head,
dirty, ahead and behind (against the configured upstream, i.e. as of the last
fetch) plus the service checks in (d); it does not cover (c), it does not
fetch before reporting, and it has no notion of the "expected SHA" the fleet
should be on.

---

## 2. Current state (verified, not assumed)

* **No CI.** No `.github/` directory exists. Nothing runs on push. The last
  merge (`bfecf95`) was conflict-resolved by hand.
* **GitHub is the source of truth** (`DOCS/NewPlan.md` §3) and `ubuntu-hub`
  keeps a bare mirror at `~/tesis.git` that is refreshed *by hand*
  (`git -C ~/tesis.git fetch --prune`); the doc suggests a cron/Timer but no
  unit exists in the repo.
* **`do-relay` compiles TeX** through the compose stack (`docker-compose.yml`),
  so the relay can drift from `main` and still serve a plausible-looking PDF.
  That is the highest-risk silent drift in the current setup.
* **Ephemeral nodes (Colab, `deploy_do.sh` droplets) clone at boot** and never
  report back. `run_colab.sh` / `remote_run.sh` write sentinels
  (`/tmp/tesis-colab.done`, `/tmp/tesis-run.done`), which is a *job* protocol,
  not a *state* protocol: a sentinel says a run finished, not which commit it
  ran from.
* **`fleet_status.sh --json`** already emits host/tailnet/herdr/repo/sentinel/
  disk data and `app.py:/fleet` renders it on the tablet — the natural read-out
  surface for whatever this document builds. Extend it rather than start over.
* **`RUNTIMES.txt`** tells us what CI can afford: the full unit suite is
  ~2 s wall and the verification gates are ~74 s. Both fit comfortably in a
  hosted runner; `--stage opt` (>30 min, >1800 s timeout on 1 vCPU) does not.

---

## 3. The options, honestly compared

### A. Node-side timer pull (no Actions in the loop)

Each node runs a `systemd` timer (or `cron`) every N minutes:
`git fetch --prune && git merge --ff-only origin/main`.

* **Pros:** zero secrets, zero inbound reachability, works on Colab-style NAT
  and on the tablet, trivial to reason about, works while GitHub is rate
  limiting API calls. Survives CI being down.
* **Cons:** latency up to the interval; every node talks to github.com
  independently (fine at this fleet size); a dirty tree must be skipped
  loudly, or an operator loses work.
* **When it wins:** this is the correct *baseline* for every long-lived node.

### B. Actions as verifier + artifact publisher (no node contact)

Workflow on `push`/`pull_request`: run `tests/run_tests.py`, build every
`TEX/*.tex`, upload PDFs and a `manifest.json` (commit SHA, file hashes,
resolved pip freeze) as run artifacts.

* **Pros:** gives the fleet one definition of "green"; catches a broken commit
  before four machines pull it; PDFs and lock files become downloadable with
  `gh run download`; no infra to keep alive.
* **Cons:** does not move anything onto a node by itself; artifacts expire
  (default 90 days) and are per-run, so they are a verification record, not a
  distribution channel.
* **When it wins:** always. This is phase 1 and it is cheap.

### C. Actions → tailnet push

Workflow joins the tailnet with `tailscale/github-action` (OAuth client →
ephemeral, tagged node), then `ssh ubuntu-hub 'SCRIPTS/sync_pull.sh'` (or a
`curl` to a tiny sync endpoint) for each target.

* **Pros:** sync is immediate on merge; you can gate on the remote command's
  exit code; no long-lived credential on the nodes beyond their tailnet identity.
* **Cons:** adds a moving part (ephemeral CI node + ACL rule) for a latency win
  measured in minutes; a compromised workflow can ssh into the hub; ephemeral
  nodes appear in `tailscale status` and must be tagged and expiring; the runner
  uses userspace networking, so only outbound/`tailscale ssh` works.
* **When it wins:** only if "members of the fleet react to `main` within
  seconds" is a real requirement. It is a nice-to-have here, not a need.

### D. Self-hosted runner on `ubuntu-hub`

Register a runner (label e.g. `hub`) that dials out to github.com; queue
hub-only jobs to it (`runs-on: [self-hosted, hub]`).

* **Pros:** no inbound port (the runner is outbound-only), no auth keys in
  Actions at all, jobs get the hub's CPU, its bare mirror, its tailnet identity
  and its sentinels for free — so "run the gates" and "probe the fleet" become
  ordinary workflow steps; `gh run watch` gives the tablet a progress view.
* **Cons:** the runner is a persistent process on the box that also hosts
  Herdr and the experiments — it needs a dedicated unprivileged user, a work
  directory outside `~/TESIS`, and its own systemd unit; a malicious PR from a
  fork could run on it (default-deny fork PRs, or restrict to
  `push`/`workflow_dispatch`).
* **When it wins:** as soon as CI needs to touch the tailnet or the GPU-less hub
  at all — which, for this repo, is immediately (gates, `/fleet` probe, LaTeX).

### E. `tailscale serve` sync endpoint on the hub

One small HTTP handler on the hub ("POST /sync → run `sync_pull.sh`, return
SHA"), published with `tailscale serve` so it is reachable **only inside the
tailnet**, then called from Actions or from the tablet.

* **Pros:** no SSH from CI, trivially auditable, the same endpoint can serve the
  `fleet_status.sh --json` payload.
* **Cons:** a new network service to write, secure and keep alive; `app.py`
  already exists, so this is really "add a POST route to `app.py` and
  `tailscale serve --bg 5001`" — small, but still code that can be wrong.
* **Never** use `tailscale funnel` for this: that is public internet exposure
  of a mutating endpoint.

### F. Third-party sync (Syncthing, Dropbox, `rsync` cron) — rejected

Bidirectional file sync over a *git working tree* fights git: it produces
conflict copies of tracked files, resurrects deleted ones, and moves `.git/`
internals between machines. Use it for `out/`/`figs/` if you must (Syncthing
over the tailnet is a reasonable choice for artifacts), never for the repo.

### Comparison

| Option | Direction | Inbound port | Secrets | Moving parts | Main failure mode | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A pull timer | node → GitHub | none | none | 1 unit/node | silent skip on dirty tree | **adopt** |
| B CI verify + artifacts | Actions only | none | none | 1 workflow | expiry of artifacts | **adopt** |
| C Actions → tailnet push | Actions → node | none (tailnet) | OAuth client | ephemeral node, ACL | CI holds hub reachability | optional |
| D runner on hub | node → GitHub | none | none | runner + unit | fork PR abuse | **adopt** |
| E serve endpoint | tailnet → hub | none (tailnet) | none | new route | mutating endpoint bugs | later, if needed |
| F file sync | bidirectional | varies | varies | lots | conflicts in `.git/` | reject |

---

## 4. Can Tailscale be used? Yes — as the transport, with caveats

Tailscale does **not** synchronize anything. It has no git awareness, no file
semantics, and no "push to all nodes". What it provides, and how each piece is
relevant:

| Tailscale feature | Role in a sync design | Caveat for this repo |
| :--- | :--- | :--- |
| MagicDNS (`ubuntu-hub`, `do-relay`, `surface-ctrl`, `tail39daae.ts.net`) | stable names inside workflows and scripts — no IPs in Actions | a Colab node gets a *new* name each session; never hard-code it |
| ACL tags (`SCRIPTS/fleet-acl.hujson`) | let a CI identity reach exactly the hub/relay on 22 (or 5001) and nothing else | needs a new `tag:ci` in `tagOwners` + one grant; do not attach `autogroup:member` to servers |
| `tailscale/github-action` (OAuth client) | ephemeral, tagged, expiring runner node; then plain `ssh`/`curl` by MagicDNS name | uses userspace networking: outbound only; OAuth client secret is a real secret — store it in repo secrets, never in this file; verify action inputs against its current README rather than trusting this doc |
| Tailscale SSH | keyless SSH from the CI node to the hub; identity/ACL decides, not `~/.ssh` | currently disabled on this tailnet (`RunSSH=false` per `fleet-acl.hujson`); enabling it is a console action, so phase 4 must not assume it |
| `tailscale serve` | expose a hub endpoint (e.g. `:5001` sync/status) to the tailnet only | `serve` is HTTP(S) only and only inside the tailnet; `funnel` is public — do not use it |
| `tailscale ping` / `status --json` | cheap preflight in a workflow or in `fleet_status.sh` | an unreachable node should fail the *notify* step, never block the merge |

**Does the CI runner even need the tailnet?** Only for option C. If you take the
recommended path (A + B + D), Actions never joins the tailnet: the hosted runner
verifies and publishes, and the hub's own runner does the tailnet-side work
under the hub's existing identity. That is strictly less attack surface than
handing an ephemeral CI node SSH rights, which is why C is marked optional.

**Can a node behind NAT be reached?** Only outbound is required for A and D, so
Colab and `deploy_do.sh` droplets need nothing from Tailscale for sync — they
clone at boot and should *pin* the SHA they ran (§8).

---

## 5. Concrete sketches

> These are proposals. Committing them means creating `.github/workflows/` and
> `SCRIPTS/sync_pull.sh`; neither exists yet.

### 5.1 `.github/workflows/ci.yml` — verification only

```yaml
name: ci
on:
  push: { branches: [main] }
  pull_request:
  workflow_dispatch:

jobs:
  tests:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix: { python: ["3.12"] }        # matches RUNTIMES.txt host
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "${{ matrix.python }}", cache: pip }
      # CPU torch wheel: CPU-only, but keeps tests/test_fv2d_torch.py in scope.
      - run: pip install -r requirements.txt
      - run: python -u tests/run_tests.py          # ~2 s wall, whole suite
      - uses: actions/upload-artifact@v4
        if: always()
        with: { name: test-log, path: runs/ }

  tex:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: sudo apt-get update && sudo apt-get install -y texlive-latex-extra texlive-fonts-recommended
      # \today makes every build differ; pin the clock so a rebuild is comparable.
      - env: { SOURCE_DATE_EPOCH: "1700000000", FORCE_SOURCE_DATE: "1" }
        run: |
          for f in TEX/*.tex; do
            ( cd TEX && pdflatex -interaction=nonstopmode -halt-on-error "$(basename "$f")" )
            ( cd TEX && pdflatex -interaction=nonstopmode -halt-on-error "$(basename "$f")" )
          done
      - uses: actions/upload-artifact@v4
        with: { name: tex-pdfs, path: TEX/*.pdf }
```

Notes that matter:

* **Do not** gate on `git diff --exit-code TEX/*.pdf`. `current_equations_algorithms.tex`
  uses `\date{\today}`, so the committed PDF can never be byte-identical to a
  fresh build unless the clock is pinned — hence `SOURCE_DATE_EPOCH` +
  `FORCE_SOURCE_DATE`. Even then, a hypothetical "PDF is stale" gate is a
  maintenance tax; prefer "compiles cleanly, zero warnings" as the gate, and
  re-commit PDFs deliberately (as done in `ecd1a22`).
  *Verified:* two clean 2-pass builds of `current_equations_algorithms.tex` with
  those two env vars set are byte-identical (`md5` match). The build must start
  from a tree with no stale `.aux`/`.out` — a fresh checkout in CI is fine, but
  a local rerun needs `rm -f TEX/*.aux TEX/*.out TEX/*.toc` first, or pass 1
  reads pass 2's leftovers and the bytes differ.
* Both PDFs are tracked (`!TEX/*.pdf` in `.gitignore`), so the artifact is a
  convenience, not the record.
* `--stage opt` must **not** be in CI. `--stage gates` (~74 s) may be, if kept
  on the self-hosted runner so hosted-runner time stays free.

### 5.2 `SCRIPTS/sync_pull.sh` — the contract every node runs

Same conventions as the rest of `SCRIPTS/` (`set -Eeuo pipefail`, sentinel in
`/tmp`, `--dry-run`, read-only default, exit code meaningful):

```
sync_pull.sh [--repo DIR] [--remote origin] [--branch main] [--dry-run] [--json]

 1. verify $repo is a git work tree on $branch (else exit 3, do nothing)
 2. refuse if the work tree is dirty (exit 4) — never stash, never reset
 3. git fetch --prune $remote
 4. if merge-base --is-ancestor HEAD $remote/$branch:
        git merge --ff-only $remote/$branch     # fast-forward only
    else exit 5 (diverged — a human decides; do not force)
 5. append {ts, host, old_sha, new_sha, ok} to /tmp/tesis-sync.log
 6. if HEAD changed and post-merge hooks exist (e.g. "rebuild TEX/"),
    run them; failure is reported, not rolled back
 7. --json: print {host, branch, sha, ahead, behind, dirty, ok}
```

Why these rules: `--ff-only` makes divergence loud instead of silently merging
an experiment branch into `main`; the dirty check is the difference between "a
timer nobody notices" and "a timer that ate my uncommitted figure tweak"
(`DOCS/HUMAN.md` is full of these); fast-forward keeps the hub's bare mirror
useful.

Pair it with a timer, e.g. `tesis-sync.timer` (10 min) and a oneshot service
running `ExecStart=%h/TESIS/SCRIPTS/sync_pull.sh`, `Nice=10`, logs to journald.
On Colab there is no timer: `run_colab.sh` calls the same script in
`--repo /content/TESIS` mode so the pinning logic is identical.

### 5.3 Optional: Actions → tailnet (only if latency matters)

```yaml
name: notify
on: { push: { branches: [main] } }
jobs:
  poke:
    runs-on: ubuntu-latest
    steps:
      - uses: tailscale/github-action@v3
        with:
          oauth-client-id: ${{ secrets.TS_OAUTH_CLIENT_ID }}
          oauth-secret:    ${{ secrets.TS_OAUTH_SECRET }}
          tags: tag:ci
          ping: ubuntu-hub
      - run: ssh -o StrictHostKeyChecking=accept-new "${{ vars.TS_SSH_USER }}@ubuntu-hub" '~/TESIS/SCRIPTS/sync_pull.sh --json'
```

Requires, in `SCRIPTS/fleet-acl.hujson`: `"tag:ci": ["autogroup:admin"]` in
`tagOwners`, and one grant `tag:ci → tag:server-hub on 22`. The CI node must be
ephemeral and tagged so it expires on its own; never reuse a user auth key.
If Tailscale SSH stays off, use `~/.ssh` keys pinned in Actions secrets and
accept the extra secret — one more reason to prefer option D.

---

## 6. The parts git cannot synchronize

1. **Dependencies.** `requirements.txt` is floor-pinned (`numpy>=1.26`,
   `torch>=2.2`). Two nodes installing a week apart get different versions and
   the "identical result" claim dies quietly. Fix: a CI job that emits a
   resolved lock (`pip freeze` artifact, or `uv pip compile requirements.txt`)
   and, once per platform (`cpu`, `cu121`), commit it. Torch wheels differ per
   node by design, so the lock must be per-platform, not global.
2. **Artifacts.** `out/`, `figs/`, `ZIPS/`, `*.npz`, `*.pt` are ignored on
   purpose (`DOCS/NewPlan.md` rule 3). Sync them as archives (`rsync -avz
   ubuntu-hub:~/TESIS/out/ ./out/`, `gh run download` for CI-produced figures)
   — never un-ignore them to "make sync easy".
3. **Commits actually used for a run.** Add `git rev-parse HEAD` and the env
   hash to every `summary.json` / `runs/runtimes.jsonl` record. Cheap, and it
   is what makes "which machine produced this number" answerable later.
4. **Service state.** `herdr server`, `docker compose up -d`, `app.py` are not
   code. A sync can leave a node on the right commit with a stale process; that
   is a *restart* step, not a *sync* step — keep them separate so a pull never
   kills a running optimization.

---

## 7. Recommended rollout

| Phase | Change | Acceptance | Cost |
| :--- | :--- | :--- | :--- |
| 0 | `SCRIPTS/sync_pull.sh`; make the existing `head`/`dirty`/`ahead`/`behind` output in `fleet_status.sh --json` render in `app.py:/fleet` as drift against the expected SHA | from the tablet, a node that is behind or dirty is visible without SSH; the expected SHA is the one the CI run published | ~1 h |
| 1 | `.github/workflows/ci.yml` (tests + TeX) | a red test or a broken `TEX/*.tex` blocks the merge; artifacts downloadable | ~1 h |
| 2 | `tesis-sync.timer` on hub + relay (+ surface if it holds a clone) | after a merge, every long-lived node converges within one interval, unattended; diverged node logs `exit 5` instead of forcing | ~1 h |
| 3 | Self-hosted runner on `ubuntu-hub` (`runs-on: [self-hosted, hub]`) for gates + `/fleet` probe | `--stage gates` runs in CI on hub hardware; no `DIGITALOCEAN_TOKEN` or SSH key enters Actions | ~2 h |
| 4 *(optional)* | `notify.yml` over Tailscale (`tag:ci`) | merge → nodes updated in seconds; ACL diff reviewed in `fleet-acl.hujson` | ~1 h + ACL edit |
| 5 *(optional)* | `tailscale serve` + `POST /sync` on `app.py` | tablet can trigger and read a sync without SSH | ~2 h |

Phases 0–1 are the ones that actually buy correctness; 2 buys convenience; 3–5
buy latency and ergonomics. Nothing after phase 1 is worth starting before
phase 1 is green on a real PR.

**What not to build:** a workflow that clones the repo into a node's directory
and overwrites it (loses local work and the hub's bare mirror), or a `rsync
--delete` of the working tree (same, worse). Git's own transport is fine; the
missing pieces are *verification* and *trigger*.

---

## 8. Failure modes and drift detection

| Symptom | Cause | Detection | Fix |
| :--- | :--- | :--- | :--- |
| Nodes run different SHAs | timer disabled / node offline | expected-SHA file vs `fleet_status.sh --json` | surface `behind` in `app.py:/fleet`, not just in a log |
| Same SHA, different numbers | unpinned deps, different torch wheel | env hash in the run record | §6.1 lock file |
| Relay serves a stale PDF | relay pulled code but never recompiled | compare `TEX/*.pdf` hash to `manifest.json` | post-pull hook: `latexmk` when `TEX/` changed |
| Pull silently skipped | dirty tree | `dirty: true` in JSON; sentinel exit 4 | operator decides: commit, stash, or `--force` explicitly |
| Diverged branch | experiment pushed from a node | exit 5 in `/tmp/tesis-sync.log` | never auto-resolve; open a PR |
| CI green, node still broken | node-specific (disk 95%, missing TeX) | `fleet_status.sh` checks disk; add a `latexmk --version` probe | per-node, not a sync problem |
| Ephemeral CI node lingers in tailnet | not tagged/ephemeral | `tailscale status` | ephemeral + short key expiry, `tag:ci` |
| Actions outage | github.com | timer still converges (phase 2 is independent of CI) | — |
| Hub offline | hardware | sentinels age out; relay can keep serving docs | do not make the hub the only sync path (it is not) |

`fleet_status.sh --json` already carries `repo.head`, `repo.dirty`, `repo.ahead`
and `repo.behind` (from the last fetch); what is missing is a *reference* — the
CI run's SHA — so `/fleet` can render "expected vs actual" instead of four raw
fields. Everything else above is derivable from that pair plus the per-run env
hash.

---

## 9. Security notes (extends `DOCS/NewPlan.md` §6)

* **No node credentials in GitHub.** Prefer (D): the hub's runner already has
  the identity and the tailnet membership, so Actions stores nothing but the
  runner's registration token — which itself needs periodic rotation.
* **Fork PRs never run on the self-hosted runner.** Default GitHub behaviour
  plus an explicit `if: github.event.pull_request.head.repo.full_name == github.repository`
  (or `pull_request_target` discipline) — a PR that can `curl` the tailnet from
  the hub is an incident.
* **`tag:ci` must be infrastructure-only.** No `tailscale serve`, no subnet
  routes, no exit node, no `autogroup:member` grant; expire the node as soon as
  the job ends.
* **Secrets that exist today and must stay out of CI:** `DIGITALOCEAN_TOKEN`,
  DeepSeek/LLM API keys, HF tokens, `~/.ssh/tesis_do_ed25519`, `.ssh_pass*`.
  `.gitignore` already blocks the common patterns; note that CI logs are public
  on a public repo, so `set -x` around a token is a disclosure (this is why
  `fleet_bootstrap.sh` redacts `--authkey` under `--dry-run`).
* **Artifacts are published, not private.** `tex-pdfs` and `test-log` from a
  public repo are downloadable by anyone with the run URL — that is fine for
  TeX, not necessarily for `runs/` containing absolute paths or hostnames.
  Decide per job: `if: github.event_name != 'pull_request'` for anything with
  local paths.
* **Turning on Tailscale SSH** (needed by option C, not by A/B/D) is a
  control-plane change; record it in `DOCS/NewPlan.md` §6 when it happens,
  along with the `ssh` ACL block that `fleet-acl.hujson` deliberately leaves
  commented out.

---

## 10. Open items

- [ ] `SCRIPTS/sync_pull.sh` (ff-only, dirty guard, `--json`, sentinel); add a
      `--fetch` option so `fleet_status.sh` can report a *fresh* ahead/behind
      instead of the state as of the last fetch
- [ ] `.github/workflows/ci.yml`: `tests/run_tests.py` on 3.12 + TeX build with
      pinned `SOURCE_DATE_EPOCH`; upload `test-log` and `tex-pdfs`
- [ ] `tesis-sync.timer` / `.service` units (hub first, then relay); document
      the interval and the "dirty tree = skip + warn" rule in `DOCS/HUMAN.md`
- [ ] self-hosted runner on `ubuntu-hub` under a dedicated user, work dir
      outside `~/TESIS`, fork PRs excluded; move `--stage gates` and the
      `/fleet` probe there
- [ ] dependency lock per platform (cpu / cu121) produced in CI; decide whether
      `requirements.txt` keeps floor pins or moves to `requirements.in`
- [ ] record `commit` + env hash in `summary.json` / `runs/runtimes.jsonl`
- [ ] decide on phase 4 (Actions → tailnet) *after* phases 0–2 have run for a
      week; if adopted, add `tag:ci` to `SCRIPTS/fleet-acl.hujson` in the same
      PR so the ACL never lags the workflow
- [ ] rotate the DO token (already open in `DOCS/NewPlan.md` §9) before any
      workflow is allowed to mention it
