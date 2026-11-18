# Synch.md — Keeping Every Machine Up-to-Date (GitHub Actions + Tailscale)

> Naming note: this project's private mesh is **Tailscale**. Some scratch
> notes call it *"Terascale"*; in this repository the two are the same thing,
> and everything below refers to the Tailscale tailnet already described in
> [`NewPlan.md`](NewPlan.md).

The fleet has a single source of truth — `https://github.com/sergiolucero/TesisNov26.git`
(`main`) — and a private Tailscale mesh that connects `ubuntu-hub`,
`do-relay`, `surface-ctrl`, and any ephemeral Colab/DO worker. This document
explains how to keep every node running **the same commit** and **the same
environment**, automatically, using two complementary mechanisms:

1. **GitHub Actions** — the *trigger and gatekeeper*. A push to `main` runs the
   test suite, and only then fans the change out to the fleet.
2. **Tailscale** — the *transport*. Actions reaches every node over the private
   mesh, so no node needs a public SSH port.

The two layers are not competitors: GitHub is the source of truth and policy,
Tailscale is the wire. You can use Actions alone (SSH to reachable hosts), the
tailnet alone (cron pulls from the hub mirror), or — recommended — both.

---

## 1. What "up-to-date" means on this fleet

| Artifact | Where it lives | How it is synced |
|---|---|---|
| Source code, configs, docs | `git` (`main`) | `git fetch && git reset --hard origin/main` on every node |
| Python environment | `requirements.txt` | `pip install -r requirements.txt` after a pull |
| Run artifacts (`out/`, `figs/`, `*.npz`, `*.pt`, `*.zip`) | **never git** (`.gitignore`) | `rsync`/`tar` over Tailscale, or `colab --zip` downloads |
| Long-lived services | Herdr server, Flask, Docker stack | restart only when their files change |
| Per-node "am I current?" signal | `git rev-parse HEAD` | surfaced by `SCRIPTS/fleet_status.sh` |

Rule (from [`NewPlan.md`](NewPlan.md)): **artifacts never enter git**. Syncing
machines means syncing *code*, not megabytes of run output. Results travel as
archives.

---

## 2. The recommended design (Actions + Tailscale together)

```
you push ──► GitHub main ──► Actions: run tests
                                  │ (only if green)
                                  ▼
                    Actions runner joins the tailnet
                    (tailscale/github-action, OAuth)
                                  │
                    ┌─────────────┼──────────────┐
                    ▼             ▼              ▼
               ubuntu-hub     do-relay       (any other worker)
               git pull       git pull       git pull
               pip install    pip install    pip install
               restart svcs   docker pull    —
```

Why this shape:

- **Tests gate the fan-out.** A broken commit never reaches the fleet; the hub
  keeps serving the last known-good revision.
- **The runner is ephemeral.** Using `tailscale/github-action` means you do
  *not* have to keep a self-hosted runner alive 24/7.
- **Each node stays behind its private address.** No inbound port 22 on the
  public internet; Actions reaches hosts by tailnet name/100.x address.

### 2.1 Prerequisites (one-time)

1. A Tailscale **OAuth client** scoped to the tailnet (Settings → OAuth
   clients, with `devices:write` and tag `tag:ci`). Store the client id and
   secret in GitHub Secrets:
   - `TS_OAUTH_CLIENT_ID`
   - `TS_OAUTH_SECRET`
2. A dedicated **deploy keypair** for the sync user. Add the public key to
   `~/.ssh/authorized_keys` on `ubuntu-hub` and `do-relay`; store the private
   key in the secret `FLEET_SSH_KEY`.
3. The repo cloned on each node at a known path (the bootstrap clones it to
   `/root/TESIS` on the hub; see [`NewPlan.md`](NewPlan.md) §2/§3).

### 2.2 The workflow

Create `.github/workflows/sync.yml`:

```yaml
name: fleet-sync

on:
  push:
    branches: [main]
  workflow_dispatch: {}        # allow a manual re-sync from the Actions tab

jobs:
  # 1. Gatekeeper: nothing fans out unless the repo is healthy.
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install numpy scipy pyyaml
      - run: python -u tests/run_tests.py

  # 2. Fan-out: join the tailnet once, then pull on every node.
  sync:
    needs: test
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        node: [ubuntu-hub, do-relay]
    steps:
      - name: Join the tailnet (ephemeral CI device)
        uses: tailscale/github-action@v3
        with:
          oauth-client-id: ${{ secrets.TS_OAUTH_CLIENT_ID }}
          oauth-secret: ${{ secrets.TS_OAUTH_SECRET }}
          tags: tag:ci
          version: latest

      - name: Pull and reinstall on ${{ matrix.node }}
        env:
          SSH_KEY: ${{ secrets.FLEET_SSH_KEY }}
        run: |
          mkdir -p ~/.ssh && chmod 700 ~/.ssh
          printf '%s\n' "$SSH_KEY" > ~/.ssh/fleet_key
          chmod 600 ~/.ssh/fleet_key
          ssh -i ~/.ssh/fleet_key \
              -o StrictHostKeyChecking=accept-new \
              -o ConnectTimeout=20 \
              root@${{ matrix.node }} 'bash -s' <<'EOF'
          set -euo pipefail
          cd /root/TESIS || git clone ubuntu-hub:tesis.git /root/TESIS && cd /root/TESIS
          git fetch --prune origin
          git checkout main
          git reset --hard origin/main
          /root/base/bin/python3 -m pip install -r requirements.txt
          echo "SYNCED $(hostname) @ $(git rev-parse --short HEAD)"
          EOF
```

Notes:

- `tailscale/github-action@v3` brings the runner onto the tailnet for the
  duration of the job and tears it down afterwards. The `tag:ci` device is
  short-lived; grant it only what it needs in the tailnet ACL.
- Use `root@ubuntu-hub`/`root@do-relay` if those are the SSH users. MagicDNS
  resolves the hostnames inside the tailnet, so no IPs are hard-coded.
- `fail-fast: false` means one sick node does not block the others; the job
  turns red so you can see *which* node failed.
- On `do-relay` the interpreter differs (its own Docker stack); adjust the
  `pip` line per node, or key it off `hostname` in the heredoc.

### 2.3 Self-hosted-runner variant (no OAuth client)

If you already keep the hub on 24/7 (this fleet does), a **self-hosted runner
on `ubuntu-hub`** is simpler and faster — the runner is already on the tailnet
and can also update itself:

```yaml
  sync:
    needs: test
    runs-on: [self-hosted, linux, hub]     # label the ubuntu-hub runner
    steps:
      - name: Pull on the hub (this runner's host)
        run: |
          set -e
          cd ~/tesis.git && git fetch --prune origin    # bare mirror first
          cd /root/TESIS && git fetch --prune origin && \
            git reset --hard origin/main && \
            /root/base/bin/python3 -m pip install -r requirements.txt
      - name: Fan out to relay over the tailnet
        run: |
          ssh -o StrictHostKeyChecking=accept-new root@do-relay 'bash -s' <<'EOF'
          set -e
          cd /root/TESIS
          git fetch --prune origin && git reset --hard origin/main
          pip install -r requirements.txt
          EOF
```

Install the runner on the hub with `gh` (bootstrap: `./SCRIPTS/gh_setup.sh`)
or the GitHub UI (`Settings → Actions → Runners → New self-hosted runner`).

---

## 3. Tailscale-only sync (no GitHub Actions)

If you do not want Actions to touch the fleet, the mesh alone is enough: every
node pulls from the **hub bare mirror** (`~/tesis.git`) or straight from GitHub
on a schedule. This is the "pull" model — each machine is responsible for
itself, and the hub is just the local cache.

### 3.1 Keep the mirror fresh

On `ubuntu-hub`, refresh the bare mirror from GitHub on a timer (or on demand):

```bash
git clone --bare https://github.com/sergiolucero/TesisNov26.git ~/tesis.git   # once
git -C ~/tesis.git fetch --prune                                               # repeat
```

### 3.2 A reusable sync script

Put this on each node as `~/tesis-sync.sh` (or commit it to the repo):

```bash
#!/usr/bin/env bash
# tesis-sync.sh — pull the latest main into the checkout and refresh the env.
set -euo pipefail

REPO_DIR="${1:-$HOME/TESIS}"
REMOTE="${2:-origin}"
PYTHON="${PYTHON:-python3}"            # hub: /root/base/bin/python3

cd "$REPO_DIR"
git fetch --prune "$REMOTE"
git checkout main 2>/dev/null || git checkout -b main --track "$REMOTE/main"
git reset --hard "$REMOTE/main"
"$PYTHON" -m pip install -r requirements.txt 2>/dev/null || true
echo "SYNCED $(hostname) @ $(git rev-parse --short HEAD)"
```

### 3.3 Schedule it (systemd timer or cron)

```bash
# cron — every 15 minutes, logs to the sentinel dir
*/15 * * * *  /root/tesis-sync.sh /root/TESIS origin >>/var/log/tesis-sync.log 2>&1
```

or a systemd timer (survives better on headless nodes):

```ini
# /etc/systemd/system/tesis-sync.service
[Unit]
Description=tesis repo sync

[Service]
Type=oneshot
ExecStart=/root/tesis-sync.sh /root/TESIS origin
```

```ini
# /etc/systemd/system/tesis-sync.timer
[Unit]
Description=run tesis sync every 15 min

[Timer]
OnBootSec=2min
OnUnitActiveSec=15min

[Install]
WantedBy=timers.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now tesis-sync.timer
```

### 3.4 Trigger on demand from any node

Because the tailnet makes every host reachable, a manual push to the whole
fleet is one command per host:

```bash
for h in ubuntu-hub do-relay; do
  ssh root@$h '/root/tesis-sync.sh /root/TESIS origin' &
done
wait
```

`SCRIPTS/fleet_status.sh` already reports each node's repo state; add
`git -C /root/TESIS rev-parse --short HEAD` to its output if you want the
exact commit in the dashboard.

---

## 4. Per-node matrix

| Node | Mechanism | Trigger | What it syncs |
|---|---|---|---|
| `ubuntu-hub` (hub) | Actions (fan-out) or cron pull | push to `main` / timer | repo + venv + Herdr service on change |
| `do-relay` (relay) | Actions fan-out over tailnet | push to `main` | repo + Docker stack + TeX rebuild on change |
| `surface-ctrl` (tablet) | manual `git pull` / VS Code | operator | docs and control-plane notes |
| Colab / DO worker | **fresh clone each session** | session start | repo + `requirements.txt` (see `Collab.md`, `deploy_do.sh`) |

Colab and DO workers are ephemeral: they never "stay" up-to-date, they start
up-to-date by cloning `main` when the session begins. That is already handled
by `run_colab.sh` / `deploy_do.sh`; no sync job should target them.

---

## 5. Security rules (non-negotiable)

- **Secrets never touch the repo.** `TS_OAUTH_CLIENT_ID`, `TS_OAUTH_SECRET`,
  `FLEET_SSH_KEY`, and any DO/DeepSeek tokens live only in GitHub Secrets or
  the shell environment — never in `sync.yml`, `NewPlan.md`, or scripts.
- **Least privilege for the CI device.** Tag the ephemeral runner `tag:ci` and
  give it SSH access to the sync user only, not to the whole tailnet.
- **Deploy key, not your personal key.** Generate a dedicated keypair; revoke
  and rotate it independently of your own.
- **Artifacts out-of-band.** `out/`, `figs/`, `*.npz`, `*.pt`, `*.zip` stay out
  of git, so a sync pull never drags binaries across the mesh.
- **The hub mirror is not a second truth.** `git fetch --prune` from `origin`
  (GitHub) only; the mirror caches, it does not host a divergent branch.

---

## 6. Bring-up checklist

- [ ] Tailscale OAuth client created; id/secret in GitHub Secrets
- [ ] `FLEET_SSH_KEY` secret set; public key in `authorized_keys` on each node
- [ ] `.github/workflows/sync.yml` added and `tests/run_tests.py` green in CI
- [ ] One `workflow_dispatch` run fans out to `ubuntu-hub` and `do-relay`
- [ ] `git rev-parse --short HEAD` matches across nodes (`fleet_status.sh`)
- [ ] (optional) `tesis-sync.timer` enabled on the hub as a fallback
- [ ] Verify no secrets are echoed in workflow logs (`set -x` off in CI)
