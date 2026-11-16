# NewPlan.md — Thesis Fleet & Simulation Coordination Architecture

**Project:** Stochastic Groundwater Remediation (2D/3D PINNs, FNOs, Monte Carlo / MODFLOW-MT3DMS)  
**Orchestrator:** `herdr` on the Ubuntu home server (`ubuntu-hub`)  
**Agents:** `pi` coding agent (DeepSeek API + local Ollama)  
**Control plane:** Tailscale (private network) + Herdr (persistent agent sessions)  
**Source of truth:** `https://github.com/sergiolucero/TesisNov26.git` (`main`)

This is the operating plan for the fleet: topology, bootstrap, repo and artifact
flow, job dispatch, and the bring-up checklist. The scripts in the repo are the
source of truth where prose and code disagree.

## 0. Design rules

1. **Private by default** — no inbound ports except 80/443 on `do-relay`
   (Nginx → Flask/LaTeX preview). Everything else is Tailscale-only.
2. **Jobs survive disconnects** — long runs are detached (`nohup`, Herdr panes,
   Colab cells) and report through sentinel files, never through a live SSH
   session.
3. **Artifacts never enter git** — `out/`, `figs/`, `*.npz`, `*.pt`, `*.zip`
   are ignored (`.gitignore`); exchange results as archives or over Tailscale.
4. **Idempotent bootstrap** — every setup step can be re-run; mutating scripts
   offer `--dry-run` where practical (`deploy_do.sh`, `fleet_bootstrap.sh`).
5. **One hub** — `ubuntu-hub` alone runs the Herdr server, the bare repo, and
   the dispatch polling; other nodes are clients or ephemeral workers.

---

## 1. Fleet Topology & Node Assignments

| Node | Tailscale Hostname | Role | Primary Workload |
| :--- | :--- | :--- | :--- |
| **Ubuntu Home Server** (Dell Latitude 5310) | `ubuntu-hub` | **Master Control & Heavy CPU Hub** | `herdr` server, master Git bare repo, Monte Carlo ensembles, MODFLOW/MT3DMS, PostGIS, Pyomo, local Ollama inference, Colab dispatch. |
| **DigitalOcean Droplet** | `do-relay` | **Always-On Web & LaTeX Node** | Docker stack (TeX Live, Jupyter, Flask, Nginx HTTPS), continuous PDF compilation, external preview hosting, backup `pi` worker. |
| **Windows Tablet** (Surface Pro) | `surface-ctrl` | **Mobile Control Plane** | Tailscale SSH / VS Code Remote client, `NewPlan.md` updates, literature review, fleet monitoring. |
| **Google Colab** | *Ephemeral T4 GPU* | **Burst GPU Compute** | `torch.autograd` PINN residual sweeps, 2D/3D Fourier Neural Operator (FNO) training via `run_colab.sh`. |

**Data path.** `surface-ctrl` (editor) → `ubuntu-hub` (orchestration, CPU) →
`do-relay` (publish/compile) or Colab/DO (GPU burst) → results archived under
`ZIPS/` or `do_results/`, never committed as loose binaries.

---

## 2. Installation & Bootstrap Commands

### A. Install Tailscale (all nodes: `ubuntu-hub`, `do-relay`, `surface-ctrl`)

Run on Ubuntu and DigitalOcean (install via MSI/winget on Windows Surface Pro:
`winget install tailscale.tailscale`):

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --hostname=$(hostname)
```

On the hub, use the repo wrapper so the role, hostname and SSH policy stay
consistent:

```bash
./SCRIPTS/fleet_bootstrap.sh --role hub          # add --authkey for headless nodes
./SCRIPTS/fleet_bootstrap.sh --verify            # report tailnet + Herdr state
```

### B. Install Herdr (`ubuntu-hub`, orchestrator)

Herdr is the terminal workspace manager for the fleet's coding agents. Its
background server keeps agent panes alive across SSH disconnects, so the hub
can host the persistent `pi` sessions the other nodes attach to.

```bash
curl -fsSL https://herdr.dev/install.sh | sh   # installs to ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"           # add to ~/.bashrc or ~/.profile
herdr --version                                # verify (stable channel)
herdr server                                   # headless daemon on the hub
herdr integration install pi                   # let Herdr track `pi` sessions
```

Check runtime state with `herdr status`; update with `herdr update`; stop
everything with `herdr server stop`. The hub role of
`SCRIPTS/fleet_bootstrap.sh` performs the install idempotently (skip it with
`--no-herdr`, or force it on another node with `--herdr`).

**Run it as a user service** so the hub survives logout and reboot. The unit
lives in `SCRIPTS/herdr-server.service`; the bootstrap installs and enables it
with `--systemd`:

```bash
./SCRIPTS/fleet_bootstrap.sh --role hub --systemd
loginctl enable-linger "$USER"                        # survive logout
systemctl --user status herdr-server.service
systemctl --user restart herdr-server.service        # after config changes
```

### C. Python environment (hub and CPU workers)

The repo expects Python ≥ 3.10 with `torch`, `numpy`, `scipy`, `matplotlib`,
`flask`, `pyyaml`. On the hub the known-good interpreter is
`/root/base/bin/python3` (see `HUMAN.md`); the default `python3` has no torch.

```bash
/root/base/bin/python3 -m pip install -r requirements.txt
/root/base/bin/python3 -u tests/run_tests.py      # 32 tests, ~2 s
```

Colab and the DO droplet install their own environment (`run_colab.sh`,
`remote_run.sh`); do not copy the hub venv to them.

---

## 3. Repo & Artifact Flow

**GitHub is the source of truth; the hub keeps a bare mirror for fast tailnet
clones and a single fetch point.**

```bash
# once, on ubuntu-hub
git clone --bare https://github.com/sergiolucero/TesisNov26.git ~/tesis.git
git -C ~/tesis.git fetch --prune                  # refresh mirror (cron/Timer)

# on any tailnet node
git clone ubuntu-hub:tesis.git TESIS
cd TESIS
git remote add github https://github.com/sergiolucero/TesisNov26.git
```

Branch policy:

| Branch | Purpose | Who pushes |
| :--- | :--- | :--- |
| `main` | Reviewed, reproducible state; gates must pass | humans, `pi` after tests |
| `hub/<task>` | Long experiment or refactor on the hub | `ubuntu-hub` agents |
| `do/<task>`, `colab/<task>` | Throw-away burst-compute branches (optional) | ephemeral nodes |

Artifact exchange (never `git add` these):

```bash
# pull results from a node
rsync -avz ubuntu-hub:~/TESIS/out/ ./out/
# bundle a run before archiving it under ZIPS/
tar -czf ZIPS/$(date -u +%Y%m%d)-<tag>.tgz out/<run> runs/<run>.json
```

`deploy_do.sh` downloads results into `do_results/`; `collab_run.py
--zip-results` writes `out/colab_results.zip` for download from the notebook.

---

## 4. Job Dispatch & Sentinel Protocol

Every dispatch lane follows the same contract: write a log, write
`<job>.exit`, then touch `<job>.done`. Poll the sentinel, not the process.

| Lane | Entry point | Sentinel | Log | Typical wall time |
| :--- | :--- | :--- | :--- | :--- |
| Hub, local | `run_optimization.py --stage gates` | none (foreground) | stdout | ~74 s |
| Hub, overnight | `run_optimization.py --stage opt` | none | stdout | >30 min (see `RUNTIMES.txt`) |
| DO droplet | `deploy_do.sh --sim pinn` | `/tmp/tesis-run.done` / `.exit` | `/tmp/tesis-run.log` | minutes–hours |
| Colab T4 | `run_colab.sh` → `collab_run.py` | `/tmp/tesis-colab.done` / `.exit` | notebook cell | preset-dependent |

### A. DigitalOcean burst lane

```bash
export DIGITALOCEAN_TOKEN=dop_v1_...        # never commit or echo this
./deploy_do.sh --sim smoke                  # cheap end-to-end check first
./deploy_do.sh --sim pinn --keep            # keep the droplet for debugging
./deploy_do.sh --ip 203.0.113.10 --sim smoke  # reuse an existing droplet
```

`deploy_do.sh` packs the worktree, creates the droplet, copies the archive,
starts `remote_run.sh` detached, polls the sentinel, downloads artifacts to
`do_results/`, and destroys the droplet unless `--keep` is passed (failures are
kept by default so you can inspect them).

### B. Colab burst lane

```python
# inside the notebook, after %cd /content/TESIS
!bash run_colab.sh --preset quick --models dataset pinn
!bash run_colab.sh --preset full  --models fno --fno-epochs 300
```

`run_colab.sh` checks the GPU, installs `requirements.txt`, then forwards all
arguments to `collab_run.py` (`smoke|quick|full` presets). Download the results
zip (or sync to Drive) before the runtime is recycled.

### C. Polling from the hub

```bash
job=/tmp/tesis-run            # or /tmp/tesis-colab
while [[ ! -f "$job.done" ]]; do sleep 10; done
echo "exit=$(cat "$job.exit")"
tail -n 40 "${job}.log" 2>/dev/null || true
```

Wrap long hub jobs in a named Herdr session so they survive disconnects and are
visible to every node:

```bash
herdr session attach opt        # one session per long experiment
herdr session list
herdr status                    # server/client state + socket path
```

---

## 5. Monitoring & Health

Run these from any tailnet node (or `surface-ctrl` over SSH):

```bash
tailscale status                          # who is up, IPs, last seen
sudo tailscale ping ubuntu-hub            # path check (direct vs DERP)
./SCRIPTS/fleet_status.sh                 # one-shot health snapshot
./SCRIPTS/fleet_status.sh --json          # same, for scripts/dashboards
./SCRIPTS/fleet_bootstrap.sh --verify     # tailnet + herdr, role-aware
herdr status                              # server, socket, update state
ssh ubuntu-hub 'df -h; free -h; uptime'   # capacity before a heavy run
```

`fleet_status.sh` reports host/load/disk, Tailscale, Herdr, repo state, and
job sentinels; it exits non-zero on a failed sentinel, a down tailnet, or
disk ≥ 95%. Pass `--role hub` on the hub so a stopped Herdr server is flagged.

Expected runtime envelope on the 1 vCPU hub container (`RUNTIMES.txt`): unit
suite 2 s, `run_2d.py --quick` forward 6 s / optimize 16 s, stochastic gates
74 s. Anything above ~10 min on the hub belongs on the DO lane or overnight.

---

## 6. Security & Secrets

- **Network:** tailnet-only SSH; `do-relay` is the only public surface (443 via
  Nginx, 80 redirect). Restrict the tailnet with an ACL/tag policy in the
  Tailscale admin console; do not use default `autogroup:member` for servers.
- **SSH:** key-based only (`~/.ssh/tesis_do_ed25519` for DO); no agent
  forwarding to ephemeral nodes.
- **Tokens:** `DIGITALOCEAN_TOKEN`, DeepSeek keys, and HF tokens live in the
  shell environment or a password manager — never in the repo, scripts, logs,
  or `NewPlan.md`.
- **Herdr:** the server socket is a local API (`~/.config/herdr/`); treat
  access to `ubuntu-hub` as access to every agent session.
- **Ephemeral nodes:** assume Colab/DO droplets are shared infrastructure;
  download artifacts, then destroy the droplet (`--keep` only for debugging).

---

## 7. Bring-Up Checklist

**`ubuntu-hub`**

- [ ] `SCRIPTS/fleet_bootstrap.sh --role hub` → Tailscale up, Herdr installed
- [ ] `herdr integration install pi` and `herdr server` running
- [ ] Python env + `tests/run_tests.py` green
- [ ] `~/tesis.git` bare mirror cloned and fetchable
- [ ] `run_optimization.py --stage gates` green (~74 s)
- [ ] `app.py --port 5099` serves `/` (expect 200)

**`do-relay`**

- [ ] `SCRIPTS/fleet_bootstrap.sh --role relay --authkey ...`
- [ ] `docker compose up -d` (TeX Live / Jupyter / Flask)
- [ ] Nginx + TLS for the preview host; `TEX/` rebuild via `latexmk`
- [ ] `remote_run.sh` smoke (`SIM_MODE=smoke`) writes the sentinel

**`surface-ctrl`**

- [ ] Tailscale + `ssh ubuntu-hub` / VS Code Remote-SSH hosts configured
- [ ] Herdr attached to the hub for agent monitoring
- [ ] `./SCRIPTS/fleet_bootstrap.sh --role surface` for hostname/auth check

**Colab**

- [ ] GPU runtime selected; `nvidia-smi` shows a T4
- [ ] `bash run_colab.sh --preset smoke` finishes and writes the sentinel
- [ ] Results zip downloaded / Drive-synced before teardown

---

## 8. Workload Routing

| Workload | Best node | Why |
| :--- | :--- | :--- |
| Unit tests, gates, 2D quick runs | `ubuntu-hub` | seconds, CPU-only |
| Stochastic SAA/CVaR benchmark | `ubuntu-hub` (short) / DO (long) | optimization loop dominates cost |
| MODFLOW/MT3DMS, PostGIS, Pyomo | `ubuntu-hub` | persistent services, local data |
| PINN/FNO training, residual sweeps | Colab T4, else DO GPU | needs CUDA, ephemeral is cheapest |
| LaTeX compile + public preview | `do-relay` | always-on, HTTPS, external access |
| Editing, review, monitoring | `surface-ctrl` | mobile, no compute |
| 2.5D axisymmetric PINN (smoke) | `ubuntu-hub`, then Colab | `N_r ~ 5×10⁴`, hours on CPU |
| Full 3D PINN / FNO3d | **rented GPU only** (24–80 GB) | `3D_plan.md`: 1–10 days, 50 GB–2 TB |

**3D decision (from `3D_plan.md`).** Start with the 2.5D axisymmetric bridge and
run its smoke tests on the hub; full 3D training does not fit the hub's CPU or
memory and belongs on a rented 24–80 GB GPU (DO GPU size or Colab Pro) after
the 2D gates are green. Never queue a 3D run on `ubuntu-hub`.

Rule of thumb: if a run needs a GPU or more than ~10 minutes, it does not run
on the hub.

---

## 9. Open Items

- [x] systemd user service for `herdr server` (`SCRIPTS/herdr-server.service`,
      installed with `--systemd`, lingering enabled on the hub)
- [x] `SCRIPTS/fleet_status.sh`: tailnet + Herdr + repo + sentinels + disk,
      with `--json` and a failing exit code
- [x] progress lines in `run_optimization.py` and `stoch/optimize.py` (stage
      timing + per-iteration pattern-search/SLSQP updates)
- [x] 3D/2.5D placement decided in §8 (2.5D on the hub first, full 3D on a
      rented GPU only)
- [ ] tailnet ACL: tag `server:hub`, `server:relay`, `client:surface`, deny
      server-to-server except SSH from the hub
- [ ] wire `fleet_status.sh --json` into the relay dashboard / tablet view
- [ ] rotate the DO token and store it in the password manager; audit
      `deploy_do.sh` excludes before the next burst run
