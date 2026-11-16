# NewPlan.md — Thesis Fleet & Simulation Coordination Architecture

**Project:** Stochastic Groundwater Remediation (2D/3D PINNs, FNOs, Monte Carlo / MODFLOW-MT3DMS)  
**Orchestrator:** `herdr` (Master Hub on Ubuntu Home Server)  
**Agents:** `pi` coding agent (DeepSeek API + Local Ollama)  

---

## 1. Fleet Topology & Node Assignments

| Node | Tailscale Hostname | Role | Primary Workload |
| :--- | :--- | :--- | :--- |
| **Ubuntu Home Server** (Dell Latitude 5310) | `ubuntu-hub` | **Master Control & Heavy CPU Hub** | `herdr` daemon, master Git bare repo, Monte Carlo ensembles, MODFLOW/MT3DMS, PostGIS, Pyomo, local Ollama inference, Colab dispatch. |
| **DigitalOcean Droplet** | `do-relay` | **Always-On Web & LaTeX Node** | Docker stack (TeX Live, Jupyter, Flask, Nginx HTTPS), continuous PDF compilation, external preview hosting, backup `pi` worker. |
| **Windows Tablet** (Surface Pro) | `surface-ctrl` | **Mobile Control Plane** | Tailscale SSH / VS Code Remote client, `NewPlan.md` updates, literature review, fleet monitoring. |
| **Google Colab** | *Ephemeral T4 GPU* | **Burst GPU Compute** | `torch.autograd` PINN residual sweeps, 2D/3D Fourier Neural Operator (FNO) training via `run_colab.sh`. |

---

## 2. Installation & Bootstrap Commands

### A. Install Tailscale (All Nodes: `ubuntu-hub`, `do-relay`, `surface-ctrl`)
Run on Ubuntu and DigitalOcean (install via MSI/winget on Windows Surface Pro: `winget install tailscale.tailscale`):
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --hostname=$(hostname)
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
```

Check runtime state with `herdr status`; update with `herdr update`; stop
everything with `herdr server stop`. The hub role of
`SCRIPTS/fleet_bootstrap.sh` performs the install idempotently (skip it with
`--no-herdr`, or force it on another node with `--herdr`).
