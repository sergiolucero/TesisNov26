# PI WEB on `quant.cl` — setup and remote access from Windows

Status: **working**. PI WEB is installed on the `quant.cl` server and listens on the
Tailscale interface. Access from Windows is done through an SSH tunnel.

Last verified: 2026-11-26 (web/API and session daemon both `current`, tunnel test returned HTTP 200).

---

## 1. What is running

| Item | Value |
| --- | --- |
| Package | `@jmfederico/pi-web` `1.202611.5` |
| Install path | `/usr/local/lib/node_modules/@jmfederico/pi-web` |
| Runtime | Node.js `26.13.1` (`/usr/bin/node`); Bun is **not** installed |
| `PI_WEB_RUNTIME` | `node` (exported in `/root/.bashrc`) |
| Pi SDK package | `PI_WEB_PI_PACKAGE_DIR=/snap/pi-coding-agent/current/lib/pi/packages/coding-agent` |
| Config file | `/root/.config/pi-web/config.json` |
| Bind address | `100.73.214.36:8504` (Tailscale IP only) |
| Services | systemd **user** units `pi-web.service`, `pi-web-sessiond.service` |
| Session socket | `/root/.pi-web/sessiond.sock` |
| Server public IPv4 | `206.81.4.82` (`quant.cl` DNS → this address) |
| Tailnet name | `ubuntu-quant` / `ubuntu-quant.tail39daae.ts.net` |

Current config:

```json
{
  "host": "100.73.214.36",
  "port": 8504,
  "allowedHosts": [],
  "hostAccess": {
    "mode": "restricted",
    "hosts": [
      "ubuntu-quant",
      "ubuntu-quant.tail39daae.ts.net"
    ]
  }
}
```

Both services auto-start with the systemd user session (lingering is enabled), so
they come back after a reboot without logging in.

---

## 2. Why `http://quant.cl:8504` does not work

`quant.cl` resolves to the **public** IP `206.81.4.82`. PI WEB has **no
authentication** and runs commands as `root` in `/root/TESIS`, so it must never be
exposed directly to the internet. It is deliberately bound to the Tailscale IP:

- `http://quant.cl:8504` → connection refused (nothing listens on the public IP)
- `Host: quant.cl:8504` to the Tailscale listener → `403` (blocked by `hostAccess`)
- Public internet → refused

Use the tunnel in section 4 or the Tailnet URL instead.

---

## 3. Server-side operations

All commands run on the `quant.cl` server as `root`:

```sh
pi-web status                 # is everything up?
pi-web start                  # start both services
pi-web restart                # restart both services
pi-web stop                   # stop both services
pi-web logs                   # service logs (journalctl --user)
pi-web doctor                 # full health check (runtime, node-pty, hosts)
```

Useful low-level checks:

```sh
systemctl --user status pi-web.service pi-web-sessiond.service
journalctl --user -u pi-web.service -n 50 --no-pager
journalctl --user -u pi-web-sessiond.service -n 50 --no-pager
ss -ltnp | grep 8504          # should show 100.73.214.36:8504
```

After editing `/root/.config/pi-web/config.json`, run `pi-web restart` (the
session daemon does not need a restart for `host`/`hostAccess` changes, but the
combined restart is harmless).

---

## 4. Access from Windows

Choose one of the two options. **Option B (SSH tunnel)** is the recommended and
tested path; it works from any network that can reach SSH on the server.

### Option A — direct Tailscale access (no tunnel)

Prerequisite: Tailscale is installed and logged into the same tailnet on the
Windows machine (`desktop-t08pmff` is currently a member).

Open in the browser:

```text
http://ubuntu-quant.tail39daae.ts.net:8504
```

or

```text
http://100.73.214.36:8504
```

This only works while the Windows machine is on the tailnet. Nothing else to do.

### Option B — SSH tunnel (recommended)

Requires the Windows built-in OpenSSH client. Check once in PowerShell:

```powershell
ssh -V
```

If missing: **Settings → Apps → Optional features → Add a feature → OpenSSH Client**.

Start the tunnel in PowerShell or Command Prompt:

```powershell
ssh -N -L 8504:100.73.214.36:8504 root@quant.cl
```

Then open in the browser:

```text
http://localhost:8504
```

Explanation:

- `-N` — open the tunnel only, no remote shell session.
- `-L 8504:100.73.214.36:8504` — listen on the Windows machine's `localhost:8504`
  and forward to the server's Tailscale address on port 8504.
- `root@quant.cl` — the SSH login (key or root password is prompted).

**Important:** the server-side target must be `100.73.214.36:8504`, *not*
`localhost:8504`. PI WEB is bound to the Tailscale IP, so the server's own
loopback does not answer.

Keep the PowerShell window open while using the UI. Press `Ctrl+C` to close the
tunnel.

Optional keepalive flags for flaky links:

```powershell
ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 8504:100.73.214.36:8504 root@quant.cl
```

If local port 8504 is already taken on Windows, forward a different port:

```powershell
ssh -N -L 18504:100.73.214.36:8504 root@quant.cl
```

then browse `http://localhost:18504`.

Verify the tunnel from Windows:

```powershell
netstat -ano | findstr 8504
```

The first connection asks to trust the host key. Type `yes`. If SSH to
`quant.cl` is blocked on your current network, tunnel through Tailscale instead:

```powershell
ssh -N -L 8504:100.73.214.36:8504 root@100.73.214.36
```

### Optional: double-click shortcut on Windows

Create `piweb-tunnel.ps1`:

```powershell
ssh -N -L 8504:100.73.214.36:8504 root@quant.cl
if ($LASTEXITCODE -ne 0) { Read-Host "Tunnel exited. Press Enter to close" }
```

Create a shortcut to:
`powershell.exe -ExecutionPolicy Bypass -File "C:\path\to\piweb-tunnel.ps1"`

Double-click it, leave the window open, then open `http://localhost:8504`.

---

## 5. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Browser cannot connect to `localhost:8504` on Windows | Tunnel window closed or failed. Re-run the `ssh -N -L ...` command and check `netstat -ano \| findstr 8504`. |
| `ssh: connect to host quant.cl port 22: ...` fails | SSH blocked by network/ISP. Use `root@100.73.214.36` (Tailscale) instead. |
| `Permission denied (publickey,password)` | Wrong key or password. `passwordauthentication yes` is enabled on the server; use the root password or install a key. |
| `bind: Address already in use` on Windows | Port 8504 is taken. Use `-L 18504:100.73.214.36:8504` and browse port 18504. |
| Page loads but says `Host not allowed` / `403` | Host header not in `hostAccess`. `localhost` and literal IPs are always allowed; custom hostnames must be added to `hostAccess.hosts` and the service restarted. |
| `pi-web status` shows a service as down | `pi-web restart`, then `pi-web logs`. |
| `pi-web doctor` mentions `node-pty` | Rebuild it: `cd /usr/local/lib/node_modules/@jmfederico/pi-web && npm install node-pty@^1.1.0 --no-save`, then `pi-web restart`. |
| `no usable runtime for PI_WEB_RUNTIME=...` | Ensure `export PI_WEB_RUNTIME=node` in `/root/.bashrc` (line 2, before the non-interactive guard) and `pi-web restart`. |
| Web UI works but terminals/sessions fail | Session daemon down or node-pty missing: `systemctl --user status pi-web-sessiond`, `pi-web doctor`. |

---

## 6. Optional: public HTTPS on `quant.cl` (NOT enabled)

Only do this if you need browser access without Tailscale or SSH. It requires
adding authentication, because PI WEB itself has none. Basic auth over plain HTTP
is not acceptable; terminate TLS at Cloudflare (or use Cloudflare Tunnel/Access)
and proxy to nginx.

1. Add the public hostname to `hostAccess` in `/root/.config/pi-web/config.json`:

   ```json
   { "hostAccess": { "mode": "restricted", "hosts": ["quant.cl"] } }
   ```

2. Create `/etc/nginx/sites-available/piweb`:

   ```nginx
   server {
       listen 8504;
       server_name quant.cl;

       auth_basic           "PI WEB";
       auth_basic_user_file /etc/nginx/piweb.htpasswd;

       location / {
           proxy_pass http://100.73.214.36:8504;
           proxy_http_version 1.1;
           proxy_set_header Host $http_host;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_read_timeout 3600s;
           proxy_send_timeout 3600s;
       }
   }
   ```

   - `$http_host` (not `$host`) preserves the public host and port for PI WEB's
     Host/Origin checks.
   - The same `location` must carry both HTTP and WebSocket traffic.

3. Enable it and set credentials:

   ```sh
   htpasswd -c /etc/nginx/piweb.htpasswd youruser
   ln -s /etc/nginx/sites-available/piweb /etc/nginx/sites-enabled/piweb
   nginx -t && systemctl reload nginx
   pi-web restart
   ```

4. Open `https://quant.cl:8504` through Cloudflare with **Full (strict)** TLS and
   Cloudflare Access in front. Cloudflare only proxies standard ports by default;
   alternatively keep nginx on 443 under a subdomain or use a Cloudflare Tunnel.

**Security warning:** without authentication in front, anyone who reaches the
port can read your files and run commands as root. Prefer Cloudflare Access or
Tailscale over bare basic auth.

---

## 7. Rollback

To go back to localhost-only access:

```json
{
  "host": "127.0.0.1",
  "port": 8504,
  "allowedHosts": []
}
```

Then `pi-web restart`. With that binding, SSH tunnels must target
`127.0.0.1:8504` server-side:

```powershell
ssh -N -L 8504:127.0.0.1:8504 root@quant.cl
```
