# TensorCost GPU Agent — Bare-metal Linux (systemd)

Installs the agent as a persistent systemd service. The service pulls the
Docker image at start-up and restarts automatically on failure.

---

## Prerequisites

- Linux with systemd.
- Docker Engine installed and the daemon running.
- NVIDIA drivers + [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed and Docker restarted after toolkit installation.
- Agent credentials from the TensorCost onboarding wizard (`TENANT_ID`, `AGENT_API_KEY`, `AGENT_KEY_ID`, `AGENT_HMAC_PEPPER`).
  - Allocate at least 16 GB of disk on the host. The agent image plus container runtime overhead exceeds the 8 GB default on AL2023 t3.micro AMIs; an actual customer hit `no space left on device` during `docker pull` in our 2026-05-18 E2E test.

---

## Install

```bash
# 1. Paste the env block from the TensorCost onboarding wizard
export BACKEND_API_URL=https://api.dev.tensorcost.com
export NVML_ENABLED=true   # required for GPU metrics — the agent defaults it off
export TENANT_ID=<your-tenant-uuid>
export AGENT_HOSTNAME=<hostname-of-this-box>
export AGENT_API_KEY=<api-key-from-wizard>
export AGENT_KEY_ID=<key-id-from-wizard>
export AGENT_HMAC_PEPPER=<signing-key-from-wizard>

# 2. Run the install script as root
sudo -E bash install.sh

# 3. Verify the service started
systemctl status tensorcost-agent
journalctl -u tensorcost-agent -f
```

---

## Optional overrides

| Variable | Default | Description |
|---|---|---|
| `AGENT_IMAGE` | `docker.io/rlgocloudera/gpu-usage:latest` | Docker image to pull |
| `COMM_MODE` | `http` | `http`, `grpc`, or `both` |
| `GRPC_TARGET` | _(none)_ | `host:port` for gRPC when `COMM_MODE=grpc` or `both` |
| `GRPC_USE_TLS` | `false` | Set `true` for any public gRPC endpoint |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARN`, or `ERROR` |

---

## Useful commands

```bash
journalctl -u tensorcost-agent -f          # follow logs
systemctl status tensorcost-agent          # service status
systemctl restart tensorcost-agent         # restart after config change
systemctl disable tensorcost-agent --now   # stop and disable
```

Credentials are stored in `/etc/tensorcost/agent.env` (mode 600, root-only).
