# TensorCost GPU Agent — Installation

The GPU agent runs on any Linux host with NVIDIA drivers. It samples
NVML metrics (GPU utilization, memory, temperature, power draw, MIG
partitions, training-phase detection) and streams them to the TensorCost
gateway over HTTPS or gRPC.

---

## Before you install: provisioning

Every agent needs a server-side record before it can authenticate. Open
**Build → Agents** (`/build/agents`) in the TensorCost console, mint a
key, and copy the one-time env block:

```sh
export TENANT_ID=<your-tenant-uuid>
export AGENT_HOSTNAME=<hostname-of-this-box>
export AGENT_KEY_ID=<key-id-from-wizard>
export AGENT_API_KEY=<api-key-from-wizard>
export AGENT_HMAC_PEPPER=<signing-key-from-wizard>
export BACKEND_API_URL=https://api.dev.tensorcost.com
export NVML_ENABLED=true   # required for GPU metrics — off by default
```

Copy and paste that block before running any of the install paths below.
These values are generated once; the plaintexts cannot be recovered after
the reveal dialog closes, so store them in your secret manager immediately.

For Terraform-based agent deploy on AWS/GCP/Azure, see [`agent/deploy/terraform/`](../agent/deploy/terraform/).

If your organization wants to set up the AWS cross-account IAM trust
before connecting Bedrock, launch this CloudFormation stack:

```
https://tensorcost-onboarding-public-845041271304.s3.amazonaws.com/latest/customer-onboarding.yml
```

---

## Install paths

### Bare-metal Linux (systemd)

Runs the agent as a persistent systemd service with automatic restart.
Best for rack servers, cloud VMs, or any GPU host that runs systemd.

See [`bare-metal/`](./bare-metal/) for the install script and unit file.

**Quick start:**

```bash
# 1. Paste the env block from the wizard (see above)

# 2. Run the install script as root
sudo BACKEND_API_URL="$BACKEND_API_URL" \
     TENANT_ID="$TENANT_ID" \
     AGENT_HOSTNAME="$AGENT_HOSTNAME" \
     AGENT_API_KEY="$AGENT_API_KEY" \
     AGENT_KEY_ID="$AGENT_KEY_ID" \
     AGENT_HMAC_PEPPER="$AGENT_HMAC_PEPPER" \
     bash bare-metal/install.sh

# 3. Verify the service started
systemctl status tensorcost-agent
journalctl -u tensorcost-agent -f
```

### Kubernetes DaemonSet

Deploys the agent as a DaemonSet so it runs on every GPU node in your
cluster. Works on AKS, EKS, GKE, and any k8s cluster with NVIDIA device
plugin installed.

See [`kubernetes/`](./kubernetes/) for the manifest and fill-in instructions.

**Quick start:**

```bash
# 1. Edit kubernetes/agent-daemonset.yaml — fill in TENANT_ID, AGENT_API_KEY,
#    AGENT_KEY_ID, AGENT_HMAC_PEPPER, and BACKEND_API_URL in the Secret section.
#    (Or supply them as a Helm values override — see kubernetes/README.md)

# 2. Apply
kubectl apply -f kubernetes/agent-daemonset.yaml

# 3. Verify
kubectl -n tensorcost get pods -l app=tensorcost-agent
kubectl -n tensorcost logs -l app=tensorcost-agent --tail=50
```

### Docker Compose (quick evaluation)

Good for trying the agent on a single machine with a GPU — no systemd,
no k8s. One file, one command.

See [`docker-compose/`](./docker-compose/).

**Quick start:**

```bash
cd docker-compose

# 1. Copy the env template and fill in your values from the wizard
cp .env.example .env
$EDITOR .env

# 2. Start
docker compose up -d

# 3. Tail logs
docker compose logs -f
```

---

## Image variants

The canonical images are published by [vaadhlabs/gpu-agents](https://github.com/vaadhlabs/gpu-agents)
CI publishes to `public.ecr.aws/tensorcost/gpu-agent` (preferred). Legacy Docker Hub
(`rlgocloudera/gpu-usage:<variant>`) and GHCR
(`ghcr.io/vaadhlabs/gpu-agent:<variant>`). The multi-cloud default is
`public.ecr.aws/tensorcost/gpu-agent:latest` (full variant). Legacy
`docker.io/rlgocloudera/gpu-usage:latest`. A public
ECR repository that requires no Docker Hub account or authentication. All
install paths default to this image.

| Tag | What's inside | When to use |
|---|---|---|
| `docker.io/rlgocloudera/gpu-usage:latest` | All cloud SDKs (full) | Docker compose / bare-metal default |
| `docker.io/rlgocloudera/gpu-usage:gcp` | GCP | `deploy/gcp` |
| `docker.io/rlgocloudera/gpu-usage:azure` | Azure | `deploy/azure` GPU mode |
| `docker.io/rlgocloudera/gpu-usage:aws` | AWS | `deploy/aws` |
| `docker.io/rlgocloudera/gpu-usage:node` | k8s node sidecar | DaemonSet |

To verify image signatures (cosign keyless OIDC) before deploying into
production, see the agent repo's `README.md` for the cosign + SBOM + SLSA
verification commands.

---

## Key environment variables

| Variable | Required | Description |
|---|---|---|
| `BACKEND_API_URL` | Yes | TensorCost API base URL, e.g. `https://api.dev.tensorcost.com` |
| `TENANT_ID` | Yes | Your tenant UUID from the onboarding wizard |
| `AGENT_HOSTNAME` | Yes | Hostname matching the server-side agent record |
| `AGENT_API_KEY` | Yes | API key for the `X-Agent-Key` HTTP header |
| `AGENT_KEY_ID` | Yes (gRPC) | Stable key identifier for HMAC verification |
| `AGENT_HMAC_PEPPER` | Yes (gRPC) | Hex-encoded 32-byte per-agent signing key |
| `NVML_ENABLED` | Yes (for GPU) | Set `true` to collect GPU metrics. **Defaults to `false`** — with it off the agent connects and heartbeats but reports no GPU data. |
| `NVML_SAMPLE_INTERVAL` | No | Seconds between NVML GPU samples (default: `10`) |
| `COMM_MODE` | No | `http`, `grpc`, or `both` (default: `http`) |
| `GRPC_TARGET` | No | `host:port` for gRPC endpoint when `COMM_MODE=grpc` or `both` |
| `GRPC_USE_TLS` | No | Set `true` for any non-loopback gRPC target (default: `false`) |
| `MONITORING_INTERVAL` | No | Seconds between cloud-billing telemetry pushes (default: `300`) — distinct from `NVML_SAMPLE_INTERVAL` |
| `LOG_LEVEL` | No | `DEBUG`, `INFO`, `WARN`, `ERROR` (default: `INFO`) |

For cloud-specific variables (AWS_ENABLED, GCP_PROJECT_ID, etc.) and
advanced options (TLS pinning, mTLS, actuator, tag allowlist) see the
agent repo's full documentation.

---

## Troubleshooting

**Agent starts but never shows up in the GPU dashboard.**

The most common cause is that the agent's `AGENT_HOSTNAME` doesn't match
the hostname stored in the server-side agent record. Check `journalctl -u
tensorcost-agent` (systemd) or `docker compose logs` for a line like
`hostname mismatch: got <X>, expected <Y>` and correct the env var.

The dashboard shows new agents with a ~2-minute lag after the first
heartbeat. If you see the agent record in the TensorCost settings page
but no metrics, wait 5 minutes, then check whether `MONITORING_INTERVAL`
was set unusually high.

**Authentication failures (`401 Unauthorized`).**

`AGENT_API_KEY` is a single-use secret emitted by the provisioning step.
If you lost the original output, reprovision via the wizard — there is no
recovery path for the plaintext.

**gRPC channel fails to connect.**

Make sure `GRPC_USE_TLS=true` for any public endpoint and that
`GRPC_TARGET` uses `host:443`, not `host:50051` (the TensorCost public
gateway terminates gRPC on 443). TLS=false is only safe on loopback or
inside a VPN segment.

**Docker: "NVML library not found" / "no NVIDIA GPU detected".**

The container needs `--gpus all` (Docker >= 19.03 with the
`nvidia-container-toolkit` installed on the host). The
`docker-compose/docker-compose.yml` already sets this; if you run
`docker run` manually, add `--gpus all --network host`.

**Support:** support@tensorcost.com
