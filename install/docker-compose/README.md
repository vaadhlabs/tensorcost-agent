# TensorCost GPU Agent — Docker Compose quickstart

This is the fastest way to get the GPU agent running on a single machine.
Good for evaluation, for dev workstations with an NVIDIA GPU, or for a
single-node deployment before you're ready for a DaemonSet.

---

## Prerequisites

- Docker Engine >= 20.10 with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.
- The Docker daemon must be restarted after the toolkit is installed.
- Agent credentials from the TensorCost onboarding wizard.
  - Allocate at least 16 GB of disk on the host. The agent image plus container runtime overhead exceeds the 8 GB default on AL2023 t3.micro AMIs; an actual customer hit `no space left on device` during `docker pull` in our 2026-05-18 E2E test.

Quick check that your GPU is visible to Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

If this fails, install the toolkit and restart Docker before continuing.

---

## Setup

```bash
# 1. Copy the env template
cp .env.example .env

# 2. Fill in your credentials (open in any editor)
$EDITOR .env

# 3. Start the agent
docker compose up -d

# 4. Confirm it's running and healthy
docker compose ps
docker compose logs -f
```

The agent should appear in the TensorCost GPU dashboard within about
2 minutes of the first successful heartbeat.

---

## Enabling cloud cost collection (optional)

The default configuration only collects NVML GPU metrics (utilization,
memory, temperature, power). To also pull cost data from a cloud billing
API, edit `.env`:

**AWS:**
```
AWS_ENABLED=true
AWS_REGION=us-east-1
```
Then uncomment the `~/.aws:/home/gpuagent/.aws:ro` volume in
`docker-compose.yml` to mount your AWS credentials, or rely on an EC2
instance profile if running on AWS.

**GCP:**
```
GCP_ENABLED=true
GCP_PROJECT_ID=your-project-id
```
Place your GCP service account key at `./gcp-key.json` and uncomment the
corresponding volume mount.

**Azure:**
```
AZURE_ENABLED=true
AZURE_SUBSCRIPTION_ID=your-subscription-id
```
Uses managed identity when running on an Azure VM. On a non-Azure host,
also set `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, and `AZURE_TENANT_ID`.

---

## Stopping and removing

```bash
docker compose down         # stop containers, keep volumes
docker compose down -v      # stop and remove volumes (loses agent state)
```
