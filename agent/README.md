# TensorCost GPU Agent

The TensorCost GPU monitoring agent. Runs on any Linux host with NVIDIA
drivers, streams GPU sampling + instance-state data to the TensorCost
gateway, and stays out of the way — stateless, containerized, no
database, ~50MB resident memory.

Points at the TensorCost platform gateway (`/api/sync/*` HTTP and optional
gRPC). It is **not** compatible with the legacy `gpu-dashboard-backend`.

## What it does

- **GPU sampling** via NVML — utilization, memory, temperature, power, MIG
  partitions, training-phase detection (idle / data-loading /
  checkpointing / eval / active-inference).
- **Instance inventory + utilization** for AWS, Azure, GCP, Kubernetes, and
  SageMaker — pushed to `/api/sync/instances`, `/api/sync/metrics`, and
  `/api/sync/health`.
- **Remote commands** — stop, start, resize, restart, and terminate via the
  actuator when `AGENT_ACTUATOR_ENABLED=true`.
- **Kubernetes awareness** — when deployed as a DaemonSet, samples each
  node's GPUs and annotates pods with GPU-time-per-pod.
- **Optional in-process LLM middleware** — library-only wrappers for OpenAI,
  Anthropic, Bedrock, and Azure OpenAI. The agent does **not** push token
  spend; AI cost is ingested server-side by cost-service and ai-service.
- **Streams to a single backend** — HTTPS + optional gRPC. No local database;
  agent restarts don't lose unsent samples beyond the heartbeat file.

## Getting started — enroll in the console, then run

A `gpu.agent` row (and API key) must exist **before** the agent can
authenticate. Customers mint that from the TensorCost console — not from
this repo.

1. Open **Setup → Agents** (Build / Agents) in the TensorCost console.
2. **Enroll agent** → pick a display name / hostname → **Mint**.
3. **Save credentials immediately.** The plaintext API key is shown once.
   Capture at least:
   - `TENANT_ID`
   - `AGENT_HOSTNAME`
   - `AGENT_KEY_ID`
   - `AGENT_API_KEY`
   - `AGENT_HMAC_PEPPER`
   - Backend endpoint (prefilled in the install snippet)
4. Copy the install snippet for **docker**, **helm**, or **curl/systemd**
   from the same dialog (or paste the env block into your secret store and
   use the install path that matches your fleet).

Internal TensorCost operators who need CLI provisioning without the UI
should use the in-repo gpu-service tooling — that path is not a customer
install guide.

## Installing

Three customer-supported paths. Values come from the console mint dialog.

### 1. Docker (fastest)

```bash
docker run -d \
  --name tensorcost-agent \
  --restart=unless-stopped \
  -e BACKEND_API_URL='https://<your-gateway>' \
  -e TENANT_ID='<tenant uuid>' \
  -e AGENT_HOSTNAME='<hostname from mint>' \
  -e AGENT_KEY_ID='<key id from mint>' \
  -e AGENT_API_KEY='<plaintext key from mint>' \
  -e AGENT_HMAC_PEPPER='<hex pepper from mint>' \
  # unique per host; the agent auto-generates if not set
  -e AGENT_INSTANCE_ID='$(hostname -s)-$(uuidgen | cut -c1-8)' \
  public.ecr.aws/g2c6m0v1/agent:latest
```

Your console snippet uses the correct image registry and endpoint for your
tenant; prefer that over editing this example by hand.

### 2. Helm (Kubernetes)

```bash
helm repo add tensorcost https://charts.tensorcost.com
helm repo update
helm install tensorcost-agent tensorcost/agent \
  --namespace tensorcost --create-namespace \
  --set backend.apiUrl='https://<your-gateway>' \
  --set agent.tenantId='<tenant uuid>' \
  --set agent.hostname='<hostname from mint>' \
  --set agent.keyId='<key id>' \
  --set agent.apiKey='<plaintext key>' \
  --set agent.hmacPepper='<hex pepper>'
```

DaemonSet mode is preferred for per-node NVML sampling. Chart README:
`deploy/helm/` when shipped with your install package.

### 3. curl + systemd (bare metal)

```bash
# 1) Save the env file (keep it 0600 root-owned).
sudo install -m 0700 -d /etc/tensorcost-agent
sudo tee /etc/tensorcost-agent/.env <<'EOF' >/dev/null
BACKEND_API_URL=https://<your-gateway>
TENANT_ID=<tenant uuid>
AGENT_HOSTNAME=<hostname>
AGENT_KEY_ID=<key id>
AGENT_API_KEY=<plaintext key>
AGENT_HMAC_PEPPER=<hex pepper>
EOF
sudo chmod 0600 /etc/tensorcost-agent/.env

# 2) Install (systemd unit + binary) from your tenant's gateway.
curl -fsSL https://<your-gateway-host>/install.sh | sudo bash
```

The exact `curl` URL is in the console snippet (it matches your gateway).

## Configuration

Everything is driven by environment variables. The container loads
`/etc/default/unified-gpu-agent` first (if present, used by the Terraform
modules' systemd unit) and falls back to direct env vars.

### Required (HTTP transport)

| Variable          | What it is                                              |
| ----------------- | ------------------------------------------------------- |
| `BACKEND_API_URL` | Base URL of the TensorCost gateway (e.g. `https://gateway.<env>.tensorcost.com`). |
| `AGENT_API_KEY`   | Plaintext API key for the HTTP `X-Agent-Key` header. Emitted by the provisioning CLI. |
| `TENANT_ID`       | Tenant UUID — must match the `--tenant` flag the CLI ran with. |
| `AGENT_HOSTNAME`  | Hostname matching `gpu.agent.hostname` server-side. Defaults to `AGENT_ID` for legacy compose files. |

`BACKEND_API_KEY` is accepted as a fallback for `AGENT_API_KEY` so older
compose files don't break — but new deployments should use the new var.

### Required (gRPC transport, optional but recommended)

| Variable             | What it is                                                       |
| -------------------- | ---------------------------------------------------------------- |
| `GRPC_TARGET`        | `host:port` for the apps-new gateway gRPC endpoint.              |
| `AGENT_KEY_ID`       | Stable identifier emitted by the provisioning CLI; used by the HMAC verifier to look up `gpu.agent.api_key_hmac`. |
| `AGENT_HMAC_PEPPER`  | Hex-encoded 32-byte signing-key blob from the provisioning CLI. Despite the name this IS the per-agent signing key, not the server-side pepper. |

### Common optional

| Variable                | Default          | Purpose                                     |
| ----------------------- | ---------------- | ------------------------------------------- |
| `COMM_MODE`             | `http`           | `http`, `grpc`, or `both`.                  |
| `GRPC_USE_TLS`          | `false`          | Set `true` for any non-loopback target.     |
| `MONITORING_INTERVAL`   | `300`            | Seconds between samples / pushes.           |
| `HEALTH_INTERVAL`       | `60`             | Heartbeat cadence (independent of monitoring cycle). |
| `AGENT_CLOUD_PROVIDER`  | —                | Surfaces in the AgentHealth DTO; used by the dashboard. |
| `AGENT_REGION`          | `AWS_REGION`     | Same — surfaces in the health DTO.          |
| `LOG_LEVEL`             | `INFO`           | `DEBUG`, `INFO`, `WARN`, `ERROR`.           |

### Cloud-specific

- **AWS**: `AWS_ENABLED=true`. Credentials via IAM role (EC2 instance
  profile, ECS task role, or IRSA on EKS) — no static keys.
- **Azure**: `AZURE_ENABLED=true` + `AZURE_SUBSCRIPTION_ID`. Auth via
  system-assigned managed identity on the VM.
- **GCP**: `GCP_ENABLED=true` + `GCP_PROJECT_ID`. Auth via Workload
  Identity (GKE) or instance service account (GCE).
- **Kubernetes**: `K8S_ENABLED=true`. When running as a pod, reads the
  in-cluster service account; external clients set `KUBECONFIG` path.

### Actuator (instance stop / start / resize / restart / terminate)

The agent can dispatch lifecycle commands on customer cloud instances
when the backend pushes them via gRPC or surfaces them via
`GET /api/sync/agent/commands`. **This path is fail-closed by default**
— even if the backend sends a `terminate` command, the agent will
log it and respond `status=rejected` unless both env vars below are
set. The shipped Terraform / Helm modules grant the agent read-only
IAM, which is a second layer of protection.

| Var                              | Default | Effect                                                                                                            |
| -------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------- |
| `AGENT_ACTUATOR_ENABLED`         | (unset) | Must be the literal string `true` (case-insensitive) to enable. `1`, `yes`, `on` all leave the gate closed.       |
| `AGENT_ACTUATOR_ALLOWED_ACTIONS` | (unset) | Comma-separated subset of `stop,start,resize,restart,terminate`. Any action not listed is rejected and logged.    |

Two gates rather than one because operators commonly want
stop+start+resize without permitting terminate. The destructive
action has to be named explicitly — there is no single-toggle that
opens the entire surface. Every reject and dispatch is logged at
INFO/WARN with the full `(command_id, action_type, instance_id,
cloud_provider)` tuple so the operator-side audit trail is intact
even when the gate blocks the call.

If you also need to revoke the IAM permissions backing the actuator,
remove the `ec2:StopInstances`, `ec2:StartInstances`,
`ec2:RebootInstances`, `ec2:TerminateInstances`, and
`ec2:ModifyInstanceAttribute` entries from your role — the shipped
Terraform modules do not include them.

For the YAML-file form of the same knobs see
[`config.yaml.example`](./config.yaml.example).

### Tag handling (egress allowlist + PII filter)

Cloud-instance tags routinely carry owner emails, project codenames,
and occasionally secrets. The agent does **not** egress raw tag values
outside of an allowlist. The choke point is `filter_tags` in
`src/transport/http_client.py` — both the HTTP transport
(`rename_instance_to_wire`) and the gRPC `_send_instances` path call
it, so the posture is identical across both wires.

Default allowlist (the keys typical cost-attribution pipelines need):

  `cost-center`, `cost_center`, `environment`, `env`, `team`, `owner`,
  `project`, `service`, `tier`

Anything else is dropped before it leaves the host. Values longer than
256 characters are truncated to `value[:253] + "..."`.

| Var                       | Default | Effect                                                                                                                         |
| ------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `AGENT_TAG_ALLOWLIST`     | (unset) | Comma-separated key list that **replaces** the default allowlist. An empty string drops every tag. Whitespace tolerated.       |
| `AGENT_TAG_HASH_UNKNOWN`  | (unset) | When set to `true` (case-insensitive), keys not in the allowlist are retained but their values are replaced by `sha256(value)[:16]` — useful when ops needs tag *cardinality* without exfiltrating literal values. |

If you need a tag visible to the cost dashboard, add the key to
`AGENT_TAG_ALLOWLIST`; if you need only the cardinality, set
`AGENT_TAG_HASH_UNKNOWN=true`. The two are independent — you can
allow `cost-center` and hash everything else.

### TLS / mTLS hardening

By default, both the HTTPS and gRPC clients trust the host's system
root store. For most cloud-VM agents that's the right call — a
public ACM-issued cert validates fine against `ca-certificates`. But
because the agent has a path that can stop / terminate customer
instances (see "Actuator" above), operators with stronger threat
models can opt into TLS certificate pinning, replace the trust root
entirely with their own CA bundle, or layer mTLS on top.

**Pinning is OFF by default and you must opt in** — this is
deliberate, because customer environments routinely run
TLS-terminating corporate proxies and a default-on pin would brick
the first deploy. The config is fail-fast: bad base64, missing files,
half-set mTLS pair, or `http://` URLs while pinning is on all crash
the agent at boot rather than silently falling back to system roots.

| Var                            | Default | Effect                                                                                                                                              |
| ------------------------------ | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `AGENT_TLS_PIN_SHA256_HTTPS`   | (unset) | Comma-separated list of base64-encoded SPKI SHA-256 hashes (RFC 7469 §2.4). The HTTPS chain to `BACKEND_API_URL` must match at least one.           |
| `AGENT_TLS_PIN_SHA256_GRPC`    | (unset) | Same shape, for the gRPC channel to `GRPC_TARGET`. Validated via a stdlib TLS probe before each channel is opened.                                  |
| `AGENT_TLS_CA_BUNDLE`          | (unset) | Path to a PEM bundle that **replaces** the system root store on both planes. Useful for the BYOC topology where the gateway cert is issued by the customer's internal CA. |
| `AGENT_MTLS_CLIENT_CERT`       | (unset) | Client cert PEM path. Must be set together with `AGENT_MTLS_CLIENT_KEY` — half-set fails at boot, no silent single-sided fallback.                  |
| `AGENT_MTLS_CLIENT_KEY`        | (unset) | Client private key PEM path.                                                                                                                        |

Pinning is layered on top of normal certificate validation — the
chain must validate AND at least one cert in the chain must match a
configured pin. Multiple pins are supported (and recommended) so
cert rotation does not require an immediate agent restart: ship the
old pin and the new pin together, rotate the cert at the gateway,
then drop the old pin on the next agent release.

To compute the pin for `gateway.tensorcost.com`:

```sh
echo | openssl s_client -connect gateway.tensorcost.com:443 -servername gateway.tensorcost.com 2>/dev/null \
  | openssl x509 -pubkey -noout \
  | openssl pkey -pubin -outform DER \
  | openssl dgst -sha256 -binary \
  | base64
```

The same recipe with `grpc.tensorcost.com:443` produces the gRPC
pin. Once cert rotation is on a known cadence, ship two pins (the
current cert and the next one) so a routine rotation never needs
an agent push.

Worked example for an EKS / Helm deployment:

```sh
AGENT_TLS_PIN_SHA256_HTTPS=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=,BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=
AGENT_TLS_PIN_SHA256_GRPC=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=,BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=
# Optional — BYOC topology pinning the gateway cert to the customer's internal CA:
AGENT_TLS_CA_BUNDLE=/etc/tensorcost/internal-ca.pem
# Optional — mTLS, where the gateway requires a client cert:
AGENT_MTLS_CLIENT_CERT=/etc/tensorcost/agent-client.crt
AGENT_MTLS_CLIENT_KEY=/etc/tensorcost/agent-client.key
```

A pin mismatch surfaces as `gRPC TLS pin mismatch:` in the agent
log (gRPC plane) or a connection error containing `TLS pin
mismatch for HTTPS gateway` (HTTPS plane). Both messages include
the leaf SPKI hash the agent saw, which is the exact value you'd
copy into the env var if the cert legitimately rotated.

## Architecture

```
┌──────────────────────┐       ┌─────────────────────────────────┐
│  Unified GPU Agent   │       │      tensorcost backend          │
│                      │ HTTP  │                                   │
│  ┌─ NVML sampler ─┐  ├──────▶│  /api/ingest/* endpoints         │
│  ├─ cost monitors ─┤  │ gRPC │     → Postgres (hot)             │
│  ├─ LLM middleware ┤  ├──────▶│     → S3/Glacier (cold)          │
│  └─ k8s informer ──┘  │       │     → dashboard + alerting       │
└──────────────────────┘       └─────────────────────────────────┘
```

Stateless by design. Every sample / cost row / LLM call is a single HTTP
POST; no queuing, no local DB. Restart kills nothing except in-flight
requests, and those retry with the next tick.

## Release artifacts

Each artifact is versioned independently so a chart bug fix doesn't force
an image rebuild, and a new agent image doesn't force a Terraform
re-release.

| Artifact         | Tag prefix                | Published to                                 |
| ---------------- | ------------------------- | -------------------------------------------- |
| Docker image     | `v<semver>`               | Docker Hub + GHCR (mirror)                   |
| Helm chart       | `helm-v<semver>`          | GHCR OCI — `ghcr.io/<owner>/charts/gpu-agent` |
| Terraform (AWS)  | `terraform-aws-v<semver>` | GitHub Release with tarball                  |
| Terraform (Azure)| `terraform-azure-v<semver>` | GitHub Release with tarball                |
| Terraform (GCP)  | `terraform-gcp-v<semver>` | GitHub Release with tarball                  |

The `appVersion` field in `deploy/helm/gpu-agent/Chart.yaml` tracks the
Docker image the chart was last tested against; the chart's own `version`
evolves independently.

See [`.github/workflows/`](./.github/workflows/) for the exact publish
logic.

## Verifying the image

Every variant is signed with [cosign](https://github.com/sigstore/cosign)
in keyless OIDC mode, accompanied by a CycloneDX SBOM, and stamped with
a SLSA build-provenance attestation. The signing identity is the GitHub
Actions workflow that built it — there is no shared signing key to leak.

### 1. Verify the signature

```sh
cosign verify docker.io/tensorcost/gpu-agent:<tag> \
  --certificate-identity-regexp '^https://github\.com/.+/gpu-agents/\.github/workflows/docker-publish\.yml@' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

Substitute the GHCR mirror (`ghcr.io/<owner>/gpu-agent:<tag>`) for the
same identity check — both registries are signed in one workflow run.
The command exits non-zero if the signature is missing, expired, or
bound to a different workflow.

### 2. Inspect the SBOM

The CycloneDX SBOM is attached as a cosign attestation with predicate
type `cyclonedx`. Pull and view it with:

```sh
cosign download attestation \
  --predicate-type=https://cyclonedx.org/bom \
  docker.io/tensorcost/gpu-agent:<tag> \
  | jq -r '.payload | @base64d | fromjson | .predicate'
```

This is the same artifact the agent's CI generated with syft against
the multi-arch manifest, so every package in every layer is enumerated.

### 3. Inspect SLSA build provenance

GitHub stamps a SLSA-v1 in-toto provenance attestation against the
manifest digest. The fastest path is the `gh` CLI:

```sh
gh attestation verify oci://docker.io/tensorcost/gpu-agent:<tag> \
  --owner <github-org>
```

For programmatic consumption, the same attestation is also pushed to
the registry — fetch with `cosign download attestation
--predicate-type=https://slsa.dev/provenance/v1`.

### 4. Pin a digest into your fleet

Once you have verified a tag, capture its `sha256:...` digest from
`cosign verify`'s output and pin it. The Terraform EC2 SSM module
accepts `agent_image_digest`:

```hcl
module "gpu_agent" {
  source             = "github.com/gpu-usage/gpu-agents//deploy/terraform/aws/ec2-agent?ref=terraform-aws-v1.2.3"
  agent_image        = "docker.io/tensorcost/gpu-agent:aws-1.2.3"
  agent_image_digest = "sha256:abcdef0123456789..."  # from cosign verify
  # ... other module inputs ...
}
```

The install script then pulls `image@sha256:...` instead of
`image:tag`, so the SSM Association's 6-hourly re-run cannot pick up a
replacement image even if a registry credential leaks.

## Development

```bash
# Clone + install dev deps
git clone git@github.com:gpu-usage/gpu-agents.git
cd gpu-agents
pip install -r requirements-full.txt -r requirements-dev.txt

# Run tests
pytest -v

# Coverage
pytest --cov=src --cov-report=term-missing

# Build a specific variant locally
docker build --build-arg VARIANT=azure -t gpu-agent:azure .

# Run the agent directly against a dev backend
# (provision an agent server-side first — see "Getting started"; then
#  source the printed env block before running)
BACKEND_API_URL=http://localhost:4105 \
TENANT_ID=... AGENT_HOSTNAME=... \
AGENT_API_KEY=... AGENT_KEY_ID=... AGENT_HMAC_PEPPER=... \
ACTIVE_MONITORS=nvml \
python main.py
```

### Releasing

1. **Docker image** (every `main` push): automatic. Tag `v1.2.3` to cut a
   semver release.
2. **Helm chart**: bump `version:` in
   `deploy/helm/gpu-agent/Chart.yaml`, then
   `git tag helm-v<same-version> && git push --tags`. The publish
   workflow fails fast if tag and Chart.yaml disagree.
3. **Terraform module**: `git tag terraform-<cloud>-v<semver> && git push --tags`.

## Contributing

1. Fork, branch off `main`.
2. `pre-commit install` (hooks run black, flake8, mypy).
3. Add tests for new behavior.
4. Open a PR — CI runs the full test matrix + a container smoke test.

## Migration from legacy agent

If you're updating an existing deployment that pointed at
`apps/gpu-dashboard-backend` (the legacy backend), the wire layer is
not compatible with the new TensorCost gateway. Plan for a re-provision,
not a DNS flip.

### What changed

| | Legacy agent | This branch (`agent-rewrite-c`) |
| --- | --- | --- |
| **gRPC service** | `gpu_agent.AgentService.AgentStream` | `tensorcost.agent.v1.AgentService.Connect` |
| **gRPC auth** | metadata `x-api-key` + `x-tenant-id` | HMAC-SHA256-signed `AgentHello` (`AGENT_KEY_ID` + `AGENT_HMAC_PEPPER`) |
| **HTTP auth header** | `X-API-Key` + `X-Tenant-ID` | `X-Agent-Key` (`AGENT_API_KEY`) |
| **HTTP envelope** | `POST /api/sync` with 15 `data_type`s | `POST /api/sync/` (metrics + instance_updates only) + 4 dedicated endpoints |
| **Metric fields** | `gpu_utilization`, `memory_utilization`, `temperature_c`, `timestamp` (ISO) | `util_pct`, `mem_pct`, `temp_c`, `ts_unix_ms` (epoch ms) — translated by the agent's transport layer |
| **Spot interruption** | `alerts` envelope (`alert_type=spot_interruption`) | `POST /api/sync/spot-interruption` (dedicated DTO) |
| **Health DTO** | `{status, monitors[], timestamp, agent_version, tenant_id}` | `{hostname, version, status, cloud_provider?, region?}` |
| **Provisioning** | hand-rolled SQL or shared tenant key | `pnpm --filter @tensorcost/gpu-service provision:agent` |

### Dropped event types

The new backend does not have ingest endpoints for these — apps-new
fills these tables via different pipelines (cost-service puller,
ai-service integrations, monitoring-service derivation):

`costs` · `ai_spend` · `inference_metrics` · `training_runs` ·
`gateway_metrics` · `cluster_health` · generic `alerts` (other than
spot interruption) · `recommendations` · `errors`

The collectors for these still load (so existing customer YAML / env
configs don't blow up on boot) but they no longer push anywhere. See
the `# ---- DROPPED:` markers in `main.py` for the exact cycles
that were switched off, and `apps-new/docs/AGENT_COMPATIBILITY.md` §2.3
for the full table.

### Migration checklist for an in-place upgrade

1. Provision a fresh `gpu.agent` row for each running agent host
   (`pnpm --filter @tensorcost/gpu-service provision:agent ...`).
2. Replace the old `BACKEND_API_KEY` / `X-API-Key` env in your compose
   / k8s Secret with the new `AGENT_API_KEY` from the CLI output.
3. If you used gRPC, also set `AGENT_KEY_ID` + `AGENT_HMAC_PEPPER`
   (gRPC will refuse to authenticate without them now).
4. Pull the new image (or rebuild from this branch).
5. If you depend on cost / ai-spend / training-run / cluster-health
   data flowing from the agent, switch to the apps-new server-side
   pipelines for those — they are NOT agent-pushed any more.

---

## Followups (out of scope for this branch)

Things this branch identified but deliberately did NOT fix:

- ~~**Cloud-native instance ID resolution.**~~ **Resolved 2026-05-07.**
  Server-side `MetricPointDto` / `InstanceUpdateDto` accept
  `cloud_id` + `cloud_provider` and `SyncService.insertMetrics`
  batch-resolves them to the local UUID via the unique
  `(tenant_id, cloud_provider, instance_id)` index on `gpu.instance`.
  The agent's `rename_metric_to_wire` / `rename_instance_to_wire` now
  detect UUID-shaped ids and route everything else through the
  cloud-id pair, with `_send_metrics` / `_send_instances` stamping
  `self.cloud_provider` as a fallback for collectors that don't carry
  it per-record.
- **Reviving the dropped event types.** If apps-new adds ingest
  endpoints for `costs` / `ai_spend` / `inference_metrics` / etc., the
  corresponding `# ---- DROPPED:` cycles in `main.py` can be re-enabled
  by switching them back to call into a new transport method.
- **`gpu.agent` ↔ `tenant` foreign key.** The provisioning CLI does not
  validate the tenant UUID exists. Operators with a typo'd `--tenant`
  will silently orphan the row.

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). Apache 2.0 was chosen over MIT
to provide an explicit patent grant, which matters for a privileged binary
that runs inside enterprise customer infrastructure.
