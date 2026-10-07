# GCP `compute-engine` — single-VM agent install on Google Compute Engine

Greenfield sub-module: Terraform creates a GCE VM, attaches a
purpose-built service account to it (Workload Identity), drops the
agent's secrets into Secret Manager scoped to that SA, and runs the
agent under `systemd` via a startup-script.

Sibling of [`../gke-workload/`](../gke-workload/) which is for tenants
whose GPUs are already on a GKE cluster. Most "I have a single GPU VM"
use cases want THIS module.

> **NO SERVICE ACCOUNT JSON KEYS.** The agent authenticates to all GCP
> APIs (Secret Manager, Artifact Registry, Cloud Monitoring, Billing
> Export queries) via the VM's attached service account through the
> metadata server. Long-lived JSON keys are GCP's #1 credential-leak
> vector — this module never creates or mounts one. The §8.3 audit
> explicitly called out the Helm chart's GCP path encouraging the JSON
> pattern; this module is the Terraform answer.

## Usage

```hcl
module "gpu_agent" {
  source = "git::https://github.com/gpu-usage/gpu-agents.git//deploy/terraform/gcp/compute-engine?ref=terraform-gcp-v1.0.0"

  name       = "acme-prod-agent"
  project_id = "acme-prod-12345"
  region     = "us-central1"
  zone       = "us-central1-a"

  # CPU-only cost-collection bastion:
  machine_type = "e2-standard-2"

  # Or a GPU host:
  # machine_type = "n1-standard-8"
  # gpu_type     = "nvidia-tesla-t4"
  # gpu_count    = 1

  backend_api_url = "https://api.tensorcost.com"
  agent_api_key   = var.agent_api_key   # sensitive
  tenant_id       = var.tenant_id

  # Optional: pin the image digest for production (see ead77a7).
  # agent_image_digest = "sha256:abcd…"

  # Optional: grant billing.viewer on a billing account for cost-export
  # billing_account_id = "01ABCD-234567-EFGH89"

  labels = {
    env  = "prod"
    team = "platform"
  }
}
```

One `terraform apply` →
service account created →
secrets written to Secret Manager →
SA gets `secretmanager.secretAccessor` scoped to JUST those two secrets →
VM boots →
startup-script reads secrets via `gcloud secrets versions access` (uses
attached SA via metadata server, no JSON key) →
Docker + (optional) NVIDIA driver + toolkit installed →
systemd unit starts →
agent registers with backend within ~5 minutes.

## How auth works (Workload Identity for VMs)

1. The module creates a GCP service account: `<name>-sa@<project>.iam.gserviceaccount.com`.
2. The VM is launched with `service_account { email = ..., scopes = ["cloud-platform"] }`.
3. Application Default Credentials on the VM resolve through the GCE
   metadata server (`http://metadata.google.internal/.../token`) using
   that SA. The SDK / `gcloud` / `docker-credential-gcr` all transparently
   use it.
4. No `GOOGLE_APPLICATION_CREDENTIALS` env var. No JSON key file. No
   `iam.serviceAccounts.keys.create` call.

To rotate "credentials": there's nothing to rotate — the SA's tokens are
already short-lived (1 h) and minted on demand. To revoke: delete the SA
or remove its IAM bindings.

## IAM minimums (granted by this module)

| Role | Scope | Why |
|---|---|---|
| `roles/monitoring.viewer` | project | Read instance/VM metrics from Cloud Monitoring |
| `roles/secretmanager.secretAccessor` | the two secrets only | Read agent_api_key + tenant_id from Secret Manager |
| `roles/billing.viewer` | billing account | Cost-export queries — only when `billing_account_id` is set |
| `var.extra_iam_roles` | project | Caller-supplied extras (`roles/bigquery.dataViewer` for BQ cost exports etc.) |

Roles deliberately NOT granted by default:
- `roles/owner`, `roles/editor` — too broad. Agent never mutates infra.
- `roles/cloudkms.admin` — agent never touches KMS keys.
- `roles/iam.serviceAccountAdmin` — agent never creates/deletes SAs.
- `roles/compute.viewer` at project — only via `extra_iam_roles` when needed.

The audit (§8.3) flagged that the Helm chart's GCP path "encourages
mounting a long-lived service account JSON". This module's IAM bindings
are intentionally narrow precisely so the SA's tokens (which are what
actually authenticate the agent) cannot be used as a foothold elsewhere
in the project.

## Pinning `agent_image_digest`

Defaults to the mutable tag `docker.io/tensorcost/gpu-agent:gcp` so the
module works out of the box. For production fleets, pin a content-
addressed digest verified against the cosign signature published by
the parallel image-signing change (commit ead77a7):

```bash
cosign verify docker.io/tensorcost/gpu-agent:gcp-1.2.3 \
  --certificate-identity-regexp '^https://github\.com/.+/gpu-agents/' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

Read the digest from the `critical.image.docker-manifest-digest` field
and set `agent_image_digest = "sha256:..."` in the module call. The
startup-script then pulls `<image>@<digest>` instead of `<image>:<tag>`
— a compromised registry credential cannot retag a malicious image
onto your fleet because the digest is content-addressed. The install
script logs a `WARNING:` when the digest is empty so the path is
visible in Cloud Logging.

The module's `variables.tf` validates the digest with the same regex
as the AWS module (`^sha256:[0-9a-f]{64}$`), keeping the three clouds
symmetric.

## Verifying the agent registered

```bash
gcloud compute ssh <instance-name> --zone=<zone> -- \
  'sudo journalctl -u unified-gpu-agent -n 50 | grep -i registered'

# or the install log:
gcloud compute ssh <instance-name> --zone=<zone> -- \
  'sudo tail -f /var/log/tensorcost-agent-install.log'
```

The `verify_registration_hint` output prints the exact one-liner.

## Inputs

See [`variables.tf`](./variables.tf). Required:

| Variable | Description |
|---|---|
| `name` | Resource name prefix (RFC1035) |
| `project_id` | GCP project ID |
| `region` / `zone` | Where to launch the VM |
| `backend_api_url` | TensorCost dashboard API base URL |
| `agent_api_key` | Tenant API key (sensitive) |
| `tenant_id` | Tenant UUID |

Common optional: `machine_type`, `gpu_type`, `gpu_count`, `network`,
`subnetwork`, `agent_image_digest`, `use_private_artifact_registry`,
`billing_account_id`, `extra_iam_roles`, `labels`.

## Outputs

| Output | Description |
|---|---|
| `instance_name` / `instance_self_link` / `instance_zone` | The VM |
| `service_account_email` | Principal the agent authenticates as |
| `secret_agent_api_key_name` / `secret_tenant_id_name` | Secret Manager resource names |
| `install_log_path` | Where the startup-script logs |
| `agent_service_name` | systemd unit (for `journalctl -u`) |
| `verify_registration_hint` | One-liner to confirm registration |
