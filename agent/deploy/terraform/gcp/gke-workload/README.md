# GCP `gke-workload` — agent on an existing GKE cluster, Workload Identity wired

Sub-module that drops the unified-gpu-agent onto a GKE cluster the
caller already owns. Deploys a Kubernetes ServiceAccount + Deployment
(or DaemonSet for per-GPU-node sampling), and wires GCP Workload
Identity end-to-end so the pod authenticates to GCP APIs (Secret
Manager, Cloud Monitoring, Billing) **without any service-account JSON
key**.

> **NO SERVICE ACCOUNT JSON KEYS, EVER.** The §8.3 audit explicitly
> called out the Helm chart's GCP path encouraging
> `gcloud iam service-accounts keys create` + mounting the resulting
> JSON. This module rejects that pattern. The K8s SA is annotated with
> `iam.gke.io/gcp-service-account=<gcp-sa>` and the GCP-side
> `roles/iam.workloadIdentityUser` binding is created for the
> `<project>.svc.id.goog[<ns>/<k8s-sa>]` principal. The pod's
> tokens are short-lived and minted by the metadata server.

## Pre-req: Workload Identity must be enabled on the cluster

```bash
gcloud container clusters update my-cluster \
  --location=us-central1 \
  --workload-pool=my-project.svc.id.goog
```

The module reads the cluster via a `data` source but does NOT mutate
the cluster's workload-identity setting (cluster-level changes belong
to whoever owns the cluster). If Workload Identity is not enabled, the
Deployment will start but the pod's Secret Manager calls will return
`IAM_PERMISSION_DENIED`.

## Pre-req: Kubernetes provider configured

This module uses the `kubernetes` provider to create resources in the
target cluster. Configure it however you normally would — typically:

```hcl
data "google_client_config" "default" {}

data "google_container_cluster" "target" {
  name     = "my-cluster"
  location = "us-central1"
}

provider "kubernetes" {
  host                   = "https://${data.google_container_cluster.target.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(data.google_container_cluster.target.master_auth[0].cluster_ca_certificate)
}
```

## Usage

```hcl
module "gpu_agent_gke" {
  source = "git::https://github.com/gpu-usage/gpu-agents.git//deploy/terraform/gcp/gke-workload?ref=terraform-gcp-v1.0.0"

  name             = "acme-prod-agent"
  project_id       = "acme-prod-12345"
  cluster_name     = "my-cluster"
  cluster_location = "us-central1"
  namespace        = "tensorcost"

  backend_api_url = "https://api.tensorcost.com"
  agent_api_key   = var.agent_api_key   # sensitive
  tenant_id       = var.tenant_id

  # Cost-collection only (default):
  # deploy_as = "deployment"

  # Or per-GPU-node NVML sampling:
  # deploy_as     = "daemonset"
  # node_selector = { "cloud.google.com/gke-accelerator" = "nvidia-tesla-t4" }
  # tolerations = [{ key = "nvidia.com/gpu", operator = "Exists", effect = "NoSchedule" }]

  # Production: pin the digest from cosign verify (see ead77a7).
  # agent_image_digest = "sha256:abcd…"
}
```

## How auth works (Workload Identity for GKE)

1. The module creates a GCP SA: `<name>-gke-sa@<project>.iam.gserviceaccount.com`.
2. The module creates a K8s SA in `var.namespace`, annotated with
   `iam.gke.io/gcp-service-account=<gcp-sa>`.
3. The module creates a `roles/iam.workloadIdentityUser` binding on
   the GCP SA for the principal
   `serviceAccount:<project>.svc.id.goog[<namespace>/<k8s-sa>]`.
4. The pod runs as the K8s SA. GKE's Workload Identity webhook
   automatically projects an OIDC token into the pod and routes
   `metadata.google.internal` → a GKE-internal proxy that exchanges
   the OIDC token for a short-lived GCP access token.
5. `gcloud`, ADC-based GCP client libraries, and the Secret Manager
   API all transparently use that token.

End result: the pod calls the Secret Manager API with a token tied to
the GCP SA. No JSON key file is ever materialised, on disk or in a K8s
Secret. The token expires hourly and is auto-refreshed.

## IAM minimums (granted by this module)

| Role | Scope | Granted to | Why |
|---|---|---|---|
| `roles/iam.workloadIdentityUser` | the GCP SA | the K8s-SA principal `<project>.svc.id.goog[<ns>/<sa>]` | Lets the K8s SA impersonate the GCP SA |
| `roles/monitoring.viewer` | project | the GCP SA | Read VM/pod metrics from Cloud Monitoring |
| `roles/secretmanager.secretAccessor` | the two secrets only | the GCP SA | Read agent_api_key + tenant_id from Secret Manager |
| `roles/billing.viewer` | billing account | the GCP SA | Cost-export queries — only when `billing_account_id` is set |
| `var.extra_iam_roles` | project | the GCP SA | Caller-supplied extras |

Roles deliberately NOT granted: `roles/owner`, `roles/editor`,
`roles/cloudkms.admin`, `roles/iam.serviceAccountAdmin`,
`roles/container.admin`, project-level `roles/secretmanager.secretAccessor`
(this module scopes to JUST the two secrets it created).

## Pinning `agent_image_digest`

Same pattern as the AWS module (commit ead77a7):

```bash
cosign verify docker.io/tensorcost/gpu-agent:gcp-1.2.3 \
  --certificate-identity-regexp '^https://github\.com/.+/gpu-agents/' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

Read the digest from `critical.image.docker-manifest-digest` and set
`agent_image_digest = "sha256:..."`. The Deployment's `image:` field
becomes `<image>@<digest>` instead of `<image>:<tag>` — kubelet pulls
a content-addressed manifest.

## DaemonSet vs Deployment

| Mode | When | Why |
|---|---|---|
| `deployment` (default) | Cost-collection only, single pod | One agent per cluster is enough for billing/cost queries |
| `daemonset` | Per-node NVML GPU sampling | Need one agent on every GPU node for utilisation telemetry |

For DaemonSet, pair with `node_selector` (e.g.
`cloud.google.com/gke-accelerator`) and `tolerations` for the
GPU node-pool's `nvidia.com/gpu` taint.

## Verifying the agent registered

```bash
kubectl -n <namespace> get pods -l app.kubernetes.io/instance=<name>
kubectl -n <namespace> logs -l app.kubernetes.io/instance=<name> --tail=200 | grep -i registered

# Workload Identity working?
kubectl -n <namespace> exec -it <pod> -- \
  curl -s -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email
# Should print: <name>-gke-sa@<project>.iam.gserviceaccount.com
```

The `verify_registration_hint` output prints the first two commands.

## Inputs

See [`variables.tf`](./variables.tf). Required:

| Variable | Description |
|---|---|
| `name` | Resource name prefix (RFC1035) |
| `project_id` | GCP project — also the cluster's project |
| `cluster_name` / `cluster_location` | The existing GKE cluster |
| `backend_api_url` | TensorCost dashboard API base URL |
| `agent_api_key` | Tenant API key (sensitive) |
| `tenant_id` | Tenant UUID |

Common optional: `namespace`, `deploy_as` (`deployment`/`daemonset`),
`node_selector`, `tolerations`, `resources`, `agent_image_digest`,
`billing_account_id`, `extra_iam_roles`, `labels`.

## Outputs

| Output | Description |
|---|---|
| `deployment_name` / `deployment_kind` / `namespace` | Where the agent runs |
| `kubernetes_service_account_name` | The K8s SA the pod runs as |
| `gcp_service_account_email` | GCP SA the K8s SA impersonates |
| `workload_identity_member` | Principal name for the WI binding (debug aid) |
| `secret_agent_api_key_name` / `secret_tenant_id_name` | Secret Manager resource names |
| `verify_registration_hint` | One-liners to confirm registration |
