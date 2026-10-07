# Terraform modules

Reusable Terraform modules for deploying the tensorcost `unified-gpu-agent`
to AWS / Azure / GCP. Each module is single-purpose — it installs the
agent onto compute the caller provisioned (or provisions a minimal demo
topology around it). Higher-level stacks like "demo environment with AKS +
Azure ML + OpenAI" live in consumer repos, not here.

## Modules

| Path | Target | Tag prefix | Status |
|---|---|---|---|
| [`aws/fargate/`](./aws/fargate/) | ECS Fargate task — cost-collection only, no GPU | `terraform-aws-v*` | Stable |
| [`aws/ec2-agent/`](./aws/ec2-agent/) | Install onto an existing EC2 instance (BYO-VM, SSM Association) | `terraform-aws-v*` | v1.1.0 |
| [`azure/`](./azure/) | Linux VM + cloud-init (greenfield) | `terraform-azure-v*` | v1.0.0 |
| [`azure/agent-extension/`](./azure/agent-extension/) | Install onto an existing Azure VM (BYO-VM, VM Extension) | `terraform-azure-v*` | v1.0.0 |
| [`gcp/compute-engine/`](./gcp/compute-engine/) | GCE VM + startup-script (greenfield, GPU-capable, Workload Identity) | `terraform-gcp-v*` | v1.0.0 |
| [`gcp/gke-workload/`](./gcp/gke-workload/) | Deployment / DaemonSet on an existing GKE cluster (Workload Identity) | `terraform-gcp-v*` | v1.0.0 |

Per-cloud sub-modules ship on the same tag — one release, multiple
consumable sub-paths. AWS has two shapes (Fargate + EC2) because the
Fargate path is cost-collection-only and doesn't cover EC2 GPU
workloads. Azure has one root + one sub-module (greenfield + BYO-VM).
GCP has two sub-modules: `compute-engine/` for greenfield single-VM
installs (cost or GPU) and `gke-workload/` for tenants whose GPUs are
on an existing GKE cluster. Both GCP sub-modules use **Workload
Identity** end-to-end — no service-account JSON keys are ever issued
or mounted (per §8.3 audit).

## Consuming a module

Modules are sourced from this repo via git-tagged refs. The `ref` MUST pin a
specific version; don't track `main` from a production deployment.

```hcl
module "gpu_agent" {
  source = "git::https://github.com/gpu-usage/gpu-agents.git//deploy/terraform/azure?ref=terraform-azure-v1.0.0"

  # …required inputs — see the module's README…
}
```

See each module's own README for required inputs, outputs, and usage
examples.

## Release cadence

Each module is versioned independently — an Azure-module bug fix doesn't
require a new Docker image, and a new agent image doesn't force a module
re-release.

| Artifact | Tag prefix | Example | Triggers |
|---|---|---|---|
| Docker image | `v*` | `v1.2.3` | Dual-push to Docker Hub + GHCR |
| Helm chart | `helm-v*` | `helm-v1.0.0` | `helm push` OCI to GHCR |
| AWS module | `terraform-aws-v*` | `terraform-aws-v1.0.0` | GitHub Release with `deploy/terraform/aws/` tarball |
| Azure module | `terraform-azure-v*` | `terraform-azure-v1.0.0` | GitHub Release with `deploy/terraform/azure/` tarball |
| GCP module | `terraform-gcp-v*` | `terraform-gcp-v1.0.0` | GitHub Release with `deploy/terraform/gcp/` tarball |

The Chart.yaml `appVersion` in the Helm chart tracks the Docker image it was
tested against; the chart's own `version` evolves independently.

## Why no HashiCorp Terraform Registry?

The HashiCorp public registry requires the repo to be named
`terraform-<provider>-<name>`, which would mean splitting into three repos
(`terraform-aws-gpu-agent`, `terraform-azurerm-gpu-agent`,
`terraform-google-gpu-agent`). Not worth the sprawl for the current size of
the project. Git-tagged refs work today; we can migrate to the registry
later without breaking consumers by keeping the git tags around.

## Air-gapped deployments

Each module exposes a `use_private_acr` / `use_private_ecr` / equivalent
flag that routes the image pull through the caller's own registry. See each
module's README for the exact variable names.
