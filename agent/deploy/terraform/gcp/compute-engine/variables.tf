// ─── Identity / placement ─────────────────────────────────────────────────

variable "name" {
  description = "Name prefix for every resource the module creates (the VM, its service account, the secrets in Secret Manager). Usually something like `<tenant-slug>-agent`."
  type        = string

  validation {
    condition     = can(regex("^[a-z]([-a-z0-9]{0,61}[a-z0-9])?$", var.name))
    error_message = "name must be RFC1035-compliant (lowercase letters/digits/hyphens, starts with a letter, max 63 chars) — GCE resource names enforce this."
  }
}

variable "project_id" {
  description = "GCP project ID where the agent VM, service account, and Secret Manager secrets are created."
  type        = string
}

variable "region" {
  description = "GCP region for the agent VM (e.g. `us-central1`). The zone is derived from `zone` below."
  type        = string
}

variable "zone" {
  description = "GCE zone within `region` to launch the VM in (e.g. `us-central1-a`). GCE VMs are zonal, not regional."
  type        = string
}

variable "network" {
  description = "Self-link or short name of the VPC network to attach the VM to (e.g. `projects/<proj>/global/networks/default` or just `default`)."
  type        = string
  default     = "default"
}

variable "subnetwork" {
  description = "Self-link or short name of the subnetwork. Optional — when empty, GCE picks an auto-mode subnet from the network."
  type        = string
  default     = ""
}

// ─── VM shape ─────────────────────────────────────────────────────────────

variable "machine_type" {
  description = "GCE machine type (e.g. `e2-standard-2`, `n1-standard-4`). For GPU workloads pair with `gpu_type`."
  type        = string
  default     = "e2-standard-2"
}

variable "gpu_type" {
  description = <<-EOT
    OPTIONAL. Accelerator type to attach (e.g. `nvidia-tesla-t4`,
    `nvidia-l4`, `nvidia-h100-80gb`). When set, the VM gets one GPU of
    that type and `on_host_maintenance` is forced to `TERMINATE` (GCE
    requires this for accelerators). Empty (the default) means CPU-only
    — agent runs in cost-collection mode without NVML sampling.
  EOT
  type        = string
  default     = ""
}

variable "gpu_count" {
  description = "Number of GPUs to attach. Ignored when gpu_type is empty."
  type        = number
  default     = 1
}

variable "boot_image" {
  description = "Source image family or self-link for the boot disk. Default is the latest Ubuntu 22.04 LTS image."
  type        = string
  default     = "ubuntu-os-cloud/ubuntu-2204-lts"
}

variable "boot_disk_size_gb" {
  description = "Boot disk size in GB. 50 GB is enough for the agent + Docker + NVIDIA driver; bump for GPU workloads that need scratch."
  type        = number
  default     = 50
}

// ─── Backend connection (consumed by the agent env file) ─────────────────

variable "backend_api_url" {
  description = "TensorCost dashboard API base URL, e.g. https://api.tensorcost.com."
  type        = string
}

variable "agent_api_key" {
  description = <<-EOT
    Tenant API key. Written to GCP Secret Manager as a secret version,
    NOT baked into the VM startup-script metadata. The agent's service
    account is granted `roles/secretmanager.secretAccessor` scoped to
    JUST this secret (not the project), and the startup-script reads it
    via `gcloud secrets versions access` at run time.

    This keeps the secret out of:
      * VM metadata (visible to anyone with `compute.instances.get`),
      * Terraform plan/apply output (still sensitive in state, but
        encrypted at rest if you use a remote backend with CMEK).
  EOT
  type        = string
  sensitive   = true
}

variable "tenant_id" {
  description = "Tenant UUID. Stored alongside agent_api_key in Secret Manager (kept symmetric — also lets ops rotate without touching VM metadata)."
  type        = string
}

variable "agent_hostname" {
  description = "Hostname the agent reports as. Empty = use the VM's instance name."
  type        = string
  default     = ""
}

variable "agent_key_id" {
  description = "Agent HMAC key id from mint (AGENT_KEY_ID). Required when grpc_host is set."
  type        = string
  sensitive   = true
  default     = ""
}

variable "agent_hmac_pepper" {
  description = "Agent HMAC pepper hex from mint (AGENT_HMAC_PEPPER). Required when grpc_host is set."
  type        = string
  sensitive   = true
  default     = ""
}

variable "grpc_host" {
  description = "Hostname for the gRPC streaming endpoint. Empty disables gRPC — agent falls back to HTTP only."
  type        = string
  default     = ""
}

variable "grpc_port" {
  description = "Port for the gRPC streaming endpoint."
  type        = number
  default     = 50051
}

variable "metrics_push_interval" {
  description = "Seconds between metric pushes to the backend."
  type        = number
  default     = 60
}

// ─── Agent image + registry ───────────────────────────────────────────────

variable "agent_image" {
  description = <<-EOT
    Container image the systemd unit pulls. Defaults to the CI-published
    public image. Override for a specific variant/version (e.g.
    `tensorcost/gpu-agent:gcp-1.2.3`) or for a private Artifact Registry
    pull (pair with `use_private_artifact_registry = true`).

    NOTE: a tag like `:gcp` or `:latest` is mutable — set
    `agent_image_digest` for production fleets to pin a content-addressed
    digest verified against the cosign signature.
  EOT
  type        = string
  default     = "docker.io/tensorcost/gpu-agent:gcp"
}

variable "agent_image_digest" {
  description = <<-EOT
    OPTIONAL but STRONGLY RECOMMENDED for production. A
    `sha256:<64-hex>` content digest of the image to pin. When set, the
    startup-script pulls `<agent_image without tag>@<digest>` so a
    compromised registry credential cannot retag a malicious image onto
    your fleet. See parallel commit ead77a7 (cosign image signing +
    SBOM) for how the digest is published.

    Verify and capture before each release:

        cosign verify docker.io/tensorcost/gpu-agent:gcp-1.2.3 \
          --certificate-identity-regexp '^https://github\.com/.+/gpu-agents/' \
          --certificate-oidc-issuer https://token.actions.githubusercontent.com

    then read the digest from `cosign verify`'s output (the
    `critical.image.docker-manifest-digest` field) and set it here.
    Empty (the default) preserves the legacy tag-pull behaviour for
    backwards compatibility, but logs a warning at install time.
  EOT
  type        = string
  default     = ""

  validation {
    condition     = var.agent_image_digest == "" || can(regex("^sha256:[0-9a-f]{64}$", var.agent_image_digest))
    error_message = "agent_image_digest must be empty or of the form sha256:<64-hex-chars>."
  }
}

variable "use_private_artifact_registry" {
  description = "When true, the startup-script runs `gcloud auth configure-docker <host>` (using the VM's Workload Identity, NOT a service-account key) before pulling. Pair with `artifact_registry_host`."
  type        = bool
  default     = false
}

variable "artifact_registry_host" {
  description = "Artifact Registry host, e.g. `us-central1-docker.pkg.dev`. Required when use_private_artifact_registry = true."
  type        = string
  default     = ""
}

// ─── IAM ───────────────────────────────────────────────────────────────────

variable "extra_iam_roles" {
  description = <<-EOT
    Additional GCP roles to grant to the agent's service account at
    project scope, on top of the module's defaults
    (`roles/monitoring.viewer`, `roles/billing.viewer` if a billing
    account is provided). Use this for tenants that store cost data in
    BigQuery (`roles/bigquery.dataViewer` on the dataset) etc.

    Format: list of role strings, e.g.
        ["roles/compute.viewer", "roles/storage.objectViewer"]
  EOT
  type        = list(string)
  default     = []
}

variable "billing_account_id" {
  description = "OPTIONAL. Billing account ID (e.g. `01ABCD-234567-EFGH89`). When set, the agent SA gets `roles/billing.viewer` on it for cost-export queries."
  type        = string
  default     = ""
}

variable "labels" {
  description = "Labels applied to the VM, service account, and secrets the module creates. GCP labels are lowercase-only; the module enforces a `managed-by` label on top."
  type        = map(string)
  default     = {}
}

locals {
  common_labels = merge(
    var.labels,
    {
      managed-by = "terraform-tensorcost-gpu-agent"
      module     = "gcp-compute-engine"
    },
  )

  resolved_agent_hostname = var.agent_hostname != "" ? var.agent_hostname : var.name
}
