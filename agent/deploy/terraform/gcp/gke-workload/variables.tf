// ─── Identity / placement ─────────────────────────────────────────────────

variable "name" {
  description = "Name prefix for the K8s SA, GCP SA, Deployment, and Secret Manager secrets. Usually `<tenant-slug>-agent`."
  type        = string

  validation {
    condition     = can(regex("^[a-z]([-a-z0-9]{0,61}[a-z0-9])?$", var.name))
    error_message = "name must be RFC1035-compliant (lowercase letters/digits/hyphens, starts with a letter, max 63 chars)."
  }
}

variable "project_id" {
  description = "GCP project ID — also the project the GKE cluster lives in (cross-project Workload Identity is supported but not by this module)."
  type        = string
}

variable "cluster_name" {
  description = "Name of the existing GKE cluster the agent runs on. The cluster MUST have Workload Identity enabled (`--workload-pool=<project>.svc.id.goog`)."
  type        = string
}

variable "cluster_location" {
  description = "Region (regional cluster) or zone (zonal cluster) of `cluster_name`. Used to look the cluster up and to construct the workload-pool identifier."
  type        = string
}

variable "namespace" {
  description = "Kubernetes namespace the Deployment + ServiceAccount land in. The module creates the namespace if it doesn't exist."
  type        = string
  default     = "tensorcost"
}

// ─── Backend connection (consumed by the agent env file) ─────────────────

variable "backend_api_url" {
  description = "TensorCost dashboard API base URL, e.g. https://api.tensorcost.com."
  type        = string
}

variable "agent_api_key" {
  description = <<-EOT
    Tenant API key. Stored in GCP Secret Manager, NOT in a Kubernetes
    Secret. The agent's pod uses Workload Identity to fetch it via the
    Secret Manager API at startup — same trust model as the
    compute-engine sub-module. K8s Secrets at rest are base64, not
    encrypted by default; Secret Manager is encrypted with Google-
    managed (or CMEK if configured) keys and audit-logged.
  EOT
  type        = string
  sensitive   = true
}

variable "tenant_id" {
  description = "Tenant UUID. Stored alongside agent_api_key in Secret Manager."
  type        = string
}

variable "agent_hostname" {
  description = "Hostname the agent reports as. Empty = use the pod's name (downward API)."
  type        = string
  default     = ""
}

variable "grpc_host" {
  description = "Hostname for the gRPC streaming endpoint. Empty disables gRPC."
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

// ─── Agent image ──────────────────────────────────────────────────────────

variable "agent_image" {
  description = <<-EOT
    Container image the Deployment runs. Defaults to the CI-published
    public image. For private Artifact Registry images, GKE nodes auto-
    authenticate when the node SA has `roles/artifactregistry.reader`
    — no imagePullSecrets needed.
  EOT
  type        = string
  default     = "docker.io/tensorcost/gpu-agent:gcp"
}

variable "agent_image_digest" {
  description = <<-EOT
    OPTIONAL but STRONGLY RECOMMENDED for production. A
    `sha256:<64-hex>` content digest. When set, the Deployment runs
    `<image>@<digest>` instead of `<image>:<tag>`. See parallel commit
    ead77a7 (cosign image signing + SLSA provenance) for how the digest
    is published, and the AWS module's variables.tf for the same regex-
    validated pattern.
  EOT
  type        = string
  default     = ""

  validation {
    condition     = var.agent_image_digest == "" || can(regex("^sha256:[0-9a-f]{64}$", var.agent_image_digest))
    error_message = "agent_image_digest must be empty or of the form sha256:<64-hex-chars>."
  }
}

// ─── DaemonSet vs single Deployment ───────────────────────────────────────

variable "deploy_as" {
  description = <<-EOT
    `daemonset` or `deployment`. Use `daemonset` for tenants who want
    NVML sampling on every GPU node in the cluster. Use `deployment`
    (the default) for cost-collection-only.
  EOT
  type        = string
  default     = "deployment"

  validation {
    condition     = contains(["deployment", "daemonset"], var.deploy_as)
    error_message = "deploy_as must be either 'deployment' or 'daemonset'."
  }
}

variable "node_selector" {
  description = "Optional nodeSelector for the pod (e.g. `{ \"cloud.google.com/gke-accelerator\" = \"nvidia-tesla-t4\" }` to target GPU node pools when running as a DaemonSet)."
  type        = map(string)
  default     = {}
}

variable "tolerations" {
  description = "Optional tolerations — typically needed when the GPU node pool has `nvidia.com/gpu` taints. List of objects with `key`, `operator`, `value`, `effect`."
  type = list(object({
    key      = optional(string)
    operator = optional(string)
    value    = optional(string)
    effect   = optional(string)
  }))
  default = []
}

variable "resources" {
  description = "Pod resource requests/limits. Defaults are sized for cost-collection workloads — bump for DaemonSet GPU sampling at high cardinality."
  type = object({
    requests = optional(map(string), { cpu = "100m", memory = "128Mi" })
    limits   = optional(map(string), { cpu = "500m", memory = "512Mi" })
  })
  default = {}
}

// ─── IAM ───────────────────────────────────────────────────────────────────

variable "extra_iam_roles" {
  description = "Additional GCP project-level roles to grant the GCP service account. Defaults are `monitoring.viewer` + scoped `secretmanager.secretAccessor`."
  type        = list(string)
  default     = []
}

variable "billing_account_id" {
  description = "OPTIONAL. Billing account ID. When set, the GCP SA gets `roles/billing.viewer` on it."
  type        = string
  default     = ""
}

variable "labels" {
  description = "Labels applied to the GCP service account, secrets, and the K8s objects this module creates."
  type        = map(string)
  default     = {}
}

locals {
  common_labels = merge(
    var.labels,
    {
      managed-by = "terraform-tensorcost-gpu-agent"
      module     = "gcp-gke-workload"
    },
  )

  workload_pool       = "${var.project_id}.svc.id.goog"
  k8s_sa_name         = "${var.name}-sa"
  gcp_sa_account_id   = "${var.name}-gke-sa"
  resolved_agent_host = var.agent_hostname != "" ? var.agent_hostname : "$(POD_NAME)"
}
