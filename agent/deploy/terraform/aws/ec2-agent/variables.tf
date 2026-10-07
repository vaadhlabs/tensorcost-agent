// ─── Existing EC2 instance the agent installs onto ───────────────────────

variable "name" {
  description = "Name prefix for every resource the module creates (SSM document, parameters, etc). Usually something like `<tenant-slug>-agent`."
  type        = string
}

variable "instance_id" {
  description = "ID of the existing EC2 instance to install the agent on. Must have an SSM-managed instance profile (AmazonSSMManagedInstanceCore) and be running."
  type        = string
}

variable "instance_iam_role_name" {
  description = <<-EOT
    Name (NOT ARN) of the IAM role attached to the EC2 instance. Used to
    attach the policies the agent needs:
      * ssm:GetParameter on the SecureString params we create
      * ce:GetCostAndUsage, ec2:Describe* for cost-collection
    Leave empty to skip policy attachment (you grant IAM yourself).
  EOT
  type        = string
  default     = ""
}

variable "region" {
  description = "AWS region of the EC2 instance. Required because SSM parameters are region-scoped."
  type        = string
}

// ─── Backend connection (consumed by agent env file) ──────────────────────

variable "backend_api_url" {
  description = "tensorcost dashboard API base URL, e.g. https://api.tensorcost.com."
  type        = string
}

variable "backend_api_key" {
  description = <<-EOT
    Tenant API key. Written to SSM Parameter Store as a SecureString
    (encrypted with the AWS-managed `alias/aws/ssm` KMS key) and
    fetched by the install script at run time. Never transits the
    SSM Association's parameters field — customers with Read-only
    SSM access to the document can't see the secret.
  EOT
  type        = string
  sensitive   = true
}

variable "tenant_id" {
  description = "Tenant UUID. Also stored as a SecureString (not strictly secret but kept symmetric with backend_api_key)."
  type        = string
}

variable "agent_hostname" {
  description = "Install identity from mint (AGENT_HOSTNAME) — pins heartbeats to gpu.agent row. Empty uses var.name."
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

// ─── Agent image + registry ───────────────────────────────────────────────

variable "agent_image" {
  description = <<-EOT
    Container image the systemd unit pulls. Defaults to the CI-published
    public image on Docker Hub. Override for a specific variant/version
    (`tensorcost/gpu-agent:aws-1.2.3`) or for a private ECR pull.

    NOTE: a tag like `:aws` or `:latest` is mutable — a compromised
    upstream maintainer can replace the image bytes a tag points at, and
    the SSM Association (default `rate(6 hours)`) would `docker pull` the
    replacement within hours. For production use, set `agent_image_digest`
    to a `sha256:...` value verified against the cosign signature; the
    install script will then pull `image@digest` instead of `image:tag`.
  EOT
  type        = string
  default     = "docker.io/tensorcost/gpu-agent:aws"
}

variable "agent_image_digest" {
  description = <<-EOT
    OPTIONAL but STRONGLY RECOMMENDED for production. A
    `sha256:<64-hex>` content digest of the image you want pinned. When
    set, the install script pulls `<agent_image without tag>@<digest>`
    so a compromised registry credential cannot retag a malicious image
    onto your fleet — the digest is content-addressed.

    Bump this on each release after verifying the cosign signature, e.g.

        cosign verify docker.io/tensorcost/gpu-agent:aws-1.2.3 \
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

variable "use_private_ecr" {
  description = "When true, the install script runs `aws ecr get-login-password | docker login` before pulling. Pair with `ecr_registry` pointing at the account/region ECR host."
  type        = bool
  default     = false
}

variable "ecr_registry" {
  description = "ECR registry host, e.g. `123456789.dkr.ecr.us-east-1.amazonaws.com`. Required when use_private_ecr = true."
  type        = string
  default     = ""
}

// ─── SSM Association cadence ──────────────────────────────────────────────

variable "association_schedule" {
  description = <<-EOT
    Rate expression for how often SSM re-runs the install. `rate(6 hours)`
    gives you drift detection + automatic restart if the service dies
    silently. Empty means run-once at Terraform apply (no reconciliation).
  EOT
  type        = string
  default     = "rate(6 hours)"
}

variable "tags" {
  description = "Tags applied to resources the module creates (SSM parameters, document, association)."
  type        = map(string)
  default     = {}
}

locals {
  common_tags = merge(
    var.tags,
    {
      ManagedBy = "terraform-tensorcost-gpu-agent"
      Module    = "aws/ec2-agent"
    },
  )

  # Fully-qualified SSM param names. We put them under `/tensorcost/<name>/…`
  # so customers can see at a glance what manages them, and so IAM grants
  # can scope to `/tensorcost/*` cleanly.
  ssm_param_prefix          = "/tensorcost/${var.name}"
  ssm_param_backend_api_key = "${local.ssm_param_prefix}/backend_api_key"
  ssm_param_tenant_id       = "${local.ssm_param_prefix}/tenant_id"
  ssm_param_agent_key_id    = "${local.ssm_param_prefix}/agent_key_id"
  ssm_param_agent_hmac      = "${local.ssm_param_prefix}/agent_hmac_pepper"
  resolved_agent_hostname   = var.agent_hostname != "" ? var.agent_hostname : var.name
}
