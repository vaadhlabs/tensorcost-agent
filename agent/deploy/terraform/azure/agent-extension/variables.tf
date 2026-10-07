// ─── Existing VM the agent will be installed onto ────────────────────────

variable "name" {
  description = "Name prefix — used to name the VM extension itself. Usually something like `<tenant-slug>-agent`."
  type        = string
}

variable "vm_id" {
  description = "Resource ID of the existing Linux VM to install the agent on. Must be a running VM with system-assigned managed identity."
  type        = string
}

variable "vm_principal_id" {
  description = <<-EOT
    System-assigned managed identity principal_id of the VM. Used to grant
    the `Cost Management Reader` + `Reader` roles the agent needs. Pass
    `azurerm_linux_virtual_machine.mine.identity[0].principal_id`.

    Leave empty (`""`) to skip role assignments — caller grants roles
    themselves. Useful when the VM already has broader permissions, or
    when you're running inside an enterprise where role-assignment IAM
    is centralized.
  EOT
  type        = string
  default     = ""
}

variable "location" {
  description = "Azure region of the VM (e.g. `eastus`). Azure requires this field on VM extensions even though it's always the VM's region."
  type        = string
}

variable "azure_subscription_id" {
  description = "Subscription ID — passed to the agent for its Cost Management queries AND used as the scope for the Cost Management Reader role."
  type        = string
}

// ─── Backend connection (consumed by agent env file) ──────────────────────

variable "backend_api_url" {
  description = "tensorcost dashboard API base URL, e.g. https://api.tensorcost.com."
  type        = string
}

variable "backend_api_key" {
  description = <<-EOT
    Tenant API key. Written to /etc/default/unified-gpu-agent on the VM
    with 0600 perms. Passed to the VM Extension via `protected_settings`
    so Azure encrypts it at rest and in transit.
  EOT
  type        = string
  sensitive   = true
}

variable "tenant_id" {
  description = "Tenant UUID. Set for clarity in agent logs."
  type        = string
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
    public image on Docker Hub. Override with a private registry URL for
    air-gapped tenants — typically paired with use_private_acr = true.
  EOT
  type        = string
  default     = "docker.io/tensorcost/gpu-agent:azure"
}

variable "use_private_acr" {
  description = <<-EOT
    When true, the install script calls `az acr login --name acr_login_server`
    (via managed identity) before pulling the image. When false, pulls are
    anonymous (works with docker.io / ghcr.io public images).
  EOT
  type        = bool
  default     = false
}

variable "acr_login_server" {
  description = "ACR login server (e.g. `acme.azurecr.io`). Required when use_private_acr = true."
  type        = string
  default     = ""
}

// ─── Re-install trigger ───────────────────────────────────────────────────
// Azure VM Extensions don't re-execute when only the script body changes —
// the extension resource is considered unchanged unless `force_update_tag`
// differs. Bump this to force a re-install (e.g. when rolling out a new
// agent image or changing backend creds).

variable "force_update_tag" {
  description = "Change this string to force the extension to re-run on the next `terraform apply`. Common choice: `timestamp()` for always-reinstall, or a deploy build number for explicit control."
  type        = string
  default     = "v1"
}

variable "tags" {
  description = "Tags applied to the VM Extension resource."
  type        = map(string)
  default     = {}
}

locals {
  common_tags = merge(
    var.tags,
    {
      ManagedBy = "terraform-tensorcost-gpu-agent"
      Module    = "azure/agent-extension"
    },
  )
}
