// ─── Identity / placement ────────────────────────────────────────────────

variable "name" {
  description = "Name prefix for every resource the module creates. Usually the tenant slug or deployment id."
  type        = string
}

variable "resource_group_name" {
  description = "Existing resource group — the module does NOT create one (single-purpose module, caller owns the RG)."
  type        = string
}

variable "location" {
  description = "Azure region (e.g. `eastus`)."
  type        = string
}

variable "subnet_id" {
  description = "Existing subnet ID the GPU VM's NIC attaches to. Caller owns the VNet/subnet."
  type        = string
}

variable "azure_subscription_id" {
  description = "Subscription ID. Passed through to the agent so its Cost Management queries target the right subscription."
  type        = string
}

// ─── Backend connection (consumed by agent cloud-init) ────────────────────

variable "backend_api_url" {
  description = "tensorcost dashboard API base URL, e.g. https://api.tensorcost.com."
  type        = string
}

variable "backend_api_key" {
  description = "Tenant API key. Written to /etc/default/unified-gpu-agent with 0600 perms on the VM."
  type        = string
  sensitive   = true
}

variable "tenant_id" {
  description = "Tenant UUID. Set for clarity in agent logs; backend could derive it from the API key."
  type        = string
}

variable "agent_key_id" {
  description = "AGENT_KEY_ID from the TensorCost mint dialog. Required for Fleet ingest with HMAC auth."
  type        = string
  default     = ""
}

variable "agent_hmac_pepper_hex" {
  description = "AGENT_HMAC_PEPPER (64-char hex) from the mint dialog. Required for Fleet ingest with HMAC auth."
  type        = string
  sensitive   = true
  default     = ""
}

variable "grpc_host" {
  description = "Hostname for the gRPC streaming endpoint. Empty disables gRPC — agent falls back to HTTP."
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
// The agent ships as a container; these choose where the VM pulls it from.
// Default = CI-published public image. Air-gapped tenants flip use_private_acr
// and point at their own registry.

variable "agent_image" {
  description = <<-EOT
    Container image the VM's systemd unit pulls. Defaults to the CI-published
    public image on Docker Hub. Override with a private registry URL for
    air-gapped tenants — typically paired with use_private_acr = true.
  EOT
  type        = string
  default     = "docker.io/tensorcost/gpu-agent:azure"
}

variable "use_private_acr" {
  description = <<-EOT
    When true, the cloud-init unit calls `az acr login --name acr_login_server`
    (via system-assigned managed identity) before pulling the image. When
    false, pulls are anonymous (works with docker.io / ghcr.io public images).
  EOT
  type        = bool
  default     = false
}

variable "acr_login_server" {
  description = "ACR login server (e.g. `acme.azurecr.io`). Required when use_private_acr = true; ignored otherwise."
  type        = string
  default     = ""
}

variable "acr_id" {
  description = <<-EOT
    Resource ID of the ACR to grant AcrPull to the VM's managed identity.
    Required when use_private_acr = true. Ignored otherwise.
  EOT
  type        = string
  default     = ""
}

// ─── VM sizing ────────────────────────────────────────────────────────────

variable "vm_size" {
  description = "Azure VM size for the primary GPU VM. Default is the smallest T4 SKU."
  type        = string
  default     = "Standard_NC4as_T4_v3"
}

variable "enable_spike_vm" {
  description = "When true, provisions an additional V100 VM for spike / burst testing. Off by default — V100s are expensive even on spot."
  type        = bool
  default     = false
}

variable "spike_vm_size" {
  description = "Azure VM size for the spike-testing VM. Read only when enable_spike_vm = true."
  type        = string
  default     = "Standard_NC6s_v3"
}

variable "use_spot" {
  description = "Run VMs on Azure Spot priority. Cuts cost ~80% but Azure can evict with 30s notice when capacity is tight."
  type        = bool
  default     = true
}

variable "deployment_mode" {
  description = <<-EOT
    gpu       — NC4as_T4_v3 spot VM with NVML + Azure monitors (needs GPU + spot quota).
    heartbeat — cheap CPU-only VM (Standard_B1s, on-demand) for Connect Agents liveness
                when GPU quota is zero. Switch back to gpu after quota is approved.
  EOT
  type        = string
  default     = "gpu"

  validation {
    condition     = contains(["gpu", "heartbeat"], var.deployment_mode)
    error_message = "deployment_mode must be gpu or heartbeat."
  }
}

variable "heartbeat_vm_size" {
  description = "VM size when deployment_mode = heartbeat. D2s_v3 uses general-purpose quota (usually non-zero)."
  type        = string
  default     = "Standard_D2s_v3"
}

variable "agent_hostname" {
  description = <<-EOT
    Install identity from the TensorCost mint dialog (shown as "Install identity (AGENT_HOSTNAME)").
    Must match gpu.agent.hostname exactly. This is NOT a DNS name or OS hostname — it is the
    opaque install pin wired into AGENT_HOSTNAME on the VM.
  EOT
  type        = string
}

variable "admin_username" {
  description = "Linux admin username on the VM."
  type        = string
  default     = "azureuser"
}

// ─── Auto-shutdown ────────────────────────────────────────────────────────

variable "auto_shutdown_enabled" {
  description = "When true, register a daily shutdown schedule on each VM. Off by default — a mid-demo shutdown is a bad surprise."
  type        = bool
  default     = false
}

variable "auto_shutdown_time" {
  description = "Daily shutdown time in HH:MM, UTC. Only honored when auto_shutdown_enabled = true."
  type        = string
  default     = "06:00"
}

// ─── Tagging ──────────────────────────────────────────────────────────────

variable "tags" {
  description = "Tags applied to every resource the module creates."
  type        = map(string)
  default     = {}
}

locals {
  common_tags = merge(
    var.tags,
    {
      ManagedBy = "terraform-tensorcost-gpu-agent"
      Module    = "azure/agent"
    },
  )
}
