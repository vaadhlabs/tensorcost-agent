/**
 * agent-extension — install the unified-gpu-agent onto an EXISTING Azure VM.
 *
 * Complements the top-level `deploy/terraform/azure/` module which
 * provisions a VM + installs the agent via cloud-init. When the customer
 * already has a VM they want to monitor, cloud-init is a non-starter
 * (it only runs on first boot), so we use an Azure VM Extension
 * instead — sanctioned, declarative, re-runnable.
 *
 * The install script is rendered via templatefile() and handed to the
 * extension via `protected_settings.script` (base64-encoded), which
 * means Azure encrypts the whole payload — including the backend
 * API key baked into it — at rest and in transit.
 *
 * Usage:
 *
 *   module "gpu_agent_byovm" {
 *     source = "git::https://github.com/gpu-usage/gpu-agents.git//deploy/terraform/azure/agent-extension?ref=terraform-azure-v1.0.0"
 *
 *     name                  = "acme-prod-agent"
 *     vm_id                 = azurerm_linux_virtual_machine.mine.id
 *     vm_principal_id       = azurerm_linux_virtual_machine.mine.identity[0].principal_id
 *     location              = azurerm_linux_virtual_machine.mine.location
 *     azure_subscription_id = data.azurerm_client_config.current.subscription_id
 *
 *     backend_api_url = "https://api.tensorcost.com"
 *     backend_api_key = var.backend_api_key
 *     tenant_id       = var.tenant_id
 *   }
 */

locals {
  install_script = templatefile("${path.module}/install-agent.sh.tftpl", {
    backend_api_url       = var.backend_api_url
    backend_api_key       = var.backend_api_key
    tenant_id             = var.tenant_id
    grpc_host             = var.grpc_host
    grpc_port             = var.grpc_port
    metrics_push_interval = var.metrics_push_interval
    azure_subscription_id = var.azure_subscription_id
    agent_image           = var.agent_image
    use_private_acr       = var.use_private_acr
    acr_login_server      = var.acr_login_server
  })
}

// The VM Extension itself. CustomScript v2.1 on Linux accepts a base64-
// encoded inline `script` via protected_settings, which gets decoded to
// /var/lib/waagent/custom-script/download/<seq>/script.sh and executed.
//
// Whole script body (including secrets) lives in protected_settings →
// encrypted by Azure. settings is empty on purpose — any non-secret
// field we put there would be visible to anyone with Reader on the VM.
resource "azurerm_virtual_machine_extension" "agent" {
  name                 = var.name
  virtual_machine_id   = var.vm_id
  publisher            = "Microsoft.Azure.Extensions"
  type                 = "CustomScript"
  type_handler_version = "2.1"

  # Bump var.force_update_tag to force re-run on the next apply.
  # Azure treats a changed tag as a "new" extension invocation.
  settings = jsonencode({
    timestamp = var.force_update_tag
  })

  protected_settings = jsonencode({
    script = base64encode(local.install_script)
  })

  tags = local.common_tags
}

// ─── Role assignments on the caller-provided managed identity ────────────
// Skipped entirely when vm_principal_id is empty — caller may want to
// grant roles out-of-band (e.g. via a centralized IAM process).

resource "azurerm_role_assignment" "cost_management_reader" {
  count                = var.vm_principal_id != "" ? 1 : 0
  scope                = "/subscriptions/${var.azure_subscription_id}"
  role_definition_name = "Cost Management Reader"
  principal_id         = var.vm_principal_id
}

// `Reader` at the subscription level is intentionally NOT granted — too
// broad for most enterprise policies. The agent only needs to enumerate
// resources within the scope it's told about via AZURE_RESOURCE_GROUP /
// AZURE_SUBSCRIPTION_ID env vars. Caller grants whatever narrower Reader
// scope (RG-level, management-group-level) fits their policy.
