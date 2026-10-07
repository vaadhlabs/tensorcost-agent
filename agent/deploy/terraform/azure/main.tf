/**
 * tensorcost unified-gpu-agent — Azure VM module.
 *
 * Provisions one GPU VM (optionally a second one for spike testing),
 * installs the unified-gpu-agent container via cloud-init, and grants
 * the VM's managed identity the Azure roles the agent needs.
 *
 * Scope kept deliberately narrow (single-purpose module):
 *   - Module does NOT create the resource group, VNet, subnet, or NSG.
 *     Caller owns the network layout and passes subnet_id in.
 *   - Module does NOT create the ACR. When use_private_acr = true,
 *     caller passes an existing acr_id + acr_login_server. Default is
 *     docker.io/tensorcost/gpu-agent:azure, no ACR needed.
 *   - Module does NOT create AKS, Azure ML, Azure OpenAI, or budget
 *     alerts. Those live in higher-level wrappers (e.g. the demo
 *     stack in github.com/gpu-usage/usage-app/deploy/azure).
 *
 * Minimal usage:
 *
 *   module "gpu_agent" {
 *     source = "git::https://github.com/gpu-usage/gpu-agents.git//deploy/terraform/azure?ref=terraform-azure-v1.0.0"
 *
 *     name                  = "acme-dev"
 *     resource_group_name   = azurerm_resource_group.main.name
 *     location              = "eastus"
 *     subnet_id             = azurerm_subnet.gpu.id
 *     azure_subscription_id = data.azurerm_client_config.current.subscription_id
 *
 *     backend_api_url = "https://api.tensorcost.com"
 *     backend_api_key = var.backend_api_key
 *     tenant_id       = var.tenant_id
 *   }
 */

data "azurerm_client_config" "current" {}

// ─── VM admin password (generated once, stored in state) ─────────────────
// Caller can retrieve via `terraform output -raw admin_password` to SSH in
// for debugging. For production, swap password auth for SSH keys by passing
// your own public key (future enhancement).

resource "random_password" "admin" {
  length      = 24
  special     = true
  min_upper   = 2
  min_lower   = 2
  min_numeric = 2
  min_special = 2
}

// ─── Cloud-init rendering ─────────────────────────────────────────────────
// Conditional `az acr login` + Azure CLI install only when the caller opts
// into ACR. See cloud-init.yaml.tftpl for the template.

locals {
  is_heartbeat = var.deployment_mode == "heartbeat"

  cloud_init_gpu = templatefile("${path.module}/cloud-init.yaml.tftpl", {
    backend_api_url       = var.backend_api_url
    backend_api_key       = var.backend_api_key
    tenant_id             = var.tenant_id
    agent_hostname        = var.agent_hostname
    agent_key_id          = var.agent_key_id
    agent_hmac_pepper_hex = var.agent_hmac_pepper_hex
    grpc_host             = var.grpc_host
    grpc_port             = var.grpc_port
    metrics_push_interval = var.metrics_push_interval
    azure_subscription_id = var.azure_subscription_id
    resource_group        = var.resource_group_name
    agent_image           = var.agent_image
    use_private_acr       = var.use_private_acr
    acr_login_server      = var.acr_login_server
  })

  cloud_init_heartbeat = templatefile("${path.module}/cloud-init-heartbeat.yaml.tftpl", {
    backend_api_url       = var.backend_api_url
    backend_api_key       = var.backend_api_key
    tenant_id             = var.tenant_id
    agent_hostname        = var.agent_hostname
    agent_key_id          = var.agent_key_id
    agent_hmac_pepper_hex = var.agent_hmac_pepper_hex
    grpc_host             = var.grpc_host
    grpc_port             = var.grpc_port
    metrics_push_interval = var.metrics_push_interval
    azure_subscription_id = var.azure_subscription_id
    resource_group        = var.resource_group_name
    agent_image           = var.agent_image
    use_private_acr       = var.use_private_acr
    acr_login_server      = var.acr_login_server
  })

  cloud_init = local.is_heartbeat ? local.cloud_init_heartbeat : local.cloud_init_gpu

  vm_size_effective = local.is_heartbeat ? var.heartbeat_vm_size : var.vm_size
  use_spot_effective = local.is_heartbeat ? false : var.use_spot

  # Tiny helper so VM/NIC resources read cleanly.
  spot_config = local.use_spot_effective ? {
    priority        = "Spot"
    eviction_policy = "Deallocate"
    max_bid_price   = -1
    } : {
    priority        = "Regular"
    eviction_policy = null
    max_bid_price   = null
  }

  vm_name_suffix = local.is_heartbeat ? "agent" : "gpu"
  os_disk_type   = local.is_heartbeat ? "Standard_LRS" : "Premium_LRS"
}

// ─── Primary GPU VM (T4 by default) ──────────────────────────────────────

resource "azurerm_public_ip" "primary" {
  name                = "${var.name}-gpu-pip"
  location            = var.location
  resource_group_name = var.resource_group_name
  allocation_method   = "Static"
  sku                 = "Standard"
  tags                = local.common_tags
}

resource "azurerm_network_interface" "primary" {
  name                = "${var.name}-gpu-nic"
  location            = var.location
  resource_group_name = var.resource_group_name

  ip_configuration {
    name                          = "primary"
    subnet_id                     = var.subnet_id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.primary.id
  }

  tags = local.common_tags
}

resource "azurerm_linux_virtual_machine" "primary" {
  name                = "${var.name}-${local.vm_name_suffix}"
  location            = var.location
  resource_group_name = var.resource_group_name
  size                = local.vm_size_effective

  priority        = local.spot_config.priority
  eviction_policy = local.spot_config.eviction_policy
  max_bid_price   = local.spot_config.max_bid_price

  disable_password_authentication = false
  admin_username                  = var.admin_username
  admin_password                  = random_password.admin.result

  // System-assigned managed identity — used by cloud-init for ACR login
  // (when use_private_acr) and by the agent's Azure cost monitor to query
  // Cost Management without any static credentials.
  identity {
    type = "SystemAssigned"
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = local.os_disk_type
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts-gen2"
    version   = "latest"
  }

  network_interface_ids = [azurerm_network_interface.primary.id]
  custom_data           = base64encode(local.cloud_init)
  tags                  = local.common_tags
}

resource "azurerm_dev_test_global_vm_shutdown_schedule" "primary" {
  count              = var.auto_shutdown_enabled ? 1 : 0
  virtual_machine_id = azurerm_linux_virtual_machine.primary.id
  location           = var.location
  enabled            = true

  daily_recurrence_time = var.auto_shutdown_time
  timezone              = "UTC"

  notification_settings {
    enabled = false
  }
}

// ─── Optional spike-testing VM (V100) ────────────────────────────────────

resource "azurerm_public_ip" "spike" {
  count               = var.enable_spike_vm ? 1 : 0
  name                = "${var.name}-spike-pip"
  location            = var.location
  resource_group_name = var.resource_group_name
  allocation_method   = "Static"
  sku                 = "Standard"
  tags                = local.common_tags
}

resource "azurerm_network_interface" "spike" {
  count               = var.enable_spike_vm ? 1 : 0
  name                = "${var.name}-spike-nic"
  location            = var.location
  resource_group_name = var.resource_group_name

  ip_configuration {
    name                          = "primary"
    subnet_id                     = var.subnet_id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.spike[0].id
  }

  tags = local.common_tags
}

resource "azurerm_linux_virtual_machine" "spike" {
  count               = var.enable_spike_vm ? 1 : 0
  name                = "${var.name}-spike"
  location            = var.location
  resource_group_name = var.resource_group_name
  size                = var.spike_vm_size

  priority        = local.spot_config.priority
  eviction_policy = local.spot_config.eviction_policy
  max_bid_price   = local.spot_config.max_bid_price

  disable_password_authentication = false
  admin_username                  = var.admin_username
  admin_password                  = random_password.admin.result

  identity {
    type = "SystemAssigned"
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts-gen2"
    version   = "latest"
  }

  network_interface_ids = [azurerm_network_interface.spike[0].id]
  custom_data           = base64encode(local.cloud_init)

  tags = merge(local.common_tags, { Purpose = "spike-testing" })
}

resource "azurerm_dev_test_global_vm_shutdown_schedule" "spike" {
  count              = var.enable_spike_vm && var.auto_shutdown_enabled ? 1 : 0
  virtual_machine_id = azurerm_linux_virtual_machine.spike[0].id
  location           = var.location
  enabled            = true

  daily_recurrence_time = var.auto_shutdown_time
  timezone              = "UTC"

  notification_settings {
    enabled = false
  }
}

// ─── Role assignments ─────────────────────────────────────────────────────
// - AcrPull on the caller-provided ACR, ONLY when use_private_acr = true.
// - Cost Management Reader on the subscription — lets the agent's Azure
//   cost monitor actually pull cost data. Without this the "Cost
//   Intelligence" page in the dashboard stays empty.
// - Reader on the resource group — lets the agent enumerate VMs / disks.

resource "azurerm_role_assignment" "primary_acr_pull" {
  count                = var.use_private_acr ? 1 : 0
  scope                = var.acr_id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_linux_virtual_machine.primary.identity[0].principal_id
}

resource "azurerm_role_assignment" "spike_acr_pull" {
  count                = var.enable_spike_vm && var.use_private_acr ? 1 : 0
  scope                = var.acr_id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_linux_virtual_machine.spike[0].identity[0].principal_id
}

resource "azurerm_role_assignment" "primary_cost_reader" {
  scope                = "/subscriptions/${var.azure_subscription_id}"
  role_definition_name = "Cost Management Reader"
  principal_id         = azurerm_linux_virtual_machine.primary.identity[0].principal_id
}

resource "azurerm_role_assignment" "spike_cost_reader" {
  count                = var.enable_spike_vm ? 1 : 0
  scope                = "/subscriptions/${var.azure_subscription_id}"
  role_definition_name = "Cost Management Reader"
  principal_id         = azurerm_linux_virtual_machine.spike[0].identity[0].principal_id
}

data "azurerm_resource_group" "target" {
  name = var.resource_group_name
}

resource "azurerm_role_assignment" "primary_rg_reader" {
  scope                = data.azurerm_resource_group.target.id
  role_definition_name = "Reader"
  principal_id         = azurerm_linux_virtual_machine.primary.identity[0].principal_id
}

resource "azurerm_role_assignment" "spike_rg_reader" {
  count                = var.enable_spike_vm ? 1 : 0
  scope                = data.azurerm_resource_group.target.id
  role_definition_name = "Reader"
  principal_id         = azurerm_linux_virtual_machine.spike[0].identity[0].principal_id
}
