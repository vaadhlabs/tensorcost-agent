# Azure Terraform module

Two install paths, pick whichever matches where the customer is:

| Customer situation | Path |
|---|---|
| Net-new Azure deploy — no existing compute | [**This module (root)**](./) — provisions a GPU VM + installs the agent via cloud-init |
| Customer already has a Linux VM | [**`agent-extension/`** sub-module](./agent-extension/) — bolts the agent onto an existing VM via an Azure VM Extension |

Both paths end up with the same systemd-managed agent container running on
the host. The root module is for greenfield; the sub-module is for BYO-VM.

---

This README covers the root (greenfield) module. For BYO-VM see
[`agent-extension/README.md`](./agent-extension/README.md).

## What this module provisions

A GPU VM on Azure with the tensorcost `unified-gpu-agent` installed via
cloud-init. Single-purpose: caller owns the resource group, VNet, and
subnet; the module owns the VM, its managed identity, and the role
assignments the agent needs.

## Usage

```hcl
module "gpu_agent" {
  source = "git::https://github.com/gpu-usage/gpu-agents.git//deploy/terraform/azure?ref=terraform-azure-v1.0.0"

  name                  = "acme-dev"
  resource_group_name   = azurerm_resource_group.main.name
  location              = "eastus"
  subnet_id             = azurerm_subnet.gpu.id
  azure_subscription_id = data.azurerm_client_config.current.subscription_id

  # From https://app.tensorcost.com → Integrations → API keys
  backend_api_url = "https://api.tensorcost.com"
  backend_api_key = var.backend_api_key
  tenant_id       = var.tenant_id
}
```

One `terraform apply` brings up a Standard_NC4as_T4_v3 VM on Spot pricing,
pulls `docker.io/tensorcost/gpu-agent:azure` (CI-published, public), and
starts streaming metrics + cost data to your dashboard within ~2 minutes of
`systemctl status unified-gpu-agent` going active.

## Private registry / air-gapped deployments

Set `use_private_acr = true` plus `acr_id` + `acr_login_server`:

```hcl
module "gpu_agent" {
  source = "…"

  # …required inputs as above…

  use_private_acr   = true
  acr_login_server  = azurerm_container_registry.main.login_server
  acr_id            = azurerm_container_registry.main.id
  agent_image       = "${azurerm_container_registry.main.login_server}/gpu-agent:latest"
}
```

The VM's managed identity gets the `AcrPull` role automatically. You'll need
to `docker buildx build --push` the image to your ACR before the VM's
systemd unit can pull it.

## What the module creates

- 1 × GPU VM (`azurerm_linux_virtual_machine`) with cloud-init and managed
  identity.
- 1 × network interface (`azurerm_network_interface`) + 1 × public IP
  (`azurerm_public_ip`) attached to the caller's subnet.
- Optional second VM (`enable_spike_vm = true`) for V100 spike / burst
  testing — off by default.
- Role assignments:
  - `Cost Management Reader` on the subscription (for the agent's Azure cost
    monitor).
  - `Reader` on the resource group (for VM / disk enumeration).
  - `AcrPull` on the caller's ACR when `use_private_acr = true`.
- Optional daily shutdown schedule (`auto_shutdown_enabled = true`).

## What the module does NOT create

- Resource group, VNet, subnet, NSG — caller's responsibility.
- ACR — caller passes `acr_id` if they want private pulls.
- AKS, Azure ML, Azure OpenAI, budget alerts — these live in higher-level
  wrappers (e.g. the demo stack in `usage-app/deploy/azure`).

## Inputs

See [`variables.tf`](./variables.tf) for the full list. Required:

| Variable | Description |
|---|---|
| `name` | Prefix for every resource the module creates. |
| `resource_group_name` | Existing resource group. |
| `location` | Azure region, e.g. `eastus`. |
| `subnet_id` | Existing subnet the VM's NIC attaches to. |
| `azure_subscription_id` | Subscription ID — passed to the agent. |
| `backend_api_url` | tensorcost dashboard API base URL. |
| `backend_api_key` | Tenant API key (sensitive). |
| `tenant_id` | Tenant UUID. |

## Outputs

| Output | Description |
|---|---|
| `primary_vm_id` / `primary_vm_name` | Identifiers for the primary GPU VM. |
| `primary_public_ip` / `primary_private_ip` | Network addresses. |
| `primary_principal_id` | Managed identity principal ID (for granting additional roles outside the module). |
| `spike_vm_id` / `spike_public_ip` | Set when `enable_spike_vm = true`, null otherwise. |
| `admin_username` / `admin_password` | SSH credentials — `admin_password` is sensitive. |
| `cloud_init_yaml` | Rendered cloud-init (sensitive — contains backend key). |

## Versioning

Published on `terraform-azure-v*` git tags. See
[`deploy/terraform/README.md`](../README.md) for the full release matrix.
