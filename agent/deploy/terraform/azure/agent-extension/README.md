# Azure `agent-extension` — install the agent onto an existing VM

Sub-module for the "bring your own VM" case: the customer already has a
Linux VM (GPU or cost-only) and wants the agent bolted on. Pairs with the
parent [`azure/`](../README.md) module which provisions a VM from scratch.

Uses an Azure **Custom Script VM Extension** — a different install
mechanism from the parent module's cloud-init, because cloud-init only
runs on first boot.

## When to use which

| Situation | Module |
|---|---|
| Net-new deploy, no existing compute | [`azure/`](../) — provisions VM + installs agent via cloud-init |
| Existing Linux VM, want agent monitoring it | **This sub-module** — installs via VM Extension |
| Existing AKS cluster | [`../../helm/gpu-agent/`](../../helm/gpu-agent/) as DaemonSet, not Terraform |
| Existing GKE/EKS cluster | Same — use the Helm chart |

## Usage

```hcl
data "azurerm_client_config" "current" {}

resource "azurerm_linux_virtual_machine" "mine" {
  # …whatever the customer already has. Must have:
  #   identity { type = "SystemAssigned" }
  # so the agent can authenticate to Cost Management without static creds.
}

module "gpu_agent_byovm" {
  source = "git::https://github.com/gpu-usage/gpu-agents.git//deploy/terraform/azure/agent-extension?ref=terraform-azure-v1.0.0"

  name                  = "acme-prod-agent"
  vm_id                 = azurerm_linux_virtual_machine.mine.id
  vm_principal_id       = azurerm_linux_virtual_machine.mine.identity[0].principal_id
  location              = azurerm_linux_virtual_machine.mine.location
  azure_subscription_id = data.azurerm_client_config.current.subscription_id

  backend_api_url = "https://api.tensorcost.com"
  backend_api_key = var.backend_api_key
  tenant_id       = var.tenant_id
}
```

The install script is idempotent — terraform apply re-runs it whenever
`force_update_tag` changes (default: `"v1"`). Bump that value to push a
config change to the VM without touching the VM resource itself.

## Requirements on the existing VM

1. **Linux, apt-based** (Ubuntu 20.04/22.04 tested; Debian should work).
2. **Running** at the time of `terraform apply`. Azure VM Extensions fail
   fast on stopped VMs.
3. **System-assigned managed identity.** Add it with:
   ```hcl
   identity { type = "SystemAssigned" }
   ```
   Free to enable, and required for the agent's Cost Management queries.
4. **Outbound HTTPS** to Docker Hub (or your private ACR) for the image
   pull, and to `api.tensorcost.com` for metric streaming. No inbound
   ports required — the agent is a pure outbound client.
5. **NVIDIA GPU optional.** If the VM has a GPU, the script installs the
   driver + container toolkit and runs the agent with `--gpus all`. If
   not (e.g. a cost-monitoring bastion on a general-purpose SKU), the
   script detects that and runs in HTTP-only / cost-only mode.

## Secrets handling

`backend_api_key` is passed through Terraform's `sensitive = true` and
then handed to the VM Extension via `protected_settings`. Azure
encrypts the whole `protected_settings` blob at rest and never renders
it in the portal. The script itself is also in `protected_settings`
(inline, base64-encoded), so even the rendered template — which has the
key baked into the `/etc/default/unified-gpu-agent` heredoc — stays
encrypted.

## Role assignments

By default the module grants the VM's managed identity:

- **Cost Management Reader** at the subscription level — lets the
  agent's Azure cost monitor pull cost data.

It does **NOT** grant `Reader` at the subscription level (too broad for
most enterprise policies). If you want the agent to enumerate VMs /
disks / etc. across a scope, grant `Reader` yourself at the right scope
(RG, resource group hierarchy, or narrower subscription) using
[`azurerm_role_assignment`](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/resources/role_assignment)
with `principal_id = var.vm_principal_id`.

Set `vm_principal_id = ""` to skip the Cost Management Reader assignment
entirely — useful when your IAM policy doesn't let the Terraform
principal manage subscription-level role assignments.

## Inputs

See [`variables.tf`](./variables.tf) for the full list. The required
ones:

| Variable | Description |
|---|---|
| `name` | Prefix for the VM extension resource. Usually `${tenant_slug}-agent`. |
| `vm_id` | Resource ID of the existing Linux VM. |
| `location` | Azure region of the VM (required by the extension API). |
| `azure_subscription_id` | Subscription ID — used both as agent config and as the scope for `Cost Management Reader`. |
| `backend_api_url` | `https://api.tensorcost.com` for most deployments. |
| `backend_api_key` | Tenant API key (sensitive). |
| `tenant_id` | Tenant UUID. |

Optional but common: `agent_image`, `use_private_acr`, `force_update_tag`.

## Outputs

| Output | Description |
|---|---|
| `extension_id` | Resource ID of the VM Extension. `terraform taint` it to force a reinstall. |
| `install_log_path` | Path on the VM where the install script logs — useful for debugging. |
| `agent_service_name` | `unified-gpu-agent.service` — feed to `systemctl` / `journalctl`. |
| `cost_management_reader_assignment_id` | ID of the role assignment (null when `vm_principal_id = ""`). |

## Troubleshooting

```bash
# SSH to the VM, then:
sudo systemctl status unified-gpu-agent          # Is the agent running?
sudo journalctl -u unified-gpu-agent -f          # Live agent logs
sudo tail -f /var/log/tensorcost-agent-install.log  # Install script log
sudo cat /etc/default/unified-gpu-agent          # Env file (0600; run as root)
```

Extension status also shows up in the Azure portal: VM → Extensions →
`<name>` → View detailed status. A non-zero exit code from the install
script surfaces here with a stack trace from waagent.
