output "deployment_mode" {
  description = "Active agent VM mode: gpu or heartbeat."
  value       = var.deployment_mode
}

output "primary_vm_id" {
  description = "Resource ID of the primary GPU VM."
  value       = azurerm_linux_virtual_machine.primary.id
}

output "primary_vm_name" {
  description = "Name of the primary GPU VM."
  value       = azurerm_linux_virtual_machine.primary.name
}

output "primary_public_ip" {
  description = "Public IP of the primary GPU VM. SSH here to debug agent issues."
  value       = azurerm_public_ip.primary.ip_address
}

output "primary_private_ip" {
  description = "Private IP of the primary GPU VM."
  value       = azurerm_network_interface.primary.private_ip_address
}

output "primary_principal_id" {
  description = "System-assigned managed identity principal ID of the primary VM. Use this to grant additional roles outside the module."
  value       = azurerm_linux_virtual_machine.primary.identity[0].principal_id
}

output "spike_vm_id" {
  description = "Resource ID of the optional spike-testing VM, or null when enable_spike_vm = false."
  value       = var.enable_spike_vm ? azurerm_linux_virtual_machine.spike[0].id : null
}

output "spike_public_ip" {
  description = "Public IP of the optional spike-testing VM, or null."
  value       = var.enable_spike_vm ? azurerm_public_ip.spike[0].ip_address : null
}

output "admin_username" {
  description = "Linux admin username on the VMs."
  value       = var.admin_username
}

output "admin_password" {
  description = "Generated admin password. Retrieve with `terraform output -raw admin_password` — SSH in for cloud-init debugging."
  value       = random_password.admin.result
  sensitive   = true
}

output "cloud_init_yaml" {
  description = "Rendered cloud-init YAML — useful for `terraform console` debugging or feeding into your own VM resource."
  value       = local.cloud_init
  sensitive   = true
}
