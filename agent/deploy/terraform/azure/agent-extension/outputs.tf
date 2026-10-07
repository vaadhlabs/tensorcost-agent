output "extension_id" {
  description = "Resource ID of the VM Extension — use this to force a re-run by tainting."
  value       = azurerm_virtual_machine_extension.agent.id
}

output "install_log_path" {
  description = "Path on the VM where the install script writes its log. Tail with `journalctl -u unified-gpu-agent` OR `sudo tail -f /var/log/tensorcost-agent-install.log`."
  value       = "/var/log/tensorcost-agent-install.log"
}

output "agent_service_name" {
  description = "systemd unit name the agent runs under. `systemctl status` / `journalctl -u` accept this directly."
  value       = "unified-gpu-agent.service"
}

output "cost_management_reader_assignment_id" {
  description = "Role-assignment ID for Cost Management Reader on the subscription (null when vm_principal_id was not provided)."
  value       = length(azurerm_role_assignment.cost_management_reader) > 0 ? azurerm_role_assignment.cost_management_reader[0].id : null
}
