output "ssm_document_name" {
  description = "Name of the SSM Document that wraps the install script. Trigger a manual re-run with `aws ssm send-command --document-name <this> --instance-ids <id>`."
  value       = aws_ssm_document.install_agent.name
}

output "ssm_association_id" {
  description = "Resource ID of the SSM Association. `aws ssm describe-association-execution-target` against this for last-run status."
  value       = aws_ssm_association.install_agent.association_id
}

output "ssm_parameter_backend_api_key_arn" {
  description = "ARN of the SecureString parameter holding the backend API key. The instance role has Read access; no one else does by default."
  value       = aws_ssm_parameter.backend_api_key.arn
}

output "ssm_parameter_tenant_id_arn" {
  description = "ARN of the SecureString parameter holding the tenant UUID."
  value       = aws_ssm_parameter.tenant_id.arn
}

output "agent_runtime_policy_arn" {
  description = "ARN of the IAM policy granting the agent its runtime permissions. null when instance_iam_role_name was empty (caller manages IAM)."
  value       = length(aws_iam_policy.agent_runtime) > 0 ? aws_iam_policy.agent_runtime[0].arn : null
}

output "install_log_path" {
  description = "Path on the instance where the install script writes its log. SSM also captures stdout in its own run history."
  value       = "/var/log/tensorcost-agent-install.log"
}

output "agent_service_name" {
  description = "systemd unit name the agent runs under. `systemctl status` / `journalctl -u` accept it directly."
  value       = "unified-gpu-agent.service"
}
