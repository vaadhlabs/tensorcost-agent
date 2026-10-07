output "service_arn" {
  description = "ARN of the ECS service running the agent."
  value       = aws_ecs_service.agent.id
}

output "service_name" {
  description = "ECS service name."
  value       = aws_ecs_service.agent.name
}

output "task_definition_arn" {
  description = "ARN of the task definition."
  value       = aws_ecs_task_definition.agent.arn
}

output "task_role_arn" {
  description = "ARN of the task role. Attach extra policies here if you need more cloud permissions."
  value       = aws_iam_role.task.arn
}

output "auth_secret_arn" {
  description = "Secrets Manager ARN holding the agent's api_key and tenant_id."
  value       = aws_secretsmanager_secret.auth.arn
}

output "log_group_name" {
  description = "CloudWatch log group for agent logs."
  value       = aws_cloudwatch_log_group.agent.name
}

output "security_group_id" {
  description = "Security group ID assigned to the agent tasks."
  value       = aws_security_group.agent.id
}
