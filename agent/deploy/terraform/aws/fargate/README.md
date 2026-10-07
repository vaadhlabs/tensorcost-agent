# GPU Agent — Terraform Module (AWS ECS Fargate)

Deploys the tensorcost GPU Cost Optimization Agent as a Fargate service.

## Minimal example

```hcl
module "gpu_agent" {
  source = "git::https://github.com/gpu-usage/unified-gpu-agent.git//deploy/terraform/agent?ref=v2.0.0"

  name        = "gpu-agent-prod"
  cluster_arn = aws_ecs_cluster.main.arn
  vpc_id      = aws_vpc.main.id
  subnet_ids  = aws_subnet.private[*].id

  api_key   = var.tensorcost_api_key    # from Secrets Manager / SSM
  tenant_id = var.tensorcost_tenant_id

  monitors = {
    aws       = true
    sagemaker = true
  }

  tags = {
    Environment = "prod"
    ManagedBy   = "terraform"
  }
}
```

## What it creates

| Resource | Purpose |
|---|---|
| `aws_secretsmanager_secret.auth` | Stores `api_key` + `tenant_id` as JSON; injected into the task. |
| `aws_iam_role.exec` | Task execution role — pulls image, reads the secret, writes logs. |
| `aws_iam_role.task` | Task role — runtime identity for calling AWS APIs (EC2, SageMaker, ...). |
| `aws_ecs_task_definition.agent` | Fargate task (CPU/mem from `cpu`/`memory` vars). |
| `aws_ecs_service.agent` | Long-running service, default 1 replica. |
| `aws_security_group.agent` | Egress-only (no inbound). |
| `aws_cloudwatch_log_group.agent` | `/ecs/${var.name}` with 30-day retention. |

## Common configurations

### AWS + SageMaker monitoring
```hcl
monitors = {
  aws       = true
  sagemaker = true
}
```
Attaches a least-privilege policy granting `ec2:Describe*`, `cloudwatch:GetMetric*`, `ce:GetCost*`, and `sagemaker:*` read permissions to the task role.

### Cross-account monitoring
The module's task role is the default identity. For cross-account, use `extra_task_policy_arns` to attach a policy granting `sts:AssumeRole` on a trusted role in the target account.

```hcl
extra_task_policy_arns = [aws_iam_policy.assume_other_accounts.arn]
```

### Private subnets with NAT (recommended)
```hcl
subnet_ids       = aws_subnet.private[*].id
assign_public_ip = false
```
Requires NAT gateway/instance for outbound to tensorcost.

### Public subnets (dev only)
```hcl
subnet_ids       = aws_subnet.public[*].id
assign_public_ip = true
```

### OpenTelemetry tracing
```hcl
telemetry = {
  otel_enabled  = true
  otel_endpoint = "http://otel-collector.internal:4317"
}
```

## Uninstall

```bash
terraform destroy -target=module.gpu_agent
```

Secrets Manager secret has `recovery_window_in_days = 0` so it's deleted immediately rather than held for 30 days.

## Related

- Helm chart: `../../helm/gpu-agent/` — for Kubernetes deployments.
- Docker Compose: `../../../docker-compose.yml` — for local dev.
