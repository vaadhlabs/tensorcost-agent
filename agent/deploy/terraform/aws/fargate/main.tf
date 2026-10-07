/**
 * tensorcost GPU Agent — AWS ECS Fargate module.
 *
 * Deploys the agent as a long-running Fargate service. Credentials are
 * stored in Secrets Manager and injected as container env vars. Cloud
 * monitoring uses the task role (no static access keys).
 *
 * Usage:
 *
 *   module "gpu_agent" {
 *     source = "./deploy/terraform/agent"
 *
 *     name          = "gpu-agent-prod"
 *     cluster_arn   = module.ecs.cluster_arn
 *     subnet_ids    = module.vpc.private_subnet_ids
 *     api_key       = var.tensorcost_api_key
 *     tenant_id     = var.tensorcost_tenant_id
 *     backend_api_url = "https://api.tensorcost.com"
 *     grpc_target     = "grpc.tensorcost.com:443"
 *
 *     monitors = {
 *       aws       = true
 *       sagemaker = true
 *     }
 *   }
 */

terraform {
  required_version = ">= 1.3.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

# ──────────────────────────────────────────────────────────────────────
# Secrets
# ──────────────────────────────────────────────────────────────────────
resource "aws_secretsmanager_secret" "auth" {
  name                    = "${var.name}-auth"
  description             = "tensorcost GPU agent API key and tenant ID"
  recovery_window_in_days = 0
  tags                    = var.tags
}

resource "aws_secretsmanager_secret_version" "auth" {
  secret_id = aws_secretsmanager_secret.auth.id
  secret_string = jsonencode({
    api_key   = var.api_key
    tenant_id = var.tenant_id
  })
}

# ──────────────────────────────────────────────────────────────────────
# IAM
# ──────────────────────────────────────────────────────────────────────
data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Task execution role — pulls image, reads Secrets Manager, writes CloudWatch Logs
resource "aws_iam_role" "exec" {
  name               = "${var.name}-exec"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
  tags               = var.tags
}

resource "aws_iam_role_policy_attachment" "exec_base" {
  role       = aws_iam_role.exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "exec_secrets" {
  name = "${var.name}-exec-secrets"
  role = aws_iam_role.exec.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = [aws_secretsmanager_secret.auth.arn]
    }]
  })
}

# Task role — what the AGENT process uses at runtime to read cloud metrics
resource "aws_iam_role" "task" {
  name               = "${var.name}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
  tags               = var.tags
}

# Cloud-monitor permissions — attached conditionally based on var.monitors
resource "aws_iam_role_policy" "aws_readonly" {
  count = var.monitors.aws ? 1 : 0
  name  = "${var.name}-aws-readonly"
  role  = aws_iam_role.task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceStatus",
        "ec2:DescribeInstanceTypes",
        "ec2:DescribeTags",
        "cloudwatch:GetMetricData",
        "cloudwatch:GetMetricStatistics",
        "cloudwatch:ListMetrics",
        "ce:GetCostAndUsage",
        "ce:GetCostForecast",
      ]
      Resource = "*"
    }]
  })
}

resource "aws_iam_role_policy" "sagemaker_readonly" {
  count = var.monitors.sagemaker ? 1 : 0
  name  = "${var.name}-sagemaker-readonly"
  role  = aws_iam_role.task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "sagemaker:ListTrainingJobs",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:ListEndpoints",
        "sagemaker:DescribeEndpoint",
        "sagemaker:ListEndpointConfigs",
      ]
      Resource = "*"
    }]
  })
}

# Additional policies (e.g. customer-managed spot interruption handling)
resource "aws_iam_role_policy_attachment" "extra" {
  for_each   = toset(var.extra_task_policy_arns)
  role       = aws_iam_role.task.name
  policy_arn = each.value
}

# ──────────────────────────────────────────────────────────────────────
# CloudWatch Logs
# ──────────────────────────────────────────────────────────────────────
resource "aws_cloudwatch_log_group" "agent" {
  name              = "/ecs/${var.name}"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}

# ──────────────────────────────────────────────────────────────────────
# Security Group — egress only
# ──────────────────────────────────────────────────────────────────────
resource "aws_security_group" "agent" {
  name        = "${var.name}-sg"
  description = "GPU agent — egress to tensorcost backend"
  vpc_id      = var.vpc_id
  tags        = var.tags

  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
}

# ──────────────────────────────────────────────────────────────────────
# ECS Task Definition + Service
# ──────────────────────────────────────────────────────────────────────
locals {
  env = concat(
    [
      { name = "BACKEND_API_URL", value = var.backend_api_url },
      { name = "GRPC_TARGET", value = var.grpc_target },
      { name = "GRPC_USE_TLS", value = tostring(var.grpc_use_tls) },
      { name = "COMM_MODE", value = var.comm_mode },
      { name = "MONITORING_INTERVAL", value = tostring(var.monitoring_interval_seconds) },
      { name = "LOG_LEVEL", value = var.log_level },
      { name = "AGENT_ID", value = var.name },
      { name = "AWS_ENABLED", value = tostring(var.monitors.aws) },
      { name = "SAGEMAKER_ENABLED", value = tostring(var.monitors.sagemaker) },
      { name = "AZURE_ENABLED", value = tostring(var.monitors.azure) },
      { name = "GCP_ENABLED", value = tostring(var.monitors.gcp) },
      { name = "K8S_ENABLED", value = tostring(var.monitors.kubernetes) },
      { name = "NVML_ENABLED", value = tostring(var.features.nvml) },
      { name = "AI_SPEND_ENABLED", value = tostring(var.features.ai_spend) },
      { name = "SPOT_HANDLER_ENABLED", value = tostring(var.features.spot_handler) },
      { name = "OTEL_ENABLED", value = tostring(var.telemetry.otel_enabled) },
    ],
    var.monitors.aws ? [
      { name = "AWS_REGION", value = var.aws_region },
      { name = "AWS_SERVICES", value = var.aws_services },
    ] : [],
    var.telemetry.otel_enabled && var.telemetry.otel_endpoint != "" ? [
      { name = "OTEL_EXPORTER_OTLP_ENDPOINT", value = var.telemetry.otel_endpoint },
      { name = "OTEL_SERVICE_NAME", value = var.name },
    ] : [],
    var.extra_environment,
  )
}

resource "aws_ecs_task_definition" "agent" {
  family                   = var.name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.exec.arn
  task_role_arn            = aws_iam_role.task.arn
  tags                     = var.tags

  container_definitions = jsonencode([{
    name      = "agent"
    image     = "${var.image_repository}:${var.image_tag}"
    essential = true
    environment = local.env
    secrets = [
      {
        name      = "BACKEND_API_KEY"
        valueFrom = "${aws_secretsmanager_secret.auth.arn}:api_key::"
      },
      {
        name      = "TENANT_ID"
        valueFrom = "${aws_secretsmanager_secret.auth.arn}:tenant_id::"
      },
    ]
    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]
    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval    = 30
      timeout     = 10
      retries     = 3
      startPeriod = 30
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.agent.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "agent"
      }
    }
  }])
}

resource "aws_ecs_service" "agent" {
  name            = var.name
  cluster         = var.cluster_arn
  task_definition = aws_ecs_task_definition.agent.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"
  tags            = var.tags

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [aws_security_group.agent.id]
    assign_public_ip = var.assign_public_ip
  }

  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  lifecycle {
    # Avoid churning the service when the task def revision changes during
    # CI deploys — rely on `aws ecs update-service --force-new-deployment`.
    ignore_changes = [task_definition]
  }
}
