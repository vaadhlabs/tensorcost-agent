# ──────────────────────────────────────────────────────────────────────
# Required
# ──────────────────────────────────────────────────────────────────────

variable "name" {
  type        = string
  description = "Unique name for this agent deployment (used as prefix for all resources)."
}

variable "cluster_arn" {
  type        = string
  description = "ARN of an existing ECS cluster to deploy into."
}

variable "vpc_id" {
  type        = string
  description = "VPC that hosts the ECS cluster."
}

variable "subnet_ids" {
  type        = list(string)
  description = "Subnets for the Fargate tasks. Use private subnets with a NAT for production."
}

variable "api_key" {
  type        = string
  description = "tensorcost tenant API key (64-char hex)."
  sensitive   = true
}

variable "tenant_id" {
  type        = string
  description = "tensorcost tenant UUID."
}

# ──────────────────────────────────────────────────────────────────────
# Backend
# ──────────────────────────────────────────────────────────────────────

variable "backend_api_url" {
  type        = string
  description = "Backend HTTPS URL."
  default     = "https://api.tensorcost.com"
}

variable "grpc_target" {
  type        = string
  description = "Backend gRPC endpoint (host:port)."
  default     = "grpc.tensorcost.com:443"
}

variable "grpc_use_tls" {
  type        = bool
  default     = true
}

variable "comm_mode" {
  type        = string
  description = "Communication mode: grpc | http | both."
  default     = "grpc"
  validation {
    condition     = contains(["grpc", "http", "both"], var.comm_mode)
    error_message = "comm_mode must be grpc, http, or both."
  }
}

# ──────────────────────────────────────────────────────────────────────
# Image
# ──────────────────────────────────────────────────────────────────────

variable "image_repository" {
  type        = string
  default     = "tensorcost/gpu-agent"
}

variable "image_tag" {
  type        = string
  description = "Image tag. Variants: latest (full), aws, gcp, azure, node."
  default     = "latest"
}

# ──────────────────────────────────────────────────────────────────────
# Task sizing
# ──────────────────────────────────────────────────────────────────────

variable "cpu" {
  type        = string
  description = "Fargate CPU units (256, 512, 1024, 2048, 4096)."
  default     = "256"
}

variable "memory" {
  type        = string
  description = "Fargate memory in MiB."
  default     = "512"
}

variable "desired_count" {
  type        = number
  default     = 1
}

variable "assign_public_ip" {
  type        = bool
  description = "Required for tasks in public subnets. Use false for private subnets with NAT."
  default     = false
}

# ──────────────────────────────────────────────────────────────────────
# Monitoring
# ──────────────────────────────────────────────────────────────────────

variable "aws_region" {
  type        = string
  default     = "us-east-1"
}

variable "aws_services" {
  type        = string
  description = "Comma-separated AWS services to monitor (ec2, sagemaker, ...)."
  default     = "ec2,sagemaker"
}

variable "monitoring_interval_seconds" {
  type        = number
  default     = 300
}

variable "log_level" {
  type        = string
  default     = "INFO"
  validation {
    condition     = contains(["DEBUG", "INFO", "WARNING", "ERROR"], var.log_level)
    error_message = "log_level must be DEBUG, INFO, WARNING, or ERROR."
  }
}

# ──────────────────────────────────────────────────────────────────────
# Monitor toggles — object so customers can set all at once
# ──────────────────────────────────────────────────────────────────────

variable "monitors" {
  type = object({
    aws        = optional(bool, false)
    sagemaker  = optional(bool, false)
    azure      = optional(bool, false)
    gcp        = optional(bool, false)
    kubernetes = optional(bool, false)
  })
  default     = {}
  description = "Which cloud-provider monitors to enable."
}

variable "features" {
  type = object({
    nvml         = optional(bool, false)
    ai_spend     = optional(bool, false)
    spot_handler = optional(bool, false)
  })
  default     = {}
  description = "Optional feature monitors."
}

# ──────────────────────────────────────────────────────────────────────
# Telemetry
# ──────────────────────────────────────────────────────────────────────

variable "telemetry" {
  type = object({
    otel_enabled  = optional(bool, false)
    otel_endpoint = optional(string, "")
  })
  default = {}
}

# ──────────────────────────────────────────────────────────────────────
# Extensibility
# ──────────────────────────────────────────────────────────────────────

variable "extra_environment" {
  type        = list(object({ name = string, value = string }))
  default     = []
  description = "Additional environment variables to set on the container."
}

variable "extra_task_policy_arns" {
  type        = list(string)
  default     = []
  description = "Additional IAM policy ARNs to attach to the task role."
}

variable "log_retention_days" {
  type        = number
  default     = 30
}

variable "tags" {
  type        = map(string)
  default     = {}
}
