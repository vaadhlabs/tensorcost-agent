/**
 * ec2-agent — install the unified-gpu-agent onto an EXISTING EC2 instance.
 *
 * Complements the sibling `aws/fargate/` module (cost-collection via ECS
 * Fargate task, no GPU). This one is for customers with GPU hosts on
 * EC2 — NVML sampling + cost collection + LLM middleware on the same
 * instance as their workload.
 *
 * Install mechanism: AWS SSM. EC2 user-data only runs at instance
 * launch, so it doesn't help for BYO-VM. SSM Run Command is imperative
 * (one-off); SSM Association is declarative + re-runnable + drift-
 * detecting, which matches Terraform's model.
 *
 * Secrets flow: backend_api_key lands in SSM Parameter Store as a
 * SecureString (encrypted with the AWS-managed SSM KMS key). The
 * install script fetches via `aws ssm get-parameter --with-decryption`
 * at run time. This keeps the secret out of the SSM document itself
 * (which IS visible to Read-only SSM users via the console).
 *
 * Prerequisites on the EC2 instance:
 *   - SSM-managed (the AmazonSSMManagedInstanceCore managed policy
 *     attached to its instance profile). Modern AMIs ship with the
 *     SSM agent pre-installed.
 *   - Running. SSM Association fails fast on stopped instances.
 *   - Outbound HTTPS to ssm.<region>.amazonaws.com and the agent
 *     image registry (docker.io / ghcr.io / your ECR).
 */

// ─── 1. Secrets in SSM Parameter Store (encrypted) ───────────────────────

resource "aws_ssm_parameter" "backend_api_key" {
  name        = local.ssm_param_backend_api_key
  description = "tensorcost backend API key — read by the unified-gpu-agent install script."
  type        = "SecureString"
  value       = var.backend_api_key
  tags        = local.common_tags
}

resource "aws_ssm_parameter" "tenant_id" {
  name        = local.ssm_param_tenant_id
  description = "tensorcost tenant UUID — read by the unified-gpu-agent install script."
  type        = "SecureString"
  value       = var.tenant_id
  tags        = local.common_tags
}

resource "aws_ssm_parameter" "agent_key_id" {
  count       = var.grpc_host != "" ? 1 : 0
  name        = local.ssm_param_agent_key_id
  description = "TensorCost agent HMAC key id — read by the unified-gpu-agent install script."
  type        = "SecureString"
  value       = var.agent_key_id
  tags        = local.common_tags
}

resource "aws_ssm_parameter" "agent_hmac_pepper" {
  count       = var.grpc_host != "" ? 1 : 0
  name        = local.ssm_param_agent_hmac
  description = "TensorCost agent HMAC pepper — read by the unified-gpu-agent install script."
  type        = "SecureString"
  value       = var.agent_hmac_pepper
  tags        = local.common_tags
}

// ─── 2. SSM Document that wraps AWS-RunShellScript with our install ──────
// We define our own document (not use AWS-RunShellScript directly)
// because:
//   - it's a stable reference — upgrading the agent is changing this
//     resource in Terraform, not mutating a managed AWS document;
//   - it lets us version the document via hash + force re-association;
//   - it's scoped: only this document can run on the target instance,
//     so a leaked SSM permission can't execute arbitrary commands.

locals {
  # If the caller pinned a sha256 digest, rewrite `repo:tag` → `repo@sha256:...`
  # so the systemd unit pulls a content-addressed reference instead of a
  # mutable tag. This is the supply-chain fix for §8.2 Blocker #4 — a
  # compromised registry credential cannot retag a malicious image onto
  # a digest, only onto a tag. The rewrite strips the trailing `:<tag>`
  # only if it's present (some callers pass bare `repo` — leave alone).
  agent_image_no_tag = (
    var.agent_image_digest != "" && length(regexall(":[^/]+$", var.agent_image)) > 0
    ? replace(var.agent_image, "/:[^/]+$/", "")
    : var.agent_image
  )
  agent_image_ref = (
    var.agent_image_digest != ""
    ? "${local.agent_image_no_tag}@${var.agent_image_digest}"
    : var.agent_image
  )

  install_script = templatefile("${path.module}/install-agent.sh.tftpl", {
    backend_api_url           = var.backend_api_url
    grpc_host                 = var.grpc_host
    grpc_port                 = var.grpc_port
    metrics_push_interval     = var.metrics_push_interval
    agent_image               = local.agent_image_ref
    agent_image_digest_pinned = var.agent_image_digest != ""
    use_private_ecr           = var.use_private_ecr
    ecr_registry              = var.ecr_registry
    ssm_param_backend_api_key = local.ssm_param_backend_api_key
    ssm_param_tenant_id       = local.ssm_param_tenant_id
    ssm_param_agent_key_id    = local.ssm_param_agent_key_id
    ssm_param_agent_hmac      = local.ssm_param_agent_hmac
    agent_hostname            = local.resolved_agent_hostname
    grpc_enabled              = var.grpc_host != ""
  })
}

resource "aws_ssm_document" "install_agent" {
  name            = "${var.name}-install"
  document_type   = "Command"
  document_format = "YAML"

  # AWS-RunShellScript schema with our rendered install baked in. Bash is
  # wrapped as the command's single argument to avoid quoting headaches in
  # the document's YAML.
  content = yamlencode({
    schemaVersion = "2.2"
    description   = "Install/update the unified-gpu-agent container on the target EC2 instance."
    mainSteps = [
      {
        action = "aws:runShellScript"
        name   = "runInstallScript"
        inputs = {
          runCommand     = split("\n", local.install_script)
          timeoutSeconds = "600" # driver install can take a few minutes
        }
      },
    ]
  })

  tags = local.common_tags
}

// ─── 3. SSM Association — binds the document to the instance ─────────────
// association_schedule = "rate(6 hours)" gives drift detection + auto-
// restart if the service dies silently. Set to "" to run once at apply.

resource "aws_ssm_association" "install_agent" {
  name             = aws_ssm_document.install_agent.name
  association_name = "${var.name}-install"

  targets {
    key    = "InstanceIds"
    values = [var.instance_id]
  }

  # Empty schedule = run once at apply. Scheduled = re-run on cadence.
  schedule_expression = var.association_schedule != "" ? var.association_schedule : null

  # Treat SSM invocation errors as Terraform errors. Without this, a
  # broken install silently succeeds from Terraform's POV.
  compliance_severity = "HIGH"

  depends_on = [
    aws_ssm_parameter.backend_api_key,
    aws_ssm_parameter.tenant_id,
  ]
}

// ─── 4. IAM — attach policies to the instance's existing role ────────────
// Skipped entirely when instance_iam_role_name is empty — caller grants
// roles out-of-band (enterprise IAM flows often need this).

data "aws_region" "current" {}
data "aws_caller_identity" "current" {}

// Policy granting the agent access to:
//   - Read the SSM SecureString parameters we created (install script)
//   - Cost Explorer (GetCostAndUsage, GetCostForecast) — agent runtime
//   - EC2 Describe* — instance enumeration for the dashboard

data "aws_iam_policy_document" "agent_runtime" {
  # Read SSM params we manage. Scoped to THIS tenant's params (not
  # /tensorcost/* wildcard — tighter blast radius).
  statement {
    actions = ["ssm:GetParameter", "ssm:GetParameters"]
    resources = concat(
      [
        aws_ssm_parameter.backend_api_key.arn,
        aws_ssm_parameter.tenant_id.arn,
      ],
      var.grpc_host != "" ? [
        aws_ssm_parameter.agent_key_id[0].arn,
        aws_ssm_parameter.agent_hmac_pepper[0].arn,
      ] : [],
    )
  }

  # Decrypt the SecureString values. The SSM params are encrypted with
  # the AWS-managed key (alias/aws/ssm); we need kms:Decrypt on that key.
  statement {
    actions   = ["kms:Decrypt"]
    resources = ["arn:aws:kms:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:alias/aws/ssm"]
    condition {
      test     = "StringEquals"
      variable = "kms:EncryptionContext:PARAMETER_ARN"
      values = concat(
        [
          aws_ssm_parameter.backend_api_key.arn,
          aws_ssm_parameter.tenant_id.arn,
        ],
        var.grpc_host != "" ? [
          aws_ssm_parameter.agent_key_id[0].arn,
          aws_ssm_parameter.agent_hmac_pepper[0].arn,
        ] : [],
      )
    }
  }

  # Cost Explorer — read-only, subscription-wide (CE doesn't support
  # resource-level scoping). Safe to grant broadly.
  statement {
    actions = [
      "ce:GetCostAndUsage",
      "ce:GetCostForecast",
      "ce:GetDimensionValues",
      "ce:GetReservationCoverage",
    ]
    resources = ["*"]
  }

  # EC2 describe — read-only enumeration of this account's instances,
  # disks, and tags. No Modify*/Terminate* — agent never touches state.
  statement {
    actions = [
      "ec2:DescribeInstances",
      "ec2:DescribeVolumes",
      "ec2:DescribeTags",
      "ec2:DescribeRegions",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "agent_runtime" {
  count       = var.instance_iam_role_name != "" ? 1 : 0
  name        = "${var.name}-agent-runtime"
  description = "Runtime permissions for the unified-gpu-agent on instance ${var.instance_id}"
  policy      = data.aws_iam_policy_document.agent_runtime.json
  tags        = local.common_tags
}

resource "aws_iam_role_policy_attachment" "agent_runtime" {
  count      = var.instance_iam_role_name != "" ? 1 : 0
  role       = var.instance_iam_role_name
  policy_arn = aws_iam_policy.agent_runtime[0].arn
}
