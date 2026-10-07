# AWS `ec2-agent` — install the agent on an existing EC2 instance

Sub-module for "I have an EC2 GPU host, bolt the agent onto it". Uses
AWS Systems Manager (SSM) Association — declarative, re-runnable, drift-
detecting.

Pairs with the sibling [`fargate/`](../fargate/) module which is for
customers that want a cost-collection-only agent on ECS Fargate (no GPU,
no NVML). Most customers with actual GPU workloads want THIS module.

## When to use which

| Situation | Module |
|---|---|
| Customer has an EC2 GPU instance (p3/p4/p5/g5/g6 etc.) | **This module** |
| Customer wants cost-only tracking without touching their GPU hosts | [`fargate/`](../fargate/) |
| Customer's GPUs are on EKS | [`../../helm/gpu-agent/`](../../helm/gpu-agent/) as DaemonSet |
| Customer wants SageMaker endpoint tracking | Use the backend's SageMaker cost monitor — no Terraform |

## Usage

```hcl
resource "aws_instance" "gpu_host" {
  # …whatever the customer already has. Must have:
  #   iam_instance_profile = <profile with AmazonSSMManagedInstanceCore>
  # so SSM can talk to it.
}

module "gpu_agent_byo" {
  source = "git::https://github.com/gpu-usage/gpu-agents.git//deploy/terraform/aws/ec2-agent?ref=terraform-aws-v1.1.0"

  name                   = "acme-prod-agent"
  instance_id            = aws_instance.gpu_host.id
  instance_iam_role_name = aws_iam_role.gpu_host.name  # role name, NOT ARN
  region                 = "us-east-1"

  backend_api_url = "https://api.tensorcost.com"
  backend_api_key = var.backend_api_key  # sensitive
  tenant_id       = var.tenant_id

  # Re-run install every 6h by default — picks up agent image upgrades
  # and heals silent service failures. Set "" to run once at apply.
  # association_schedule = "rate(6 hours)"
}
```

One `terraform apply` → SSM Association binds the install document to
the instance → SSM agent (pre-installed on modern AMIs) runs the script
→ Docker + NVIDIA driver + toolkit install if needed → systemd service
starts → agent streams metrics + costs to the dashboard within ~5 min.

## How secrets are handled

`backend_api_key` lives in **SSM Parameter Store** as a `SecureString`
(encrypted with the AWS-managed `alias/aws/ssm` KMS key). The install
script fetches it at run time via `aws ssm get-parameter --with-decryption`.

This keeps the secret OUT of:
- The SSM Document itself (visible in the console to anyone with
  `ssm:GetDocument`).
- The SSM Association's `parameters` field (ditto).
- Terraform state in cleartext (still sensitive in state, but encrypted
  at rest if you're using S3 + KMS for the remote state).

The IAM policy this module attaches scopes `ssm:GetParameter` to just
the two SecureString ARNs, and includes a `kms:EncryptionContext`
condition on `kms:Decrypt` — so even if the role is re-used for
something else, it can only decrypt THESE specific parameters.

## Prerequisites

1. **SSM-managed instance.** Attach the AWS-managed policy
   `AmazonSSMManagedInstanceCore` to the instance's IAM role. Modern
   Ubuntu / Amazon Linux AMIs ship with the SSM agent installed and
   running; if yours doesn't, install it first.
2. **Running instance.** SSM Association fails fast on stopped instances.
3. **Outbound HTTPS** to `ssm.<region>.amazonaws.com` (SSM control
   plane) and the image registry (docker.io / ghcr.io / ECR).
4. **NVIDIA GPU optional.** If the host has one, the script installs
   driver + container toolkit. If not (e.g. a cost-monitoring bastion
   on a t3.medium), it runs the agent in cost-only mode.

## IAM policies attached (to `instance_iam_role_name`)

- `ssm:GetParameter` + `ssm:GetParameters` on the two SecureStrings only.
- `kms:Decrypt` on `alias/aws/ssm`, scoped to those parameter ARNs.
- Cost Explorer reads: `ce:GetCostAndUsage`, `ce:GetCostForecast`,
  `ce:GetDimensionValues`, `ce:GetReservationCoverage`.
- EC2 describe: `ec2:DescribeInstances`, `ec2:DescribeVolumes`,
  `ec2:DescribeTags`, `ec2:DescribeRegions`. Read-only — no mutating
  actions.

Set `instance_iam_role_name = ""` to skip policy attachment and grant
the policies yourself. Useful for enterprise IAM flows where a central
team owns role-policy attachments.

## Re-running the install

The Association re-runs on `var.association_schedule` (default
`rate(6 hours)`). This gives you drift detection — if the service dies
or the image cache is evicted, the next scheduled run puts things back.

To force an immediate re-run (e.g. after rotating `backend_api_key`):

```bash
aws ssm start-associations-once \
  --association-ids $(terraform output -raw ssm_association_id)
```

## Troubleshooting

```bash
# SSH to the instance, then:
sudo systemctl status unified-gpu-agent            # agent service state
sudo journalctl -u unified-gpu-agent -f            # live agent logs
sudo tail -f /var/log/tensorcost-agent-install.log # install-script log
```

Or via SSM (no SSH required):

```bash
aws ssm start-session --target <instance-id>
# once in: sudo journalctl -u unified-gpu-agent -n 100
```

Association run history shows up in the Systems Manager console under
State Manager → Associations → <name>. Each execution has a CloudWatch-
style output for both stdout and stderr of the install script.

## Inputs

See [`variables.tf`](./variables.tf) for the full list. Required:

| Variable | Description |
|---|---|
| `name` | Prefix for all created resources. |
| `instance_id` | Existing EC2 instance ID. |
| `region` | AWS region (SSM parameters are region-scoped). |
| `backend_api_url` | tensorcost dashboard API base URL. |
| `backend_api_key` | Tenant API key (sensitive). |
| `tenant_id` | Tenant UUID. |

Optional: `instance_iam_role_name`, `agent_image`, `use_private_ecr`,
`ecr_registry`, `association_schedule`.

## Outputs

| Output | Description |
|---|---|
| `ssm_document_name` | Name of the install SSM Document. |
| `ssm_association_id` | ID of the Association — feed to `start-associations-once` for manual re-runs. |
| `ssm_parameter_backend_api_key_arn` / `..._tenant_id_arn` | ARNs of the SecureString parameters. |
| `agent_runtime_policy_arn` | ARN of the IAM policy attached to the instance role (null if `instance_iam_role_name = ""`). |
| `install_log_path` | Path on the instance where the install script logs. |
| `agent_service_name` | `unified-gpu-agent.service` — for `systemctl` / `journalctl`. |
