# AWS Terraform modules

Two sub-modules, each for a different deployment shape:

| Sub-module | Target | When to use |
|---|---|---|
| [**`fargate/`**](./fargate/) | ECS Fargate task | Cost-collection agent only. No GPU (Fargate doesn't support GPU). Good for tenants whose GPU spend is on SageMaker / Bedrock / third-party SaaS — the Fargate task queries Cost Explorer + writes to the dashboard; no NVML needed. |
| [**`ec2-agent/`**](./ec2-agent/) | Install on existing EC2 (any SKU) | Full agent: NVML + cost + LLM middleware. The most common shape — customers with GPU training or inference running on EC2. Uses SSM Association for declarative, re-runnable install. |

Both sub-modules ship on the same `terraform-aws-v*` tag — one release,
two consumable sub-paths. See [`../README.md`](../README.md) for the
full release matrix.

## Choosing a sub-module, quick rubric

- **GPU workloads on EC2?** → `ec2-agent/`
- **GPU workloads on EKS?** → use the [Helm chart](../../helm/gpu-agent/), not Terraform
- **GPU workloads on SageMaker?** → `fargate/` (the task polls SageMaker APIs)
- **No GPU, just want to see Bedrock / SageMaker spend?** → `fargate/`, OR consider the
  server-side Bedrock log ingester on the tensorcost backend (zero install)

Not sure? `ec2-agent/` is the more complete agent experience — start
there unless the customer specifically doesn't have EC2 GPU hosts.
