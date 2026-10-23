# TensorCost GPU Agent — Kubernetes DaemonSet

Deploys the agent as a DaemonSet so it runs on every node that matches
the `nvidia.com/gpu: "true"` label. Each pod samples NVML metrics for the
GPUs on its node and reports back to the TensorCost gateway.

---

## Prerequisites

- `nvidia-device-plugin` (or the equivalent for your distribution) installed
  and GPU nodes labeled with `nvidia.com/gpu: "true"`.
- `kubectl` access with permissions to create `Namespace`, `Secret`,
  `ConfigMap`, `ServiceAccount`, `ClusterRole`, `ClusterRoleBinding`, and
  `DaemonSet` resources in the target cluster.
- Agent credentials from the TensorCost onboarding wizard (`TENANT_ID`,
  `AGENT_API_KEY`, `AGENT_KEY_ID`, `AGENT_HMAC_PEPPER`).
  - Allocate at least 16 GB of disk on each GPU node. The agent image plus container runtime overhead exceeds the 8 GB default on AL2023 t3.micro AMIs; an actual customer hit `no space left on device` during `docker pull` in our 2026-05-18 E2E test.

---

## What to fill in

Open `agent-daemonset.yaml` and replace every placeholder marked with
`<angle-brackets>`:

| Placeholder | Where to get it |
|---|---|
| `<your-tenant-uuid>` | TensorCost onboarding wizard |
| `<your-agent-api-key>` | Onboarding wizard — `AGENT_API_KEY` |
| `<your-agent-key-id>` | Onboarding wizard — `AGENT_KEY_ID` |
| `<your-agent-hmac-pepper>` | Onboarding wizard — `AGENT_HMAC_PEPPER` |
| `<gateway-url>` | Your TensorCost API URL, e.g. `https://api.dev.tensorcost.com` |

The values in the Secret are stored as plain strings (not base64) — `kubectl`
handles the encoding when `stringData:` is used. Do not base64-encode them
yourself.

---

## Deploy

```bash
# Apply everything in one go
kubectl apply -f agent-daemonset.yaml

# Confirm the DaemonSet rolled out on all GPU nodes
kubectl -n tensorcost get daemonset tensorcost-agent
kubectl -n tensorcost get pods -l app=tensorcost-agent -o wide

# Tail logs from one pod
kubectl -n tensorcost logs -l app=tensorcost-agent --tail=100 -f
```

---

## EKS-specific notes

On EKS the standard credential chain for the AWS SDK is:
1. IRSA (IAM Roles for Service Accounts) — recommended.
2. EC2 instance profile on the node — works, but grants over-broad access.

To use IRSA, annotate the `tensorcost-agent` ServiceAccount with the role
ARN and attach a policy that allows `ec2:DescribeInstances`,
`ce:GetCostAndUsage`, and `cloudwatch:GetMetricData` for your GPU fleet.
The DaemonSet manifest already has the ServiceAccount in place; add the
annotation after applying:

```bash
kubectl -n tensorcost annotate serviceaccount tensorcost-agent \
  eks.amazonaws.com/role-arn=arn:aws:iam::<ACCOUNT_ID>:role/<ROLE_NAME>
```

Then restart the DaemonSet pods:

```bash
kubectl -n tensorcost rollout restart daemonset tensorcost-agent
```

---

## Upgrading the image

The DaemonSet uses `imagePullPolicy: Always` so a rollout picks up the
latest tag automatically. To upgrade:

```bash
kubectl -n tensorcost rollout restart daemonset tensorcost-agent
kubectl -n tensorcost rollout status daemonset tensorcost-agent
```

To pin a specific digest (recommended for production):

```bash
# Get the current digest
docker pull docker.io/rlgocloudera/gpu-usage:node
docker inspect --format='{{index .RepoDigests 0}}' docker.io/rlgocloudera/gpu-usage:node
# → docker.io/rlgocloudera/gpu-usage@sha256:…

# Edit agent-daemonset.yaml: change the image: line to the digest form
# Then re-apply
kubectl apply -f agent-daemonset.yaml
```

---

## Removing the agent

```bash
kubectl delete -f agent-daemonset.yaml
```

This removes the DaemonSet, Secret, ConfigMap, ServiceAccount, ClusterRole,
and ClusterRoleBinding. The `tensorcost` namespace is NOT deleted — do that
manually if you want a clean slate (`kubectl delete namespace tensorcost`).
