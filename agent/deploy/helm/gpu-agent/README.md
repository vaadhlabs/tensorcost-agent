# gpu-agent Helm chart

Deploys the tensorcost Unified GPU Cost Optimization Agent to a Kubernetes cluster.

## TL;DR

```bash
helm install gpu-agent ./deploy/helm/gpu-agent \
  --namespace tensorcost --create-namespace \
  --set auth.inline.apiKey=YOUR_API_KEY \
  --set auth.inline.tenantId=YOUR_TENANT_UUID \
  --set monitors.kubernetes.enabled=true
```

## Prerequisites

- Kubernetes ≥ 1.24
- Helm ≥ 3.8
- A tensorcost tenant API key and tenant UUID (see dashboard → Integrations)

## Credentials

The chart expects backend credentials in a Kubernetes Secret with two keys: `api-key` and `tenant-id`. Two ways to provide them:

### Option 1: Existing Secret (production)

```bash
kubectl create secret generic gpu-agent-auth \
  --namespace tensorcost \
  --from-literal=api-key=YOUR_API_KEY \
  --from-literal=tenant-id=YOUR_TENANT_UUID

helm install gpu-agent ./deploy/helm/gpu-agent \
  --set auth.existingSecret=gpu-agent-auth
```

### Option 2: Inline (dev only)

```bash
helm install gpu-agent ./deploy/helm/gpu-agent \
  --set auth.inline.apiKey=YOUR_API_KEY \
  --set auth.inline.tenantId=YOUR_TENANT_UUID
```

## Common configurations

### Kubernetes monitoring (in-cluster)

Watches pods/nodes/deployments in the cluster the agent runs in:

```yaml
monitors:
  kubernetes:
    enabled: true
    namespaces: "default,ml-workloads"
    inCluster: true  # uses the pod's service account + auto-created ClusterRole
```

### AWS monitoring with IRSA (EKS)

Use IAM Roles for Service Accounts — no static credentials needed:

```yaml
serviceAccount:
  create: true
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/gpu-agent

monitors:
  aws:
    enabled: true
    region: us-east-1
    services: "ec2,sagemaker"
```

### Per-node GPU sampling via NVML

Run as a DaemonSet on GPU nodes only:

```yaml
workload:
  kind: DaemonSet

features:
  nvml:
    enabled: true

nodeSelector:
  nvidia.com/gpu.present: "true"

tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

### GCP monitoring

```bash
# First create a Secret with the service account JSON:
kubectl create secret generic gcp-creds \
  --from-file=key.json=/path/to/service-account.json
```

```yaml
monitors:
  gcp:
    enabled: true
    projectId: my-gcp-project
    credentialsSecret: gcp-creds
```

### Azure monitoring

```bash
kubectl create secret generic azure-creds \
  --from-literal=AZURE_CLIENT_ID=... \
  --from-literal=AZURE_CLIENT_SECRET=... \
  --from-literal=AZURE_TENANT_ID=...
```

```yaml
monitors:
  azure:
    enabled: true
    subscriptionId: "subscription-uuid"
    resourceGroup: "my-rg"
    credentialsSecret: azure-creds
```

### OpenTelemetry tracing

```yaml
telemetry:
  otel:
    enabled: true
    exporterEndpoint: "http://otel-collector:4317"  # or your collector
```

## Values reference

See `values.yaml` for the full list of configurable values with inline documentation.

## Uninstall

```bash
helm uninstall gpu-agent -n tensorcost
```

Note: Uninstalling does NOT delete the auth Secret if you created it manually (`auth.existingSecret`). Delete it explicitly:

```bash
kubectl delete secret gpu-agent-auth -n tensorcost
```
