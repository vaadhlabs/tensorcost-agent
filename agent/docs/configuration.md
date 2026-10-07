# GPU Agent Configuration Guide

## Running Locally

```bash
pip install -r requirements.txt
python main.py --config config.yaml
```

## Testing

```bash
pip install pytest pytest-cov pytest-mock moto boto3 responses freezegun
pytest tests/ -v                          # Run all tests
pytest tests/ -v --cov=src --cov-report=term  # With coverage
pytest tests/test_aws_monitor.py -v       # Run specific test file
```

## Key Directories

```
src/
  monitors/
    aws_monitor.py         # EC2 GPU instances, CloudWatch, Cost Explorer, Spot pricing
    azure_monitor.py       # Azure VMs, Consumption API, Spot pricing
    gcp_monitor.py         # GCE instances, BigQuery billing, Spot pricing
    kubernetes_monitor.py  # K8s GPU nodes + pods
    sagemaker_monitor.py   # SageMaker training jobs
  proto/
    agent.proto            # gRPC service definition
  grpc_client.py           # gRPC bidirectional streaming client
  nvml_sampler.py          # Local GPU metrics via NVIDIA NVML (10s sampling)
  alerting.py              # Slack + email alert manager
  ai_spend_monitor.py      # LLM token cost tracking
main.py                    # Main orchestrator
```

## Communication Modes

Set `COMM_MODE` environment variable:

| Mode | Description |
|------|-------------|
| `http` | POST /api/sync (default, works through API Gateway) |
| `grpc` | Bidirectional stream on port 50051 (preferred, instant commands) |
| `both` | gRPC preferred with HTTP fallback (recommended) |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `BACKEND_API_URL` | Yes (http/both) | Backend URL (e.g., `https://api-st.tensorcost.com`) |
| `BACKEND_API_KEY` | Yes | Tenant API key from platform |
| `TENANT_ID` | No | Tenant ID (default: resolved from API key) |
| `GRPC_SERVER_HOST` | Yes (grpc/both) | gRPC server host |
| `GRPC_SERVER_PORT` | No | gRPC port (default: 50051) |
| `COMM_MODE` | No | `http`, `grpc`, or `both` (default: `http`) |
| `MONITORING_INTERVAL` | No | Seconds between collection cycles (default: 60) |
| `AWS_REGION` | Yes (AWS) | AWS region to monitor |
| `AWS_ACCESS_KEY_ID` | Yes (AWS) | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | Yes (AWS) | AWS credentials |
| `AZURE_SUBSCRIPTION_ID` | Yes (Azure) | Azure subscription |
| `GCP_PROJECT_ID` | Yes (GCP) | GCP project ID |
| `GCP_BILLING_DATASET` | No (GCP) | BigQuery billing dataset for real cost data |
| `AI_SPEND_ENABLED` | No | Enable LLM cost tracking (default: false) |

## Multi-Cloud Setup

Enable monitors in `config.yaml`:

```yaml
monitoring:
  aws:
    enabled: true
    regions: ["us-east-1", "us-west-2"]
  azure:
    enabled: false
  gcp:
    enabled: false
  kubernetes:
    enabled: false
  sagemaker:
    enabled: false
```
