# TensorCost Agent (open source)

Apache-2.0 collection layer for **GPU telemetry** and **LLM usage metadata**. Run standalone (Prometheus / OTLP / FOCUS-shaped export) or connect to the [TensorCost](https://tensorcost.com) control plane for attribution, routing, and savings proof.

| Component | Path | Package |
|-----------|------|---------|
| GPU agent | [`agent/`](./agent/) | Docker `ghcr.io/tensorcost/agent` |
| Node SDK | [`sdks/node/`](./sdks/node/) | npm `@tensorcost/sdk` |
| Python SDK | [`sdks/python/`](./sdks/python/) | PyPI `tensorcost` |

**Ingest contract (v1):** [docs/ingest-contract.md](./docs/ingest-contract.md)

## 5-minute quickstart (standalone GPU metrics)

**Goal:** see GPU utilization locally without a TensorCost account.

### Option A — Docker (NVML host)

```bash
docker run --rm -d --name tc-agent \
  --gpus all \
  -e HOST_GPU_TELEMETRY_ENABLED=true \
  -e BACKEND_API_URL= \
  -p 9090:9090 \
  ghcr.io/tensorcost/agent:latest
curl -s localhost:9090/metrics | head
```

You should see `tensorcost_gpu_utilization_percent` (and related series when NVML reports them).

### Option B — Helm (Kubernetes)

```bash
helm upgrade --install gpu-agent ./agent/deploy/helm/gpu-agent \
  --set auth.inline.apiKey= \
  --set backend.apiUrl= \
  --set env.PROMETHEUS_TEXT_ENABLED=true
kubectl port-forward svc/gpu-agent 9090:9090
curl -s localhost:9090/metrics | head
```

### Option C — FOCUS-shaped usage file

```bash
export FOCUS_USAGE_FILE=/tmp/gpu-usage.csv
export HOST_GPU_TELEMETRY_ENABLED=true
cd agent && python main.py
# After one monitoring cycle, inspect /tmp/gpu-usage.csv
```

## Connect to TensorCost (commercial)

1. Mint an agent key in the TensorCost console.
2. Set `BACKEND_API_URL=https://api.tensorcost.com` and `AGENT_API_KEY=...`.
3. Metrics flow to the control plane per [docs/ingest-contract.md](./docs/ingest-contract.md).

LLM SDKs:

```typescript
import OpenAI from "openai";
import { wrap } from "@tensorcost/sdk";
const client = wrap(new OpenAI(), { apiKey: process.env.TENSORCOST_API_KEY });
```

```python
from openai import OpenAI
from tensorcost import wrap
client = wrap(OpenAI(), api_key="...")
```

## Privacy

Nothing is sent unless you configure a backend URL, remote_write endpoint, OTLP endpoint, or FOCUS file path. See agent [`.env.example`](./agent/.env.example).

## Trademarks

TensorCost name and logo are trademarks of Vaadh Labs. See [NOTICE](./NOTICE).

## License

Apache-2.0 — see [LICENSE](./agent/LICENSE).
