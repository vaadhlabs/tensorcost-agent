# TensorCost ingest contract (v1)

This document describes how the open-source GPU agent and LLM SDKs talk to the **commercial TensorCost control plane**. The control plane is not open source; this contract is stable and additive-only until v2.

## Versioning

- **v1** (this document): additive JSON/protobuf fields only.
- Agents and SDKs should ignore unknown response fields.

## GPU agent — HTTP sync

Authentication: header `X-Agent-Key` (plaintext key minted in the TensorCost console).

| Method | Path | Body |
|--------|------|------|
| POST | `/api/gpu/sync/` | Batch: `data_type` = `metrics` or `instance_updates` |
| POST | `/api/gpu/sync/agent/health` | Agent health heartbeat |
| POST | `/api/gpu/sync/spot-interruption` | Spot interruption event |
| GET | `/api/gpu/sync/agent/commands` | Poll remote commands (optional actuator) |
| POST | `/api/gpu/sync/agent/command-result` | Command result |

### Metric point (HTTP JSON)

Wire fields (after agent-side rename):

- `instance_id` **or** (`cloud_id` + `cloud_provider`)
- `ts_unix_ms`, `util_pct`, `mem_pct`, `temp_c`
- Optional: `gpu_index`, `power_w`, `mem_bw_pct`, `memory_used_mb`, `memory_total_mb`, ECC and clock fields

See [proto/agent.proto](./proto/agent.proto) for the canonical gRPC `MetricPoint` message (same semantics).

## GPU agent — gRPC

Package: `tensorcost.agent.v1`

- **Connect** (bidirectional stream): `AgentHello` (HMAC-signed), `MetricBatch`, `InstanceUpdate`, `CommandResult`
- **Heartbeat**: unary liveness

Proto file: [docs/proto/agent.proto](./proto/agent.proto)

Environment:

- `GRPC_TARGET`, `AGENT_KEY_ID`, `AGENT_HMAC_PEPPER`, `COMM_MODE=http|grpc|both`

## LLM SDK — observations

Authentication: Bearer JWT from `POST /api/inference-proxy/sdk-token/exchange` (API key or OAuth per tenant).

| Method | Path | Response |
|--------|------|----------|
| POST | `/api/inference-proxy/observation` | 202 Accepted |
| POST | `/api/inference-proxy/otlp/v1/traces` | 202 Accepted |

### Observation JSON (required fields)

- `sdk_version`, `provider`, `model`, `operation`, `status`
- `request_at` (ISO-8601)
- `correlation_id`

Optional: `response_at`, token counts, `latency_ms`, `sdk_added_ms`, `trace_id`, `parent_span_id`, `environment`, `error_message`.

Providers: `openai`, `anthropic`, `bedrock`, `azure_openai`, `vertex`.

SDKs ship in `sdks/node` (`@tensorcost/sdk`) and `sdks/python` (`tensorcost`).

## Standalone mode (no control plane)

The GPU agent can run without any of the endpoints above:

- Prometheus text: `http://<host>:9090/metrics` (`PROMETHEUS_TEXT_ENABLED`, default on)
- Optional `PROMETHEUS_REMOTE_WRITE_URL`
- Optional OTLP metrics: `OTEL_METRICS_ENABLED=true` and `OTEL_EXPORTER_OTLP_ENDPOINT`
- Optional local FOCUS-shaped usage file: `FOCUS_USAGE_FILE`

No telemetry is sent unless the operator configures a destination.
