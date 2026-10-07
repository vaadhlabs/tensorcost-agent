# tensorcost — Python SDK

One-line wrapper for [TensorCost](https://tensorcost.com). Adds fire-and-forget cost observability and optional inference control to OpenAI, Anthropic, AWS Bedrock, and Google Vertex AI clients without changing how your code calls the underlying provider.

**Current version:** 1.3.0 — period budgets + agent-run span linking (upgrade from 1.2.x if you use daily/monthly caps or Router · Agent runs). See [CHANGELOG](./CHANGELOG.md), [SDK compliance](https://github.com/vaadhlabs/tensorcost/blob/main/docs/features/sdk-compliance.md), and [Migrating from 0.x](#migrating-from-0x).

## Install

```bash
pip install tensorcost
```

Runtime dependency: `httpx` only.

## Quick start (observe)

The default layer is **`observe`**: telemetry only, provider call unchanged.

```python
from openai import OpenAI
from tensorcost import wrap

client = wrap(
    OpenAI(api_key="sk-..."),
    api_key="tc_live_...",
)

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hello"}],
)
```

Anthropic works identically:

```python
from anthropic import Anthropic
from tensorcost import wrap

client = wrap(
    Anthropic(api_key="sk-ant-..."),
    api_key="tc_live_...",
)
client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=128,
    messages=[{"role": "user", "content": "hello"}],
)
```

Create a long-lived SDK key in the TensorCost console (**Setup → Keys**). The SDK exchanges it for a short-lived JWT via `POST /api/inference-proxy/sdk-token/exchange` and refreshes before expiry.

## Control layers (`max_layer`)

v1.0 uses a single ladder for how much TensorCost may intervene on a call:

`off → observe → govern → steer → route`

A layer is a **ceiling**, not a mode switch. The effective layer is always:

```text
min(your max_layer, console published layer)
```

When routing is paused in the console, the published ceiling is capped at `govern`.

| Layer | Provider path | Behavior |
|---|---|---|
| `off` | Direct | No SDK wiring; client returned unchanged |
| `observe` | Direct | Telemetry only (**default** when `max_layer` is omitted) |
| `govern` | Direct | Metadata-only `POST /v1/admit` before the provider call |
| `steer` | Via proxy | Proxy on the path; may steer without full live routing |
| `route` | Via proxy | Full applied-mode routing through inference-proxy |

```python
from openai import OpenAI
from tensorcost import wrap

client = wrap(
    OpenAI(api_key="sk-..."),
    api_key="tc_live_...",
    max_layer="route",
    proxy_url="https://api.my-instance.tensorcost.com",
)
```

`wrap()` raises `MissingConfigError` when `max_layer` is `steer` or `route` and no `proxy_url` is available. **`govern` does not require `proxy_url`** — admit uses `base_url`.

### Console-published ceiling

Operators set the tenant ceiling in the console (**Policy → Routing**) or via `PATCH /api/ai/router/layer` (`published_layer`, default `observe`). The SDK fetches the effective value from `GET /api/inference-proxy/v1/sdk-layer`:

- **30s cache** while the endpoint is reachable
- **250ms cap** on a cold fetch — a hung TensorCost cannot block the model call
- **Stale-while-revalidate** after TTL: serve the last snapshot immediately and refresh in the background
- **Decays to `govern`** after 5 minutes unreachable (fail-safe: still run admit, never silently upgrade to route)

Publishing `steer` or `route` in the console requires a **content-retention data grant**. Without a grant, publish returns **409**.

### Govern admission

At `govern` and above, the SDK calls `POST /api/inference-proxy/v1/admit` with metadata only — **no prompt or completion bodies**:

- **204** + `x-tc-decision: layer=govern; action=allow` → proceed to the provider
- **403** + `x-tc-decision: layer=govern; action=refuse; reason=…` → call blocked

Admit is **fail-open** on transport errors: the provider call proceeds and an `admit_unavailable` observation is emitted when transport recovers. The admit HTTP call is capped at **500ms**.

### Compliance profiles (1.2.0+)

Enterprise tenants with `feature.compliance_profiles` assign frameworks (GDPR, HIPAA,
PCI-DSS, EU AI Act, residency) in **Policy → Compliance profiles**. The SDK loads
the compiled snapshot from `GET /v1/sdk-layer` and enforces it at govern and above.

- **`team_id`** on `wrap(..., team_id="…")` or **`TENSORCOST_TEAM_ID`** — narrows to a
  team overlay; must match a team-bound SDK token when the key is scoped to a team.
- **In-process DLP** — pattern detectors run locally before the provider; matched
  secrets never leave your process or hit TensorCost.
- **Observations** stamp `compliance_frameworks` and `dlp_action` (never secret text).
- **`TensorCostComplianceDeniedError`** / **`TensorCostComplianceTeamMismatchError`**
  on refuse paths.

See [SDK compliance profiles](https://github.com/vaadhlabs/tensorcost/blob/main/docs/features/sdk-compliance.md).

### Route layer prerequisites

`max_layer="route"` is necessary but not sufficient for live provider swaps. ADR-0016 gates still apply on the proxy side (LaunchDarkly flag, `AI_ROUTING_LIVE_EXECUTION_PERMITTED`, non-`dry_run` policy with rollout, routing not paused). When gated off, traffic pass-throughs to the original provider.

See [ADR-0026](https://github.com/vaadhlabs/tensorcost/blob/main/docs/adr/0026-sdk-control-layer-ceilings.md) and [inference-proxy applied mode](https://github.com/vaadhlabs/tensorcost/blob/main/docs/features/inference-proxy-applied-mode.md).

## Cold start (`prewarm`)

At Lambda/container init, the first steer/route call can otherwise pay for JWT exchange and sdk-layer fetch on the hot path. Call `prewarm()` once at module scope — it shares the same process-wide runtime that `wrap()` uses, so a later `wrap()` hits a warm cache.

`wrap()` also background-warms that shared runtime on first use; explicit `prewarm()` is recommended when you know traffic is about to arrive.

```python
from openai import OpenAI
from tensorcost import prewarm, wrap

prewarm(
    api_key="tc_live_...",
    base_url="https://api.tensorcost.com",
    max_layer="route",
)

client = wrap(
    OpenAI(api_key="sk-..."),
    api_key="tc_live_...",
    max_layer="route",
    proxy_url="https://...",
)
```

`prewarm()` is best-effort: failures are swallowed and the first real call retries the fetches.

## Migrating from 0.x

| 0.x | 1.0 |
|---|---|
| `applied_mode=True` | `max_layer="route"` (deprecated alias still works with a warning) |
| Default `x-tc-max-layer: route` on proxy calls | Default is now `observe` |
| Boolean on/off | Full layer ladder + console ceiling |

```python
# Before (0.6.x)
wrap(client, applied_mode=True, proxy_url="...")

# After (1.0)
wrap(client, max_layer="route", proxy_url="...")
```

Recommended rollout: **`observe` → `govern` → `route`**, raising the console `published_layer` only after each stage is validated.

## Configuration

`wrap()` resolves: explicit kwargs → environment variables → defaults.

| Variable | Purpose | Required |
|---|---|---|
| `TENSORCOST_API_KEY` | Long-lived SDK key from **Setup → Keys** | Yes |
| `TENSORCOST_BASE_URL` | Backend base URL (admit, sdk-layer, token exchange) | No (default `https://api.tensorcost.com`) |
| `TENSORCOST_PROXY_URL` | Inference-proxy base URL | Required when `max_layer` is `steer` or `route` |
| `TENSORCOST_TENANT_ID` | Explicit tenant id | No |
| `TENSORCOST_ENVIRONMENT` | Environment tag on every observation | No |
| `TENSORCOST_CONNECTION_ID` | Provider-connection id | No |
| `TENSORCOST_AGENT_ID` / `TENSORCOST_WORKFLOW_ID` | Agent-run attribution defaults | No |
| `TENSORCOST_CUSTOMER` / `TENSORCOST_FEATURE` | Chargeback tag defaults | No |
| `TC_APPLICATION` / `TENSORCOST_APPLICATION` | Routing-policy application scope (proxy) | No |
| `TC_TAGS` / `TENSORCOST_TAGS` | Comma-separated routing tags (proxy) | No |

```python
client = wrap(
    OpenAI(api_key="sk-..."),
    api_key="tc_live_...",
    max_layer="govern",
    application="customer-support",
    tags=["tier:enterprise"],
)
```

## Proxy hardening

Applies when `max_layer` is `govern`, `steer`, or `route`.

### Retries

```python
from tensorcost import wrap, RetryConfig

client = wrap(
    OpenAI(api_key="sk-..."),
    max_layer="route",
    proxy_url="https://...",
    retry=RetryConfig(max_attempts=5, base_delay_ms=500.0, max_delay_ms=30_000.0),
)
```

5xx, network errors, and 429 are retried with jittered exponential backoff. 4xx (except 429) are never retried.

### Timeouts

Proxy timeouts are split so a wedged TCP connection fails open quickly while slow model bodies still get a full budget:

```python
client = wrap(
    OpenAI(api_key="sk-..."),
    max_layer="route",
    proxy_url="https://...",
    headers_timeout_s=2.0,  # wait for response headers (default 2.0)
    timeout_s=60.0,         # non-streaming body read (default 60.0)
    idle_timeout_s=30.0,    # streaming idle (default 30.0)
    fail_open_enabled=True,
    on_lifecycle_event=lambda e: print(e.kind, e.model),
)
```

| Option | Default | `TensorCostTimeoutError.kind` | On timeout (steer/route) |
|---|---|---|---|
| `headers_timeout_s` | 2s | `"headers"` | Streaming only (future SSE client) |
| `timeout_s` | 60s | `"total"` | TTFB + body for buffered JSON; fail-open when enabled |
| `idle_timeout_s` | 30s | `"idle"` | Reserved for future steer SSE client |

Non-streaming steer/route waits up to **`timeout_s` for headers** (proxy may buffer upstream). **`stream=True`:** direct provider when fail-open enabled; `MissingConfigError` when disabled (1.1.1).

The headers deadline targets wedged proxy connections, not slow upstream models — body read uses `timeout_s` independently once headers arrive.

Typed errors: `TensorCostError`, `TensorCostNetworkError`, `TensorCostTimeoutError`,
`TensorCostProxyError`, `TensorCostQuotaError`, `TensorCostProviderError`,
`TensorCostRunBudgetExceededError`, `TensorCostPeriodBudgetExceededError`,
`TensorCostComplianceDeniedError`, `TensorCostComplianceTeamMismatchError`.

### Period budgets (1.3.0+)

Daily or monthly caps (tenant default, per-agent, or per-team) are configured in
the console under **Router · Agent runs**. On applied-mode (`steer` / `route`)
calls the proxy returns `403` with `code: "PERIOD_BUDGET_EXCEEDED"`. The SDK
raises `TensorCostPeriodBudgetExceededError` and does **not** fall back to the
provider.

```python
from tensorcost import TensorCostPeriodBudgetExceededError

try:
    client.chat.completions.create(...)
except TensorCostPeriodBudgetExceededError as err:
    # err.scope, err.period, err.cap_cents, err.spent_cents
    raise
```

Per-run caps raise `TensorCostRunBudgetExceededError` (`RUN_BUDGET_EXCEEDED`).

Govern refusals and proxy decisions appear in `x-tc-decision` and in observation metadata as `decision_action` / `decision_reason` when present.

After consecutive proxy 5xx failures, the circuit opens and requests fail-open to the provider. Set `fail_open_enabled=False` when direct provider access is unavailable.

## Agent-run attribution

```python
client = wrap(
    OpenAI(api_key="sk-..."),
    agent_id="support-bot",
)

task_client = client.with_meta(workflow_id="ticket-48291")
task_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize this ticket"}],
)
```

Proxy calls forward `x-tc-agent-id` / `x-tc-workflow-id`. `with_meta()` works on OpenAI and Anthropic wrapped clients; Bedrock and Vertex accept wrap-time defaults only.

`customer` / `feature` are separate chargeback dimensions (max 128 chars each).

**Span linking (1.3.0+):** when an agent framework publishes a model-call link,
observations include `trace_id` and `parent_span_id` so ai-service nests the
model_call under the agent trace and run rollups subtract proxied child spend.
Plain `wrap()` without a link still attributes by `agent_id` / `workflow_id`.

## Fail-open guarantee

If TensorCost is unreachable, slow, or returns an error, your underlying provider call still completes normally. The SDK logs a warning via `logging.getLogger("tensorcost")`.

**Duplicate spend:** fail-open after a failed proxy may double-bill if the proxy already reached the provider. Use `fail_open_enabled=False` for in-VPC sidecar-only steer.

```python
from tensorcost import flush_observations

flush_observations(api_key=..., timeout_s=3.0)  # before Lambda return
```

See [`docs/demo/data-path-one-pager.md`](https://github.com/vaadhlabs/tensorcost/blob/main/docs/demo/data-path-one-pager.md) for the provider matrix.

Exceptions:

- Missing `TENSORCOST_API_KEY` → `MissingConfigError` at `wrap()` time
- Missing `proxy_url` when `max_layer` is `steer` or `route` → `MissingConfigError`
- Govern **refusals** (403 from `/admit`) → call blocked by design
- Steer **run-budget** / **period-budget** refusals → not fail-opened

## What gets sent

Metadata only — never prompt or completion content:

- SDK version, provider, model, operation, modality
- Timestamps, token counts, correlation UUID, status
- Environment, agent/workflow ids, chargeback tags when configured
- `decision_action` / `decision_reason` when `x-tc-decision` is present

Proxied requests include `x-tc-sdk-version`, `x-tc-sdk-capabilities`, and `x-tc-max-layer`.

Authentication uses a short-lived JWT from `POST /api/inference-proxy/sdk-token/exchange`.

## Supported providers & operations

| Provider | Package | Layers | Operations |
|---|---|---|---|
| OpenAI | `openai` ≥ 1.0 | observe → route | chat, completions, embeddings, responses (+ streaming) |
| Anthropic | `anthropic` ≥ 0.20 | observe → route | messages (+ streaming) |
| Azure OpenAI | `openai` (`AzureOpenAI`) | observe → route | same as OpenAI (`provider: openai`) |
| AWS Bedrock | `boto3` bedrock-runtime | observe, govern only | invoke, converse (+ streaming placeholders) |
| Google Vertex | `google-genai` (`vertexai=True`) | observe, govern only | generate_content (+ streaming) |

**Bedrock and Vertex:** signing happens inside the AWS / Google client before the proxy can intercept — **`max_layer="steer"` or `"route"` raises `MissingConfigError`**.

```python
import boto3
from tensorcost import wrap

bedrock = wrap(
    boto3.client("bedrock-runtime", region_name="us-east-1"),
    api_key="tc_live_...",
    max_layer="govern",  # steer/route not supported
)
```

## Node vs Python parity (1.1)

| Feature | Node (`@tensorcost/sdk`) | Python (`tensorcost`) |
|---|---|---|
| Control layers | ✓ | ✓ |
| Govern `/admit` | ✓ | ✓ |
| SDK layer fetch (250ms cold cap + SWR) | ✓ | ✓ |
| `prewarm()` / shared runtime cache | ✓ | ✓ |
| Proxy headers timeout (fail-open) | `headersTimeoutMs` | `headers_timeout_s` |
| Framework adapters | `wrapLangChain`, etc. | — |
| `maxGrants` / grant intersection | ✓ | not yet |
| `deployment` / `promptTemplateId` | ✓ | not yet |

## Development

```bash
pip install -e '.[dev]'
pytest
```

See [PUBLISHING.md](PUBLISHING.md) for the PyPI release runbook.

## Further reading

- [ADR-0026 — SDK control-layer ceilings](https://github.com/vaadhlabs/tensorcost/blob/main/docs/adr/0026-sdk-control-layer-ceilings.md)
- [Inference proxy SDK feature doc](https://github.com/vaadhlabs/tensorcost/blob/main/docs/features/inference-proxy-sdk.md)
- [Applied mode / live routing](https://github.com/vaadhlabs/tensorcost/blob/main/docs/features/inference-proxy-applied-mode.md)
- [CHANGELOG](./CHANGELOG.md)

## License

Apache-2.0. See [LICENSE](LICENSE) for the full text.
