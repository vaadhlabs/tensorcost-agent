# Changelog

All notable changes to the `tensorcost` Python SDK are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## Unreleased

## 1.3.0 — 2026-09-26

Minor release (parity with Node 1.3.0): period-budget refusals and agent-run
span linking. **Upgrade from 1.2.x** if you enforce daily/monthly agent or
team caps, or attribute spend under **Router · Agent runs**.

### Added

**`TensorCostPeriodBudgetExceededError`** — maps proxy `403` /
`PERIOD_BUDGET_EXCEEDED`. Terminal — `should_fall_back_to_provider` returns
`False`.

**Agent-run span linking** — observations may carry `trace_id` and
`parent_span_id` for model_call nesting under agent traces (rollup deducts
proxied child spend).

**Public exports** — `TensorCostRunBudgetExceededError` and
`TensorCostPeriodBudgetExceededError` are exported from `tensorcost`.

### Changed

Terminal proxy refusal handling includes `PERIOD_BUDGET_EXCEEDED` on
streaming and non-streaming applied-mode paths.

## 1.2.1 — 2026-09-24

Patch release (parity with Node 1.2.1). **Upgrade from 1.2.0** if you use
compliance profiles, guardrails, run budgets, or applied-mode proxy calls.

### Fixed

**Terminal proxy refusals (streaming + non-streaming)** — shared
`_terminal_proxy_refusal_error()` maps all five 403 refusal codes to typed
errors; non-streaming compliance refusals no longer fail-open to the provider.

**Streaming lifecycle** — keep the httpx stream context open until the proxy
response iterator completes (fixes `httpx.StreamClosed` on streaming calls).

**Fail-open guard** — `should_fall_back_to_provider` treats compliance denial
errors as terminal (no direct-provider bypass).

## 1.2.0 — 2026-09-22

SDK compliance profiles (parity with Node 1.2.0). Requires Enterprise
`feature.compliance_profiles` and console assignments under **Policy → Compliance
profiles**.

### Added

**Compliance snapshot** — compiled profile from `GET /v1/sdk-layer`.

**Team scoping** — `wrap(..., team_id=...)` or `TENSORCOST_TEAM_ID`; team-bound
tokens must match.

**In-process DLP** — OpenAI and Anthropic at govern+; local refuse/redact before
provider.

**Compliance errors** — `TensorCostComplianceDeniedError`,
`TensorCostComplianceTeamMismatchError` (exported from `tensorcost`).

**Observation stamps** — `compliance_frameworks`, `dlp_action`.

See [SDK compliance profiles](https://github.com/vaadhlabs/tensorcost/blob/main/docs/features/sdk-compliance.md).

## 1.1.1 — 2026-09-19

Enterprise architect-hour fixes (parity with Node 1.1.1).

### Fixed

**Circuit HALF_OPEN probe** — successful probe no longer triggers a second direct
provider call.

**Buffered JSON headers timeout** — non-streaming steer/route uses full
`timeout_s` for TTFB.

**Single-flight HALF_OPEN probe** — concurrent OPEN callers bypass until probe
completes.

**Proxy headers wait** — removed global 8-worker thread pool; httpx stream on
caller thread.

### Added

**`flush_observations()`** — drain queued observations before Lambda freeze.

**`stream=True` through steer proxy** — refused (direct when fail-open enabled).

## 1.1.0 — 2026-09-19

Agent hot-path hardening: fail-open faster on wedged proxy connections, optional
cold-start warmup, and non-blocking sdk-layer refresh.

### Added

**`prewarm()`** — prefetch JWT + console published layer at Lambda/container
init. Shares the same process-wide runtime cache as `wrap()`.

**`headers_timeout_s`** (default 2.0) — separate deadline for proxy response
headers vs body read (`timeout_s`, default 60.0). Missing headers within the
budget fail-open to the direct provider without retry.

### Changed

**Sdk-layer fetch** — 250ms cap on cold `GET /sdk-layer`; stale-while-revalidate
after the 30s TTL so refresh never blocks a model call.

**Admit** — HTTP timeout tightened to 500ms (still fail-open on errors).

**`wrap()`** — background-warms the shared runtime on first use; accepts
`headers_timeout_s`.

### Fixed

Headers timeout no longer caps the model body read — body uses `timeout_s`
independently once response headers arrive.

## 1.0.0 — 2026-09-13

Major release: control-layer ceilings replace the `applied_mode` boolean.
Effective layer is always `min(code.max_layer, console published layer)`.

### Added

**Control layers (`max_layer`)** — `off`, `observe`, `govern`, `steer`, `route`.
At `govern`, metadata-only `POST /admit` may refuse before the provider is
called. SDK fetches console ceiling via `GET /sdk-layer` (30s cache).

**Wire headers** — `x-tc-sdk-version`, `x-tc-sdk-capabilities`, `x-tc-max-layer`
on proxied requests; parses `x-tc-decision` on responses.

**Bedrock + Vertex (observe-only)** — already in 0.5.x/0.6.x; now documented
alongside the layer ladder. Steer/route remain unsupported for signed transports.

### Changed

**Breaking:** `applied_mode=True` is deprecated — use `max_layer="route"`.
Default proxy header `x-tc-max-layer` is now `observe` (was `route`).

### Fixed

Circuit breaker re-probes proxy after fail-open; cold-start layer cache no
longer skips decay window.

## 0.6.0 — 2026-08-22

A-04 chargeback attribution — stamp end-customer cohort and product feature on
every observation and forward them through applied-mode proxy headers.

### Added

**`customer` / `feature` kwargs** — optional chargeback dimensions (max 128
chars each). Pass to ``wrap(customer=..., feature=...)`` or set
``TENSORCOST_CUSTOMER`` / ``TENSORCOST_FEATURE`` env vars. Values land as
top-level ``customer`` / ``feature`` fields on observations (not inside
metadata) and as ``x-tc-customer`` / ``x-tc-feature`` on applied-mode proxy
requests.

## 0.5.0 — 2026-08-04

### Added

**Google Vertex AI support (observe-only)** — `wrap()` now detects
``google.genai.Client`` instances constructed with ``vertexai=True`` and
monkey-patches ``models.generate_content`` and
``generate_content_stream`` to emit fire-and-forget observations with
``provider: "vertex"``. Token counts are read from
``usage_metadata.prompt_token_count`` / ``candidates_token_count``;
streaming calls emit a placeholder observation (null token counts).
Gemini Developer API clients (``vertexai=False``) are not detected.

Applied mode is not supported for Vertex — ``wrap(client,
applied_mode=True)`` raises ``MissingConfigError`` immediately.

**AWS Bedrock support (observe-only)** — `wrap()` now detects
`boto3.client("bedrock-runtime")` clients (via
`meta.service_model.service_name`) and monkey-patches `invoke_model`,
`converse`, `invoke_model_with_response_stream`, and `converse_stream`
to emit fire-and-forget observations. Token counts are extracted from
the Anthropic-on-Bedrock, Amazon Nova/Titan, and Converse API usage
shapes; streaming calls emit a placeholder observation (null token
counts) since per-chunk accounting isn't implemented yet. Mirrors the
existing Node SDK Bedrock wrapper.

Applied mode is not supported for Bedrock — `wrap(client,
applied_mode=True)` raises `MissingConfigError` immediately, since
Bedrock's SigV4-signed requests can't be intercepted at the proxy layer.

## 0.4.0 — 2026-05-13

Applied-mode hardening. Five new subsystems sit between `wrap()` and the proxy
POST; all are opt-in with safe defaults and zero impact on the observe-only
path.

### Added

**Typed error hierarchy** — every failure from the proxy path is a subclass of
`TensorCostError`. Customer code can `isinstance`-branch without importing raw
httpx exception types.

- `TensorCostError` (base)
- `TensorCostNetworkError` — could not reach the proxy at all
- `TensorCostTimeoutError` — wall-clock or idle timeout fired; carries `timeout_kind: str`
- `TensorCostProxyError` — proxy returned 5xx
- `TensorCostQuotaError` — proxy returned 429; carries `retry_after_ms`
- `TensorCostProviderError` — proxy forwarded a 4xx from the upstream AI provider

All carry `status`, `request_id`, `attempt`, and `root_cause` (avoiding the
clash with `Exception.__cause__`).

**Retries with jittered exponential backoff** — configurable via `RetryConfig`:

```python
from tensorcost import wrap, RetryConfig

client = wrap(
    OpenAI(api_key="sk-..."),
    retry=RetryConfig(max_attempts=5, base_delay_ms=500.0, max_delay_ms=30_000.0),
)
```

Retry rules: 5xx + network errors + 429 always retry; 4xx (except 429) never
retry; 429 with a `Retry-After` header uses that value, capped at `max_delay_ms`.

**Request timeouts** — non-streaming proxy calls have a 60-second default total
timeout, controllable via `timeout_s`. Surfaces as `TensorCostTimeoutError`.

```python
client = wrap(OpenAI(api_key="sk-..."), timeout_s=30.0)
```

**Lifecycle telemetry hooks** — optional `on_lifecycle_event` callback fires on
each stage. Events contain only request metadata — never prompt content or
credentials:

```python
from tensorcost import wrap, BeforeRequestEvent, AfterResponseEvent

def my_callback(event):
    print(event.kind, event.model, event.attempt_number)

client = wrap(OpenAI(api_key="sk-..."), on_lifecycle_event=my_callback)
```

Five event kinds: `before_request`, `after_response`, `on_retry`, `on_error`,
`on_fallback`. Errors in the callback are swallowed.

**Fail-open circuit breaker** — after 3 consecutive proxy 5xx responses the
circuit opens and all requests route directly to the provider. Closes again
after 5 consecutive probe successes. Controlled by `fail_open_enabled` (default
`True`):

```python
# In-VPC: no direct provider access — raise on proxy failure instead of
# falling back.
client = wrap(OpenAI(api_key="sk-..."), fail_open_enabled=False)
```

When `fail_open_enabled=True` (default) and the circuit is open, an
`on_fallback` lifecycle event fires before the direct-provider call.

### No breaking changes

All new options default to values that preserve 0.3.x behaviour.
`applied_mode` still defaults to `False`; the observe-only path is unchanged.

## 0.3.1 — 2026-05-12

- Changed: Anthropic provider sends native Anthropic body shape to the proxy (was: OpenAI-translated). Wire-shape translation now lives in the proxy (`inference-proxy-service`); both SDKs send the original upstream's shape with the base URL in `x-tc-provider-url`.
- No customer-visible API change.

## 0.3.0 — 2026-05-12

### Added

- `applied_mode` and `proxy_url` config options on `wrap()`. When `applied_mode=True`,
  `chat.completions.create` (OpenAI) and `messages.create` (Anthropic) route through
  the TensorCost inference-proxy at `{proxy_url}/api/inference-proxy/v1/chat/completions`.
  The proxy decides per-request whether to route or pass through; the SDK doesn't
  participate in that decision.
- `proxy_url` falls back to `$TENSORCOST_PROXY_URL`. If `applied_mode=True` and no
  proxy URL is available from either source, `MissingConfigError` is raised at `wrap()`
  time — not at first call.
- Two custom headers set on every proxied request: `x-tc-provider-url` (original
  upstream base URL) and `x-tc-provider-auth` (auth value for that provider — Bearer
  for OpenAI, x-api-key for Anthropic).
- Fail-open on proxy errors: if the proxy call fails for any reason, the SDK falls
  back to the direct provider call and logs a warning. The customer's call always
  completes.
- Observe-only telemetry (`POST /api/inference-proxy/observation`) still fires on
  every call regardless of applied-mode state.
- `TENSORCOST_PROXY_URL` added to the env-scrub fixture in the test suite.

### No breaking changes

The 0.2.x observe-only API is unchanged. `applied_mode` defaults to `False`.

See `docs/features/inference-proxy-applied-mode.md` for the proxy design.

## 0.2.1 — 2026-05-07

### Added

- `environment` and `connection_id` config options on `wrap()` (Phase A4 batches 3 + 4).
  Both are optional and fall back to `$TENSORCOST_ENVIRONMENT` / `$TENSORCOST_CONNECTION_ID`.
  When set, they're stamped on every observation envelope.

## 0.1.1 — 2026-05-05

### Fixed
- Observation transport now POSTs to `/api/inference-proxy/observation`
  (and exchanges tokens at `/api/inference-proxy/sdk-token/exchange`),
  matching the inference-proxy-service controller mount per the
  `/api/<svc>/<feature>` routing convention. The 0.1.0 paths
  (`/api/proxy/...`) returned 404 against every TensorCost deployment,
  so customer-side observations never landed.

## 0.1.0 — 2026-05-05

Initial release.

### Added
- `wrap(client, *, api_key=None, base_url=None, tenant_id=None, fail_open=True)`
  — single public entry point. Returns the same client object with the
  appropriate `create` method monkey-patched.
- OpenAI provider support: `chat.completions.create` and
  `completions.create`.
- Anthropic provider support: `messages.create`.
- Fire-and-forget transport (`httpx` + bounded `ThreadPoolExecutor`)
  with back-pressure: drops oldest observations when saturated rather
  than blocking the caller.
- Short-lived JWT exchange via
  `POST /api/inference-proxy/sdk-token/exchange` (the 0.1.0 paste-up
  said `/api/proxy/...`; corrected to the real route in 0.1.1), cached
  in memory and refreshed before expiry.
- Fail-open by default: any failure in the observation pipeline is
  logged and swallowed; the customer's underlying provider call always
  completes.
- `py.typed` marker — fully type-checked.
- `MissingConfigError` is the single non-swallowed failure mode (the
  SDK was never configured).

### Not in this release (deferred)
- Streaming responses, tool / function calling, vision, async clients
  — all targeted for v1.0.
- AWS Bedrock, Azure OpenAI — targeted for v0.1.
- Google Vertex AI, LangChain / LlamaIndex adapters — targeted for v1.0.
