# Changelog

All notable changes to the `tensorcost` Python SDK are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## 1.5.0 — 2026-10-26

**1.4.0 could route around a hard spend cap when fail-open was on (the
default); upgrade.** When a money budget's hard cap refused a call (403
with `code: "BUDGET_HARD_CAP"`), 1.4.0 read the refusal as a provider error,
and with `fail_open_enabled` on it sent the call to the provider anyway. The
cap was enforced by the proxy and then skipped by the SDK.

This is a minor release, not a patch: it adds public error classes and
makes `wrap()` raise at startup without a tenant id. Read the upgrade notes.

### Upgrading from 1.4.x

- **`wrap()` requires a tenant id at `govern`, `steer` and `route`.** The
  token exchange needs it, so without one the SDK never talked to TensorCost
  at all: no `/v1/admit`, no proxy call, no observation, and every call went
  straight to the provider while the README called the variable optional.
  `wrap()` now raises `MissingConfigError` at those layers when neither
  `tenant_id` nor `TENSORCOST_TENANT_ID` is set, whatever `fail_open_enabled` says (1.4.0 skipped
  TensorCost silently, or with `fail_open_enabled=False` in route raised
  `TensorCostProxyError` on every call). Find the id in the TensorCost
  console under **Developers › SDK setup**, in the `TENSORCOST_TENANT_ID`
  line of the env block. `observe` without one keeps working and logs one
  warning that nothing is being sent.
- **Unknown refusals are terminal.** A refusal code this version doesn't
  know stops the call as `TensorCostRefusedError`; it no longer falls back
  to the provider. Details below.
- **A revoked key fails closed.** A TensorCost key the token exchange,
  `/v1/admit` or the proxy rejects is `TensorCostKeyRejectedError`, and the
  call isn't sent. 1.4.0 sent it to the provider anyway when fail-open was on.

### Changed: an unknown refusal is now terminal

1.4.0 decided whether a failed proxy call could go straight to the provider
with a list of refusals that must not. Any refusal missing from that list,
including ones added on the server later, fell back. 1.5.0 turns the rule
around: a call goes around TensorCost only when TensorCost is unavailable.

- Falls back (with `fail_open_enabled` on): network errors and connection
  refused (`TensorCostNetworkError`), timeouts (`TensorCostTimeoutError`),
  5xx from the proxy or an open circuit (`TensorCostProxyError`), and a 429
  or 402 from the proxy with no refusal body (`TensorCostQuotaError`; the
  429 is TensorCost's rate limit, the 402 the plan's monthly request
  ceiling). The same outages on the token exchange count too.
- Never falls back: any 4xx carrying a TensorCost refusal, whatever the
  code. A code this version doesn't know arrives as `TensorCostRefusedError`
  with `code`, `reason` and the server's message intact. A refusal is never
  retried, even on a 429.
- A provider's own 4xx relayed by the proxy (an OpenAI 400, say) doesn't
  fall back either. It is raised as the provider SDK's own error, built by
  the client itself (`client._make_status_error_from_response(response)`,
  which is how openai-python and anthropic-python build `BadRequestError`,
  `AuthenticationError` and the rest; neither has a public equivalent), with
  the original `TensorCostProviderError` on its `tensorcost` attribute. A
  client that can't build one gets the `TensorCostProviderError` (`status`,
  `body`, `headers`). Tested against openai 2.32.0 and anthropic 0.96.0.
- A 4xx that isn't JSON (a firewall's HTML page, an empty body) is
  `TensorCostGatewayRejectedError`: the request was turned away before it
  reached TensorCost or the provider. Terminal.
- The proxy's own NestJS error body (`{ statusCode, message }`) is
  TensorCost's, not the provider's: a 401 there is
  `TensorCostKeyRejectedError`, anything else a `TensorCostRefusedError`
  with the message.

A refusal is any 4xx whose body has an upper-case code at `error.code` or a
top-level `code` (`{"error": {"code": "SOME_CODE", ...}}` or
`{"code": "SOME_CODE", ...}`), or whose `x-tc-decision` header says
`action=refuse`. Lower-case codes (OpenAI's `invalid_api_key` and the like)
belong to the provider.

The govern layer follows the same rule: `/v1/admit` fails open only on an
outage, and only while `fail_open_enabled` is on. Before, admit failed open on
everything except a 403, whatever `fail_open_enabled` said.

### Added

**`TensorCostRefusedError`**, the base class of every refusal:
`TensorCostRunBudgetExceededError`, `TensorCostRunStoppedError`,
`TensorCostPeriodBudgetExceededError`, `TensorCostModelGovernanceDeniedError`,
`TensorCostGuardrailHardStopError`, `TensorCostComplianceDeniedError` and
`TensorCostComplianceTeamMismatchError` all extend it now, and all carry
`code`, `reason` and `detail` (the refusal object the server sent). Catch
`TensorCostRefusedError` to handle every refusal, current and future.

**`TensorCostBudgetHardCapError`** for a money budget's hard cap. It extends
`TensorCostPeriodBudgetExceededError`, so an `except` written against 1.4.0 for
the current wire shape still matches. Fields: `budget_id`, `budget_name`,
`burn_pct`, `scope`, `period`, `cap_cents`, `spent_cents`, `reason`,
`budget_code` (always `"BUDGET_HARD_CAP"`), `retryable` and
`retry_after_seconds`. `retryable` is true when the budget's spend hasn't been
counted yet (`reason: "spend_not_known_yet"` or `retryable: true` on the
wire), usually for a few seconds after a budget is created or its scope
changes, and the message then says "the budget's spend hasn't been counted
yet; retry in 5 s". Otherwise it says "the budget's hard cap is reached". The
SDK doesn't retry for you. The shapes it accepts, on the chat path and on
`/v1/admit`:

```json
{ "error": { "code": "PERIOD_BUDGET_EXCEEDED", "budget_code": "BUDGET_HARD_CAP",
             "budget_id": "b-1", "budget_name": "Prod AI", "burn_pct": 1.02,
             "scope": "budget:b-1", "period": "monthly",
             "cap_cents": 10000, "spent_cents": 10200, "retryable": false,
             "message": "..." } }

{ "error": { "code": "PERIOD_BUDGET_EXCEEDED", "budget_code": "BUDGET_HARD_CAP",
             "reason": "spend_not_known_yet", "retryable": true,
             "retry_after_seconds": 5, "budget_id": "b-2", ... } }
  + header  Retry-After: 5

{ "error": { "code": "BUDGET_HARD_CAP", "budget_id": "b1", "burn_pct": 0,
             "reason": "spend_not_known_yet", "message": "..." } }   (older servers)
```

`retry_after_seconds` reads the `Retry-After` header (seconds or an HTTP date)
first, then `retry_after_seconds` or `retry_after` in the body. A
`PERIOD_BUDGET_EXCEEDED` without `budget_code` is still a plain
`TensorCostPeriodBudgetExceededError`.

**`TensorCostGatewayRejectedError`** (non-JSON 4xx, above) and
**`TensorCostKeyRejectedError`** (the token exchange, `/v1/admit` or the
proxy rejected the TensorCost key: "the TensorCost key was rejected; check
it's current"). Both are terminal. 1.4.0 raised `httpx.HTTPStatusError` for a revoked
key, and with fail-open on, the call then went to the provider.

### Fixed

**Anthropic clients run in govern mode.** The proxy has no Anthropic Messages
route yet, so steer/route for an Anthropic client could only fail open around
TensorCost. `max_layer="steer"` or `"route"` (or `applied_mode=True`) on an
Anthropic client now runs as `"govern"`: every call is checked with
`/v1/admit` before it goes to Anthropic. One warning per process says so;
set `max_layer="govern"` to silence it.

**Bedrock and Vertex say govern isn't enforced.** There is no `/v1/admit`
check for them yet, so `max_layer="govern"` only observes. The SDK now logs
one warning per process saying so, and the README no longer lists govern as
supported for them.

**Govern-layer refusals are typed by their code.** 1.4.0 threw
`TensorCostModelGovernanceDeniedError` ("request refused at govern layer")
for every `/v1/admit` refusal except a run stop, budget caps included.

**Streaming proxy errors.** A 5xx or 429 on the streaming path was reported
as `TensorCostProviderError`, and a connection failure as a raw `httpx`
exception; they are `TensorCostProxyError` / `TensorCostQuotaError` /
`TensorCostNetworkError` / `TensorCostTimeoutError` now.

### Not in 1.5.0: `application` and `tags` on observations

The server's observation endpoint reads `application` and `tags`, and the
SDK sends them only as `x-tc-application` / `x-tc-tags` on proxied calls.
Adding them to observations is built but switched off: byoc-v0.1.40
validates observations strictly and doesn't know the fields, so it would
reject every observation that carried them. They come in a later release, alongside the server release that accepts
them.

### Compatibility

- Server: no minimum version. 1.5.0 reads both the money-budget shape the
  proxy sends since the change that added `budget_code` and the older
  `code: "BUDGET_HARD_CAP"` shape. Nothing new is sent to the server, so
  byoc-v0.1.40 keeps accepting every request and observation.
- `wrap()` at `govern`, `steer` or `route` without a tenant id throws
  `MissingConfigError` (see above). Set `TENSORCOST_TENANT_ID`.
- `should_fall_back_to_provider()` returns False for `TensorCostProviderError`
  and for errors that aren't `TensorCost*` outages (1.4.0 returned True).
- A relayed provider 4xx: with fail-open on, 1.4.0 sent the call to the
  provider a second time and you got the provider SDK's error from that
  second call; with fail-open off you got `TensorCostProxyError`. 1.5.0
  raises the same provider error type, built from the relayed response,
  after one call.
- A TensorCost 4xx that isn't an outage (a rejected key, a refusal with an
  unknown code, a firewall page) now stops the call instead of going around
  it.
- `fail_open_enabled=False` now also fails the call when `/v1/admit` is down.
- `admit_request()` raises every refusal as its typed error; it used to
  return `(False, header)` for anything but `RUN_STOPPED`, and the wrappers
  then raised `TensorCostProviderError`. With `fail_open=False` it raises the
  typed outage error instead of `httpx.HTTPStatusError`.
- `TensorCostModelGovernanceDeniedError`, `TensorCostGuardrailHardStopError`,
  `TensorCostRefusedError`, `TensorCostBudgetHardCapError`,
  `TensorCostGatewayRejectedError` and `TensorCostKeyRejectedError` are now
  exported from `tensorcost`.
- Anthropic clients configured for steer/route make one `/v1/admit` call per
  request instead of a proxy call.

## 1.4.0 — 2026-10-25

### Added

**`TensorCostRunStoppedError`** — maps proxy `403` / `RUN_STOPPED` (an operator
stopped the agent run), on the applied-mode chat path and on the govern-layer
`/v1/admit`. Carries `workflow_id`, `stopped_by`, `stopped_at` and `reason`.
Terminal: `should_fall_back_to_provider()` returns False for it, so with
`fail_open_enabled` (the default) a stopped run no longer goes to the
provider directly. Before this, the 403 surfaced as a generic provider error
and fail-open called the provider anyway.

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
