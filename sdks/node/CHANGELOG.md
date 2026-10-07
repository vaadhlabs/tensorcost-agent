# Changelog

## Unreleased

## 1.3.0 — 2026-09-26

Minor release: period-budget refusals and agent-run span linking so proxied
sub-agent spend rolls up without double-counting. **Upgrade from 1.2.x** if
you enforce daily/monthly agent or team caps, or attribute spend under
**Router · Agent runs**.

### Added

**`TensorCostPeriodBudgetExceededError`** — maps proxy `403` /
`PERIOD_BUDGET_EXCEEDED` (daily or monthly agent/team/tenant caps). Terminal:
must not fail-open to the provider or the cap becomes a suggestion.

**Agent-run span linking** — observations may carry `trace_id` and
`parent_span_id` so ai-service inserts the model_call under the agent trace.
When `@tensorcost/agent-sdk` (or equivalent) publishes a model-call link,
proxied child spend is subtracted from sub-agent totals in run rollups.

**`agentId` / `workflowId` attribution** — unchanged API; documented more
clearly for Router · Agent runs and period-budget scopes.

### Changed

Terminal proxy refusal set now includes `PERIOD_BUDGET_EXCEEDED` alongside
run-budget, governance, guardrail, and compliance codes (streaming and
non-streaming).

## 1.2.1 — 2026-09-24

Patch release: applied-mode proxy refusal handling and streaming hardening
shipped after 1.2.0 was published. **Upgrade from 1.2.0** if you use
compliance profiles, guardrails, run budgets, or streaming through the proxy.

### Fixed

**Terminal proxy refusals (streaming)** — map all five 403 refusal codes
(`RUN_BUDGET_EXCEEDED`, `MODEL_GOVERNANCE_DENIED`, `GUARDRAIL_HARD_STOP`,
`COMPLIANCE_DENIED`, `COMPLIANCE_TEAM_MISMATCH`) to typed errors instead of
falling through to the direct provider when fail-open is on.

**Compliance on streaming** — `COMPLIANCE_DENIED` and
`COMPLIANCE_TEAM_MISMATCH` were missing from the streaming path in 1.2.0.

**Stream idle timeout** — `idleTimeoutMs` on hardening opts; wraps the proxy
stream iterator so wedged upstream bodies fail-open instead of hanging.

**`TensorCostRunBudgetExceededError`** — constructor args aligned with the
proxy refusal payload.

**DLP snapshot parser** — narrow `DlpAction` typing in compliance profile
deserialization.

## 1.2.0 — 2026-09-22

SDK compliance profiles: tenant floor + team overlays, in-process DLP, and
observation stamps. Requires Enterprise `feature.compliance_profiles` and
console assignments under **Policy → Compliance profiles**.

### Added

**Compliance snapshot** — `GET /v1/sdk-layer` returns compiled `compliance` alongside
`published_layer`; SDK refreshes with existing sdk-layer cache rules.

**Team scoping** — `wrap({ teamId })` or `TENSORCOST_TEAM_ID` narrows the effective
profile to a team overlay; team-bound SDK tokens must match or admit returns 403
`compliance_team_mismatch`.

**In-process DLP** — local pattern detectors on OpenAI and Anthropic paths at
govern+; matched secrets never sent to TensorCost or the provider when configured
to refuse.

**Compliance errors** — `TensorCostComplianceDeniedError`,
`TensorCostComplianceTeamMismatchError`.

**Observation stamps** — `compliance_frameworks`, `dlp_action` on telemetry (never
matched secret text).

See [SDK compliance profiles](https://github.com/vaadhlabs/tensorcost/blob/main/docs/features/sdk-compliance.md).

## 1.1.1 — 2026-09-19

Enterprise architect-hour fixes: fail-open correctness, honest streaming stance,
observation flush for Lambda.

### Fixed

**Circuit HALF_OPEN probe** — successful probe no longer triggers a second direct
provider call (`proxyFailed || circuitWasOpen` → `proxyFailed` only).

**Buffered JSON headers timeout** — non-streaming steer/route waits up to
`timeoutMs` for proxy headers (avoids fail-open double-bill while proxy buffers
upstream).

**Single-flight HALF_OPEN probe** — concurrent OPEN callers bypass until the
probe completes.

### Added

**`flushObservations()`** — best-effort drain of fire-and-forget observation POSTs
before Lambda freeze.

**`stream: true` through steer proxy** — refused (direct provider when fail-open
enabled; `TensorCostConfigError` when fail-open disabled).

## 1.1.0 — 2026-09-19

Agent hot-path hardening: fail-open faster on wedged proxy connections, optional
cold-start warmup, and non-blocking sdk-layer refresh.

### Added

**`prewarm()`** — prefetch JWT + console published layer at Lambda/container
init. Shares the same process-wide runtime cache as `wrap()`.

**`headersTimeoutMs`** (default 2s) — separate deadline for proxy response
headers vs body read (`timeoutMs`, default 60s). Missing headers within the
budget fail-open to the direct provider without retry.

### Changed

**Sdk-layer fetch** — 250ms cap on cold `GET /sdk-layer`; stale-while-revalidate
after the 30s TTL so refresh never blocks a model call.

**Admit** — HTTP timeout tightened to 500ms (still fail-open on errors).

**`wrap()`** — background-warms the shared runtime on first use.

### Fixed

Body JSON parse no longer leaves an unhandled rejection when the body-read
deadline fires before `response.json()` settles.

## 1.0.0 — 2026-09-13

Major release: control-layer ceilings replace the `appliedMode` boolean.
Effective layer is always `min(code.maxLayer, console published layer)`.

### Added

**Control layers (`maxLayer`)** — `off`, `observe`, `govern`, `steer`, `route`.
At `govern`, metadata-only `POST /admit` may refuse before the provider is
called. SDK fetches console ceiling via `GET /sdk-layer` (30s cache).

**Wire headers** — `x-tc-sdk-version`, `x-tc-sdk-capabilities`, `x-tc-max-layer`
on proxied requests; parses `x-tc-decision` on responses.

**Framework adapters** — `wrapLangChain`, `wrapLlamaIndex`, `wrapCrewAI`,
`wrapVercelAi`.

**Google Vertex AI (observe-only)** — `@google/genai` clients with
`vertexai: true`; `models.generateContent` / streaming variants.

**Grant intersection** — `maxGrants` caps console data grants from code;
prompt cache writes gated on `retain_completions`.

### Changed

**Breaking:** `appliedMode` is deprecated — use `maxLayer: "route"`. Default
proxy header `x-tc-max-layer` is now `observe` (was `route`).

### Fixed

Circuit breaker re-probes proxy after fail-open; cold-start layer cache no
longer skips decay window.

## 0.7.0 — 2026-08-22

A-04 chargeback attribution — stamp end-customer cohort and product feature on
every observation and forward them through applied-mode proxy headers.

### Added

**`customer` / `feature` options** — optional chargeback dimensions (max 128
chars each). Set at wrap time via `wrap({ customer, feature })` or
`TENSORCOST_CUSTOMER` / `TENSORCOST_FEATURE` env vars. Per-call overrides via
`.withMeta({ customer, feature })` on OpenAI and Anthropic clients. Values land
as top-level `customer` / `feature` fields on observations (not inside
`metadata`) and as `x-tc-customer` / `x-tc-feature` in applied mode.

Bedrock and Vertex clients support wrap-time defaults only (no `withMeta()`).

## 0.6.0 — 2026-05-28

SDK metadata story — lets you group traces by prompt template or deployment in
the TensorCost dashboard.

### Added

**`deployment` option** — stamp a git SHA, CI run id, or release label on every
observation emitted by a wrapped client. The backend groups by this value when
you select "Group by deployment" in the Traces view.

```ts
const client = wrap(new OpenAI({ apiKey: "..." }), {
  apiKey: process.env.TENSORCOST_API_KEY,
  deployment: process.env.GIT_SHA ?? "local",
});
```

Falls back to `TENSORCOST_DEPLOYMENT` env var. When absent, no deployment field
is written and the group-by row shows as "(unset)".

**`promptTemplateId` option** — per-client default template marker. Useful when
one SDK client always drives the same call-site template.

```ts
const client = wrap(new Anthropic({ apiKey: "..." }), {
  apiKey: process.env.TENSORCOST_API_KEY,
  promptTemplateId: "onboarding-email-v2",
});
```

Falls back to `TENSORCOST_PROMPT_TEMPLATE_ID` env var.

**`client.withMeta(overrides)` helper** — per-call metadata override. Returns a
thin proxy of the wrapped client with the metadata overridden for that single
call. The wrap-time `deployment` default is still inherited unless you explicitly
pass a new `deployment` here; `promptTemplateId` is also overridable.

```ts
// One client, multiple templates:
const baseClient = wrap(new OpenAI({ apiKey: "..." }), {
  apiKey: process.env.TENSORCOST_API_KEY,
  deployment: process.env.GIT_SHA,
});

// Template override per call-site:
const result = await (baseClient as any)
  .withMeta({ promptTemplateId: "personalization-rerank-v3" })
  .chat.completions.create({ model: "gpt-4o", messages });
```

Both fields are stored as `metadata.deployment` and `metadata.prompt_template_id`
in the `trace_span` row. The backend's `GET /api/ai/traces?group_by=deployment`
and `group_by=prompt_template` endpoints index these keys.

### No breaking changes

All new options are optional. Existing `wrap()` calls without metadata options
are completely unaffected — the metadata field is absent from the observation
envelope when nothing is configured.

## 0.5.0 — 2026-05-13

Phase A4 batch 5 (environment + connectionId GA). See 0.2.1 for the original
`connectionId` addition; this entry is a placeholder for any 0.5.x patches.

## 0.4.0 — 2026-05-13

Applied-mode hardening. Five new subsystems sit between `wrap()` and the
proxy POST; all are opt-in with safe defaults and zero impact on the
observe-only path.

### Added

**Typed error hierarchy** — every failure from the proxy path is a subclass of
`TensorCostError`. Customer code can `instanceof`-branch without importing raw
HTTP exception types.

- `TensorCostError` (base)
- `TensorCostNetworkError` — could not reach the proxy at all
- `TensorCostTimeoutError` — wall-clock or idle timeout fired; carries `timeoutKind: "total" | "idle"`
- `TensorCostProxyError` — proxy returned 5xx
- `TensorCostQuotaError` — proxy returned 429; carries `retryAfterMs`
- `TensorCostProviderError` — proxy forwarded a 4xx from the upstream AI provider

All carry `status`, `requestId`, `attempt`, and `rootCause` (avoiding the clash
with ES2022 `Error.cause`).

**Retries with jittered exponential backoff** — configurable via `RetryConfig`:

```ts
const client = wrap(new OpenAI(...), {
  retry: { maxAttempts: 5, baseDelayMs: 500, maxDelayMs: 30_000 },
});
```

Retry rules: 5xx + network errors + 429 always retry; 4xx (except 429) never
retry; 429 with a `Retry-After` header uses that value, capped at `maxDelayMs`.

**Request timeouts** — non-streaming proxy calls have a 60-second default total
timeout, controllable via `timeoutMs`. Surfaces as `TensorCostTimeoutError`.

```ts
const client = wrap(new OpenAI(...), { timeoutMs: 30_000 });
```

**Lifecycle telemetry hooks** — optional `onLifecycleEvent` callback fires on
each stage. Events contain only request metadata — never prompt content or
credentials:

```ts
const client = wrap(new OpenAI(...), {
  onLifecycleEvent(event) {
    console.log(event.kind, event.model, event.attemptNumber);
  },
});
```

Five event kinds: `before_request`, `after_response`, `on_retry`, `on_error`,
`on_fallback`. Errors in the callback are swallowed.

**Fail-open circuit breaker** — after 3 consecutive proxy 5xx responses the
circuit opens and all requests route directly to the provider. Closes again
after 5 consecutive probe successes. Controlled by `failOpenEnabled` (default
`true`):

```ts
// In-VPC: no direct provider access — raise on proxy failure instead of
// falling back.
const client = wrap(new OpenAI(...), { failOpenEnabled: false });
```

When `failOpenEnabled: true` (default) and the circuit is open, an
`on_fallback` lifecycle event fires before the direct-provider call.

### No breaking changes

All new options default to values that preserve 0.3.x behaviour.
`appliedMode` still defaults to `false`; the observe-only path is unchanged.

## 0.3.0 — 2026-05-12

- Added: `appliedMode` config flag. When true and `proxyUrl` (or `TENSORCOST_PROXY_URL`) is set, requests route through the TensorCost inference-proxy. See `docs/features/inference-proxy-applied-mode.md`.
- Added: `TensorCostConfigError` — thrown at `wrap()` time when `appliedMode=true` and no proxy URL is configured.
- AWS Bedrock clients throw immediately at `wrap()` time when `appliedMode=true` (v1 limitation — SigV4 signing is not interceptable in the same pattern; observe-only mode still works).
- No breaking changes to the 0.2.x observe-only API.

## 0.2.1 — 2026-05-05

Phase A4 batch 4 — `connectionId` support. Stamps `connection_id` on every observation envelope when the SDK is initialised with `wrap(client, { connectionId })` or `TENSORCOST_CONNECTION_ID`. The backend uses this as a fallback for environment resolution when the `environment` field is absent.

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

- `wrap(client, options?)` one-line entry point.
- OpenAI provider — wraps `chat.completions.create` and `completions.create`.
- Anthropic provider — wraps `messages.create`.
- Fire-and-forget HTTP transport over native `fetch` (Node 18+).
- Short-lived JWT exchange against `POST /api/inference-proxy/sdk-token/exchange` (the 0.1.0 paste-up said `/api/proxy/...`; corrected to the real route in 0.1.1), cached + auto-refreshed.
- Bounded queue (1024) with drop-on-saturation backpressure.
- Fail-open by default. The single exception: missing `apiKey` throws `MissingConfigError` at `wrap()` time.
- Zero runtime dependencies.
- TypeScript-first; ships with `.d.ts`.
