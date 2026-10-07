# @tensorcost/sdk

One-line wrapper for TensorCost — fire-and-forget LLM observability and optional inference control for OpenAI, Anthropic, AWS Bedrock, and Google Vertex AI clients in Node.js.

**Current version:** 1.3.0 — period budgets + agent-run span linking (upgrade from 1.2.x if you use daily/monthly caps or Router · Agent runs). See [CHANGELOG](./CHANGELOG.md), [SDK compliance](https://github.com/vaadhlabs/tensorcost/blob/main/docs/features/sdk-compliance.md), and [Migrating from 0.x](#migrating-from-0x).

## Install

```bash
npm install @tensorcost/sdk
# or
pnpm add @tensorcost/sdk
```

Requires Node.js 18+ (native `fetch`).

## Architecture diagrams

Engineering slides: [docs/diagrams/audiences/engineering.md](../../docs/diagrams/audiences/engineering.md).

<!-- tc-diagram:slide-engineering-flow -->
![SDK observe vs route request paths](../../docs/diagrams/exports/slide-engineering-flow.png)
<!-- /tc-diagram -->

## Quick start (observe)

The default layer is **`observe`**: telemetry only, provider call unchanged.

```ts
import OpenAI from "openai";
import { wrap } from "@tensorcost/sdk";

const client = wrap(new OpenAI({ apiKey: process.env.OPENAI_API_KEY }), {
  apiKey: process.env.TENSORCOST_API_KEY,
});

const resp = await client.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "hi" }],
});
```

Anthropic works the same way:

```ts
import Anthropic from "@anthropic-ai/sdk";
import { wrap } from "@tensorcost/sdk";

const client = wrap(new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY }), {
  apiKey: process.env.TENSORCOST_API_KEY,
});

await client.messages.create({
  model: "claude-3-5-sonnet-20241022",
  max_tokens: 256,
  messages: [{ role: "user", content: "hi" }],
});
```

Create a long-lived SDK key in the TensorCost console (**Setup → Keys**). The SDK exchanges it for a short-lived JWT via `POST /api/inference-proxy/sdk-token/exchange` and refreshes ~60s before expiry.

## Control layers (`maxLayer`)

v1.0 uses a single ladder for how much TensorCost may intervene on a call:

`off → observe → govern → steer → route`

A layer is a **ceiling**, not a mode switch. The effective layer is always:

```text
min(your maxLayer, console published layer)
```

When routing is paused in the console, the published ceiling is capped at `govern`.

| Layer | Provider path | Behavior |
|---|---|---|
| `off` | Direct | No SDK wiring; client returned unchanged |
| `observe` | Direct | Telemetry only (**default** when `maxLayer` is omitted) |
| `govern` | Direct | Metadata-only `POST /v1/admit` before the provider call; may refuse unsafe calls |
| `steer` | Via proxy | Proxy on the path; may steer without full live routing |
| `route` | Via proxy | Full applied-mode routing through inference-proxy |

```ts
const client = wrap(new OpenAI({ apiKey: process.env.OPENAI_API_KEY }), {
  apiKey: process.env.TENSORCOST_API_KEY,
  maxLayer: "route",
  proxyUrl: process.env.TENSORCOST_PROXY_URL, // required for steer | route
});
```

`wrap()` throws `TensorCostConfigError` when `maxLayer` is `steer` or `route` and no `proxyUrl` / `TENSORCOST_PROXY_URL` is set. **`govern` does not require `proxyUrl`** — admit uses `baseUrl`.

### Console-published ceiling

Operators set the tenant ceiling in the console (**Policy → Routing**) or via `PATCH /api/ai/router/layer` (`published_layer`, default `observe`). The SDK fetches the effective value from `GET /api/inference-proxy/v1/sdk-layer`:

- **30s cache** while the endpoint is reachable
- **250ms cap** on a cold fetch — a hung TensorCost cannot block the model call
- **Stale-while-revalidate** after TTL: serve the last snapshot immediately and refresh in the background
- **Decays to `govern`** after 5 minutes unreachable (fail-safe: still run admit, never silently upgrade to route)

Publishing `steer` or `route` in the console requires a **content-retention data grant** (`retain_completions`, etc.). Steer/route without a grant returns **409**.

### Govern admission

At `govern` and above, the SDK calls `POST /api/inference-proxy/v1/admit` with metadata only — **no prompt or completion bodies**:

- **204** + `x-tc-decision: layer=govern; action=allow` → proceed to the provider
- **403** + `x-tc-decision: layer=govern; action=refuse; reason=…` → call blocked (budget, model governance, etc.)

Admit is **fail-open** on transport errors: the provider call proceeds and an `admit_unavailable` observation is emitted when transport recovers. The admit HTTP call is capped at **500ms**.

### Compliance profiles (1.2.0+)

Enterprise tenants with `feature.compliance_profiles` assign frameworks (GDPR, HIPAA,
PCI-DSS, EU AI Act, residency) in **Policy → Compliance profiles**. The SDK loads
the compiled snapshot from `GET /v1/sdk-layer` and enforces it at govern and above.

- **`teamId`** on `wrap({ teamId: "…" })` or **`TENSORCOST_TEAM_ID`** — narrows to a
  team overlay; must match a team-bound SDK token when the key is scoped to a team.
- **In-process DLP** — pattern detectors run locally before the provider; matched
  secrets never leave your process or hit TensorCost.
- **Observations** stamp `compliance_frameworks` and `dlp_action` (never secret text).
- **`TensorCostComplianceDeniedError`** / **`TensorCostComplianceTeamMismatchError`**
  on refuse paths.

See [SDK compliance profiles](https://github.com/vaadhlabs/tensorcost/blob/main/docs/features/sdk-compliance.md).

### Route layer prerequisites

`maxLayer: "route"` is necessary but not sufficient for live provider swaps. ADR-0016 gates still apply on the proxy side: LaunchDarkly flag, `AI_ROUTING_LIVE_EXECUTION_PERMITTED`, non-`dry_run` policy with rollout, and routing not paused. When gated off, traffic pass-throughs to the original provider and the decision is recorded.

See [ADR-0026](https://github.com/vaadhlabs/tensorcost/blob/main/docs/adr/0026-sdk-control-layer-ceilings.md) and [inference-proxy applied mode](https://github.com/vaadhlabs/tensorcost/blob/main/docs/features/inference-proxy-applied-mode.md).

## Cold start (`prewarm`)

At Lambda/container init, the first steer/route call can otherwise pay for JWT exchange and sdk-layer fetch on the hot path. Call `prewarm()` once at module scope — it shares the same process-wide runtime that `wrap()` uses, so a later `wrap()` hits a warm cache.

`wrap()` also background-warms that shared runtime on first use; explicit `prewarm()` is recommended when you know traffic is about to arrive (e.g. before the handler is invoked).

```ts
import { prewarm, wrap } from "@tensorcost/sdk";

await prewarm({
  apiKey: process.env.TENSORCOST_API_KEY,
  baseUrl: process.env.TENSORCOST_BASE_URL,
  maxLayer: "route",
});

export const client = wrap(new OpenAI({ apiKey: process.env.OPENAI_API_KEY }), {
  apiKey: process.env.TENSORCOST_API_KEY,
  maxLayer: "route",
  proxyUrl: process.env.TENSORCOST_PROXY_URL,
});
```

`prewarm()` is best-effort: failures are swallowed and the first real call retries the fetches.

## Migrating from 0.x

| 0.x | 1.0 |
|---|---|
| `appliedMode: true` | `maxLayer: "route"` (deprecated alias still works with a console warning) |
| Default `x-tc-max-layer: route` on proxy calls | Default is now `observe` |
| Boolean on/off | Full layer ladder + console ceiling |

```ts
// Before (0.6.x)
wrap(client, { appliedMode: true, proxyUrl: "..." });

// After (1.0)
wrap(client, { maxLayer: "route", proxyUrl: "..." });
```

Recommended rollout: **`observe` → `govern` → `route`**, raising the console `published_layer` only after each stage is validated.

## Configuration

`wrap()` resolves options in order: explicit kwargs → environment variables → defaults.

| Variable | Purpose | Required |
|---|---|---|
| `TENSORCOST_API_KEY` | Long-lived SDK key from **Setup → Keys** | Yes |
| `TENSORCOST_BASE_URL` | Backend base URL (admit, sdk-layer, token exchange) | No (default `https://api.tensorcost.com`) |
| `TENSORCOST_PROXY_URL` | Inference-proxy base URL | Required when `maxLayer` is `steer` or `route` |
| `TENSORCOST_TENANT_ID` | Explicit tenant id (multi-tenant tests) | No |
| `TENSORCOST_ENVIRONMENT` | Environment tag on every observation | No |
| `TENSORCOST_CONNECTION_ID` | Provider-connection id for environment resolution | No |
| `TENSORCOST_AGENT_ID` / `TENSORCOST_WORKFLOW_ID` | Agent-run attribution defaults | No |
| `TENSORCOST_CUSTOMER` / `TENSORCOST_FEATURE` | Chargeback tag defaults | No |
| `TENSORCOST_DEPLOYMENT` | Git SHA / release label (`metadata.deployment`) | No |
| `TENSORCOST_PROMPT_TEMPLATE_ID` | Default prompt-template id | No |
| `TC_APPLICATION` / `TENSORCOST_APPLICATION` | Routing-policy application scope (proxy) | No |
| `TC_TAGS` / `TENSORCOST_TAGS` | Comma-separated routing tags (proxy) | No |

```ts
const client = wrap(new OpenAI(...), {
  apiKey: "tc_live_xxx",
  baseUrl: "https://api.tensorcost.com",
  maxLayer: "govern",
  deployment: process.env.GIT_SHA,
  promptTemplateId: "support-chat-v2",
  application: "customer-support",
  tags: ["tier:enterprise"],
});
```

## Proxy hardening

Applies when `maxLayer` is `govern`, `steer`, or `route` (admit + proxy paths).

### Retries

```ts
import { wrap, RetryConfig } from "@tensorcost/sdk";

const client = wrap(new OpenAI(...), {
  maxLayer: "route",
  proxyUrl: "...",
  retry: { maxAttempts: 5, baseDelayMs: 500, maxDelayMs: 30_000 },
});
```

5xx, network errors, and 429 are retried with jittered exponential backoff. `Retry-After` on 429 is honoured. 4xx (except 429) are never retried.

### Timeouts

Proxy timeouts are split so a wedged TCP connection fails open quickly while slow model bodies still get a full budget:

```ts
const client = wrap(new OpenAI(...), {
  maxLayer: "route",
  proxyUrl: "...",
  headersTimeoutMs: 2_000, // wait for response headers (default 2_000)
  timeoutMs: 60_000,       // non-streaming body read (default 60_000)
  idleTimeoutMs: 30_000,   // streaming idle (default 30_000)
});
```

| Option | Default | `TensorCostTimeoutError.kind` | On timeout (steer/route) |
|---|---|---|---|
| `headersTimeoutMs` | 2s | `"headers"` | Streaming only (when SSE client ships); **not used for buffered JSON** |
| `timeoutMs` | 60s | `"total"` | TTFB + body for non-streaming proxy calls; fail-open when `failOpenEnabled: true` |
| `idleTimeoutMs` | 30s | `"idle"` | Reserved for future steer SSE client |

For **non-streaming** steer/route, the SDK waits up to **`timeoutMs` for response headers** because the proxy may buffer the full upstream response before sending headers — using a 2s headers cap would fail-open and **double-bill** while the proxy still completes.

**Streaming (`stream: true`):** not implemented through the steer proxy client (1.1.1). With `failOpenEnabled: true` (default), calls go **direct** to the provider. With `failOpenEnabled: false`, `TensorCostConfigError` is thrown.

### Lifecycle hooks

```ts
import { wrap, LifecycleEvent } from "@tensorcost/sdk";

const client = wrap(new OpenAI(...), {
  maxLayer: "route",
  proxyUrl: "...",
  onLifecycleEvent(event: LifecycleEvent) {
    // "before_request" | "after_response" | "on_retry" | "on_error" | "on_fallback"
    console.log(event.kind, event.model, event.attemptNumber);
  },
});
```

Payloads contain request metadata only — never prompt content, response bodies, or credentials.

### Circuit breaker

After consecutive proxy 5xx failures (default 3), the circuit opens and requests route directly to the provider (fail-open). After consecutive successful probe calls (default 5), the circuit closes.

Set `failOpenEnabled: false` when direct provider access is unavailable (e.g. in-VPC sidecar-only). Proxy failures then surface as `TensorCostProxyError` instead of falling back.

HALF_OPEN recovery uses a **single-flight probe** (1.1.1) — a successful probe does **not** invoke the provider twice.

### Typed errors

```ts
import {
  TensorCostError,
  TensorCostNetworkError,
  TensorCostTimeoutError,
  TensorCostProxyError,
  TensorCostQuotaError,
  TensorCostProviderError,
  TensorCostRunBudgetExceededError,
  TensorCostPeriodBudgetExceededError,
  TensorCostModelGovernanceDeniedError,
  TensorCostGuardrailHardStopError,
} from "@tensorcost/sdk";
```

Govern refusals and proxy decisions are also reflected in the `x-tc-decision` header and attached to observations as `decision_action` / `decision_reason` metadata when present.

### Period budgets (1.3.0+)

Daily or monthly caps (tenant default, per-agent, or per-team) are configured in
the console under **Router · Agent runs**. On applied-mode (`steer` / `route`)
calls the proxy returns `403` with `code: "PERIOD_BUDGET_EXCEEDED"`. The SDK
throws `TensorCostPeriodBudgetExceededError` and **does not** fall back to the
provider.

```ts
import { TensorCostPeriodBudgetExceededError } from "@tensorcost/sdk";

try {
  await client.chat.completions.create({ /* … */ });
} catch (err) {
  if (err instanceof TensorCostPeriodBudgetExceededError) {
    // err.scope, err.period, err.capCents, err.spentCents
  }
  throw err;
}
```

Per-run caps still raise `TensorCostRunBudgetExceededError` (`RUN_BUDGET_EXCEEDED`).

## Data grants (`maxGrants`)

Code can cap which console data grants apply:

```ts
import { wrap, intersectGrants } from "@tensorcost/sdk";

const client = wrap(new OpenAI(...), {
  apiKey: "...",
  maxGrants: ["telemetry", "retain_completions"],
});
```

Effective grants = `intersect(maxGrants, console.data_grants)`. Omit `maxGrants` to allow all console grants. An empty array allows none. Grant vocabulary: `telemetry`, `retain_completions`, `retain_prompts`, `retain_tools`.

## Trace metadata & attribution

### Deployment & prompt templates

Group traces in the dashboard by deploy or template:

```ts
const client = wrap(new OpenAI(...), {
  apiKey: "...",
  deployment: process.env.GIT_SHA ?? "local",
  promptTemplateId: "onboarding-email-v2",
});

// Per-call override (OpenAI + Anthropic)
await client.withMeta({ promptTemplateId: "research-agent-v5" }).chat.completions.create({ ... });
```

### Agent runs

Tag spend for **Router · Agent runs**:

```ts
const client = wrap(new OpenAI(...), {
  apiKey: "...",
  agentId: "support-bot",
});

const scoped = client.withMeta({ workflowId: crypto.randomUUID() });
await scoped.chat.completions.create({ model: "gpt-4o-mini", messages: [...] });
```

`customer` / `feature` are separate chargeback dimensions — not substitutes for `agentId` / `workflowId`. Bedrock and Vertex accept wrap-time defaults only (no `withMeta()`).

**Span linking (1.3.0+):** when an agent framework wrapper (for example the
internal `@tensorcost/agent-sdk` link helpers) publishes a model-call link,
observations include `trace_id` and `parent_span_id`. The proxy and ai-service
nest the model_call under the agent trace so run rollups subtract proxied
child spend instead of double-counting sub-agent totals. Plain `wrap()` without
an agent link still attributes by `agentId` / `workflowId` alone.

## Fail-open guarantee

If TensorCost is slow, down, or misconfigured, your customer call still succeeds. Internal errors are swallowed and logged via `console.warn` (or your lifecycle hook). Exceptions:

- Missing `TENSORCOST_API_KEY` → `MissingConfigError` at `wrap()` time
- Missing `proxyUrl` when `maxLayer` is `steer` or `route` → `TensorCostConfigError` at `wrap()` time
- Govern **refusals** (403 from `/admit`) → call blocked by design
- Steer **run-budget** refusals → not fail-opened
- Steer **period-budget** refusals (`PERIOD_BUDGET_EXCEEDED`) → not fail-opened

### Duplicate spend (honest)

Fail-open may call the provider **after** a failed proxy attempt. If the proxy already forwarded to the provider before failing, you can be billed twice. Mitigations:

- **Sidecar / in-VPC steer:** set `failOpenEnabled: false` — no silent direct fallback
- **Buffered JSON:** SDK 1.1.1+ waits `timeoutMs` for headers (not 2s) to avoid aborting a healthy slow proxy call

Set `failOpen: false` only in tests to surface observation-pipeline errors.

### Lambda: flush observations

```ts
import { flushObservations } from "@tensorcost/sdk";

export const handler = async (event) => {
  // ... model call via wrapped client ...
  await flushObservations({ apiKey: process.env.TENSORCOST_API_KEY }, 3_000);
};
```

## SDK version matrix (1.0.0 vs 1.1.x vs 1.2.0)

| Feature | 1.0.0 | 1.1.0+ | 1.1.1+ | 1.2.0+ |
|---------|-------|--------|--------|--------|
| `prewarm()` | No | Yes | Yes | Yes |
| `headersTimeoutMs` / split timeouts | No | Yes | Yes | Yes |
| Admit 500ms cap | Partial | Yes | Yes | Yes |
| Buffered JSON headers = `timeoutMs` | No | No | **Yes** | **Yes** |
| Probe success no double-call | No | No | **Yes** | **Yes** |
| `stream: true` steer refusal | No | No | **Yes** | **Yes** |
| `flushObservations()` | No | No | **Yes** | **Yes** |
| Compliance profiles + in-process DLP | No | No | No | **Yes** |
| `teamId` / team-bound SDK tokens | No | No | No | **Yes** |

## What is captured

Metadata only — never prompt or completion content:

- SDK version, provider, model, operation, modality
- Request / response timestamps and token counts
- Status (`success` / `error`) and error class/message on failure
- Correlation UUID, environment, agent/workflow ids, chargeback tags
- `decision_action` / `decision_reason` when `x-tc-decision` is present

Proxied requests include `x-tc-sdk-version`, `x-tc-sdk-capabilities`, and `x-tc-max-layer`.

## Supported providers & operations

| Provider | Package | Layers | Operations |
|---|---|---|---|
| OpenAI | `openai` ≥ 4.x | observe → route | chat, completions, embeddings, responses |
| Anthropic | `@anthropic-ai/sdk` | observe → route | messages |
| Azure OpenAI | `openai` (`AzureOpenAI`) | observe → route | same as OpenAI (`provider: "openai"`) |
| AWS Bedrock | `@aws-sdk/client-bedrock-runtime` | observe, govern only | InvokeModel, Converse (token placeholders) |
| Google Vertex | `@google/genai` (`vertexai: true`) | observe, govern only | generateContent (token placeholders) |

**Steer/route streaming:** SDK routes `stream: true` **direct** to the provider (or errors if fail-open off). Proxy server supports SSE; SDK SSE client not shipped yet.

Full matrix: [`docs/demo/data-path-one-pager.md`](https://github.com/vaadhlabs/tensorcost/blob/main/docs/demo/data-path-one-pager.md).

**Bedrock and Vertex:** SigV4 / Vertex signing happens inside the AWS / Google SDK before the proxy can intercept — **`steer` and `route` are not supported**. Use `maxLayer: "govern"` or lower.

Bedrock peer dependency is optional:

```ts
import { BedrockRuntimeClient, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import { wrap } from "@tensorcost/sdk";

const client = wrap(new BedrockRuntimeClient({ region: "us-east-1" }), {
  apiKey: process.env.TENSORCOST_API_KEY,
});
```

## Framework adapters

Thin passthroughs that call `wrap()` on the underlying LLM client:

```ts
import { ChatOpenAI } from "@langchain/openai";
import { wrapLangChain } from "@tensorcost/sdk";

const llm = wrapLangChain(new ChatOpenAI({ model: "gpt-4o-mini" }), {
  apiKey: process.env.TENSORCOST_API_KEY,
  maxLayer: "observe",
});
```

Also exported: `wrapLlamaIndex`, `wrapCrewAI`, `wrapVercelAi`.

## Further reading

- [ADR-0026 — SDK control-layer ceilings](https://github.com/vaadhlabs/tensorcost/blob/main/docs/adr/0026-sdk-control-layer-ceilings.md)
- [Inference proxy SDK feature doc](https://github.com/vaadhlabs/tensorcost/blob/main/docs/features/inference-proxy-sdk.md)
- [Applied mode / live routing](https://github.com/vaadhlabs/tensorcost/blob/main/docs/features/inference-proxy-applied-mode.md)
- [CHANGELOG](./CHANGELOG.md)

## License

Apache-2.0
