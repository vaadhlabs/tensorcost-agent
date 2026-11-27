# TensorCost SDK for Node.js

[![npm](https://img.shields.io/npm/v/@tensorcost/sdk.svg)](https://www.npmjs.com/package/@tensorcost/sdk)
[![license](https://img.shields.io/npm/l/@tensorcost/sdk.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Wrap your OpenAI, Anthropic, AWS Bedrock or Google Vertex AI client in one line, and every call shows up in [TensorCost](https://tensorcost.com): what it cost, which model, team, customer and agent it was for. Your code keeps calling the provider exactly as before.

When you want more than visibility, the same wrapper can enforce budgets and policies before a call goes out, and route calls to a cheaper model through the TensorCost proxy.

- **One line.** `wrap(client)` and nothing else changes.
- **Never in your way.** Telemetry is fire-and-forget. If TensorCost is slow or down, your call still goes through.
- **Metadata only.** Prompts and completions never leave your process.

## Install

```bash
npm install @tensorcost/sdk
```

Node.js 18 or later.

## Quick start

```ts
import OpenAI from "openai";
import { wrap } from "@tensorcost/sdk";

const client = wrap(new OpenAI(), {
  apiKey: process.env.TENSORCOST_API_KEY,
  tenantId: process.env.TENSORCOST_TENANT_ID,
});

const reply = await client.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "Hello" }],
});
```

Anthropic works the same way:

```ts
import Anthropic from "@anthropic-ai/sdk";
import { wrap } from "@tensorcost/sdk";

const client = wrap(new Anthropic(), {
  apiKey: process.env.TENSORCOST_API_KEY,
  tenantId: process.env.TENSORCOST_TENANT_ID,
});

await client.messages.create({
  model: "claude-sonnet-4-5",
  max_tokens: 256,
  messages: [{ role: "user", content: "Hello" }],
});
```

### Your keys

In the TensorCost console:

- **Connect › API keys** creates the SDK key (`TENSORCOST_API_KEY`).
- **Connect › SDK setup** shows your tenant id (`TENSORCOST_TENANT_ID`).

The SDK trades the key for a short-lived token and refreshes it on its own.

## Configuration

Options passed to `wrap()` win over environment variables, which win over defaults.

| Environment variable | What it's for | Required |
|---|---|---|
| `TENSORCOST_API_KEY` | Your SDK key | Yes |
| `TENSORCOST_TENANT_ID` | Your tenant id | Yes |
| `TENSORCOST_BASE_URL` | TensorCost API (default `https://api.tensorcost.com`) | No |
| `TENSORCOST_PROXY_URL` | TensorCost proxy, for the `steer` and `route` layers | For `steer` / `route` |
| `TENSORCOST_ENVIRONMENT` | Environment tag on every call (`production`, `staging`, …) | No |
| `TENSORCOST_APPLICATION` | Application the calls belong to | No |
| `TENSORCOST_TAGS` | Comma-separated tags | No |
| `TENSORCOST_TEAM_ID` | Team the calls belong to | No |
| `TENSORCOST_AGENT_ID` / `TENSORCOST_WORKFLOW_ID` | Default agent and run | No |
| `TENSORCOST_CUSTOMER` / `TENSORCOST_FEATURE` | Default customer and feature, for chargeback | No |
| `TENSORCOST_DEPLOYMENT` | Release or git SHA | No |
| `TENSORCOST_PROMPT_TEMPLATE_ID` | Default prompt template | No |
| `TENSORCOST_FLUSH_ON_EXIT` | Set to `0` to skip the drain at process exit | No |
| `TENSORCOST_SPOOL_DIR` | Directory for the disk spool (off when unset) | No |

```ts
const client = wrap(new OpenAI(), {
  application: "customer-support",
  tags: ["tier:enterprise"],
  deployment: process.env.GIT_SHA,
});
```

Without a tenant id, `observe` warns once and sends nothing; `govern` and above throw `MissingConfigError` when you call `wrap()`.

## Control layers

How much TensorCost may do with a call is a ladder:

| Layer | Path | What happens |
|---|---|---|
| `off` | Direct | Nothing. The client comes back unwrapped. |
| `observe` | Direct | Telemetry only. **The default.** |
| `govern` | Direct | A metadata-only check before each call. Budgets, approved models, guardrails and compliance profiles can refuse it. |
| `steer` | Via proxy | The call goes through the TensorCost proxy, which can apply policy to it. |
| `route` | Via proxy | The proxy can send the call to another model under your routing policies. |

The layer you set is a ceiling, not a switch. TensorCost uses the lower of your `maxLayer` and the layer published in the console (**Authority › Routing**), so you can turn control up or down without a deploy.

```ts
const client = wrap(new OpenAI(), {
  maxLayer: "route",
  proxyUrl: process.env.TENSORCOST_PROXY_URL,
});
```

`steer` and `route` need a `proxyUrl`; `wrap()` throws `TensorCostConfigError` without one. `govern` doesn't.

A sensible rollout is `observe`, then `govern`, then `route`, raising the console's layer only once each step looks right. A call at `route` only moves to another model when a routing policy for it is switched on; until then it goes to your provider unchanged, and the decision is recorded.

The SDK caches the console's layer for 30 seconds and never waits more than 250 ms for it. If TensorCost can't be reached for five minutes, it settles at `govern`: it keeps checking, and never raises itself to `route` on its own.

### Data grants

`maxGrants` caps which of the console's data grants apply to this client. An empty list allows none; leaving it out allows all of them.

```ts
const client = wrap(new OpenAI(), {
  maxGrants: ["telemetry", "retain_completions"],
});
```

### Compliance profiles

When your workspace assigns compliance profiles (GDPR, HIPAA, PCI-DSS, the EU AI Act, data residency), the SDK enforces them at `govern` and above. Pattern detectors run inside your process before the provider is called, so a matched secret never leaves it and never reaches TensorCost. Set `teamId` to apply a team's profile.

## Attribution

Every call can say who and what it was for. Set defaults on `wrap()`, or per call with `withMeta()`:

```ts
const client = wrap(new OpenAI(), { agentId: "support-bot" });

await client
  .withMeta({ workflowId: crypto.randomUUID(), customer: "acme", feature: "ticket-triage" })
  .chat.completions.create({ model: "gpt-4o-mini", messages });
```

| Field | Shows up as |
|---|---|
| `application`, `tags` | Spend by application and tag |
| `agentId`, `workflowId` | Agent runs and run caps |
| `customer`, `feature` | Cost per customer and feature |
| `deployment`, `promptTemplateId` | Spend by release and prompt template |

`withMeta()` works on OpenAI and Anthropic clients. Bedrock and Vertex take the defaults set on `wrap()`.

## Reliability

If TensorCost is down, your call still reaches the provider. The SDK goes around TensorCost only when the call provably never reached the provider: TensorCost couldn't be reached at all (connection refused, DNS, a TLS handshake failure), the circuit is open, or TensorCost answered before forwarding (a 429 or 402 with no refusal in it, the proxy's or the gateway's own 503), or TensorCost couldn't connect to the provider itself (a 502 with `x-tc-upstream-error: provider_unreachable`, which the proxy has already retried once and doesn't retry again), and only while `failOpenEnabled` is on (the default). Anything TensorCost actually answers as a decision reaches your code as an error, and is never retried or sent around it.

A failure after the call may have reached the provider is never sent again, through TensorCost or around it: a timeout once the request was sent, a dropped connection, a 5xx the provider sent (marked `x-tc-error-origin: provider`, raised as the provider SDK's own error), any other 5xx TensorCost can't prove it sent before forwarding, and anything with `x-should-retry: false`. Those reach your code as the error, with `mayHaveReachedProvider` set to true, because resending could bill the call twice. Through 1.7.0 the SDK retried those and then called the provider directly, so one call could run up to six times.

- **Retries.** Only calls that provably never reached the provider are retried, with jittered backoff, honouring `Retry-After` on a 429. The admission check retries a bare 429 too, inside its 500 ms budget (see below).
- **Timeouts.** The check before a call is capped at 500 ms. Through the proxy, a buffered call waits up to `timeoutMs` (60 s); a streamed one waits `headersTimeoutMs` (2 s) for the stream to start and ends after `idleTimeoutMs` (30 s) without data. With TensorCost unreachable, only the first calls of a process wait for the published layer (250 ms at most); the token exchange counts against the budget of whatever waits on it, and after it fails calls stop waiting on it for 5 s.
- **Circuit breaker.** After three proxy failures in a row, calls go straight to the provider. Every 5 s one call probes the proxy with a single attempt, never retried: if it fails, that call goes to the provider at once and the circuit stays open for another 5 s. Once probes succeed calls go through the proxy again. A call never retries through a circuit that opened while it was running. Transitions reach `onLifecycleEvent` as `on_circuit_state`; they are not sent to TensorCost.

```ts
const client = wrap(new OpenAI(), {
  maxLayer: "route",
  proxyUrl: process.env.TENSORCOST_PROXY_URL,
  retry: { maxAttempts: 5, baseDelayMs: 500, maxDelayMs: 30_000 },
  timeoutMs: 60_000,
  onLifecycleEvent: (e) => console.log(e.kind, e.model),
});
```

Where calls must never go around TensorCost (a proxy inside your VPC, say), set `failOpenEnabled: false` and proxy failures surface as `TensorCostProxyError`.

### Unchecked calls, and when TensorCost is pacing you

A bare 429 from TensorCost (no refusal code in it) means TensorCost is pacing this process, not that it is down. Before 1.9.0 the govern layer treated it as an outage and let the call through, unrecorded. Now the admission check retries it inside its own 500 ms budget: after `Retry-After` when that fits, otherwise after a short jittered backoff, and every call to the same TensorCost waits out a `Retry-After` it has already been given rather than asking again. The proxy route already retried a 429 (`retry`, honouring `Retry-After`) before going around TensorCost. A bare 402 (the plan's request ceiling) is not retried.

If TensorCost is still pacing you when the budget or the retries run out, the fail mode applies exactly as for an outage. With `failOpenEnabled` on (the default) the call goes to the provider without a budget or governance check; with it off the call is rejected with a `TensorCostQuotaError` (govern) or a `TensorCostProxyError` (route) and the provider is never called.

Every call that goes to the provider unchecked is recorded, whatever the reason:

- Its observation carries `decision_action: allow` and a `decision_reason`: `admit_throttled`, `admit_plan_limit` or `admit_unavailable` at govern; `proxy_throttled`, `proxy_plan_limit`, `proxy_unavailable` or `circuit_open` through the proxy.
- `on_fallback` fires with the same `reason`.
- A warning is logged per reason, at most once a minute, with how many calls went through.
- `uncheckedCalls()` returns the count per reason since the process started.

```ts
import { uncheckedCalls } from "@tensorcost/sdk";

// from a health check
const { admit_throttled, proxy_throttled } = uncheckedCalls();
```

### Observation delivery

Observations are sent in the background and never slow your call down. One that fails to send (a network error, a timeout, a 5xx, a 408 or a 429) waits in memory and is retried with backoff: 1 s, doubling to 60 s, with jitter, and `Retry-After` honoured on a 429. While TensorCost is unreachable only one request at a time probes it. Each retry resends the same body with the same `correlation_id`, and ingestion dedupes on it, so nothing is counted twice. A 4xx other than 408 and 429 (a 400, 401, 403 or 422, say) is not retried, since the same body would get the same answer: it is counted in `observationStats().rejected` and logged once, then at most once a minute. An observation is given up on after an hour. When delivery recovers after an outage the SDK logs it once, and `observationStats().deliveredAfterRetry` counts what was held back and then delivered.

Up to `maxBufferedObservations` (10 000) or `maxBufferedBytes` (32 MiB) wait at once. Past that the oldest goes, with at most one warning a minute, and `observationStats().dropped` counts it.

When the process exits normally, or gets SIGTERM, the SDK spends up to 2 s sending what is queued. It never calls `process.exit()`: if your app handles SIGTERM itself, the SDK only drains alongside your handler, and if nothing else handles it, the SDK re-sends SIGTERM once the drain is done, so the process still dies by the signal. Set `flushOnExit: false` to skip all of this.

With `spoolDir` set, observations that would otherwise be lost (the buffer was full, or they were still undelivered at exit) are written there as JSONL and sent on the next start. The directory is capped at `spoolMaxBytes` (64 MiB) by deleting the oldest files. Processes can share one directory.

```ts
import { observationStats, wrap } from "@tensorcost/sdk";

const client = wrap(new OpenAI(), { spoolDir: "/var/lib/myapp/tensorcost" });
// later, from a health check
const { buffered, dropped, spooled } = observationStats();
```

## Errors

Every error is a `TensorCostError`. Refusals all extend `TensorCostRefusedError`, so one `catch` covers them, including any refusal this version doesn't know by name:

| Error | When |
|---|---|
| `TensorCostBudgetHardCapError` | A budget's hard cap was reached. `retryable` is true when its spend isn't counted yet. |
| `TensorCostPeriodBudgetExceededError` | A daily or monthly cap was reached (`scope`, `period`, `capCents`, `spentCents`). |
| `TensorCostRunBudgetExceededError` | A run reached its cap. |
| `TensorCostRunStoppedError` | Someone stopped the run (`stoppedBy`, `stoppedAt`, `reason`). |
| `TensorCostModelGovernanceDeniedError` | The model isn't approved for this team. |
| `TensorCostGuardrailHardStopError` | A guardrail stopped the call. |
| `TensorCostComplianceDeniedError` | A compliance profile refused the call. |

```ts
import { TensorCostRefusedError } from "@tensorcost/sdk";

try {
  await client.chat.completions.create({ model: "gpt-4o", messages });
} catch (err) {
  if (err instanceof TensorCostRefusedError) {
    console.warn("TensorCost refused the call:", err.code, err.reason);
  }
  throw err;
}
```

A provider's own error (a bad request, say) reaches you as the provider SDK's error, as it would without TensorCost. Configuration problems throw when you call `wrap()`: `MissingConfigError` without a key or tenant id, `TensorCostConfigError` without a `proxyUrl` at `steer` or `route`.

## Serverless

`prewarm()` at module scope fetches the token and the console's layer before the first request arrives, so the first call doesn't wait for them. A Lambda is frozen between invocations rather than exited, so the exit drain never runs there: before the handler returns, call `flushObservations()`, which sends anything queued or waiting to retry. If TensorCost is unreachable it gives up after one attempt instead of using the whole timeout, and what is left goes out on a later invocation. It returns how many observations are still waiting (0 when everything landed).

```ts
import { flushObservations, prewarm, wrap } from "@tensorcost/sdk";

await prewarm({ maxLayer: "route" });
const client = wrap(new OpenAI(), { maxLayer: "route", proxyUrl: process.env.TENSORCOST_PROXY_URL });

export const handler = async (event) => {
  const reply = await client.chat.completions.create({ model: "gpt-4o-mini", messages: event.messages });
  await flushObservations({}, 3_000);
  return reply;
};
```

## Supported providers

| Provider | Client | Layers | Calls |
|---|---|---|---|
| OpenAI | `openai` 4.x | `observe` to `route` | Chat completions, streamed or not; legacy completions (observed) |
| Azure OpenAI | `openai` (`AzureOpenAI`) | `observe` to `route` | As OpenAI |
| Anthropic | `@anthropic-ai/sdk` | `observe`, `govern` | Messages |
| AWS Bedrock | `@aws-sdk/client-bedrock-runtime` | `observe` | InvokeModel, Converse |
| Google Vertex AI | `@google/genai` (`vertexai: true`) | `observe` | generateContent |

Only the calls listed are wrapped. Anything else on the client (embeddings, for example) goes straight to the provider and isn't observed.

Bedrock and Vertex sign requests inside their own SDKs, so they can't go through the proxy: `steer` and `route` aren't available, and `govern` is accepted but not enforced yet (the SDK warns once). Anthropic at `steer` or `route` runs at `govern`.

```ts
import { BedrockRuntimeClient } from "@aws-sdk/client-bedrock-runtime";

const bedrock = wrap(new BedrockRuntimeClient({ region: "us-east-1" }));
```

### Frameworks

```ts
import { ChatOpenAI } from "@langchain/openai";
import { wrapLangChain } from "@tensorcost/sdk";

const llm = wrapLangChain(new ChatOpenAI({ model: "gpt-4o-mini" }));
```

`wrapLlamaIndex`, `wrapCrewAI` and `wrapVercelAi` work the same way.

## What's sent

Metadata only, never prompt or completion text:

- provider, model, operation, token counts, timings and status;
- the environment, application, tags, team, agent, run, customer and feature you set;
- the provider account the call was billed under, read from the provider's response: the Anthropic or OpenAI organization id, the Azure OpenAI resource, the Vertex project, or the AWS account when your credentials name it. Never your provider key.

TensorCost uses the account to match each call with the provider's own bill, so discounts and credits reach your cost figures.

## Learn more

- [SDK guide](https://docs.tensorcost.com/developer-tools)
- [TensorCost documentation](https://docs.tensorcost.com)
- [tensorcost.com](https://tensorcost.com)

## License

Apache-2.0
