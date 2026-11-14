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

If TensorCost is slow or down, your call still reaches the provider. The SDK only goes around TensorCost when it can't get an answer: a network error, a timeout, a 5xx, or a 429 or 402 with no refusal in it, and only while `failOpenEnabled` is on (the default). Anything TensorCost actually answers reaches your code as an error, and is never retried or sent around it.

- **Retries.** 5xx, network errors and 429s are retried with jittered backoff, honouring `Retry-After`.
- **Timeouts.** The check before a call is capped at 500 ms. Through the proxy, a buffered call waits up to `timeoutMs` (60 s); a streamed one waits `headersTimeoutMs` (2 s) for the stream to start and ends after `idleTimeoutMs` (30 s) without data.
- **Circuit breaker.** After three proxy 5xx failures in a row, calls go straight to the provider until the proxy answers again.

```ts
const client = wrap(new OpenAI(), {
  maxLayer: "route",
  proxyUrl: process.env.TENSORCOST_PROXY_URL,
  retry: { maxAttempts: 5, baseDelayMs: 500, maxDelayMs: 30_000 },
  timeoutMs: 60_000,
  onLifecycleEvent: (e) => console.log(e.kind, e.model),
});
```

One honest caveat: if the proxy forwarded a call to the provider and then failed, going around it can bill the call twice. Where calls must never go around TensorCost (a proxy inside your VPC, say), set `failOpenEnabled: false` and proxy failures surface as `TensorCostProxyError`.

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

`prewarm()` at module scope fetches the token and the console's layer before the first request arrives, so the first call doesn't wait for them. Before a Lambda handler returns, `flushObservations()` sends anything still queued.

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
