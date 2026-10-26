# TensorCost Node SDK Integration

`@tensorcost/sdk` wraps your existing LLM client in one line. After that one
change, TensorCost observes every inference call — which model, how many
tokens, latency, success/error — without touching any other code in your app.
Your app keeps working exactly as before.

---

## Install

```bash
npm install @tensorcost/sdk
# or
pnpm add @tensorcost/sdk
# or
yarn add @tensorcost/sdk
```

Node 18+ required (uses native `fetch`).

---

## The one-line change

Before:

```js
import OpenAI from "openai";
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
```

After:

```js
import OpenAI from "openai";
import { wrap } from "@tensorcost/sdk";

const client = wrap(new OpenAI({ apiKey: process.env.OPENAI_API_KEY }));
```

`wrap()` mutates the client in place and returns it. Every call you make
through that client is now observed by TensorCost. Nothing else in your
codebase needs to change.

---

## Configuration

`wrap()` reads from environment variables by default:

| Variable | Required | Description |
|---|---|---|
| `TENSORCOST_API_KEY` | Yes | Long-lived API key from the TensorCost console |
| `TENSORCOST_TENANT_ID` | **Yes** | Your tenant id (console: **Developers › SDK setup**). Without it the SDK never reaches TensorCost; from 1.5.0 `wrap()` refuses `govern` and above without it |
| `TENSORCOST_BASE_URL` | No | Backend base URL (default: `https://api.tensorcost.com`) |
| `TENSORCOST_ENVIRONMENT` | No | Environment tag stamped on every observation (e.g. `staging`) |
| `TENSORCOST_CONNECTION_ID` | No | Provider-connection identifier for per-connection cost attribution |
| `TENSORCOST_PROXY_URL` | Required when `maxLayer` is `steer` or `route` | Inference-proxy base URL |

You can also pass these directly to `wrap()`:

```js
const client = wrap(new OpenAI(...), {
  apiKey: "tc_live_xxx",
  environment: "production",
});
```

---

## Example apps

- [`examples/openai/`](./examples/openai/) — wrap an `OpenAI()` client; before/after comparison
- [`examples/anthropic/`](./examples/anthropic/) — wrap an `Anthropic()` client
- [`examples/azure-openai/`](./examples/azure-openai/) — wrap an `AzureOpenAI()` client (OpenAI-compatible; observations report `provider_id: openai`)
- [`examples/bedrock/`](./examples/bedrock/) — wrap a `BedrockRuntimeClient` (observe-only; applied mode is not supported for Bedrock)
- [`examples/vertex/`](./examples/vertex/) — wrap a `GoogleGenAI` Vertex client (observe-only)

Each example is a self-contained Node.js app with its own `package.json`
and a short README explaining what it demonstrates.

For dashboard **Vertex spend** (backend polling, not SDK wrap), use
[`../direct-providers/vertex/`](../direct-providers/vertex/).

---

## What gets captured

TensorCost captures only metadata — never prompt text, completion content,
or credentials:

- Provider and model name
- Operation type (`chat.completions`, `messages`, etc.)
- Request and response timestamps
- Input and output token counts (from the provider's `usage` field)
- Status (`success` or `error`) and error class/message on failure
- A correlation UUID per request

---

## Fail-open guarantee

If TensorCost is unreachable, slow, or misconfigured, your app still works.
The SDK swallows internal observation errors and surfaces them via
`console.warn`. The one exception: a missing API key throws
`MissingConfigError` at `wrap()` call time, because silently dropping every
observation is worse than a loud startup failure.

---

## Steer and route layers (optional — Layer 2)

In addition to fire-and-forget telemetry, the SDK supports **`maxLayer: "steer"`**
or **`maxLayer: "route"`** to route inference calls through the TensorCost
inference proxy. The proxy makes per-request routing decisions (model substitution,
provider failover, cost-cap enforcement) and returns the response transparently.

To enable:

```js
const client = wrap(new OpenAI({ apiKey: process.env.OPENAI_API_KEY }), {
  apiKey: "tc_live_xxx",
  maxLayer: "route",
  proxyUrl: "https://proxy.tensorcost.com", // or set TENSORCOST_PROXY_URL
});
```

Steer/route is not supported for native AWS Bedrock or Vertex client wraps (observe/govern only).

---

## Auth and credentials

See [`auth.md`](./auth.md) for how the `TENSORCOST_API_KEY` is issued,
how to rotate it, and how the JWT handshake works.

## Troubleshooting

See [`troubleshooting.md`](./troubleshooting.md) for common failure modes
and how to diagnose them.
