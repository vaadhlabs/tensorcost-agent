# TensorCost Python SDK Integration

`tensorcost` (PyPI) wraps your existing LLM client in one line. After that
one change, TensorCost observes every inference call — which model, how
many tokens, latency, success/error — without touching any other code in
your app. Your app keeps working exactly as before.

---

## Install

```bash
pip install tensorcost
```

Runtime dependencies: `httpx` only. Python 3.9+.

---

## The one-line change

Before:

```python
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
```

After:

```python
from openai import OpenAI
from tensorcost import wrap

client = wrap(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
```

`wrap()` mutates the client in place and returns it. Every call you make
through that client is now observed by TensorCost. Nothing else in your
codebase needs to change.

---

## Configuration

`wrap()` resolves config in this order:

1. Explicit kwargs: `wrap(client, api_key=..., base_url=..., tenant_id=...)`
2. Environment variables:

| Variable | Required | Description |
|---|---|---|
| `TENSORCOST_API_KEY` | Yes | Long-lived API key from the TensorCost console |
| `TENSORCOST_TENANT_ID` | **Yes** | Your tenant id (console: **Developers › SDK setup**). Without it the SDK never reaches TensorCost; from 1.4.1 `wrap()` refuses `govern` and above without it |
| `TENSORCOST_BASE_URL` | No | Backend base URL (default: `https://api.tensorcost.com`) |
| `TENSORCOST_ENVIRONMENT` | No | Environment tag stamped on every observation (e.g. `staging`) |
| `TENSORCOST_PROXY_URL` | Required when `applied_mode=True` | Inference-proxy base URL |

3. Defaults: `base_url` defaults to `https://api.tensorcost.com`.

If no `api_key` is found, `wrap()` raises `MissingConfigError` — the one
failure the SDK does not swallow, because silently dropping every
observation is worse than a loud startup failure.

---

## Example apps

- [`examples/openai/`](./examples/openai/) — wrap an `OpenAI()` client
- [`examples/anthropic/`](./examples/anthropic/) — wrap an `Anthropic()` client
- [`examples/azure-openai/`](./examples/azure-openai/) — wrap the OpenAI
  Python client pointed at an Azure OpenAI deployment (see note below)
- [`examples/bedrock/`](./examples/bedrock/) — wrap a
  `boto3.client("bedrock-runtime")` client (observe-only; applied mode not
  supported for Bedrock)
- [`examples/vertex/`](./examples/vertex/) — wrap a
  `google.genai.Client(vertexai=True, ...)` client (observe-only; applied
  mode not supported for Vertex)

Each example is a self-contained script with its own `requirements.txt`
and a short README explaining what it demonstrates.

---

## Vertex AI note

SDK wrap for Vertex is **observe-only** (Bedrock-style). Install
`google-genai` separately — it is not a hard dependency of `tensorcost`.

For **Vertex spend in the dashboard** (backend polling, not per-call SDK
observations), use [`../direct-providers/vertex/`](../direct-providers/vertex/).

---

## Coming soon / not in this package

- **Gemini Developer API wrap** (`Client(api_key=...)`, non-Vertex mode)
- Streaming token accounting improvements for Bedrock/Vertex streams

---

## Azure OpenAI note

There is no separate Azure client to wrap. Azure OpenAI is
OpenAI-API-compatible, so you point the standard `openai` Python client's
`AzureOpenAI` class at your Azure endpoint/deployment and wrap it exactly
like the plain OpenAI example. Because the underlying client is `openai`,
observations from this path land in TensorCost tagged with provider_id
`openai`, not a distinct `azure_openai` provider_id — the SDK detects the
client class, not the backing endpoint. If you need spend broken out by
Azure resource, use the [Azure OpenAI direct-provider connection](../direct-providers/azure-openai/)
instead (or in addition) — that path polls Azure directly and attributes
spend to `azure_openai` explicitly.

---

## Bedrock note

`wrap()` accepts a `boto3.client("bedrock-runtime")` client and patches
`invoke_model` / `converse` (plus their streaming counterparts) in
place. This path is **observe-only** — passing `applied_mode=True` with
a Bedrock client raises `MissingConfigError` at `wrap()` time, since
Bedrock's SigV4-signed requests can't be intercepted at the proxy layer
the way an HTTP `base_url` redirect works for OpenAI/Anthropic. See
[`examples/bedrock/`](./examples/bedrock/) for the full example.

---

## What gets captured

TensorCost captures only metadata — never prompt text, completion content,
or credentials:

- Provider and model name
- Operation type (`chat.completions.create`, `messages.create`, etc.)
- Request and response timestamps
- Input and output token counts (from the provider's `usage` field)
- Status (`success` or `error`) and error class/message on failure
- A correlation UUID per request

---

## Fail-open guarantee

If TensorCost is unreachable, slow, or misconfigured, your app still
works. The SDK swallows internal observation errors and logs them via
`logging.getLogger("tensorcost")`. The one exception: a missing API key
raises `MissingConfigError` at `wrap()` call time.

---

## Applied mode (optional — Layer 2)

In addition to fire-and-forget telemetry, the SDK supports an "applied
mode" where it routes inference calls through the TensorCost inference
proxy. See the [package README](../../apps/tensorcost/packages/sdk-python/README.md)
for the full retry/timeout/circuit-breaker configuration surface.

```python
client = wrap(
    OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    api_key="tc-...",
    applied_mode=True,
    proxy_url="https://proxy.tensorcost.com",  # or TENSORCOST_PROXY_URL
)
```

---

## Auth and credentials

See [`auth.md`](./auth.md) for how the `TENSORCOST_API_KEY` is issued,
how to rotate it, and how the SDK authenticates.

## Troubleshooting

See [`troubleshooting.md`](./troubleshooting.md) for common failure modes
and how to diagnose them.
