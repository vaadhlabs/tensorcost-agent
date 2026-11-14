# TensorCost SDK for Python

[![PyPI](https://img.shields.io/pypi/v/tensorcost.svg)](https://pypi.org/project/tensorcost/)
[![Python](https://img.shields.io/pypi/pyversions/tensorcost.svg)](https://pypi.org/project/tensorcost/)
[![license](https://img.shields.io/pypi/l/tensorcost.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Wrap your OpenAI, Anthropic, AWS Bedrock or Google Vertex AI client in one line, and every call shows up in [TensorCost](https://tensorcost.com): what it cost, which model, team, customer and agent it was for. Your code keeps calling the provider exactly as before.

When you want more than visibility, the same wrapper can enforce budgets and policies before a call goes out, and route calls to a cheaper model through the TensorCost proxy.

- **One line.** `wrap(client)` and nothing else changes.
- **Never in your way.** Telemetry is fire-and-forget. If TensorCost is slow or down, your call still goes through.
- **Metadata only.** Prompts and completions never leave your process.

## Install

```bash
pip install tensorcost
```

Python 3.9 or later. The only dependency is `httpx`.

## Quick start

```python
import os
from openai import OpenAI
from tensorcost import wrap

client = wrap(
    OpenAI(),
    api_key=os.environ["TENSORCOST_API_KEY"],
    tenant_id=os.environ["TENSORCOST_TENANT_ID"],
)

reply = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Anthropic works the same way:

```python
from anthropic import Anthropic
from tensorcost import wrap

client = wrap(Anthropic())  # reads TENSORCOST_API_KEY and TENSORCOST_TENANT_ID

client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Your keys

In the TensorCost console:

- **Connect › API keys** creates the SDK key (`TENSORCOST_API_KEY`).
- **Connect › SDK setup** shows your tenant id (`TENSORCOST_TENANT_ID`).

The SDK trades the key for a short-lived token and refreshes it on its own.

## Configuration

Arguments passed to `wrap()` win over environment variables, which win over defaults.

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
| `TENSORCOST_AUTO_WORKFLOW_ID` | Set to `1` to give each wrapped client with an agent a fresh run id | No |
| `TENSORCOST_CUSTOMER` / `TENSORCOST_FEATURE` | Default customer and feature, for chargeback | No |

```python
client = wrap(
    OpenAI(),
    application="customer-support",
    tags=["tier:enterprise"],
)
```

Without a tenant id, `observe` warns once and sends nothing; `govern` and above raise `MissingConfigError` when you call `wrap()`.

## Control layers

How much TensorCost may do with a call is a ladder:

| Layer | Path | What happens |
|---|---|---|
| `off` | Direct | Nothing. The client comes back unwrapped. |
| `observe` | Direct | Telemetry only. **The default.** |
| `govern` | Direct | A metadata-only check before each call. Budgets, approved models, guardrails and compliance profiles can refuse it. |
| `steer` | Via proxy | The call goes through the TensorCost proxy, which can apply policy to it. |
| `route` | Via proxy | The proxy can send the call to another model under your routing policies. |

The layer you set is a ceiling, not a switch. TensorCost uses the lower of your `max_layer` and the layer published in the console (**Authority › Routing**), so you can turn control up or down without a deploy.

```python
client = wrap(
    OpenAI(),
    max_layer="route",
    proxy_url=os.environ["TENSORCOST_PROXY_URL"],
)
```

`steer` and `route` need a `proxy_url`; `wrap()` raises `MissingConfigError` without one. `govern` doesn't.

A sensible rollout is `observe`, then `govern`, then `route`, raising the console's layer only once each step looks right. A call at `route` only moves to another model when a routing policy for it is switched on; until then it goes to your provider unchanged, and the decision is recorded.

The SDK caches the console's layer for 30 seconds and never waits more than 250 ms for it. If TensorCost can't be reached for five minutes, it settles at `govern`: it keeps checking, and never raises itself to `route` on its own.

### Compliance profiles

When your workspace assigns compliance profiles (GDPR, HIPAA, PCI-DSS, the EU AI Act, data residency), the SDK enforces them at `govern` and above. Pattern detectors run inside your process before the provider is called, so a matched secret never leaves it and never reaches TensorCost. Set `team_id` to apply a team's profile.

## Attribution

Every call can say who and what it was for. Set defaults on `wrap()`, or per call with `with_meta()`:

```python
client = wrap(OpenAI(), agent_id="support-bot")

client.with_meta(workflow_id="ticket-48291", customer="acme", feature="ticket-triage").chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize this ticket"}],
)
```

| Field | Shows up as |
|---|---|
| `application`, `tags` | Spend by application and tag |
| `agent_id`, `workflow_id` | Agent runs and run caps |
| `customer`, `feature` | Cost per customer and feature |

`with_meta()` works on OpenAI and Anthropic clients. Bedrock and Vertex take the defaults set on `wrap()`.

## Reliability

If TensorCost is slow or down, your call still reaches the provider. The SDK only goes around TensorCost when it can't get an answer: a network error, a timeout, a 5xx, or a 429 or 402 with no refusal in it, and only while `fail_open_enabled` is on (the default). Anything TensorCost actually answers reaches your code as an error, and is never retried or sent around it.

- **Retries.** 5xx, network errors and 429s are retried with jittered backoff.
- **Timeouts.** The check before a call is capped at 500 ms. Through the proxy, a call waits up to `timeout_s` (60 s).
- **Circuit breaker.** After repeated proxy 5xx failures, calls go straight to the provider until the proxy answers again.
- **Streaming.** At `steer` and `route`, `stream=True` calls go straight to the provider (with `fail_open_enabled=False` they raise `MissingConfigError` instead).

```python
from tensorcost import RetryConfig, wrap

client = wrap(
    OpenAI(),
    max_layer="route",
    proxy_url=os.environ["TENSORCOST_PROXY_URL"],
    retry=RetryConfig(max_attempts=5, base_delay_ms=500.0, max_delay_ms=30_000.0),
    timeout_s=60.0,
    on_lifecycle_event=lambda e: print(e.kind, e.model),
)
```

One honest caveat: if the proxy forwarded a call to the provider and then failed, going around it can bill the call twice. Where calls must never go around TensorCost (a proxy inside your VPC, say), set `fail_open_enabled=False` and proxy failures raise instead.

Warnings go to the `tensorcost` logger.

## Errors

Every error is a `TensorCostError`. Refusals all extend `TensorCostRefusedError`, so one `except` covers them, including any refusal this version doesn't know by name:

| Error | When |
|---|---|
| `TensorCostBudgetHardCapError` | A budget's hard cap was reached. `retryable` is true when its spend isn't counted yet. |
| `TensorCostPeriodBudgetExceededError` | A daily or monthly cap was reached (`scope`, `period`, `cap_cents`, `spent_cents`). |
| `TensorCostRunBudgetExceededError` | A run reached its cap. |
| `TensorCostRunStoppedError` | Someone stopped the run. |
| `TensorCostModelGovernanceDeniedError` | The model isn't approved for this team. |
| `TensorCostGuardrailHardStopError` | A guardrail stopped the call. |
| `TensorCostComplianceDeniedError` | A compliance profile refused the call. |

```python
from tensorcost import TensorCostRefusedError

try:
    client.chat.completions.create(model="gpt-4o", messages=messages)
except TensorCostRefusedError as err:
    print("TensorCost refused the call:", err.code, err.reason)
    raise
```

A provider's own error (a bad request, say) reaches you as the provider SDK's error, as it would without TensorCost. Configuration problems raise `MissingConfigError` when you call `wrap()`.

## Serverless

`prewarm()` at import time fetches the token and the console's layer before the first request arrives, so the first call doesn't wait for them. Before a Lambda handler returns, `flush_observations()` sends anything still queued.

```python
from tensorcost import flush_observations, prewarm, wrap

prewarm(max_layer="route")
client = wrap(OpenAI(), max_layer="route", proxy_url=os.environ["TENSORCOST_PROXY_URL"])

def handler(event, context):
    reply = client.chat.completions.create(model="gpt-4o-mini", messages=event["messages"])
    flush_observations(timeout_s=3.0)
    return reply.choices[0].message.content
```

## Supported providers

| Provider | Client | Layers | Calls |
|---|---|---|---|
| OpenAI | `openai` 1.x | `observe` to `route` | Chat, completions, embeddings, responses, streamed or not |
| Azure OpenAI | `openai` (`AzureOpenAI`) | `observe` to `route` | As OpenAI |
| Anthropic | `anthropic` | `observe`, `govern` | Messages, streamed or not |
| AWS Bedrock | `boto3` (`bedrock-runtime`) | `observe` | invoke_model, converse, and their streams |
| Google Vertex AI | `google-genai` (`vertexai=True`) | `observe` | generate_content, streamed or not |

Bedrock and Vertex sign requests inside their own SDKs, so they can't go through the proxy: `steer` and `route` raise `MissingConfigError`, and `govern` is accepted but not enforced yet (the SDK warns once). Anthropic at `steer` or `route` runs at `govern`.

```python
import boto3
from tensorcost import wrap

bedrock = wrap(boto3.client("bedrock-runtime", region_name="us-east-1"))
```

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
