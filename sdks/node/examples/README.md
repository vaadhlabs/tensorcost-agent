# Runaway agent

A small agent that gets more expensive on purpose. Each step it sends a bigger
prompt than the last (the context grows by `--growth` per step), so its input
tokens and the spend on its run keep climbing. You use it to watch TensorCost do
its job: show the run's burn rate, apply a run cap, and stop the run.

It calls the model through the TensorCost proxy with `wrap({ maxLayer: "route" })`,
tagged with a run id (`--workflow-id`) and an agent id (`--agent-id`). It prints one
line per call:

```
step   1  input_tokens=121  status=200
step   2  input_tokens=171  status=200
step   3  input_tokens=~246  status=403
403 RUN_STOPPED: Agent run <id> was stopped by <who> at <time>: <reason>
```

When the proxy refuses a call with a 403 (an operator stopped the run, or the run hit
its hard cap) it prints `403 <code>: <message>` and exits 3. Any other error exits 1,
and so does a usage mistake. Reaching `--max-steps` exits 0. A `~` before the token
count means the call was refused, so the number is an estimate from the request size.

## Run it

From `sdks/node`:

```sh
export TENSORCOST_API_KEY=...        # your TensorCost API key
export TENSORCOST_TENANT_ID=...      # the tenant the key belongs to
export TENSORCOST_PROVIDER_URL=https://sandbox.inference.tensorcost.com/v1/chat/completions

npx tsx examples/runaway-agent.ts --model <a model TensorCost has a rate for>
```

The example is not part of the published package (`files` ships only `dist`), and it
imports the SDK from `../src`. In your own code you would import `wrap` from
`@tensorcost/sdk`.

### The sandbox must be on for your tenant

`sandbox.inference.tensorcost.com` is not a real provider. The proxy answers the call
itself, prices it at the model's real rate, and runs every check a real call goes
through, so caps and stops behave exactly as they would in production without a
provider bill. It only does that for a tenant with the
`inference-proxy.sandbox.enabled` flag on. With the flag off the proxy replies
`400 sandbox_not_enabled`. The SDK keeps that 400 but not its body, so the example
prints the error, a hint naming the flag, and exits 1. Ask TensorCost to turn the flag
on for your tenant.

The sandbox answers non-streaming calls only, which is all this agent makes. For the
Anthropic shape use `--provider anthropic` and
`https://sandbox.inference.tensorcost.com/v1/messages`.

## Flags

`--model` is the only flag you must always pass. The key, tenant and provider URL are
required too, but can come from the environment.

| Flag | Environment | Default |
| --- | --- | --- |
| `--model` | | required |
| `--api-key` | `TENSORCOST_API_KEY` | required |
| `--tenant-id` | `TENSORCOST_TENANT_ID` | required (the SDK uses it to exchange the key for a token) |
| `--provider-url` | `TENSORCOST_PROVIDER_URL` | required |
| `--base-url` | `TENSORCOST_BASE_URL` | `https://api.tensorcost.com` |
| `--proxy-url` | `TENSORCOST_PROXY_URL` | the base URL |
| `--provider` | | `openai` (or `anthropic`) |
| `--workflow-id` | | `runaway-<uuid>` |
| `--agent-id` | | `research-agent` (a label) |
| `--growth` | | `1.5` |
| `--delay-ms` | | `2000` |
| `--max-steps` | | `200` |

`TENSORCOST_PROXY_URL` and `TENSORCOST_TENANT_ID` are the names the SDK itself reads,
so setting them for your other TensorCost code works here too. The proxy is normally
served from the same host as the base URL, which is why `--proxy-url` defaults to it.

## What to look at

Pass your own `--workflow-id` (for example `--workflow-id demo-run-1`) so you know which
run to look for, start the agent, then open **Router · Agent runs** in the console and
find that id. Set a hard cap on the run and the next call is refused. Stop the run and
the next call is refused with the name of who stopped it and why. Either way the agent
halts on the 403 and never falls back to calling a provider directly.

## Tests

`runaway-agent.spec.ts` drives the loop in-process through the real `wrap()` against a
stubbed backend. Type-check the example with `npx tsc --noEmit -p examples/tsconfig.json`.
