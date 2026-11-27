# Changelog

All notable changes to the `tensorcost` Python SDK are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## Unreleased

**A 429 from TensorCost is pacing, not an outage, and a call it lets
through unchecked is never silent.** A self-hosted load test found the
proxy's admission check capped at 600 calls a minute for the install's one
tenant. The SDK treated the bare 429 it answered over that limit as an
outage: with `fail_open_enabled` on (the default) every call over the limit
went to the provider with no budget or governance check, recorded as
`admit_unavailable` like an outage, and nothing was logged. The server limit
is fixed separately; this release fixes what the SDK does with the 429.

- **The admission check retries a bare 429** (one with no refusal code)
  inside its existing 500 ms budget. A `Retry-After` that fits is waited
  out; a longer one gives up at once, and every later call to the same
  TensorCost waits it out (or gives up) without asking again, instead of
  sending a request it has been told will be refused. Without
  `Retry-After` it backs off from 25 ms with jitter. `Retry-After: 0`
  waits the backoff, so it can't turn into a busy loop. A bare 402 (the
  plan's request ceiling) is not retried.
- **Still throttled when the budget runs out, the fail mode applies
  exactly as for an outage.** With `fail_open_enabled` on, the call goes to the
  provider; with it off, the call raises the 429 as a
  `TensorCostQuotaError` and the provider is never called. A 429
  with a refusal code in it was, and still is, a refusal.
- **The proxy route** already retried a 429 (`retry`, honouring
  `Retry-After`) before going around TensorCost. What changes is that the
  fall-back is now recorded.
- **Every call that goes to the provider unchecked says why.** Its
  observation carries `decision_action: allow` and a `decision_reason`:
  `admit_throttled`, `admit_plan_limit` or `admit_unavailable` at govern;
  `proxy_throttled`, `proxy_plan_limit`, `proxy_unavailable` or
  `circuit_open` through the proxy (those had no reason at all before).
  `on_fallback` fires with the same `reason`, now also at govern. A warning
  is logged per reason at most once a minute with the number of calls, and
  `unchecked_calls()` returns the count per reason since the process started.

## 1.8.2 — unreleased

**A call TensorCost couldn't deliver to the provider now goes to the
provider directly.** A newer inference proxy answers a provider it got no
answer from in one of three ways, each with `x-tc-upstream-error`. One of
them, `provider_unreachable` (a 502 with no `x-should-retry`), means the
proxy couldn't connect to the provider at all: the connection, DNS lookup or
TLS handshake failed before anything was sent, and failed again on the
proxy's own retry. Nothing ran and nothing was billed. Through 1.8.1 the SDK
treated that answer like the other two, as a call that may have reached the
provider: it surfaced the error and never fell open, so a provider that
TensorCost's network couldn't reach failed every call even though your own
network could reach it.

Now that answer surfaces as a `TensorCostProxyError` with
`may_have_reached_provider` False, and with `fail_open_enabled` on (the
default) the wrapper calls the provider directly. It is not retried through
the proxy, which has already retried the provider once. It does count as a
circuit failure: when the proxy's own network can't reach the provider,
every call gets this answer after the proxy's two connect attempts, and the
open circuit then sends calls straight to the provider instead of making
each one wait out those attempts first. The rule needs all of the 502,
`x-tc-error-origin: provider` and `x-tc-upstream-error:
provider_unreachable`; the header with any other origin, or none, is
handled as before. The `on_error`
lifecycle event for it carries `reason="provider_unreachable"`. With
`fail_open_enabled=False` the error is raised.

`provider_connection_lost` (502) and `provider_timeout` (504) carry
`x-should-retry: false` because the provider may have run the call, and are
handled exactly as before: never retried, never sent around TensorCost. A
`provider_unreachable` that also carries `x-should-retry: false` (a routed
call, where an earlier route may have run) is treated the same way.

## 1.8.1 — 2026-11-24

**Spend made while TensorCost is down is no longer lost.** In 1.8.0 an
observation whose token exchange or POST failed (a connection error, a
timeout, a 5xx from a gateway in front of a proxy that was scaled to zero, a
429) went into a retry queue that only the next successful delivery
replayed. Nothing retried it on a timer, `flush_observations()` didn't touch
it, and it was lost at exit, so a call the provider answered and billed
during the outage often left no spend row. Now a failed observation
waits in a bounded in-memory buffer and is resent with backoff, 1 s doubling
to 60 s with jitter (`Retry-After` sets the wait on a 429 or 503), until it
lands or an hour has passed. The token exchange is retried on the same
schedule. While TensorCost is unreachable one request at a time probes it,
so a recovering proxy sees one request rather than the whole buffer, and
once it answers the buffer drains on its own, with no call to `flush_observations()`
needed. The body is serialised once and resent byte for byte, so every
attempt carries the same `correlation_id`; ingestion dedupes the spend row,
the trace and the budget counters on it, so a resend after a lost response
is not counted twice. A 4xx other than 408 and 429 is not retried: it is
logged once, then at most once a minute with a count.

The buffer holds `max_buffered_observations` (10 000) or `max_buffered_bytes` (32 MiB). When it is full the
oldest observation goes, with at most one warning a minute. `observation_stats()` reports
what is in flight and buffered, `dropped`, `rejected`, and `delivered_after_retry` (held back
by a failure, then delivered); the SDK also logs once when delivery
recovers.

`flush_observations()` now sends what is waiting to retry as well as what is queued,
cuts a pending backoff short to probe once, and returns how many
observations are still undelivered (0 when everything landed). It returns as
soon as the next retry would land past its deadline, so a short-lived job
calling it during an outage pays for one attempt, not the whole timeout.

On a normal exit the SDK spends up to 2 s sending what is left (flush_on_exit=False or
`TENSORCOST_FLUSH_ON_EXIT=0` turns this off). Python uses `atexit`, which runs on a normal exit and under servers that turn SIGTERM into a clean shutdown (gunicorn, uvicorn); a bare SIGTERM skips it. A transport forked after it started (gunicorn `--preload`) starts fresh in the child. An opt-in disk
spool (`spool_dir` or `TENSORCOST_SPOOL_DIR`, off by default) keeps
observations that would otherwise be dropped, or that are still undelivered
at exit, as JSONL for the next start; both SDKs read and write the same
format.

**The half-open probe is one attempt.** With the circuit open, the first call
after the 5 s window probes the proxy. In 1.8.0 that call went through the
full retry schedule against a proxy that was still down (five attempts, 5 to
7.6 s) before falling back. Now the probe is a single attempt: if it fails,
the call goes to the provider at once and the circuit stays open for the
next window. A call whose own failures open the circuit stops retrying there
too and falls back, rather than spending its remaining attempts on the proxy.

**Sub-second timeouts read correctly.** A `timeout_s` of 0.4 was reported
as "timed out after 0s"; messages now say 0.4s (and 50ms below a tenth of a
second).

## 1.8.0 — 2026-11-24

**A call that may already have reached the provider is never sent again.**
Through 1.7.0 the SDK retried every 5xx, timeout and network error from the
proxy up to five times and then called the provider directly, ignoring
`x-should-retry`. A call the provider had answered and billed, which then
failed or timed out at the proxy or lost its response on the way back, could
run six times and be billed six times. Now the SDK retries, or falls back to
the provider, only when the attempt provably never got there: TensorCost
couldn't be reached at all (connection refused, DNS, a TLS handshake
failure), the circuit is open, or TensorCost answered before forwarding (its
own 429 or 402, the proxy's or the gateway's own 503, the gateway's 502 for
its own connect failure to the proxy). The proxy and gateway say which with
`x-tc-error-origin`. Everything else reaches your code as the error, with the
new `may_have_reached_provider` set to true: a timeout once the request was
sent, a reset connection, a 5xx with no origin header, a 500 or 502 from the
proxy, a gateway 502 or 504 for a proxy that failed mid-call, and anything
with `x-should-retry: false`. A status marked `x-tc-error-origin: provider`
(the provider's 5xx or 429, or the proxy's 504 for a provider that didn't
answer in time) is the provider's answer and is raised as the provider SDK's
own error type, like a relayed 4xx, and no longer counts against the
circuit. Retries that remain keep the same bound and backoff.

What changes for you: when the proxy hangs (it accepts the connection and
never answers), the calls that hit it now fail with `TensorCostTimeoutError`
instead of going direct after the timeout. Each still counts against the
circuit, so after three the circuit opens and calls go straight to the
provider before anything is sent; one call in five seconds probes the proxy,
and that call surfaces its error if the proxy is still hung. The five-second
wait between probes is new in this SDK (`CircuitBreakerConfig.probe_interval_s`,
default 5.0): an open circuit used to let the very next call probe, which
under the new rule would have failed every call while the proxy hangs. The same holds
for a 5xx from something in front of TensorCost that doesn't say where it
came from (a load balancer's own 503 page, say).

**Observations carry the provider's response id.** OpenAI's `response.id`
(plain, `with_raw_response` and streamed calls), Anthropic's
`message.id`, Bedrock's AWS request id and Vertex's `response_id` go on the
observation as `provider_response_id`, so an SDK row can be matched to the
same call seen by the proxy or a cost import. A value over 200 characters is
left off rather than losing the observation.

**Observations name the model the provider served.** An Azure deployment
named for its job ("support-bot") was recorded under that name, which no
price list has, so the call was stored at $0. The response's model now goes
on the observation as `response_model`, beside `model`: from `response.model` on a
plain call and under `with_raw_response`, from `message.model` on Anthropic, and on a stream from the last chunk that names one. An older
server ignores the field.

**A long provider error no longer loses the call's spend row.** The proxy
capped `error_message` at 200 characters and answered a longer one with 400,
which the SDK treats as final, so the observation was dropped. The SDK now
shortens `error_message` to 199 UTF-16 units plus an ellipsis, and
`sdk_version` to 120, before posting. Identifiers are sent as you set them.

Also fixes from a field test of 1.7.0 (openai 3.57.0) against a self-hosted install.

**Route mode returns openai's own types.** `create()` returned the proxy's
JSON as a dict and a streamed call yielded dicts. The proxy's response is now
replayed through a copy of the customer's client (`with_options` with a
replaying HTTP client), so the call returns a `ChatCompletion`, a `Stream` of
`ChatCompletionChunk`s, or under `with_raw_response` the raw-response wrapper,
exactly as unwrapped. `extra_headers`, `extra_query`, `extra_body` and
`timeout` are request options again instead of JSON fields (OpenAI answered
400 "Unrecognized request argument supplied: extra_headers", which broke every
`with_raw_response` call). The proxy connect timeout is the 2 s headers budget,
not `timeout_s`.

**Usage is read from every response shape.** `with_raw_response` calls and
streamed calls (openai's final usage chunk, Anthropic's message_start /
message_delta) were recorded as 0/0 tokens. A streamed call's observation is
posted when the stream ends.

**TensorCost down costs a call at most its stated budget.** The token exchange
ran on the caller's thread before the 0.25 s sdk-layer cap applied, so a
blackholed install made every call wait 6.5-8.5 s. The layer refresh (token
plus GET) now runs on its own thread and a call waits at most 0.25 s for it;
the govern path's token exchange and /admit share the 0.5 s admit budget.

**on_fallback for every way a route client goes direct**, with a `reason`:
`proxy_unavailable`, `circuit_open`, or `layer_decayed` (the layer couldn't be
fetched and the client decayed to govern, which fired nothing before).

**Observations carry the team and survive an outage.** `x-tc-team-id` is sent
on every observation POST. Deliveries that fail because TensorCost is
unreachable wait in a bounded in-memory queue (5,000, oldest dropped first,
`dropped_observations` counts them) and are replayed after the next delivery
that works. An `admit_unavailable` call is now marked on its own observation,
with its agent and real token usage, instead of a separate empty row.

**Warn-only budgets are surfaced.** A 204 from /admit with
`X-TensorCost-Budget-Would-Refuse` raises the new `on_budget_would_refuse`
lifecycle event (`OnBudgetWouldRefuseEvent.budget_ids`) and logs one warning
per budget id, at most once a minute.

## 1.7.0 — 2026-11-14

**Observations name the account the provider billed the call under.**
TensorCost can price a settled day's calls at what the provider actually
billed for them, discounts included, but only for a provider-day where
every metered call says which billing account it went to. The proxy
already labels the calls it makes; calls the SDK made itself arrived
without a label, so any day with SDK traffic stayed at the metered price.
1.7.0 sends `billing_account` on every observation for a call the SDK made
directly (observe, govern, and the fail-open fallback), in the format the
provider's own records use:

- Anthropic: the `anthropic-organization-id` response header.
- OpenAI: the `openai-organization` response header (the org, not the
  project).
- Azure OpenAI: the resource, the endpoint host's first label, lower-cased.
- Vertex (`google-genai`): the project in the URL the client sends to.
  Express mode (an API key) has no project and sends no field.
- Bedrock: the 12-digit account of the credentials that signed the call,
  when the AWS SDK knows it (`AWS_ACCOUNT_ID`, `aws_account_id` in the
  shared config, SSO, assumed roles), read without refreshing them.
  There is no STS call; plain access keys without an account send no
  field.

The headers come from a response event hook on the client's own
`httpx.Client` (`httpx2.Client` in anthropic 2.x and openai 3.x), added
once per client. It copies the headers of the response the wrapped call
got; the call's arguments, its path and what it returns are untouched, and
no request is added. Error responses and streams are labelled too. An
`AsyncClient` is left alone (this SDK wraps sync clients only). A missing or
unreadable header leaves the field out; it never fails the call. A client
pointed at a host that isn't the provider's (a gateway) sends no field,
matching how the server reads the same headers. Calls the proxy made carry
no field from the SDK; the proxy labels those itself.

Servers since 2026-10-25 (the ones 1.6.0 needs) drop an observation field
they don't know instead of rejecting the observation, so an older server
loses only this field.

**The missing-tenant error points at the right page.** `wrap()` without a
tenant id said the console shows it under "Developers › SDK setup"; the
page is Connect › SDK setup.

## 1.6.0 — 2026-11-12

**Observations carry the application and tags you wrap with.** 1.5.x held
`application` and `tags` back from the observations it posts for calls
made directly (observe and govern), so every such call landed in the
console as application "sdk:tier2" and per-application chargeback of SDK
traffic didn't work. 1.6.0 sends them, the way proxied calls already
carried them as `x-tc-application` / `x-tc-tags`.

**Upgrade the server with it if you run BYOC.** byoc-v0.1.40 validates
observations strictly and has neither field, so it rejects an
observation that carries them; current TensorCost servers accept both.

## 1.5.1 — 2026-11-12

**Azure OpenAI calls are reported as Azure OpenAI.** `AzureOpenAI` / `AsyncAzureOpenAI` is a
subclass of `OpenAI`, so 1.5.0 and earlier reported its calls with
provider `openai`. Azure spend then landed under OpenAI in the console, a
budget scoped to Azure OpenAI never counted it, and it couldn't be paired
with the Azure bill line for the same call. Observations and the govern
check now say `azure_openai` for a client on `*.openai.azure.com`,
`*.cognitiveservices.azure.com` or `*.services.ai.azure.com`, or of the
`AzureOpenAI` / `AsyncAzureOpenAI` class. The proxy request and the `on_fallback` lifecycle
event keep `openai`. Servers have accepted `azure_openai` since before
byoc-v0.1.40.

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
