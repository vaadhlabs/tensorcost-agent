# Changelog

## Unreleased

**A 429 from TensorCost is pacing, not an outage, and a call it lets
through unchecked is never silent.** A self-hosted load test found the
proxy's admission check capped at 600 calls a minute for the install's one
tenant. The SDK treated the bare 429 it answered over that limit as an
outage: with `failOpenEnabled` on (the default) every call over the limit
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
  exactly as for an outage.** With `failOpenEnabled` on, the call goes to the
  provider; with it off, the call is rejected with the 429 as a
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
  `uncheckedCalls()` returns the count per reason since the process started.

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
`mayHaveReachedProvider: false`, and with `failOpenEnabled` on (the
default) the wrapper calls the provider directly. It is not retried through
the proxy, which has already retried the provider once. It does count as a
circuit failure: when the proxy's own network can't reach the provider,
every call gets this answer after the proxy's two connect attempts, and the
open circuit then sends calls straight to the provider instead of making
each one wait out those attempts first. The rule needs all of the 502,
`x-tc-error-origin: provider` and `x-tc-upstream-error:
provider_unreachable`; the header with any other origin, or none, is
handled as before. The `on_error`
lifecycle event for it carries `reason: "provider_unreachable"`. With
`failOpenEnabled: false` the error is thrown.

`provider_connection_lost` (502) and `provider_timeout` (504) carry
`x-should-retry: false` because the provider may have run the call, and are
handled exactly as before: never retried, never sent around TensorCost. A
`provider_unreachable` that also carries `x-should-retry: false` (a routed
call, where an earlier route may have run) is treated the same way.

## 1.8.1 — 2026-11-24

**Spend made while TensorCost is down is no longer lost.** Through 1.8.0 an
observation was posted once. If the token exchange or the POST failed (a
connection error, a timeout, a 5xx from a gateway in front of a proxy that
was scaled to zero, a 429), the SDK logged a warning and the observation was
gone, so a call the provider answered and billed left no spend row, and
`flushObservations()` after recovery had nothing to send. Now a failed observation
waits in a bounded in-memory buffer and is resent with backoff, 1 s doubling
to 60 s with jitter (`Retry-After` sets the wait on a 429 or 503), until it
lands or an hour has passed. The token exchange is retried on the same
schedule. While TensorCost is unreachable one request at a time probes it,
so a recovering proxy sees one request rather than the whole buffer, and
once it answers the buffer drains on its own, with no call to `flushObservations()`
needed. The body is serialised once and resent byte for byte, so every
attempt carries the same `correlation_id`; ingestion dedupes the spend row,
the trace and the budget counters on it, so a resend after a lost response
is not counted twice. A 4xx other than 408 and 429 is not retried: it is
logged once, then at most once a minute with a count.

The buffer holds `maxBufferedObservations` (10 000) or `maxBufferedBytes` (32 MiB). When it is full the
oldest observation goes, with at most one warning a minute. `observationStats()` reports
what is in flight and buffered, `dropped`, `rejected`, and `deliveredAfterRetry` (held back
by a failure, then delivered); the SDK also logs once when delivery
recovers.

`flushObservations()` now sends what is waiting to retry as well as what is queued,
cuts a pending backoff short to probe once, and returns how many
observations are still undelivered (0 when everything landed). It returns as
soon as the next retry would land past its deadline, so a short-lived job
calling it during an outage pays for one attempt, not the whole timeout.

On a normal exit the SDK spends up to 2 s sending what is left (flushOnExit: false or
`TENSORCOST_FLUSH_ON_EXIT=0` turns this off). Node drains on `beforeExit` and SIGTERM and never calls `process.exit()`; when your app has its own SIGTERM handler the SDK only drains alongside it, and when it has none the SDK re-sends SIGTERM after the drain so the process still dies by the signal. An opt-in disk
spool (`spoolDir` or `TENSORCOST_SPOOL_DIR`, off by default) keeps
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
new `mayHaveReachedProvider` set to true: a timeout once the request was
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
and that call surfaces its error if the proxy is still hung. The same holds
for a 5xx from something in front of TensorCost that doesn't say where it
came from (a load balancer's own 503 page, say).

**Observations carry the provider's response id.** OpenAI's `response.id`
(plain, `withResponse()`/`asResponse()` and streamed calls), Anthropic's
`message.id`, Bedrock's AWS request id and Vertex's `response_id` go on the
observation as `provider_response_id`, so an SDK row can be matched to the
same call seen by the proxy or a cost import. A value over 200 characters is
left off rather than losing the observation.

**Observations name the model the provider served.** An Azure deployment
named for its job ("support-bot") was recorded under that name, which no
price list has, so the call was stored at $0. The response's model now goes
on the observation as `response_model`, beside `model`: from the body on a
plain call, `withResponse()` and `asResponse()`, from `message.model` on
Anthropic, and on a stream from the last chunk that names one. An older
server ignores the field.

**A long provider error no longer loses the call's spend row.** The proxy
capped `error_message` at 200 characters and answered a longer one with 400,
which the SDK treats as final, so the observation was dropped. The SDK now
shortens `error_message` to 199 UTF-16 units plus an ellipsis, and
`sdk_version` to 120, before posting. Identifiers are sent as you set them.

Also fixes from the 1.7.0 run against a self-hosted install (openai 7.62.0,
@anthropic-ai/sdk 1.12.0).

**Route mode works with a default `new OpenAI()` client.** The SDK sent the
client's base URL, `https://api.openai.com/v1`, as `x-tc-provider-url`. The
proxy appends `/v1/...` itself, so every call asked for `/v1/v1/...` and
came back 404. The SDK now drops a trailing API version (`/v1`, `/v2`) the
way the Python SDK always has: `https://api.openai.com`, and Azure's
`/openai/v1` becomes `/openai`; deployment paths are untouched. A 4xx with
no error body that carries `x-tc-decision` is now
`TensorCostProxyStatusError`, which says the proxy or the provider behind it
answered. It used to be `TensorCostGatewayRejectedError`, which blamed a
firewall in front of TensorCost. That class is kept for 4xx answers without
the header.

**`create()` returns an APIPromise again.** The wrapper replaced
`chat.completions.create` and `messages.create` with an async function, so
`.withResponse()`, `.asResponse()` and `._thenUnwrap()` threw, and with
them the provider SDKs' own helpers built on those (Anthropic's
`messages.stream()`, openai's `chat.completions.parse()`). In observe mode,
once the published layer is known (normally from the warmup `wrap()`
starts), the call is made in the same tick and you get the client's own
APIPromise back. When TensorCost has to answer first (govern's admission
check, route's proxy, or the first calls of a process before the layer is
known) you get a `TensorCostAPIPromise` with the same methods; in route
mode `withResponse()` gives the proxy's response, `x-tc-decision` included.
Every way of reading the call records it once.

**Streams are recorded with their tokens.** A streamed call used to be
recorded when the stream started, with no usage. It is now recorded when the
stream ends, with the usage it carried: the final chunk with
`stream_options.include_usage` on OpenAI, `message_start` and
`message_delta` on Anthropic, in observe, govern and route. The SDK watches
the iterator you read from and never reads ahead of you. A stream you stop
reading is recorded when you stop, with what it had. Bedrock and Vertex
streams are unchanged (still recorded without tokens).

**Circuit-breaker transitions are no longer sent as calls.** They were
posted as observations, which the server stores as spend events, so an
outage added 0-token "calls". They now go to `onLifecycleEvent` as
`on_circuit_state` and are logged when the circuit opens or closes again.
The same was true of a govern admission check that failed open: it posted
an extra empty observation. Now the call's own observation carries
`decision_reason: admit_unavailable`. And an open circuit now waits 5 s
before probing the proxy again, instead of probing on the very next call,
so a hung proxy no longer costs every call the proxy timeout.

**TensorCost being down costs a call its stated budget, once.** With
TensorCost unreachable every call waited out the 2 s token exchange twice,
once for the published layer and once for the compliance snapshot. Now:
only the first calls of a cold process wait for the published layer, at most
250 ms in all, token included; after that the layer is refreshed in the
background. The token exchange is bounded by the budget of whatever waits on
it (500 ms for govern's admission check, the proxy's headers timeout in
route mode), keeps running in the background to fill the cache, and after a
failure calls are told at once for 5 s instead of waiting on another
exchange. A token that is about to expire is refreshed in the background
while the current one is still used.

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
- Vertex (`@google/genai`): the project in the URL the client sends to.
  Express mode (an API key) has no project and sends no field.
- Bedrock: the 12-digit account of the credentials that signed the call,
  when the AWS SDK knows it (`AWS_ACCOUNT_ID`, `aws_account_id` in the
  shared config, SSO, assumed roles). There is no STS call; plain access
  keys without an account send no field.

The headers come from `asResponse()` on the promise the openai /
`@anthropic-ai/sdk` client returned, which hands back the response the call
already got: no extra request, and the caller's result is the same object.
Error observations carry it too when the provider answered (the APIError
keeps its headers), and streams are labelled at stream start. A missing or
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

**Azure OpenAI calls are reported as Azure OpenAI.** `AzureOpenAI` is a
subclass of `OpenAI`, so 1.5.0 and earlier reported its calls with
provider `openai`. Azure spend then landed under OpenAI in the console, a
budget scoped to Azure OpenAI never counted it, and it couldn't be paired
with the Azure bill line for the same call. Observations and the govern
check now say `azure_openai` for a client on `*.openai.azure.com`,
`*.cognitiveservices.azure.com` or `*.services.ai.azure.com`, or of the
`AzureOpenAI` class. The proxy request and the `on_fallback` lifecycle
event keep `openai`. Servers have accepted `azure_openai` since before
byoc-v0.1.40.

## 1.5.0 — 2026-10-26

**1.4.0 could route around a hard spend cap when fail-open was on (the
default); upgrade.** When a money budget's hard cap refused a call (403
with `code: "BUDGET_HARD_CAP"`), 1.4.0 read the refusal as a provider error,
and with `failOpenEnabled` on it sent the call to the provider anyway. The
cap was enforced by the proxy and then skipped by the SDK.

This is a minor release, not a patch: it adds public error classes and
makes `wrap()` throw at startup without a tenant id. Read the upgrade notes.

### Upgrading from 1.4.x

- **`wrap()` requires a tenant id at `govern`, `steer` and `route`.** The
  token exchange needs it, so without one the SDK never talked to TensorCost
  at all: no `/v1/admit`, no proxy call, no observation, and every call went
  straight to the provider while the README called the variable optional.
  `wrap()` now throws `MissingConfigError` at those layers when neither
  `tenantId` nor `TENSORCOST_TENANT_ID` is set, whatever `failOpenEnabled` says (1.4.0 skipped
  TensorCost silently, or with `failOpenEnabled: false` in route threw
  `TensorCostProxyError` on every call). Find the id in the TensorCost
  console under **Developers › SDK setup**, in the `TENSORCOST_TENANT_ID`
  line of the env block. `observe` without one keeps working and logs one
  warning that nothing is being sent.
- **Unknown refusals are terminal.** A refusal code this version doesn't
  know stops the call as `TensorCostRefusedError`; it no longer falls back
  to the provider. Details below.
- **A revoked key fails closed.** A TensorCost key the token exchange,
  `/v1/admit` or the proxy rejects is `TensorCostKeyRejectedError`, and the
  call isn't sent. 1.4.0 let the call through to the provider when fail-open
  was on (in steer/route with an openai@4 client it crashed instead, see
  Fixed).
- **Route-mode embeddings, responses and legacy completions now go straight
  to the provider.** Only `chat.completions` is proxied. 1.4.0 pointed the
  whole client's `baseURL` at the proxy, so those calls went there too,
  carrying the provider key as the bearer. Now they go to the provider and
  are not proxied; legacy completions are observed, embeddings and
  responses aren't wrapped at all.

### Changed: an unknown refusal is now terminal

1.4.0 decided whether a failed proxy call could go straight to the provider
with a list of refusals that must not. Any refusal missing from that list,
including ones added on the server later, fell back. 1.5.0 turns the rule
around: a call goes around TensorCost only when TensorCost is unavailable.

- Falls back (with `failOpenEnabled` on): network errors and connection
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
  fall back either. It is thrown as the provider SDK's own error, built by
  the client itself with `client.makeStatusError(...)`, the factory the
  client uses for a direct call (`BadRequestError`, `AuthenticationError` and
  the rest), with the original `TensorCostProviderError` on its `tensorcost`
  property. The headers go in as a plain object first, which openai@4 reads,
  then as a `Headers` instance, which current Stainless clients read with
  `headers.get(...)`. Tested against openai@4.104.0 and @anthropic-ai/sdk
  0.65.0, and against a stub of the openai v5/v6 `headers.get` contract;
  openai v5/v6 itself wasn't run. A client that can't build one gets the
  `TensorCostProviderError` (`status`, `body`, `headers`).
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
outage, and only while `failOpenEnabled` is on. Before, admit failed open on
everything except a 403, whatever `failOpenEnabled` said.

### Added

**`TensorCostRefusedError`**, the base class of every refusal:
`TensorCostRunBudgetExceededError`, `TensorCostRunStoppedError`,
`TensorCostPeriodBudgetExceededError`, `TensorCostModelGovernanceDeniedError`,
`TensorCostGuardrailHardStopError`, `TensorCostComplianceDeniedError` and
`TensorCostComplianceTeamMismatchError` all extend it now, and all carry
`code`, `reason` and `detail` (the refusal object the server sent). Catch
`TensorCostRefusedError` to handle every refusal, current and future.

**`TensorCostBudgetHardCapError`** for a money budget's hard cap. It extends
`TensorCostPeriodBudgetExceededError`, so a `catch` written against 1.4.0 for
the current wire shape still matches. Fields: `budgetId`, `budgetName`,
`burnPct`, `scope`, `period`, `capCents`, `spentCents`, `reason`,
`budgetCode` (always `"BUDGET_HARD_CAP"`), `retryable` and
`retryAfterSeconds`. `retryable` is true when the budget's spend hasn't been
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

`retryAfterSeconds` reads the `Retry-After` header (seconds or an HTTP date)
first, then `retry_after_seconds` or `retry_after` in the body. A
`PERIOD_BUDGET_EXCEEDED` without `budget_code` is still a plain
`TensorCostPeriodBudgetExceededError`.

**`TensorCostGatewayRejectedError`** (non-JSON 4xx, above) and
**`TensorCostKeyRejectedError`** (the token exchange, `/v1/admit` or the
proxy rejected the TensorCost key: "the TensorCost key was rejected; check
it's current"). Both are terminal. 1.4.0 threw a plain `Error` for a revoked
key, and with fail-open on, the call then went to the provider.

### Fixed

**steer/route with a real openai@4 client.** 1.4.0 rewrote the client's
`baseURL` to the proxy and replaced its `defaultHeaders()` method with a
plain object. Every call that reached the client's own request path died
with `TypeError: this.defaultHeaders is not a function`: every fail-open
fallback (proxy 5xx, 429, 402, network error, token-exchange outage), and
every call when a route-configured client ran below steer (published layer
govern or observe, TensorCost unreachable for more than 5 minutes). Had the
fallback run, it would have gone to the proxy URL, not the provider. 1.5.0
doesn't modify the client at all: the SDK sends the proxied call itself and
reads the provider URL and key from the client per call, and the fallback
goes to the client's own `baseURL`.

**Real Anthropic clients are detected as Anthropic.** The @anthropic-ai/sdk
client also has a legacy `completions.create`, so 1.4.0 detected it as an
OpenAI client and never wrapped `messages.create`: Claude calls were not
observed, admitted or stopped.

**Anthropic clients run in govern mode.** The proxy has no Anthropic Messages
route yet, so steer/route for an Anthropic client could only fail open around
TensorCost. `maxLayer: "steer"` or `"route"` (or `appliedMode: true`) on an
Anthropic client now runs as `"govern"`: every call is checked with
`/v1/admit` before it goes to Anthropic. One warning per process says so;
set `maxLayer: "govern"` to silence it.

**Bedrock and Vertex say govern isn't enforced.** There is no `/v1/admit`
check for them yet, so `maxLayer: "govern"` only observes. The SDK now logs
one warning per process saying so, and the README no longer lists govern as
supported for them.

**Govern-layer refusals are typed by their code.** 1.4.0 threw
`TensorCostModelGovernanceDeniedError` ("request refused at govern layer")
for every `/v1/admit` refusal except a run stop, budget caps included.

**Streaming proxy errors.** After retries ran out, a 5xx or 429 on the
streaming path was reported as `TensorCostProviderError`; it is
`TensorCostProxyError` / `TensorCostQuotaError` now.

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
- `shouldFallBackToProvider()` returns false for `TensorCostProviderError`
  and for errors that aren't `TensorCost*` outages (1.4.0 returned true).
- A relayed provider 4xx with an openai@4 client: in 1.4.0 the fail-open
  retry crashed with the `defaultHeaders` TypeError above, so callers never
  got `openai.BadRequestError` from it; with fail-open off they got
  `TensorCostProxyError`. 1.5.0 throws the openai error type the client
  builds, after one call.
- A TensorCost 4xx that isn't an outage (a rejected key, a refusal with an
  unknown code, a firewall page) now stops the call instead of going around
  it.
- `failOpenEnabled: false` now also fails the call when `/v1/admit` is down.
- `AdmitResult.refusal` is set on every refusal, not only `RUN_STOPPED`, and
  can be a `TensorCostGatewayRejectedError` or `TensorCostKeyRejectedError`.
- Anthropic clients configured for steer/route make one `/v1/admit` call per
  request instead of a proxy call.
- `wrap()` no longer sets `baseURL`, `defaultHeaders`, `x-tc-provider-url` or
  `x-tc-provider-auth` on the client. If you read those back, read your own
  config instead.

## 1.4.0 — 2026-10-25

### Added

**`TensorCostRunStoppedError`** — maps proxy `403` / `RUN_STOPPED` (an operator
stopped the agent run). Carries `workflowId`, `stoppedBy`, `stoppedAt` and
`reason`. Terminal: it never falls back to the provider, so a stopped run
stays stopped. Before this, a stop surfaced as a generic provider error and
applied mode's fail-open path would call the provider directly.

### Fixed

**Govern layer: a stopped run throws `TensorCostRunStoppedError`.** The proxy
now refuses a stopped run on `/v1/admit` too. The OpenAI and Anthropic
wrappers throw the stop with its detail instead of a generic "refused at
govern layer" governance error, and never call the provider.

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
