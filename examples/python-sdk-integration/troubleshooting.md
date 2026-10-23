# Troubleshooting — Python SDK integration

## My wrapped calls aren't showing up in the dashboard

The most common reasons:

**1. The API key is wrong or missing.**

`wrap()` raises `MissingConfigError` immediately if `TENSORCOST_API_KEY`
is absent (and no `api_key` kwarg was passed). If it raises but your app
catches and swallows it, you'll see normal client behavior with no
observations. Check application startup logs for `MissingConfigError`.

If the key is present but wrong (e.g. a `tc_test_` key in production),
the token-exchange call returns `401` and the SDK swallows it silently
(fail-open) and logs a warning via `logging.getLogger("tensorcost")`.
Raise that logger's level to `DEBUG` to see it:

```python
import logging
logging.getLogger("tensorcost").setLevel(logging.DEBUG)
```

**2. You wrapped the client after already making calls.**

`wrap()` patches the client's methods at call time. Any calls you made
before `wrap()` are not observed. Make sure `wrap()` happens at
initialization, before your first inference call.

**3. Dashboard lag.**

After the first successful observation, data appears in the TensorCost
dashboard with a lag of up to 2 minutes. Wait 2 minutes and refresh. If
you see nothing after 5 minutes, observations are likely not reaching
the backend — see the network section below.

**4. The `base_url` points at the wrong environment.**

If `TENSORCOST_BASE_URL=https://api.dev.tensorcost.com`, your
observations go to the dev backend, not production. The production URL
is `https://api.tensorcost.com`.

---

## Network connectivity / firewall

The SDK makes outbound HTTPS requests to `TENSORCOST_BASE_URL` (default
`https://api.tensorcost.com`) via `httpx`. If your environment blocks
outbound HTTPS to arbitrary hosts, add `api.tensorcost.com` (or
`api.dev.tensorcost.com` for dev) to your egress allowlist.

If your environment runs behind an HTTPS-intercepting proxy, configure
`httpx`'s standard `HTTPX_CA_BUNDLE` / `SSL_CERT_FILE` environment
variables with your proxy's CA bundle. The SDK does not disable TLS
verification.

---

## The first wrapped call fails with a network error

The SDK fails open by default: it catches observation errors and logs
them, and your underlying provider call still succeeds — TensorCost is
never on the critical path in observe-only mode.

To see the underlying error during development, raise the logger level
(see above) — the SDK does not expose a `fail_open=False` toggle for
observe-mode network errors; the fail-open behavior there is unconditional.
Applied-mode failures have a separate `fail_open_enabled` control (see
below).

---

## Applied mode — proxy errors

When `applied_mode=True`, the SDK routes inference requests through the
TensorCost proxy. If the proxy is unreachable or returns 5xx repeatedly,
the circuit breaker opens after 3 consecutive failures and routes
requests directly to the upstream provider. Watch for `on_fallback`
lifecycle events:

```python
from tensorcost import wrap, LifecycleEvent

def on_event(event: LifecycleEvent) -> None:
    if event.kind == "on_fallback":
        print("proxy unavailable, falling back to direct provider", event)

client = wrap(
    OpenAI(...),
    applied_mode=True,
    proxy_url=os.environ["TENSORCOST_PROXY_URL"],
    on_lifecycle_event=on_event,
)
```

If you're in a network environment where direct provider access is not
available (e.g. inside a VPC where only the proxy can reach the
internet), set `fail_open_enabled=False` so proxy failures raise
`TensorCostProxyError` instead of silently falling back.

---

## Azure OpenAI: observations show `provider_id: openai`, not `azure_openai`

This is expected for the SDK-wrap path — see the note in the
[README](./README.md#azure-openai-note). The SDK detects the client
class (`openai.AzureOpenAI` still uses the `openai` package's request
machinery), not the backing endpoint. If you need spend attributed to
`azure_openai` explicitly, set up the
[Azure OpenAI direct-provider connection](../direct-providers/azure-openai/)
in addition to (or instead of) the SDK wrap.

---

## `TypeError: wrap() takes 1 positional argument`

Check your installed version — `wrap()` accepts keyword-only config
arguments after the client. Update: `pip install -U tensorcost`.

---

## Getting help

- Email: support@tensorcost.com
- Trust portal: https://security.tensorcost.com
