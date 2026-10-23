# Auth and API key management

## How the API key is issued

When you create an account on TensorCost, the platform issues a
**long-lived tenant API key** from the Settings → API Keys page. The key
is a randomly generated opaque string prefixed `tc_live_` (production)
or `tc_test_` (dev/test environments).

The key is tied to your tenant and has no built-in expiry, but you can
rotate it at any time from the console.

## Using the key with the SDK

Set the `TENSORCOST_API_KEY` environment variable before starting your app:

```bash
export TENSORCOST_API_KEY=tc_live_xxxxxxxxxxxx
```

Or pass it directly to `wrap()`:

```python
from tensorcost import wrap
client = wrap(OpenAI(...), api_key="tc_live_xxxxxxxxxxxx")
```

The environment variable approach is preferred — it keeps the key out of
your source code and is easier to rotate without a code change.

## How the SDK authenticates

Observe-mode telemetry authenticates with a short-lived JWT obtained via
`POST /api/inference-proxy/sdk-token/exchange`. The long-lived
`TENSORCOST_API_KEY` never leaves the SDK process after the first
exchange — it is cached in memory and used to mint a new JWT once the
previous one expires.

In applied mode (`applied_mode=True`), the API key is additionally sent
as `x-tc-api-key` on each request to the inference proxy
(`TENSORCOST_PROXY_URL`). The proxy validates it against the same key
database as the observation endpoint.

## Multi-environment usage

If you run the same application code in multiple environments (staging,
production, etc.), differentiate observations with the `environment`
kwarg or `TENSORCOST_ENVIRONMENT`:

```python
# staging
client = wrap(OpenAI(...), environment="staging")

# production
client = wrap(OpenAI(...), environment="production")
```

Both share the same tenant API key. The `environment` tag is stamped on
every observation and is visible as a filter in the TensorCost dashboard.

## Rotating the API key

1. Go to Settings → API Keys in the TensorCost console.
2. Click "Generate new key" — the old key remains valid during the transition window.
3. Update your secret store (AWS Secrets Manager, Vault, k8s Secret, etc.)
   and redeploy your app.
4. Revoke the old key from the console once you've confirmed the new one is active.

There is no automated rotation mechanism in the SDK itself — key rotation
is an operator-level procedure.

## Applied mode — provider auth

The customer's provider API key (OpenAI key, Azure OpenAI key, Anthropic
key, etc.) is NOT forwarded to TensorCost in default observe-only mode.
In applied mode, the underlying provider credential is included in the
`x-tc-provider-auth` header to the proxy so the proxy can make the
upstream call on your behalf. The proxy does not log or store provider
keys beyond the lifetime of the request.
