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

```js
import { wrap } from "@tensorcost/sdk";
const client = wrap(new OpenAI(...), { apiKey: "tc_live_xxxxxxxxxxxx" });
```

The environment variable approach is preferred — it keeps the key out of
your source code and is easier to rotate without a code change.

## How the SDK authenticates

Every observation the SDK emits is a `POST` to
`https://api.tensorcost.com/api/proxy/observation` (or your
`TENSORCOST_BASE_URL`) with the key in the `Authorization: Bearer <key>`
header. The endpoint is idempotent — duplicate observations with the same
correlation UUID are deduplicated server-side.

## Multi-environment usage

If you run the same application code in multiple environments (staging,
production, etc.), you can differentiate observations with the
`environment` option:

```js
// staging
const client = wrap(new OpenAI(...), {
  apiKey: process.env.TENSORCOST_API_KEY,
  environment: "staging",
});

// production
const client = wrap(new OpenAI(...), {
  apiKey: process.env.TENSORCOST_API_KEY,
  environment: "production",
});
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

## Applied mode — proxy authentication

When `maxLayer` is `steer` or `route`, the SDK also sends the API key as
`x-tc-api-key` on each request to the inference proxy (`TENSORCOST_PROXY_URL`).
The proxy validates it against the same key database as the observation endpoint.

The customer's provider API key (OpenAI key, Anthropic key, etc.) is NOT
forwarded to TensorCost in default observe-only mode. In applied mode, the
key is included in the `x-tc-provider-auth` header to the proxy so the proxy
can make the upstream call on your behalf. The proxy does not log or store
provider keys beyond the lifetime of the request.
