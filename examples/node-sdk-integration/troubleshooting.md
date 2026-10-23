# Troubleshooting — Node SDK integration

## My wrapped calls aren't showing up in the dashboard

The most common reasons:

**1. The API key is wrong or missing.**

`wrap()` throws `MissingConfigError` immediately if `TENSORCOST_API_KEY`
is absent. If it throws but your app catches and swallows it, you'll see
the normal client behavior with no observations. Look for
`MissingConfigError` in your application logs.

If the key is present but wrong (e.g. a `tc_test_` key in production),
the observation POST returns `401` and the SDK swallows it silently (fail-open).
Enable debug logging to see this:

```js
const client = wrap(new OpenAI(...), {
  apiKey: process.env.TENSORCOST_API_KEY,
  failOpen: false,  // temporarily; re-enables throw on observation error
});
```

**2. You wrapped the client after already making calls.**

`wrap()` patches the client methods at call time. Any calls you made
before `wrap()` are not observed. Make sure `wrap()` happens at
initialization, before your first inference call.

**3. Dashboard lag.**

After the first successful observation POST, data appears in the
TensorCost dashboard with a lag of up to 2 minutes (the backend's
observation-processing pipeline runs in near-real-time but not
instantaneously). Wait 2 minutes and refresh. If you see nothing after
5 minutes, the observations are likely not reaching the backend — see
the network troubleshooting section below.

**4. The `baseUrl` points at the wrong environment.**

If you set `TENSORCOST_BASE_URL=https://api.dev.tensorcost.com`, your
observations go to the dev backend, not production. The production URL is
`https://api.tensorcost.com`. Check which dashboard you're looking at and
which `TENSORCOST_BASE_URL` your app is using.

---

## Network connectivity / firewall

The SDK makes outbound HTTPS POSTs to `TENSORCOST_BASE_URL` (default
`https://api.tensorcost.com`) from inside your app process. If your
environment blocks outbound HTTPS to arbitrary hosts, add `api.tensorcost.com`
(or `api.dev.tensorcost.com` for dev) to your egress allowlist.

The SDK uses Node's native `fetch` (Node 18+). If your environment runs
behind an HTTPS proxy, set `HTTPS_PROXY` or `NODE_EXTRA_CA_CERTS` as
appropriate. The SDK does not set `NODE_TLS_REJECT_UNAUTHORIZED=0` — if
your proxy does TLS interception you must supply the proxy CA bundle via
`NODE_EXTRA_CA_CERTS`.

---

## The first wrapped call fails with a network error

If `failOpen: true` (the default), the SDK catches the error and logs it
via `console.warn`. Your LLM call still succeeds — TensorCost is never on
the critical path in observe-only mode.

If you need to see the error clearly during development:

```js
const client = wrap(new OpenAI(...), {
  failOpen: false,
});
```

This makes the SDK throw instead of swallowing network errors, so you can
see exactly what's failing.

---

## Applied mode — proxy errors

When `maxLayer` is `steer` or `route`, the SDK routes inference requests through the
TensorCost proxy. If the proxy is unreachable or returns 5xx, the circuit
breaker opens after 3 consecutive failures and routes requests directly
to the upstream provider (fail-open circuit). Watch for `on_fallback` events:

```js
const client = wrap(new OpenAI(...), {
  maxLayer: "route",
  proxyUrl: process.env.TENSORCOST_PROXY_URL,
  onLifecycleEvent(event) {
    if (event.kind === "on_fallback") {
      console.warn("proxy unavailable, falling back to direct provider", event);
    }
  },
});
```

If you're in a network environment where direct provider access is not
available (e.g., inside a VPC where only the proxy can reach the internet),
set `failOpenEnabled: false` to raise an error instead of silently falling
back.

---

## AWS Bedrock: observations not showing cost breakdown

Bedrock cost tracking works through observe-only mode. The `usage` field in
the Bedrock `InvokeModel` response contains `inputTokens` and `outputTokens`;
the SDK reads these and includes them in the observation. If you see
observations in the dashboard but no token counts, check that you're passing
the response through the patched method — Bedrock's `send()` is the
intercepted call. Using `client.send(new InvokeModelCommand(...))` is
correct; bypassing `client.send()` by constructing the HTTP request manually
won't be observed.

Applied mode is not supported for Bedrock clients. `wrap()` will throw if
you pass `maxLayer: "route"` with a `BedrockRuntimeClient`.

---

## TypeScript: "could not find a declaration file for module @tensorcost/sdk"

The package ships `dist/index.d.ts` in the `types` export condition. If
your TypeScript config does not pick it up, check:

1. `"moduleResolution": "bundler"` or `"moduleResolution": "node16"` in
   your `tsconfig.json` — the package uses the `exports` field in
   `package.json`, which older `"moduleResolution": "node"` does not support.
2. The package was installed correctly: `ls node_modules/@tensorcost/sdk/dist/`
   should show `index.js` and `index.d.ts`.

---

## Getting help

- Email: support@tensorcost.com
- Trust portal: https://security.tensorcost.com
