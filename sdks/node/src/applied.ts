/**
 * Applied-mode wiring — Layer 2.
 *
 * When the customer opts into applied mode, the SDK routes every inference
 * request through the TensorCost inference-proxy instead of directly to the
 * upstream AI provider. The proxy decides per-request whether to re-route
 * or pass through unchanged; the SDK doesn't make that call.
 *
 * The mechanism is straightforward: at `wrap()` time we read the
 * provider's original baseURL and auth from the client, permanently
 * redirect the client's baseURL to the proxy endpoint, and inject two
 * custom headers the proxy expects:
 *
 *   x-tc-provider-url   — the original provider base URL
 *   x-tc-provider-auth  — the auth header value to forward upstream
 *
 * Both the OpenAI SDK and the Anthropic SDK expose `baseURL` and
 * `defaultHeaders` as first-class properties, so we can set them
 * without monkey-patching the HTTP layer.
 *
 * Bedrock is not supported in v1. The AWS SDK uses SigV4 signing on
 * regional endpoints and doesn't expose baseURL / defaultHeaders in
 * the same shape. Passing a Bedrock-shaped client with appliedMode=true
 * throws `TensorCostConfigError` at wrap time.
 *
 * Observe-only telemetry (the 0.2.x substrate) continues to fire
 * after every call regardless of this setting.
 */

import { TensorCostConfigError } from "./config.js";

/** Proxy chat-completions path — fixed in v1. */
const PROXY_PATH = "/api/inference-proxy/v1/chat/completions";

/**
 * Client shapes we understand. All detection is duck-typed; no import
 * of the actual openai or @anthropic-ai/sdk packages.
 */

/** Header value the proxy expects for OpenAI + Azure OpenAI. */
function bearerAuth(apiKey: string): string {
  return `Bearer ${apiKey}`;
}

/** Header value the proxy expects for Anthropic. */
function anthropicAuth(apiKey: string): string {
  return `x-api-key ${apiKey}`;
}

/**
 * Read the original baseURL from a duck-typed client. Both the OpenAI
 * and Anthropic SDKs expose a `baseURL` string property on the client
 * instance (the actual HTTP origin, e.g. "https://api.openai.com/v1").
 * Returns null when the property is absent or not a string.
 */
function readBaseUrl(client: Record<string, unknown>): string | null {
  const raw = client["baseURL"];
  if (typeof raw === "string" && raw.length > 0) {
    return raw.replace(/\/+$/, "");
  }
  return null;
}

/**
 * Read the API key from the client. OpenAI SDK exposes `client.apiKey`;
 * Anthropic SDK exposes `client.apiKey` too. Returns null when absent.
 */
function readApiKey(client: Record<string, unknown>): string | null {
  const raw = client["apiKey"];
  return typeof raw === "string" && raw.length > 0 ? raw : null;
}

/**
 * Detect whether this client looks like an Azure OpenAI client (the
 * openai package's `AzureOpenAI` subclass). Azure uses `api-key` header
 * auth and a custom endpoint URL, so the auth value we forward differs
 * from vanilla OpenAI.
 *
 * Detection: Azure clients have an `azure` property set to true, OR
 * their baseURL contains `.openai.azure.com`.
 */
function isAzureOpenAI(client: Record<string, unknown>): boolean {
  if (client["azure"] === true) return true;
  const url = readBaseUrl(client) ?? "";
  return url.includes(".openai.azure.com");
}

/**
 * Azure OpenAI uses an `api-key` header rather than `Authorization: Bearer`.
 * The proxy forwards `x-tc-provider-auth` verbatim as the upstream auth
 * header, so we need to format it as the proxy expects: "api-key <value>".
 *
 * Note: The proxy currently only supports `Authorization` and `api-key`
 * style headers. This value is forwarded in the `x-tc-provider-auth`
 * header and then sent by the proxy as the upstream Authorization.
 */
function azureAuth(apiKey: string): string {
  return `api-key ${apiKey}`;
}

/**
 * Check a client can be used in applied mode. Called once by `wrap()`.
 *
 * Since 1.5.0 it changes nothing on the client (see the note at the end);
 * the wrapper sends the proxied call itself with the values from
 * `providerTarget()`.
 *
 * @param client   - The customer's AI SDK client (duck-typed).
 * @param proxyUrl - The TensorCost proxy base URL (already stripped of trailing slash).
 * @param provider - "openai" | "anthropic", as detected by detectProvider().
 *
 * @throws TensorCostConfigError
 *   - If the client doesn't expose `baseURL` (can't determine the original
 *     provider URL to forward in `x-tc-provider-url`).
 *   - If the client doesn't expose `apiKey` (can't build `x-tc-provider-auth`).
 *   - If the provider is "bedrock" (not yet supported in v1).
 */
export function installAppliedMode(
  client: unknown,
  proxyUrl: string,
  provider: "openai" | "anthropic",
  scope?: {
    application?: string | null;
    tags?: string[];
    customer?: string | null;
    feature?: string | null;
    teamId?: string | null;
  },
): void {
  if (provider as string === "bedrock") {
    // Bedrock uses SigV4 signing on regional HTTPS endpoints. The AWS
    // SDK does not expose baseURL / defaultHeaders in the same shape,
    // so we can't intercept the request path in v1. Pass-through-only
    // mode is the correct posture; applied mode for Bedrock is a
    // follow-up workstream.
    throw new TensorCostConfigError(
      "TensorCost applied mode is not yet supported for AWS Bedrock clients. " +
        "Set appliedMode: false to continue with observe-only mode, or use " +
        "an OpenAI / Anthropic client.",
    );
  }

  const c = client as Record<string, unknown>;

  const originalBaseUrl = readBaseUrl(c);
  if (!originalBaseUrl) {
    throw new TensorCostConfigError(
      "TensorCost applied mode requires a client with a readable `baseURL` " +
        "property (the original provider endpoint). The client passed to " +
        "wrap() does not expose one.",
    );
  }

  const apiKey = readApiKey(c);
  if (!apiKey) {
    throw new TensorCostConfigError(
      "TensorCost applied mode requires a client with a readable `apiKey` " +
        "property to forward provider auth. The client passed to wrap() does " +
        "not expose one.",
    );
  }

  // Nothing on the client is changed. Until 1.5.0 this rewrote
  // `client.baseURL` to the proxy and replaced `client.defaultHeaders` with
  // a plain object. openai@4 has `defaultHeaders()` as a method, so every
  // call that reached the client's own request path (the fail-open
  // fallback, and every call below steer) died with "this.defaultHeaders is
  // not a function"; and the fallback, when it worked, went to the proxy
  // URL rather than the provider. The SDK sends the proxied call itself, so
  // the client only needs to be read, and `providerTarget()` reads it per
  // call so a rotated key is picked up.
  void scope;
}

function providerAuthFor(
  c: Record<string, unknown>,
  provider: "openai" | "anthropic",
  apiKey: string,
): string {
  if (provider === "anthropic") return anthropicAuth(apiKey);
  if (isAzureOpenAI(c)) return azureAuth(apiKey);
  return bearerAuth(apiKey);
}

/**
 * Where the proxy should forward this client's call: the client's own base
 * URL and its auth, formatted for `x-tc-provider-url` / `x-tc-provider-auth`.
 *
 * A client that already carries those as plain `defaultHeaders` entries
 * (callers who set them by hand) keeps them.
 */
export function providerTarget(
  client: unknown,
  provider: "openai" | "anthropic",
  defaultUrl: string,
): { providerUrl: string; providerAuth: string } {
  const c = (client && typeof client === "object" ? client : {}) as Record<string, unknown>;
  const dh = c["defaultHeaders"];
  const manual =
    dh && typeof dh === "object" && !Array.isArray(dh) ? (dh as Record<string, unknown>) : {};
  const manualUrl = typeof manual["x-tc-provider-url"] === "string" ? manual["x-tc-provider-url"] : null;
  const manualAuth =
    typeof manual["x-tc-provider-auth"] === "string" ? manual["x-tc-provider-auth"] : null;
  const apiKey = readApiKey(c);
  return {
    providerUrl: manualUrl ?? readBaseUrl(c) ?? defaultUrl,
    providerAuth: manualAuth ?? (apiKey ? providerAuthFor(c, provider, apiKey) : ""),
  };
}

export const PROXY_PATH_SUFFIX = PROXY_PATH;

export const __test__ = {
  readBaseUrl,
  readApiKey,
  isAzureOpenAI,
  bearerAuth,
  anthropicAuth,
  azureAuth,
};
