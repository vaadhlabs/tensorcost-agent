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
 * Install applied-mode routing on the client. Called once by `wrap()`.
 *
 * Mutates `client.baseURL` to point at the proxy endpoint and injects
 * `x-tc-provider-url` / `x-tc-provider-auth` into `client.defaultHeaders`.
 * These headers survive for the lifetime of the client — every subsequent
 * request automatically carries them.
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

  // Build the auth header value to forward. The proxy sends this value
  // as the upstream Authorization (or api-key) header verbatim.
  let providerAuth: string;
  if (provider === "anthropic") {
    providerAuth = anthropicAuth(apiKey);
  } else if (isAzureOpenAI(c)) {
    providerAuth = azureAuth(apiKey);
  } else {
    providerAuth = bearerAuth(apiKey);
  }

  // Redirect the client's base URL to the proxy. The path suffix
  // (/api/inference-proxy/v1) is appended here so that the OpenAI-compatible
  // client doesn't double-prepend its own /v1 path. Both the OpenAI and
  // Anthropic SDKs append the method path to baseURL; the proxy endpoint is
  // OpenAI-compatible, so the client will build:
  //   baseURL + "/chat/completions"
  // We set baseURL to the proxy's parent path to get the right URL.
  const cleanProxyUrl = proxyUrl.replace(/\/+$/, "");
  c["baseURL"] = `${cleanProxyUrl}/api/inference-proxy/v1`;

  // Inject the custom headers into the client's defaultHeaders object.
  // Both SDKs merge defaultHeaders with every outgoing request.
  const existing = c["defaultHeaders"];
  const headers: Record<string, string> =
    existing && typeof existing === "object" && !Array.isArray(existing)
      ? (existing as Record<string, string>)
      : {};
  headers["x-tc-provider-url"] = originalBaseUrl;
  headers["x-tc-provider-auth"] = providerAuth;
  if (scope?.application) {
    headers["x-tc-application"] = scope.application;
  }
  if (scope?.tags && scope.tags.length > 0) {
    headers["x-tc-tags"] = scope.tags.join(",");
  }
  if (scope?.customer) {
    headers["x-tc-customer"] = scope.customer;
  }
  if (scope?.feature) {
    headers["x-tc-feature"] = scope.feature;
  }
  if (scope?.teamId) {
    headers["x-tc-team-id"] = scope.teamId;
  }
  c["defaultHeaders"] = headers;
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
