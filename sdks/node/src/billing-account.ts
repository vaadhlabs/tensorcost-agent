/**
 * The account the provider billed a call under, as the provider names it.
 *
 * TensorCost lets a provider's discount reach the counted figures only for
 * a provider-day where every metered call names its billing account, and
 * one call without it keeps the whole day at the metered price. Proxied
 * calls are labelled by the proxy; this is the same label for the calls an
 * SDK makes itself, so the observation carries it as `billing_account`.
 *
 * The formats and the host rules are the backend's, copied from
 * @tensorcost/contracts billing-account.ts (`billingAccountOfCall`), and
 * must stay identical to it: the two sides only meet if both write the
 * same string.
 *
 *   anthropic     `anthropic-organization-id` response header
 *   openai        `openai-organization` response header (not the project)
 *   azure_openai  the resource: the endpoint host's first label, lower-cased
 *   vertex        the project in /projects/<id>/ of the request URL
 *   bedrock       the 12-digit AWS account of the credentials that signed it
 *
 * Every reader here returns null rather than throw. A missing or unreadable
 * header costs the call its label, never the call.
 */

/** What a fetch Response's headers or a plain header record look like. */
export type HeaderBag =
  | { get(name: string): string | null }
  | Readonly<Record<string, string | string[] | number | undefined | null>>;

const MAX_LEN = 128;

/** Trimmed, at most 128 characters, or null (the proxy's DTO caps it at 128). */
export function normalizeBillingAccount(v: unknown): string | null {
  if (typeof v !== "string") return null;
  const s = v.trim();
  return s.length > 0 && s.length <= MAX_LEN ? s : null;
}

function header(headers: HeaderBag | null | undefined, name: string): string | null {
  if (!headers || typeof headers !== "object") return null;
  if (typeof (headers as { get?: unknown }).get === "function") {
    return normalizeBillingAccount((headers as { get(n: string): string | null }).get(name));
  }
  const want = name.toLowerCase();
  for (const [k, v] of Object.entries(headers as Record<string, unknown>)) {
    if (k.toLowerCase() !== want) continue;
    return normalizeBillingAccount(Array.isArray(v) ? v[0] : v);
  }
  return null;
}

const isHost = (host: string, domain: string) => host === domain || host.endsWith(`.${domain}`);

function billingAccountOfCallUnsafe(url: string, headers: HeaderBag | null | undefined): string | null {
  let u: URL;
  try {
    u = new URL(url);
  } catch {
    return null;
  }
  const host = u.hostname.toLowerCase();
  if (isHost(host, "anthropic.com")) return header(headers, "anthropic-organization-id");
  if (isHost(host, "openai.com")) return header(headers, "openai-organization");
  if (
    host.endsWith(".openai.azure.com") ||
    host.endsWith(".cognitiveservices.azure.com") ||
    host.endsWith(".services.ai.azure.com")
  ) {
    return normalizeBillingAccount(host.split(".")[0]);
  }
  if (host === "aiplatform.googleapis.com" || host.endsWith("-aiplatform.googleapis.com")) {
    const m = /\/projects\/([^/]+)\//.exec(u.pathname);
    return m ? normalizeBillingAccount(decodeURIComponent(m[1]!)) : null;
  }
  return null;
}

/**
 * The billing account a call went to, from its URL and the provider's
 * response headers. Null where neither names one, for a host we don't know
 * (a gateway, the TensorCost proxy itself), and when reading throws.
 */
export function billingAccountOfCall(
  url: string | null | undefined,
  headers: HeaderBag | null | undefined,
): string | null {
  if (typeof url !== "string" || url.length === 0) return null;
  try {
    return billingAccountOfCallUnsafe(url, headers);
  } catch {
    return null;
  }
}

/** A 12-digit AWS account id, or null. */
export function awsAccountId(v: unknown): string | null {
  return typeof v === "string" && /^\d{12}$/.test(v.trim()) ? v.trim() : null;
}

/** Resolves with what `p` resolves to, or null if it rejects or hasn't settled by the next macrotask. */
function settledOrNull<T>(p: PromiseLike<T>): Promise<T | null> {
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(null), 0);
    Promise.resolve(p).then(
      (v) => {
        clearTimeout(timer);
        resolve(v);
      },
      () => {
        clearTimeout(timer);
        resolve(null);
      },
    );
  });
}

/** The client's base URL (openai, @anthropic-ai/sdk and AzureOpenAI all expose `baseURL`). */
export function clientBaseUrl(client: unknown): string | null {
  try {
    const v = (client as { baseURL?: unknown } | null)?.baseURL;
    return typeof v === "string" ? v : null;
  } catch {
    return null;
  }
}

/**
 * The billing account of a call the provider answered, read off the
 * APIPromise the provider SDK returned.
 *
 * `asResponse()` on the openai / @anthropic-ai/sdk APIPromise hands back the
 * response the call already received (it shares the promise the parsed body
 * comes from), so it adds no request and doesn't touch what the caller gets.
 * Call this after the call itself has resolved: the response is in hand by
 * then and this settles within microtasks. Anything else (no asResponse, a
 * rejection, a header read that throws) is null.
 */
export async function billingAccountOfApiPromise(
  call: unknown,
  url: string | null,
): Promise<string | null> {
  try {
    if (!url) return null;
    // Azure and Vertex name the account in the URL; no response needed.
    const fromUrl = billingAccountOfCall(url, null);
    if (fromUrl !== null) return fromUrl;
    const asResponse = (call as { asResponse?: unknown } | null)?.asResponse;
    if (typeof asResponse !== "function") return null;
    const res = (await settledOrNull(
      (asResponse as () => PromiseLike<unknown>).call(call),
    )) as { headers?: HeaderBag } | null;
    return billingAccountOfCall(url, res?.headers);
  } catch {
    return null;
  }
}

/**
 * The billing account on a provider error. The openai and @anthropic-ai/sdk
 * APIError keeps the response headers (a Headers object, or a plain record
 * on openai@4); an error the provider never answered has none, and on
 * Azure the URL still names the resource the call was aimed at.
 */
export function billingAccountOfError(err: unknown, url: string | null): string | null {
  try {
    const headers = (err as { headers?: unknown } | null)?.headers;
    return billingAccountOfCall(url, (headers ?? null) as HeaderBag | null);
  } catch {
    return null;
  }
}

/** `{ billing_account }` when there is one, `{}` otherwise, for spreading into an observation. */
export function billingAccountField(account: string | null): { billing_account?: string } {
  return account !== null ? { billing_account: account } : {};
}

/**
 * The URL a @google/genai Vertex client sends generateContent to, built the
 * way its ApiClient builds it: `<baseUrl>/<apiVersion>/projects/<project>/
 * locations/<location>/...`. Null for the Gemini API, Vertex express mode
 * (an API key, no project in the path) and anything that isn't that client.
 */
export function vertexCallUrl(client: unknown): string | null {
  try {
    const api = (client as { apiClient?: Record<string, unknown> } | null)?.apiClient;
    const o = api?.clientOptions as
      | { vertexai?: unknown; apiKey?: unknown; project?: unknown; location?: unknown }
      | undefined;
    if (!api || !o || o.vertexai !== true || o.apiKey || typeof o.project !== "string" || !o.project) {
      return null;
    }
    const getRequestUrl = api.getRequestUrl;
    if (typeof getRequestUrl !== "function") return null;
    const base = (getRequestUrl as () => unknown).call(api);
    if (typeof base !== "string") return null;
    const location = typeof o.location === "string" && o.location ? o.location : "global";
    return `${base.replace(/\/+$/, "")}/projects/${o.project}/locations/${location}/publishers/google/models`;
  } catch {
    return null;
  }
}

/**
 * The AWS account of the credentials that signed a Bedrock call, from the
 * Smithy context the AWS SDK v3 client filled in for it (the identity its
 * auth-scheme middleware resolved, before the build step this SDK's
 * middleware sits in). No STS call: the SDK only knows the account when
 * the credentials carry it (AWS_ACCOUNT_ID, aws_account_id in the shared
 * config, SSO, assumed roles); plain access keys without it leave the
 * field out.
 */
export function bedrockAccountOfContext(context: unknown): string | null {
  try {
    const smithy = (context as Record<string, unknown> | null)?.["__smithy_context"] as
      | { selectedHttpAuthScheme?: { identity?: { accountId?: unknown } } }
      | undefined;
    return awsAccountId(smithy?.selectedHttpAuthScheme?.identity?.accountId);
  } catch {
    return null;
  }
}
