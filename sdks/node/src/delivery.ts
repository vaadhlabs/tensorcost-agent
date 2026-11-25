/**
 * Did a failed proxied call reach the provider?
 *
 * A proxied call is sent once to TensorCost, which forwards it to the
 * provider. When that goes wrong the SDK has two ways to try again: retry
 * through the proxy, or fall open and call the provider directly. Either one
 * sends the call to the provider a second time, so either one is safe only
 * when the first attempt provably never got there. Through 1.7.0 the SDK
 * retried every 5xx, timeout and network error up to five times and then fell
 * back, so a call the provider had already answered and billed (the proxy
 * then failed, timed out, or lost the response) could run six times.
 *
 * This module answers the question from what the SDK can see, and answers
 * "maybe" whenever it can't prove "no":
 *
 *   - A send that threw. Only an error from setting up the connection proves
 *     nothing was written: refused, DNS, unreachable, a connect timeout, a
 *     TLS handshake or certificate failure. A reset, a closed socket or a
 *     timeout may come after the body was written and the proxy forwarded it.
 *
 *   - A non-2xx answer. The proxy marks every error with `x-tc-error-origin`:
 *     `provider` when the status is the provider's (relayed, or the proxy's
 *     answer about a provider that did not answer in time), `proxy` when it
 *     is the proxy's own. The gateway marks its own answers `gateway`. The
 *     proxy also sends `x-should-retry: false` (the OpenAI and Anthropic SDK
 *     convention) on every error it chose not to resend because the provider
 *     may have run the call. See inference-proxy-service error-origin.ts and
 *     the gateway's proxy.middleware.ts.
 *
 *     `proxy` alone does not prove "not sent": the proxy also marks its 502
 *     for a provider answer it could not translate, and any unhandled error
 *     after the forward. What does prove it is narrower, below.
 */

/** Set by inference-proxy-service and the gateway on every 4xx/5xx. */
export const ERROR_ORIGIN_HEADER = "x-tc-error-origin";
/** OpenAI/Anthropic SDK convention; "false" means do not send this again. */
export const SHOULD_RETRY_HEADER = "x-should-retry";
/**
 * Set by inference-proxy-service on its answer for a provider send that got
 * no answer (unanswered-send.ts): `provider_unreachable`,
 * `provider_connection_lost` or `provider_timeout`.
 */
export const UPSTREAM_ERROR_HEADER = "x-tc-upstream-error";

/** What a non-2xx answer says about where the call got to. */
export type ErrorAnswer =
  /** TensorCost answered before forwarding: safe to retry, safe to fall back. */
  | "unsent"
  /** The provider's own answer, relayed: surface it as the provider's error. */
  | "provider"
  /** Can't tell: the provider may have run and billed it. Never resend. */
  | "maybe_sent";

/** Error codes that only happen while connecting, before a byte is written. */
const CONNECT_PHASE_CODES = new Set([
  "ECONNREFUSED",
  "ENOTFOUND",
  "EAI_AGAIN",
  "EAI_FAIL",
  "EAI_NONAME",
  "ENETUNREACH",
  "EHOSTUNREACH",
  "EHOSTDOWN",
  "EADDRNOTAVAIL",
  "UND_ERR_CONNECT_TIMEOUT",
  "ERR_INVALID_URL",
  // TLS certificate verification and the handshake itself. Not a prefix
  // match on ERR_SSL_: OpenSSL also raises those after the handshake, while
  // the response is read, when the request may already have run. Same list
  // the proxy uses for its own sends (proxy.service.ts TLS_HANDSHAKE_CODES).
  "CERT_HAS_EXPIRED",
  "CERT_NOT_YET_VALID",
  "CERT_REVOKED",
  "CERT_UNTRUSTED",
  "CERT_SIGNATURE_FAILURE",
  "CERT_REJECTED",
  "DEPTH_ZERO_SELF_SIGNED_CERT",
  "SELF_SIGNED_CERT_IN_CHAIN",
  "HOSTNAME_MISMATCH",
  "UNABLE_TO_GET_ISSUER_CERT",
  "UNABLE_TO_GET_ISSUER_CERT_LOCALLY",
  "UNABLE_TO_VERIFY_LEAF_SIGNATURE",
  "ERR_TLS_CERT_ALTNAME_INVALID",
  "ERR_TLS_HANDSHAKE_TIMEOUT",
  "ERR_SSL_WRONG_VERSION_NUMBER",
  "ERR_SSL_UNSUPPORTED_PROTOCOL",
  "ERR_SSL_NO_PROTOCOLS_AVAILABLE",
  "ERR_SSL_TLSV1_ALERT_PROTOCOL_VERSION",
  "ERR_SSL_SSLV3_ALERT_HANDSHAKE_FAILURE",
  "ERR_SSL_TLSV1_ALERT_UNKNOWN_CA",
]);

/** Node system errors name the call that failed; these two run before any write. */
const CONNECT_SYSCALLS = new Set(["connect", "getaddrinfo"]);

/**
 * The same codes as they appear in a message, for an error that carries no
 * `code` ("connect ECONNREFUSED 10.0.0.5:443", or the gateway relaying its
 * own connect failure in a 502 body). Connect-phase codes only.
 */
const CONNECT_MESSAGE_RE =
  /\b(ECONNREFUSED|ENOTFOUND|EAI_AGAIN|ENETUNREACH|EHOSTUNREACH|EHOSTDOWN|EADDRNOTAVAIL|UND_ERR_CONNECT_TIMEOUT)\b/;

/**
 * True when a fetch that threw provably wrote nothing: the error, or one in
 * its cause chain (undici wraps the socket error in TypeError("fetch
 * failed")), is a connect-phase failure. False for everything else,
 * including an abort or timeout, a reset (ECONNRESET, EPIPE, undici's "other
 * side closed"), or an error the SDK doesn't recognise: those can happen
 * after the body was written and forwarded.
 */
export function fetchErrorSentNothing(err: unknown): boolean {
  const seen = new Set<unknown>();
  const walk = (e: unknown, depth: number): boolean => {
    if (e === null || typeof e !== "object" || depth > 6 || seen.has(e)) return false;
    seen.add(e);
    const rec = e as { code?: unknown; syscall?: unknown; name?: unknown; message?: unknown; cause?: unknown };
    if (rec.name === "AbortError" || rec.name === "TimeoutError") return false;
    if (typeof rec.code === "string" && CONNECT_PHASE_CODES.has(rec.code)) return true;
    if (typeof rec.syscall === "string" && CONNECT_SYSCALLS.has(rec.syscall)) return true;
    if (typeof rec.message === "string" && CONNECT_MESSAGE_RE.test(rec.message)) return true;
    return walk(rec.cause, depth + 1);
  };
  return walk(err, 0);
}

/**
 * Classify a non-2xx, non-refusal answer from the proxy (or the gateway in
 * front of it). `body` is the parsed error body when the caller read it.
 *
 * In order:
 *   0. The proxy's 502 `x-tc-upstream-error: provider_unreachable` without
 *      `x-should-retry: false` (isProviderUnreachable): the proxy could not
 *      connect to the provider (connect, DNS or TLS, after its own one
 *      retry), so nothing of the call left it and nothing was billed. The
 *      proxy marks it origin `provider`, so it has to come before rule 1.
 *      A routed call's answer carries `x-should-retry: false` (an earlier
 *      route may have run) and stays "maybe_sent".
 *   1. origin `provider`: the provider's answer. Never resent: it would
 *      answer the same, or (a 5xx, the proxy's 504) it may already have run.
 *   2. `x-should-retry: false`: whoever sent it says don't. Always honoured.
 *   3. 429 and 402 not from the provider: TensorCost's rate limit and plan
 *      ceiling, both checked before the call is forwarded. An unmarked 429
 *      (a proxy older than the origin header) counts too: a 429 is a
 *      rejection, and nobody runs or bills one.
 *   4. 503 from the gateway (its breaker is open: no forward attempt) or the
 *      proxy (every 503 it sends is about something it needs before
 *      forwarding: the SDK token, the team, the provider connection), or
 *      any 503 with `x-should-retry: true`.
 *   5. 502 from the gateway carrying its own connect failure to the proxy
 *      (UPSTREAM_PROXY_ERROR, "connect ECONNREFUSED ..."): the proxy was
 *      never reached.
 *   6. Anything else is "maybe_sent". That includes a 500 or 502 marked
 *      `proxy` (an unhandled error, or a provider answer the proxy could not
 *      translate, both after the forward), a gateway 502/504 for a proxy that
 *      reset or timed out mid-call, and any 5xx with no origin header at all.
 *      The provider may have run those, so the SDK surfaces the error rather
 *      than risk billing the call twice.
 */
export function classifyErrorAnswer(status: number, headers: Headers, body: unknown): ErrorAnswer {
  const origin = headers.get(ERROR_ORIGIN_HEADER)?.trim().toLowerCase() ?? null;
  const shouldRetry = headers.get(SHOULD_RETRY_HEADER)?.trim().toLowerCase() ?? null;

  if (isProviderUnreachable(status, headers)) return "unsent";
  if (origin === "provider") return "provider";
  if (shouldRetry === "false") return "maybe_sent";
  if (status === 429 || status === 402) return "unsent";
  if (status === 503 && (origin === "gateway" || origin === "proxy" || shouldRetry === "true")) {
    return "unsent";
  }
  if (status === 502 && origin === "gateway" && gatewayCouldNotConnect(body)) return "unsent";
  return "maybe_sent";
}

/**
 * The proxy's 502 for a provider it could not connect to, with nothing sent
 * and no `x-should-retry: false`. "unsent" to classifyErrorAnswer, but not
 * an outage of TensorCost: the proxy is healthy and already retried the
 * provider once, so the proxy client neither retries it nor counts it
 * against the circuit, and the wrapper calls the provider directly when
 * fail-open is on. `provider_connection_lost` and `provider_timeout` may
 * have run, carry `x-should-retry: false`, and never match.
 */
export function isProviderUnreachable(status: number, headers: Headers): boolean {
  if (status !== 502) return false;
  const kind = headers.get(UPSTREAM_ERROR_HEADER)?.trim().toLowerCase();
  const shouldRetry = headers.get(SHOULD_RETRY_HEADER)?.trim().toLowerCase();
  return kind === "provider_unreachable" && shouldRetry !== "false";
}

/** The gateway's 502 body for a connect failure to the proxy. */
function gatewayCouldNotConnect(body: unknown): boolean {
  if (body === null || typeof body !== "object") return false;
  const error = (body as { error?: unknown }).error;
  if (error === null || typeof error !== "object") return false;
  const { code, message } = error as { code?: unknown; message?: unknown };
  return code === "UPSTREAM_PROXY_ERROR" && typeof message === "string" && CONNECT_MESSAGE_RE.test(message);
}
