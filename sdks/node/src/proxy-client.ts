/**
 * Applied-mode proxy HTTP client.
 *
 * This is the single place that combines retries, timeouts, typed errors,
 * telemetry hooks, and the fail-open circuit breaker. The provider wrappers
 * (openai.ts, anthropic.ts) call `proxyRequest()` instead of hitting the
 * proxy directly.
 *
 * The contract:
 *   - Returns the parsed response body on success.
 *   - On all non-success outcomes, throws a subclass of TensorCostError.
 *   - Never swallows errors — the provider wrapper's fail-open logic
 *     (fall back to direct call on proxy failure) lives in the provider
 *     wrapper, not here.
 *
 * Timeout handling:
 *   - `headersTimeoutMs` (default 2s): abort if the proxy never returns
 *     response headers (TCP accepted but wedged). Fail-open to direct.
 *   - `timeoutMs` (default 60s): wall-clock budget for the full response
 *     body after headers (model time through the proxy).
 *   - Streaming through steer is refused at the provider wrapper; proxy
 *     server supports SSE but the SDK client does not consume streams yet.
 *
 * Retry logic (1.8.0, delivery.ts):
 *   - Retries, and lets the wrapper fall back, only when the attempt
 *     provably never reached the provider: a connect-phase network error,
 *     a 429 or 402 from TensorCost, a 503 from the gateway or the proxy, a
 *     gateway 502 that says it couldn't connect to the proxy.
 *   - Never resends a call that may have reached the provider: a timeout,
 *     a reset, a 5xx the proxy didn't prove was its own pre-forward answer,
 *     anything with `x-should-retry: false`. Those surface with
 *     `mayHaveReachedProvider: true`. Through 1.7.0 every 5xx, timeout and
 *     network error was retried five times and then fell back, so one call
 *     the provider had already billed could run six times.
 *   - 1.8.2: the proxy's 502 `x-tc-upstream-error: provider_unreachable`
 *     (it couldn't connect to the provider, nothing was sent) is not
 *     retried, counts as a circuit failure, and surfaces with
 *     `mayHaveReachedProvider: false` so fail-open goes direct.
 *   - Respects Retry-After on 429. Stops at maxAttempts, with backoff.
 *   - Never retries the circuit's half-open probe, or once the circuit has
 *     opened during the call: those fail over to the provider at once.
 *   - Never retries a TensorCost refusal, whatever its status.
 *
 * How a non-2xx response is classified (unchanged split from 1.4.0, plus
 * the refusal check in front of it):
 *   - 4xx whose body carries a TensorCost refusal code, or whose
 *     x-tc-decision says action=refuse → a TensorCostRefusedError
 *     subclass. Terminal. Not a circuit failure.
 *   - 429 otherwise → TensorCostQuotaError (TensorCost's rate limit).
 *   - 402 otherwise → TensorCostQuotaError, status 402 (the plan's monthly
 *     proxy-request ceiling). Not retried.
 *   - 5xx → TensorCostProxyError (proxy or upstream unavailable).
 *   - 429, 402 or 5xx marked `x-tc-error-origin: provider` →
 *     TensorCostProviderError: the provider's answer, never resent.
 *   - any other 4xx with a JSON body → TensorCostProviderError: the
 *     provider's own error, relayed verbatim by the proxy. The wrappers
 *     rethrow it as the provider SDK's own error type when they can.
 *   - any other 4xx without one (HTML, text, empty) →
 *     TensorCostProxyStatusError when it carries x-tc-decision (the proxy
 *     answered), else TensorCostGatewayRejectedError: turned away before
 *     TensorCost.
 *
 * Circuit breaker:
 *   - Maintained on the CircuitBreaker instance passed in.
 *   - When open, throws TensorCostProxyError immediately with a clear
 *     message so the provider wrapper can activate the direct-provider
 *     fallback.
 */

import {
  TensorCostError,
  TensorCostNetworkError,
  TensorCostTimeoutError,
  TensorCostProxyError,
  TensorCostQuotaError,
  TensorCostProviderError,
  TensorCostGatewayRejectedError,
  TensorCostProxyStatusError,
  unexplained4xx,
  TensorCostKeyRejectedError,
  TensorCostRefusedError,
} from "./errors.js";
import {
  decisionSaysRefuse,
  parseRefusalBody,
  providerMessageFrom,
  refusalError,
} from "./refusal.js";
import {
  DEFAULT_RETRY_CONFIG,
  computeDelay,
  isRetryableStatus,
  retryAfterMs,
  sleep,
} from "./retry.js";
import type { RetryConfig } from "./retry.js";
import { classifyErrorAnswer, fetchErrorSentNothing, isProviderUnreachable } from "./delivery.js";
import type { ErrorAnswer } from "./delivery.js";
import type { CircuitBreaker } from "./circuit.js";
import { emitEvent } from "./telemetry.js";
import type { LifecycleEventCallback } from "./telemetry.js";
import type { Provider } from "./telemetry.js";
import {
  PROXY_HEADERS_TIMEOUT_MS,
  proxyPathForOperation,
  type ControlLayer,
} from "./layer.js";
import { SDK_VERSION, sdkCapabilityHeader } from "./version.js";

export interface ProxyRequestOptions {
  proxyUrl: string;
  bearerToken: string;
  providerUrl: string;
  providerAuth: string;
  body: Record<string, unknown>;
  model: string;
  provider: Provider;
  operation: string;
  correlationId: string;
  environment: string | null;

  /**
   * Per-agent-run attribution, forwarded as `x-tc-agent-id` /
   * `x-tc-workflow-id`. These are what let the proxy scope a spend cap to
   * a run; without the workflow id it has no run to cap. Same values the
   * observation envelope carries as `agent_id` / `workflow_id`.
   */
  agentId?: string | undefined;
  workflowId?: string | undefined;
  /** A-04 chargeback → `x-tc-customer`. */
  customer?: string | undefined;
  /** A-04 chargeback → `x-tc-feature`. */
  feature?: string | undefined;

  /** Routing-policy application scope → `x-tc-application`. */
  application?: string | null | undefined;
  /** Routing-policy tags → comma-separated `x-tc-tags`. */
  tags?: string[] | undefined;
  /** Compliance team scope → `x-tc-team-id`. */
  teamId?: string | undefined;

  retry?: RetryConfig;
  timeoutMs?: number;
  /** Max wait for proxy response headers before fail-open. Default 2s. */
  headersTimeoutMs?: number;
  onLifecycleEvent?: LifecycleEventCallback;
  circuit: CircuitBreaker;
  maxLayer?: ControlLayer;
  /**
   * When true (steer fail-open), do not retry a connect failure to the
   * proxy: fail at once so the wrapper can go direct. TensorCost's own
   * 429/503 still retry. Timeouts are never retried (delivery.ts).
   */
  failOpenFast?: boolean;

  /** Injected fetch implementation (for testing). Defaults to globalThis.fetch. */
  fetchImpl?: typeof fetch;
}

export interface ProxyRequestResult {
  body: unknown;
  decisionHeader: string | null;
  /** The proxy's response (body already read), for withResponse() / asResponse(). */
  response: Response;
}

async function readJsonWithDeadline(
  resp: Response,
  deadlineMs: number,
  attempt: number,
): Promise<unknown> {
  const remaining = deadlineMs - Date.now();
  if (remaining <= 0) {
    throw new TensorCostTimeoutError(
      "tensorcost: proxy response body timed out",
      "total",
      { attempt, mayHaveReachedProvider: true },
    );
  }
  // The proxy answered 2xx: the provider has run the call. A body that
  // doesn't arrive in time is never a reason to send it again.
  const jsonPromise = resp.json() as Promise<unknown>;
  void jsonPromise.catch(() => undefined);
  return Promise.race([
    jsonPromise,
    sleep(remaining).then(() => {
      void resp.body?.cancel().catch(() => undefined);
      throw new TensorCostTimeoutError(
        "tensorcost: proxy response body timed out",
        "total",
        { attempt, mayHaveReachedProvider: true },
      );
    }),
  ]);
}

/**
 * Whether a failed attempt that never reached the provider may be followed
 * by another. Never for the half-open probe: it is one attempt, and when it
 * fails the call falls back to the provider at once and the circuit stays
 * open for the next probe window. Never once failures have opened the
 * circuit mid-call either, since every attempt after that is a wait on a
 * proxy the breaker has already given up on. Through 1.8.0 a probe retried
 * up to maxAttempts times through a dead proxy (5-7.6 s per call) before
 * falling back.
 */
function mayRetry(
  attempt: number,
  config: RetryConfig,
  circuit: CircuitBreaker,
  probe: boolean,
): boolean {
  if (probe || circuit.currentState !== "CLOSED") return false;
  return attempt + 1 < config.maxAttempts;
}

/**
 * Execute a single proxied request with retries, timeouts, telemetry,
 * and circuit-breaker integration.
 *
 * Throws a `TensorCostError` subclass on any failure.
 * Returns the parsed JSON body on success.
 */
export async function proxyRequest(
  opts: ProxyRequestOptions,
): Promise<ProxyRequestResult> {
  const retryConfig = opts.retry ?? DEFAULT_RETRY_CONFIG;
  const timeoutMs = opts.timeoutMs ?? 60_000;
  const headersTimeoutMs = opts.headersTimeoutMs ?? PROXY_HEADERS_TIMEOUT_MS;
  // Buffered JSON responses: the proxy may not send headers until upstream
  // completes. Use the full body budget for TTFB so a slow model isn't cut
  // off while the proxy is still working; since 1.8.0 a timeout never fails
  // open either way (delivery.ts). Streaming uses the short headers budget.
  const isBufferedJson = opts.body.stream !== true;
  const effectiveHeadersTimeoutMs = isBufferedJson
    ? timeoutMs
    : Math.min(headersTimeoutMs, timeoutMs);
  const fetchImpl = opts.fetchImpl ?? globalThis.fetch.bind(globalThis);
  const callStart = Date.now();
  // The circuit was not CLOSED as this call began: it is the half-open probe.
  const probe = opts.circuit.currentState !== "CLOSED";

  // Circuit bypass is decided in the provider wrapper (shouldProbe / shouldBypass).
  // Do not re-check here — the single-flight probe runs while HALF_OPEN +
  // probeInFlight, which must still reach the proxy HTTP call.

  const path =
    proxyPathForOperation(opts.operation) ??
    "/api/inference-proxy/v1/chat/completions";
  const url = `${opts.proxyUrl.replace(/\/+$/, "")}${path}`;
  const headers: Record<string, string> = {
    Authorization: `Bearer ${opts.bearerToken}`,
    "Content-Type": "application/json",
    "x-tc-sdk-version": SDK_VERSION,
    "x-tc-sdk-capabilities": sdkCapabilityHeader(),
    "x-tc-max-layer": opts.maxLayer ?? "observe",
    "x-tc-provider-url": opts.providerUrl,
    "x-tc-provider-auth": opts.providerAuth,
    "x-tc-correlation-id": opts.correlationId,
  };
  if (opts.environment) {
    headers["x-tc-environment"] = opts.environment;
  }
  if (opts.agentId) {
    headers["x-tc-agent-id"] = opts.agentId;
  }
  if (opts.workflowId) {
    headers["x-tc-workflow-id"] = opts.workflowId;
  }
  if (opts.customer) {
    headers["x-tc-customer"] = opts.customer;
  }
  if (opts.feature) {
    headers["x-tc-feature"] = opts.feature;
  }
  if (opts.application) {
    headers["x-tc-application"] = opts.application;
  }
  if (opts.tags && opts.tags.length > 0) {
    headers["x-tc-tags"] = opts.tags.join(",");
  }
  if (opts.teamId) {
    headers["x-tc-team-id"] = opts.teamId;
  }

  let lastError: TensorCostError | null = null;

  for (let attempt = 0; attempt < retryConfig.maxAttempts; attempt++) {
    const elapsedMs = Date.now() - callStart;

    emitEvent(opts.onLifecycleEvent, {
      kind: "before_request",
      provider: opts.provider,
      model: opts.model,
      operation: opts.operation,
      attemptNumber: attempt + 1,
      elapsedMs,
    });

    let headersTimedOut = false;
    const headersController = new AbortController();
    const headersTimer = setTimeout(() => {
      headersTimedOut = true;
      headersController.abort();
    }, effectiveHeadersTimeoutMs);

    let resp: Response;
    try {
      resp = await fetchImpl(url, {
        method: "POST",
        headers,
        body: JSON.stringify(opts.body),
        signal: headersController.signal,
      });
    } catch (err) {
      clearTimeout(headersTimer);

      // A timeout once fetch began. The body may already have been written
      // and forwarded, and the provider may be running the call right now,
      // so this is never retried and never falls back: a slow model would
      // otherwise be billed once per attempt and once more direct. The
      // circuit still counts it, so a wedged proxy opens the circuit and
      // the calls after this one go direct before anything is sent.
      if (
        err instanceof Error &&
        (err.name === "AbortError" || err.name === "TimeoutError")
      ) {
        const headersKind: "headers" | "total" =
          headersTimedOut && effectiveHeadersTimeoutMs === headersTimeoutMs
            ? "headers"
            : "total";
        const tcErr = new TensorCostTimeoutError(
          headersKind === "headers"
            ? `tensorcost: proxy did not respond with headers within ${headersTimeoutMs}ms`
            : `tensorcost: proxy request timed out after ${timeoutMs}ms`,
          headersKind,
          { cause: err, attempt: attempt + 1, mayHaveReachedProvider: true },
        );
        lastError = tcErr;
        opts.circuit.recordFailure();
        emitEvent(opts.onLifecycleEvent, {
          kind: "on_error",
          provider: opts.provider,
          model: opts.model,
          operation: opts.operation,
          attemptNumber: attempt + 1,
          elapsedMs: Date.now() - callStart,
          errorClass: tcErr.constructor.name,
          status: null,
        });
        throw tcErr;
      }

      // Network error. Only a connect-phase failure (refused, DNS, TLS
      // handshake) proves nothing was written; that is TensorCost being
      // unreachable, and fail-open is the product's promise for it. A reset
      // or a closed socket may come after the proxy forwarded the call, so
      // it surfaces instead (delivery.ts fetchErrorSentNothing).
      const sentNothing = fetchErrorSentNothing(err);
      const tcErr = new TensorCostNetworkError(
        `tensorcost: network error reaching proxy: ${(err as Error).message}`,
        { cause: err as Error, attempt: attempt + 1, mayHaveReachedProvider: !sentNothing },
      );
      lastError = tcErr;
      opts.circuit.recordFailure();

      if (sentNothing && !opts.failOpenFast && mayRetry(attempt, retryConfig, opts.circuit, probe)) {
        const delay = computeDelay(attempt, retryConfig);
        emitEvent(opts.onLifecycleEvent, {
          kind: "on_retry",
          provider: opts.provider,
          model: opts.model,
          operation: opts.operation,
          attemptNumber: attempt + 1,
          elapsedMs: Date.now() - callStart,
          reason: "network_error",
          status: null,
          delayMs: delay,
        });
        await sleep(delay);
        continue;
      }

      emitEvent(opts.onLifecycleEvent, {
        kind: "on_error",
        provider: opts.provider,
        model: opts.model,
        operation: opts.operation,
        attemptNumber: attempt + 1,
        elapsedMs: Date.now() - callStart,
        errorClass: tcErr.constructor.name,
        status: null,
      });
      throw tcErr;
    } finally {
      clearTimeout(headersTimer);
    }

    const requestId = resp.headers.get("x-tc-request-id");
    const bodyDeadline = callStart + timeoutMs;

    // 2xx — success.
    if (resp.ok) {
      opts.circuit.recordSuccess();
      emitEvent(opts.onLifecycleEvent, {
        kind: "after_response",
        provider: opts.provider,
        model: opts.model,
        operation: opts.operation,
        attemptNumber: attempt + 1,
        elapsedMs: Date.now() - callStart,
        status: resp.status,
        requestId,
      });
      return {
        body: await readJsonWithDeadline(resp, bodyDeadline, attempt + 1),
        decisionHeader: resp.headers.get("x-tc-decision"),
        response: resp,
      };
    }

    // 4xx — read the body once. A TensorCost refusal is terminal whatever
    // its status (a 429 refusal included), so it is checked before the
    // quota arm and is never retried or counted against the circuit: the
    // proxy is healthy and answered. Counting it would eventually trip the
    // breaker and route the customer straight past their own cap.
    let errorBody: unknown = null;
    if (resp.status >= 400) {
      errorBody = await readErrorBody(resp, bodyDeadline);
    }
    if (resp.status >= 400 && resp.status < 500) {
      const refusal = parseRefusalBody(errorBody);
      if (refusal !== null || decisionSaysRefuse(resp.headers.get("x-tc-decision"))) {
        const tcErr = refusalError(refusal, {
          status: resp.status,
          requestId,
          attempt: attempt + 1,
          retryAfterHeader: resp.headers.get("retry-after"),
        });
        emitEvent(opts.onLifecycleEvent, {
          kind: "on_error",
          provider: opts.provider,
          model: opts.model,
          operation: opts.operation,
          attemptNumber: attempt + 1,
          elapsedMs: Date.now() - callStart,
          errorClass: tcErr.constructor.name,
          status: resp.status,
        });
        throw tcErr;
      }
    }

    // Where did the call get to? (delivery.ts) Only "unsent" may be
    // retried or fall back; "provider" is the provider's own answer.
    const answer: ErrorAnswer | null =
      resp.status === 402 || resp.status === 429 || resp.status >= 500
        ? classifyErrorAnswer(resp.status, resp.headers, errorBody)
        : null;
    if (answer === "provider") {
      // The proxy is healthy and relayed the provider: not a circuit failure.
      const tcErr = providerAnswerError(resp, errorBody, requestId, attempt + 1);
      emitEvent(opts.onLifecycleEvent, {
        kind: "on_error",
        provider: opts.provider,
        model: opts.model,
        operation: opts.operation,
        attemptNumber: attempt + 1,
        elapsedMs: Date.now() - callStart,
        errorClass: tcErr.constructor.name,
        status: resp.status,
      });
      throw tcErr;
    }

    // The proxy couldn't connect to the provider and nothing was sent
    // (delivery.ts isProviderUnreachable). Not retried through the proxy,
    // which already retried the provider once. A circuit failure, so a proxy
    // whose egress is broken gets bypassed instead of costing every call
    // its connect timeouts. Surfaced as unsent so the wrapper can go direct.
    if (answer === "unsent" && isProviderUnreachable(resp.status, resp.headers)) {
      const tcErr = providerUnreachableError(resp, requestId, attempt + 1);
      opts.circuit.recordFailure();
      emitEvent(opts.onLifecycleEvent, {
        kind: "on_error",
        provider: opts.provider,
        model: opts.model,
        operation: opts.operation,
        attemptNumber: attempt + 1,
        elapsedMs: Date.now() - callStart,
        errorClass: tcErr.constructor.name,
        status: resp.status,
        reason: "provider_unreachable",
      });
      throw tcErr;
    }

    // 402 without a refusal body — the plan's monthly proxy-request ceiling
    // (Scale COGS gate). TensorCost declining to carry the call is not a
    // refusal of the call, and the server means for it to go direct, so it
    // is an outage for fail-open purposes. Not retried: it won't clear soon.
    if (resp.status === 402) {
      const tcErr = new TensorCostQuotaError(
        "tensorcost: proxy returned 402 (plan request ceiling reached)",
        retryAfterMs(resp) ?? null,
        { status: 402, requestId, attempt: attempt + 1, mayHaveReachedProvider: answer !== "unsent" },
      );
      emitEvent(opts.onLifecycleEvent, {
        kind: "on_error",
        provider: opts.provider,
        model: opts.model,
        operation: opts.operation,
        attemptNumber: attempt + 1,
        elapsedMs: Date.now() - callStart,
        errorClass: tcErr.constructor.name,
        status: 402,
      });
      throw tcErr;
    }

    // 429 — quota error, may have Retry-After.
    if (resp.status === 429) {
      const raMs = retryAfterMs(resp);
      const tcErr = new TensorCostQuotaError(
        `tensorcost: proxy returned 429 Too Many Requests`,
        raMs ?? null,
        { status: 429, requestId, attempt: attempt + 1, mayHaveReachedProvider: answer !== "unsent" },
      );
      lastError = tcErr;
      // Don't count quota errors against the circuit (the proxy is healthy).

      if (answer === "unsent" && mayRetry(attempt, retryConfig, opts.circuit, probe)) {
        const delay = computeDelay(attempt, retryConfig, raMs);
        emitEvent(opts.onLifecycleEvent, {
          kind: "on_retry",
          provider: opts.provider,
          model: opts.model,
          operation: opts.operation,
          attemptNumber: attempt + 1,
          elapsedMs: Date.now() - callStart,
          reason: "quota_429",
          status: 429,
          delayMs: delay,
        });
        await sleep(delay);
        continue;
      }

      emitEvent(opts.onLifecycleEvent, {
        kind: "on_error",
        provider: opts.provider,
        model: opts.model,
        operation: opts.operation,
        attemptNumber: attempt + 1,
        elapsedMs: Date.now() - callStart,
        errorClass: tcErr.constructor.name,
        status: 429,
      });
      throw tcErr;
    }

    // 5xx — TensorCost's own failure (the provider's was handled above).
    // Retried, and allowed to fall back, only when TensorCost proves it
    // answered before forwarding. Any other 5xx (a 500 or 502 the proxy
    // sent after the forward, a gateway 502/504 for a proxy that reset or
    // timed out mid-call, a 5xx with no origin header) may follow a call
    // the provider already ran and billed, so it is surfaced to the caller
    // instead, as the provider's SDK would surface a 5xx it was told not to
    // retry.
    if (resp.status >= 500) {
      const tcErr = new TensorCostProxyError(
        `tensorcost: proxy returned ${resp.status}`,
        { status: resp.status, requestId, attempt: attempt + 1, mayHaveReachedProvider: answer !== "unsent" },
      );
      lastError = tcErr;
      opts.circuit.recordFailure();

      if (answer === "unsent" && mayRetry(attempt, retryConfig, opts.circuit, probe)) {
        const delay = computeDelay(attempt, retryConfig);
        emitEvent(opts.onLifecycleEvent, {
          kind: "on_retry",
          provider: opts.provider,
          model: opts.model,
          operation: opts.operation,
          attemptNumber: attempt + 1,
          elapsedMs: Date.now() - callStart,
          reason: "5xx",
          status: resp.status,
          delayMs: delay,
        });
        await sleep(delay);
        continue;
      }

      emitEvent(opts.onLifecycleEvent, {
        kind: "on_error",
        provider: opts.provider,
        model: opts.model,
        operation: opts.operation,
        attemptNumber: attempt + 1,
        elapsedMs: Date.now() - callStart,
        errorClass: tcErr.constructor.name,
        status: resp.status,
      });
      throw tcErr;
    }

    // Any other status — the provider's own error, relayed by the proxy
    // (always JSON), or, when the body isn't JSON, something in front of
    // TensorCost turning the request away. Non-retriable, and never a
    // fallback: the provider would answer the same way again.
    const tcErr = nonRefusal4xx(resp, errorBody, requestId, attempt + 1);
    emitEvent(opts.onLifecycleEvent, {
      kind: "on_error",
      provider: opts.provider,
      model: opts.model,
      operation: opts.operation,
      attemptNumber: attempt + 1,
      elapsedMs: Date.now() - callStart,
      errorClass: tcErr.constructor.name,
      status: resp.status,
    });
    throw tcErr;
  }

  // Should be unreachable — every loop arm either returns or throws.
  throw lastError ?? new TensorCostProxyError("tensorcost: max attempts reached");
}

/** The proxy's 502 for a provider it could not connect to: unsent, so fail-open may go direct. */
function providerUnreachableError(resp: Response, requestId: string | null, attempt: number): TensorCostProxyError {
  return new TensorCostProxyError(
    `tensorcost: proxy returned ${resp.status} (it could not reach the provider; nothing was sent)`,
    { status: resp.status, requestId, attempt, mayHaveReachedProvider: false },
  );
}

/**
 * A non-refusal 4xx: the provider's relayed error when the body is JSON, a
 * gateway rejection when it isn't (HTML, plain text, or empty).
 */
function nonRefusal4xx(
  resp: Response,
  errorBody: unknown,
  requestId: string | null,
  attempt: number,
):
  | TensorCostProviderError
  | TensorCostGatewayRejectedError
  | TensorCostProxyStatusError
  | TensorCostRefusedError
  | TensorCostKeyRejectedError {
  if (errorBody === null || typeof errorBody === "string") {
    return unexplained4xx(
      resp.status,
      resp.headers.get("x-tc-decision"),
      typeof errorBody === "string" ? errorBody.slice(0, 500) : null,
      { requestId, attempt },
    );
  }
  // NestJS's own error body ({ statusCode, message, error: "Unauthorized" })
  // is TensorCost's, not a provider's: providers never send `statusCode`.
  // A 401 there is the proxy rejecting the TensorCost token.
  const nest = errorBody as Record<string, unknown>;
  if (typeof nest["statusCode"] === "number" && typeof nest["error"] !== "object") {
    if (resp.status === 401) return new TensorCostKeyRejectedError(401, { requestId, attempt });
    const msg = typeof nest["message"] === "string" ? nest["message"] : null;
    return new TensorCostRefusedError(
      `tensorcost: the proxy rejected the request (${resp.status}${msg ? `: ${msg}` : ""})`,
      { code: null, detail: nest },
      { status: resp.status, requestId, attempt },
    );
  }
  return providerAnswerError(resp, errorBody, requestId, attempt);
}

/**
 * The provider's own answer, relayed by the proxy: a 4xx with a JSON body,
 * or any status marked `x-tc-error-origin: provider`. The call reached the
 * provider, so it is never resent; the wrappers rethrow it as the provider
 * SDK's own error type when they can.
 */
function providerAnswerError(
  resp: Response,
  errorBody: unknown,
  requestId: string | null,
  attempt: number,
): TensorCostProviderError {
  const providerMsg = providerMessageFrom(errorBody);
  const headers: Record<string, string> = {};
  resp.headers.forEach((v, k) => {
    headers[k] = v;
  });
  return new TensorCostProviderError(
    `tensorcost: upstream provider returned ${resp.status}` +
      (providerMsg !== null ? `: ${providerMsg}` : ""),
    {
      status: resp.status,
      requestId,
      attempt,
      body: errorBody,
      headers,
      mayHaveReachedProvider: true,
    },
  );
}

/**
 * Read a non-2xx body as JSON when it is JSON, else as text, inside the
 * call's body deadline. Error bodies are small; this only guards against a
 * proxy that sends headers and then stalls.
 */
async function readErrorBody(resp: Response, deadlineMs: number): Promise<unknown> {
  const remaining = Math.max(1, deadlineMs - Date.now());
  let timer: ReturnType<typeof setTimeout> | undefined;
  const text = await Promise.race([
    resp.text().catch(() => ""),
    new Promise<string>((resolve) => {
      timer = setTimeout(() => {
        void resp.body?.cancel().catch(() => undefined);
        resolve("");
      }, remaining);
    }),
  ]).finally(() => clearTimeout(timer));
  if (text === "") return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

/**
 * Should a failed proxy attempt fall back to calling the provider
 * directly?
 *
 * Only when TensorCost itself is unavailable. Fail-open exists so that
 * TensorCost being down never takes the customer's inference down with
 * it, so the answer is yes for exactly these, and no for everything else:
 *
 *   - TensorCostNetworkError   couldn't connect, DNS, connection refused
 *   - TensorCostTimeoutError   no headers, or no body inside the budget
 *   - TensorCostProxyError     5xx from the proxy, or the circuit is open
 *   - TensorCostQuotaError     429 (rate limit) or 402 (plan ceiling) from
 *                              the proxy, with no refusal body
 *
 * This is an allow-list on purpose. 1.4.0 had the inverse, a list of
 * refusal classes that must NOT fall back, and every refusal code missing
 * from it fell back by default. That is how a money budget's hard cap
 * (BUDGET_HARD_CAP) got routed around. Now a refusal this SDK has never
 * heard of is terminal, a relayed provider 4xx surfaces as itself, and an
 * error the SDK didn't classify at all is surfaced rather than retried
 * against the provider.
 *
 * Since 1.8.0 an outage falls back only when the call provably never
 * reached the provider (`mayHaveReachedProvider` false, delivery.ts). A
 * timeout, a reset, or a 5xx the proxy may have sent after forwarding is
 * surfaced instead: calling the provider again would bill the call twice.
 */
export function shouldFallBackToProvider(err: unknown): boolean {
  const outage =
    err instanceof TensorCostNetworkError ||
    err instanceof TensorCostTimeoutError ||
    err instanceof TensorCostProxyError ||
    err instanceof TensorCostQuotaError;
  return outage && !err.mayHaveReachedProvider;
}

export interface ProxyStreamResult {
  /** Raw SSE body from the proxy — same wire shape as the upstream provider. */
  stream: ReadableStream<Uint8Array>;
  decisionHeader: string | null;
  /** The proxy's response (its body is `stream`), for withResponse(). */
  response: Response;
}

/** Abort SSE consumption when no chunk arrives within idleTimeoutMs. */
export function wrapStreamWithIdleTimeout(
  body: ReadableStream<Uint8Array>,
  idleTimeoutMs: number,
): ReadableStream<Uint8Array> {
  if (!Number.isFinite(idleTimeoutMs) || idleTimeoutMs <= 0) {
    return body;
  }
  const reader = body.getReader();
  let idleTimer: ReturnType<typeof setTimeout> | null = null;

  const clearIdleTimer = (): void => {
    if (idleTimer !== null) {
      clearTimeout(idleTimer);
      idleTimer = null;
    }
  };

  const armIdleTimer = (controller: ReadableStreamDefaultController<Uint8Array>): void => {
    clearIdleTimer();
    idleTimer = setTimeout(() => {
      void reader.cancel("idle timeout");
      controller.error(
        new TensorCostTimeoutError(
          `tensorcost: proxy stream idle for ${idleTimeoutMs}ms`,
          "idle",
        ),
      );
    }, idleTimeoutMs);
  };

  return new ReadableStream({
    start(controller) {
      armIdleTimer(controller);
      void (async () => {
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              clearIdleTimer();
              controller.close();
              return;
            }
            armIdleTimer(controller);
            controller.enqueue(value);
          }
        } catch (err) {
          clearIdleTimer();
          controller.error(err);
        }
      })();
    },
    cancel(reason) {
      clearIdleTimer();
      return reader.cancel(reason);
    },
  });
}

/**
 * Steer/route a streaming request through the inference proxy.
 * Uses the short headers timeout; no JSON body read. Retries only before
 * headers arrive, and only when the attempt provably never reached the
 * provider (same rules as proxyRequest, delivery.ts). Once SSE starts, the
 * stream is returned as-is for the provider wrapper to consume.
 */
export async function proxyRequestStream(
  opts: ProxyRequestOptions & { idleTimeoutMs?: number },
): Promise<ProxyStreamResult> {
  const retryConfig = opts.retry ?? DEFAULT_RETRY_CONFIG;
  const timeoutMs = opts.timeoutMs ?? 60_000;
  const headersTimeoutMs = opts.headersTimeoutMs ?? PROXY_HEADERS_TIMEOUT_MS;
  const idleTimeoutMs = opts.idleTimeoutMs ?? 30_000;
  const fetchImpl = opts.fetchImpl ?? globalThis.fetch.bind(globalThis);
  const callStart = Date.now();
  // The circuit was not CLOSED as this call began: it is the half-open probe.
  const probe = opts.circuit.currentState !== "CLOSED";

  const path =
    proxyPathForOperation(opts.operation) ??
    "/api/inference-proxy/v1/chat/completions";
  const url = `${opts.proxyUrl.replace(/\/+$/, "")}${path}`;
  const headers: Record<string, string> = {
    Authorization: `Bearer ${opts.bearerToken}`,
    "Content-Type": "application/json",
    Accept: "text/event-stream",
    "x-tc-sdk-version": SDK_VERSION,
    "x-tc-sdk-capabilities": sdkCapabilityHeader(),
    "x-tc-max-layer": opts.maxLayer ?? "observe",
    "x-tc-provider-url": opts.providerUrl,
    "x-tc-provider-auth": opts.providerAuth,
    "x-tc-correlation-id": opts.correlationId,
  };
  if (opts.environment) headers["x-tc-environment"] = opts.environment;
  if (opts.agentId) headers["x-tc-agent-id"] = opts.agentId;
  if (opts.workflowId) headers["x-tc-workflow-id"] = opts.workflowId;
  if (opts.customer) headers["x-tc-customer"] = opts.customer;
  if (opts.feature) headers["x-tc-feature"] = opts.feature;
  if (opts.application) headers["x-tc-application"] = opts.application;
  if (opts.tags && opts.tags.length > 0) headers["x-tc-tags"] = opts.tags.join(",");
  if (opts.teamId) headers["x-tc-team-id"] = opts.teamId;

  let lastError: TensorCostError | null = null;

  for (let attempt = 0; attempt < retryConfig.maxAttempts; attempt++) {
    emitEvent(opts.onLifecycleEvent, {
      kind: "before_request",
      provider: opts.provider,
      model: opts.model,
      operation: opts.operation,
      attemptNumber: attempt + 1,
      elapsedMs: Date.now() - callStart,
    });

    let headersTimedOut = false;
    const headersController = new AbortController();
    const headersTimer = setTimeout(() => {
      headersTimedOut = true;
      headersController.abort();
    }, Math.min(headersTimeoutMs, timeoutMs));

    let resp: Response;
    try {
      resp = await fetchImpl(url, {
        method: "POST",
        headers,
        body: JSON.stringify(opts.body),
        signal: headersController.signal,
      });
    } catch (err) {
      clearTimeout(headersTimer);
      // No headers in time. The request was written and the proxy may have
      // forwarded it (a stream's headers wait on the provider's), so this is
      // never retried and never falls back; the circuit counts it, so the
      // calls after it go direct once a wedged proxy opens the circuit.
      if (
        err instanceof Error &&
        (err.name === "AbortError" || err.name === "TimeoutError")
      ) {
        const tcErr = new TensorCostTimeoutError(
          headersTimedOut
            ? `tensorcost: proxy did not respond with headers within ${headersTimeoutMs}ms`
            : `tensorcost: proxy request timed out after ${timeoutMs}ms`,
          headersTimedOut ? "headers" : "total",
          { cause: err, attempt: attempt + 1, mayHaveReachedProvider: true },
        );
        opts.circuit.recordFailure();
        throw tcErr;
      }
      // Only a connect-phase failure proves nothing was written.
      const sentNothing = fetchErrorSentNothing(err);
      const tcErr = new TensorCostNetworkError(
        `tensorcost: network error reaching proxy: ${(err as Error).message}`,
        { cause: err as Error, attempt: attempt + 1, mayHaveReachedProvider: !sentNothing },
      );
      lastError = tcErr;
      opts.circuit.recordFailure();
      if (sentNothing && !opts.failOpenFast && mayRetry(attempt, retryConfig, opts.circuit, probe)) {
        await sleep(computeDelay(attempt, retryConfig));
        continue;
      }
      throw tcErr;
    } finally {
      clearTimeout(headersTimer);
    }

    const requestId = resp.headers.get("x-tc-request-id");

    if (resp.ok) {
      if (!resp.body) {
        // A 2xx: the provider answered. An empty body is no reason to resend.
        throw new TensorCostProxyError("tensorcost: proxy returned empty stream body", {
          status: resp.status,
          requestId,
          attempt: attempt + 1,
          mayHaveReachedProvider: true,
        });
      }
      opts.circuit.recordSuccess();
      emitEvent(opts.onLifecycleEvent, {
        kind: "after_response",
        provider: opts.provider,
        model: opts.model,
        operation: opts.operation,
        attemptNumber: attempt + 1,
        elapsedMs: Date.now() - callStart,
        status: resp.status,
        requestId,
      });
      return {
        stream: wrapStreamWithIdleTimeout(resp.body, idleTimeoutMs),
        decisionHeader: resp.headers.get("x-tc-decision"),
        response: resp,
      };
    }

    // Same classification as proxyRequest: refusal first (terminal, never
    // retried), then the provider's own answer, then 402 / 429 / 5xx
    // (outages, retried only when provably unsent), then the provider's
    // relayed 4xx.
    let errorBody: unknown = null;
    if (resp.status >= 400) {
      errorBody = await readErrorBody(resp, callStart + timeoutMs);
    }
    if (resp.status >= 400 && resp.status < 500) {
      const refusal = parseRefusalBody(errorBody);
      if (refusal !== null || decisionSaysRefuse(resp.headers.get("x-tc-decision"))) {
        const tcErr = refusalError(refusal, {
          status: resp.status,
          requestId,
          attempt: attempt + 1,
          retryAfterHeader: resp.headers.get("retry-after"),
        });
        emitEvent(opts.onLifecycleEvent, {
          kind: "on_error",
          provider: opts.provider,
          model: opts.model,
          operation: opts.operation,
          attemptNumber: attempt + 1,
          elapsedMs: Date.now() - callStart,
          errorClass: tcErr.constructor.name,
          status: resp.status,
        });
        throw tcErr;
      }
    }

    const answer: ErrorAnswer | null =
      resp.status === 402 || resp.status === 429 || resp.status >= 500
        ? classifyErrorAnswer(resp.status, resp.headers, errorBody)
        : null;
    if (answer === "provider") {
      throw providerAnswerError(resp, errorBody, requestId, attempt + 1);
    }
    // See proxyRequest: unreachable provider, nothing sent, not retried, a circuit failure.
    if (answer === "unsent" && isProviderUnreachable(resp.status, resp.headers)) {
      const tcErr = providerUnreachableError(resp, requestId, attempt + 1);
      opts.circuit.recordFailure();
      emitEvent(opts.onLifecycleEvent, {
        kind: "on_error",
        provider: opts.provider,
        model: opts.model,
        operation: opts.operation,
        attemptNumber: attempt + 1,
        elapsedMs: Date.now() - callStart,
        errorClass: tcErr.constructor.name,
        status: resp.status,
        reason: "provider_unreachable",
      });
      throw tcErr;
    }
    const unsent = answer === "unsent";
    if (resp.status === 402) {
      throw new TensorCostQuotaError(
        "tensorcost: proxy returned 402 (plan request ceiling reached)",
        retryAfterMs(resp) ?? null,
        { status: 402, requestId, attempt: attempt + 1, mayHaveReachedProvider: !unsent },
      );
    }
    if (resp.status === 429) {
      const raMs = retryAfterMs(resp);
      if (unsent && mayRetry(attempt, retryConfig, opts.circuit, probe)) {
        await sleep(computeDelay(attempt, retryConfig, raMs));
        continue;
      }
      throw new TensorCostQuotaError(
        "tensorcost: proxy returned 429 Too Many Requests",
        raMs ?? null,
        { status: 429, requestId, attempt: attempt + 1, mayHaveReachedProvider: !unsent },
      );
    }
    if (resp.status >= 500) {
      // See proxyRequest: a 5xx not proven pre-forward is surfaced, never resent.
      opts.circuit.recordFailure();
      if (unsent && mayRetry(attempt, retryConfig, opts.circuit, probe)) {
        await sleep(computeDelay(attempt, retryConfig));
        continue;
      }
      throw new TensorCostProxyError(`tensorcost: proxy returned ${resp.status}`, {
        status: resp.status,
        requestId,
        attempt: attempt + 1,
        mayHaveReachedProvider: !unsent,
      });
    }

    throw nonRefusal4xx(resp, errorBody, requestId, attempt + 1);
  }

  throw lastError ?? new TensorCostProxyError("tensorcost: max attempts reached");
}

/** Parse OpenAI-compatible SSE into an async iterable for streaming wrappers. */
export function sseChunksFromStream(
  body: ReadableStream<Uint8Array>,
): AsyncIterable<Record<string, unknown>> {
  return {
    async *[Symbol.asyncIterator]() {
      const reader = body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith("data:")) continue;
            const data = trimmed.slice(5).trim();
            if (data === "[DONE]" || !data) continue;
            try {
              yield JSON.parse(data) as Record<string, unknown>;
            } catch {
              /* skip malformed chunk */
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    },
  };
}

export { TensorCostError };
