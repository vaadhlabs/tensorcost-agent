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
 * Retry logic:
 *   - Retries on 5xx + network errors + 429.
 *   - Respects Retry-After on 429.
 *   - Stops at maxAttempts.
 *   - Does not retry on 4xx (except 429).
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
  TensorCostRunBudgetExceededError,
  TensorCostPeriodBudgetExceededError,
  TensorCostModelGovernanceDeniedError,
  TensorCostGuardrailHardStopError,
  TensorCostComplianceDeniedError,
  TensorCostComplianceTeamMismatchError,
} from "./errors.js";
import {
  DEFAULT_RETRY_CONFIG,
  computeDelay,
  isRetryableStatus,
  retryAfterMs,
  sleep,
} from "./retry.js";
import type { RetryConfig } from "./retry.js";
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
   * When true (steer fail-open), do not retry proxy *network* errors —
   * fail immediately so the wrapper can go direct. 429/5xx still retry.
   * Timeouts still retry: a timeout usually means the upstream model is slow.
   */
  failOpenFast?: boolean;

  /** Injected fetch implementation (for testing). Defaults to globalThis.fetch. */
  fetchImpl?: typeof fetch;
}

export interface ProxyRequestResult {
  body: unknown;
  decisionHeader: string | null;
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
      { attempt },
    );
  }
  const jsonPromise = resp.json() as Promise<unknown>;
  void jsonPromise.catch(() => undefined);
  return Promise.race([
    jsonPromise,
    sleep(remaining).then(() => {
      void resp.body?.cancel().catch(() => undefined);
      throw new TensorCostTimeoutError(
        "tensorcost: proxy response body timed out",
        "total",
        { attempt },
      );
    }),
  ]);
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
  // completes. Use the full body budget for TTFB so we do not fail-open
  // (and double-bill) while the proxy is still working. Streaming uses the
  // short headers budget once SSE client support lands.
  const isBufferedJson = opts.body.stream !== true;
  const effectiveHeadersTimeoutMs = isBufferedJson
    ? timeoutMs
    : Math.min(headersTimeoutMs, timeoutMs);
  const fetchImpl = opts.fetchImpl ?? globalThis.fetch.bind(globalThis);
  const callStart = Date.now();

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

      // Headers never arrived — proxy wedged; fail-open without retry.
      if (
        headersTimedOut &&
        err instanceof Error &&
        (err.name === "AbortError" || err.name === "TimeoutError")
      ) {
        const headersKind: "headers" | "total" =
          effectiveHeadersTimeoutMs === headersTimeoutMs
            ? "headers"
            : "total";
        const tcErr = new TensorCostTimeoutError(
          headersKind === "headers"
            ? `tensorcost: proxy did not respond with headers within ${headersTimeoutMs}ms`
            : `tensorcost: proxy request timed out after ${timeoutMs}ms`,
          headersKind,
          { cause: err, attempt: attempt + 1 },
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

      // AbortController fired → timeout.
      if (
        err instanceof Error &&
        (err.name === "AbortError" || err.name === "TimeoutError")
      ) {
        const tcErr = new TensorCostTimeoutError(
          `tensorcost: proxy request timed out after ${timeoutMs}ms`,
          "total",
          { cause: err, attempt: attempt + 1 },
        );
        lastError = tcErr;
        opts.circuit.recordFailure();

        // Total timeouts may be slow model — retriable unless fail-open-fast.
        if (!opts.failOpenFast && attempt + 1 < retryConfig.maxAttempts) {
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
      }

      // Network error (ECONNREFUSED, DNS failure, etc.)
      const tcErr = new TensorCostNetworkError(
        `tensorcost: network error reaching proxy: ${(err as Error).message}`,
        { cause: err as Error, attempt: attempt + 1 },
      );
      lastError = tcErr;
      opts.circuit.recordFailure();

      if (!opts.failOpenFast && attempt + 1 < retryConfig.maxAttempts) {
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
      };
    }

    // 429 — quota error, may have Retry-After.
    if (resp.status === 429) {
      const raMs = retryAfterMs(resp);
      const tcErr = new TensorCostQuotaError(
        `tensorcost: proxy returned 429 Too Many Requests`,
        raMs ?? null,
        { status: 429, requestId, attempt: attempt + 1 },
      );
      lastError = tcErr;
      // Don't count quota errors against the circuit (the proxy is healthy).

      if (attempt + 1 < retryConfig.maxAttempts) {
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

    // 5xx — proxy infrastructure error.
    if (resp.status >= 500) {
      const tcErr = new TensorCostProxyError(
        `tensorcost: proxy returned ${resp.status}`,
        { status: resp.status, requestId, attempt: attempt + 1 },
      );
      lastError = tcErr;
      opts.circuit.recordFailure();

      if (attempt + 1 < retryConfig.maxAttempts) {
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

    // 403 with our own budget code — the run hit its spend cap.
    //
    // Checked before the generic 4xx arm because it is NOT a provider
    // error and, critically, must not lead to the direct-provider
    // fallback. The body is read defensively: a 403 that is not ours
    // (e.g. an auth failure) falls through to the generic arm below.
    if (resp.status === 403) {
      let code: string | undefined;
      let detail: Record<string, unknown> | undefined;
      try {
        const parsed = (await resp.clone().json()) as Record<string, unknown>;
        const errObj = parsed?.["error"] as Record<string, unknown> | undefined;
        if (errObj && typeof errObj["code"] === "string") {
          code = errObj["code"];
          detail = errObj;
        }
      } catch {
        /* non-JSON 403 — fall through to the generic 4xx arm */
      }

      if (code === "RUN_BUDGET_EXCEEDED" || code === "PERIOD_BUDGET_EXCEEDED") {
        const numOrNull = (v: unknown): number | null =>
          typeof v === "number" ? v : null;
        const payload = detail;
        const tcErr =
          code === "PERIOD_BUDGET_EXCEEDED"
            ? new TensorCostPeriodBudgetExceededError(
                typeof payload?.["message"] === "string"
                  ? `tensorcost: ${payload["message"]}`
                  : "tensorcost: agent period spend cap reached",
                typeof payload?.["scope"] === "string" ? payload["scope"] : null,
                typeof payload?.["period"] === "string" ? payload["period"] : null,
                numOrNull(payload?.["cap_cents"]),
                numOrNull(payload?.["spent_cents"]),
                { status: 403, requestId, attempt: attempt + 1 },
              )
            : new TensorCostRunBudgetExceededError(
                typeof payload?.["message"] === "string"
                  ? `tensorcost: ${payload["message"]}`
                  : "tensorcost: agent run spend cap reached",
                typeof payload?.["workflow_id"] === "string"
                  ? payload["workflow_id"]
                  : null,
                numOrNull(payload?.["cap_cents"]),
                numOrNull(payload?.["spent_cents"]),
                { status: 403, requestId, attempt: attempt + 1 },
              );
        // Not a circuit failure — the proxy is healthy and answered
        // correctly. Counting this would eventually trip the breaker and
        // route the customer straight past their own cap.
        emitEvent(opts.onLifecycleEvent, {
          kind: "on_error",
          provider: opts.provider,
          model: opts.model,
          operation: opts.operation,
          attemptNumber: attempt + 1,
          elapsedMs: Date.now() - callStart,
          errorClass: tcErr.constructor.name,
          status: 403,
        });
        throw tcErr;
      }

      if (code === "MODEL_GOVERNANCE_DENIED") {
        const tcErr = new TensorCostModelGovernanceDeniedError(
          typeof detail?.["message"] === "string"
            ? `tensorcost: ${detail["message"]}`
            : "tensorcost: model denied by tenant governance",
          typeof detail?.["provider"] === "string" ? detail["provider"] : null,
          typeof detail?.["model"] === "string" ? detail["model"] : null,
          typeof detail?.["governance_status"] === "string"
            ? detail["governance_status"]
            : null,
          { status: 403, requestId, attempt: attempt + 1 },
        );
        emitEvent(opts.onLifecycleEvent, {
          kind: "on_error",
          provider: opts.provider,
          model: opts.model,
          operation: opts.operation,
          attemptNumber: attempt + 1,
          elapsedMs: Date.now() - callStart,
          errorClass: tcErr.constructor.name,
          status: 403,
        });
        throw tcErr;
      }

      if (code === "GUARDRAIL_HARD_STOP") {
        const tcErr = new TensorCostGuardrailHardStopError(
          typeof detail?.["message"] === "string"
            ? `tensorcost: ${detail["message"]}`
            : "tensorcost: request blocked by hard_stop guardrail",
          typeof detail?.["policy_id"] === "string" ? detail["policy_id"] : null,
          { status: 403, requestId, attempt: attempt + 1 },
        );
        emitEvent(opts.onLifecycleEvent, {
          kind: "on_error",
          provider: opts.provider,
          model: opts.model,
          operation: opts.operation,
          attemptNumber: attempt + 1,
          elapsedMs: Date.now() - callStart,
          errorClass: tcErr.constructor.name,
          status: 403,
        });
        throw tcErr;
      }

      if (code === "COMPLIANCE_DENIED") {
        const tcErr = new TensorCostComplianceDeniedError(
          typeof detail?.["message"] === "string"
            ? `tensorcost: ${detail["message"]}`
            : "tensorcost: request refused by compliance policy",
          code,
          { status: 403, requestId, attempt: attempt + 1 },
        );
        emitEvent(opts.onLifecycleEvent, {
          kind: "on_error",
          provider: opts.provider,
          model: opts.model,
          operation: opts.operation,
          attemptNumber: attempt + 1,
          elapsedMs: Date.now() - callStart,
          errorClass: tcErr.constructor.name,
          status: 403,
        });
        throw tcErr;
      }

      if (code === "COMPLIANCE_TEAM_MISMATCH") {
        const tcErr = new TensorCostComplianceTeamMismatchError(
          typeof detail?.["message"] === "string"
            ? `tensorcost: ${detail["message"]}`
            : "tensorcost: SDK team scope does not match token binding",
          { status: 403, requestId, attempt: attempt + 1 },
        );
        emitEvent(opts.onLifecycleEvent, {
          kind: "on_error",
          provider: opts.provider,
          model: opts.model,
          operation: opts.operation,
          attemptNumber: attempt + 1,
          elapsedMs: Date.now() - callStart,
          errorClass: tcErr.constructor.name,
          status: 403,
        });
        throw tcErr;
      }
    }

    // 4xx (not 429) — provider error forwarded by the proxy. Non-retriable.
    const tcErr = new TensorCostProviderError(
      `tensorcost: upstream provider returned ${resp.status}`,
      { status: resp.status, requestId, attempt: attempt + 1 },
    );
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

/**
 * Should a failed proxy attempt fall back to calling the provider
 * directly?
 *
 * The applied-mode fail-open contract exists so that TensorCost being
 * down never takes the customer's inference down with it. Every transport
 * failure — timeout, network error, 5xx, open circuit — is therefore a
 * "go around us" signal and returns true.
 *
 * A policy refusal (run budget, model governance, guardrail hard_stop) is
 * when the proxy is healthy and the refusal IS the product. Falling back
 * would issue the provider call anyway and turn enforcement into a log line.
 */
export function shouldFallBackToProvider(err: unknown): boolean {
  return !(
    err instanceof TensorCostRunBudgetExceededError ||
    err instanceof TensorCostPeriodBudgetExceededError ||
    err instanceof TensorCostModelGovernanceDeniedError ||
    err instanceof TensorCostGuardrailHardStopError ||
    err instanceof TensorCostComplianceDeniedError ||
    err instanceof TensorCostComplianceTeamMismatchError
  );
}

export interface ProxyStreamResult {
  /** Raw SSE body from the proxy — same wire shape as the upstream provider. */
  stream: ReadableStream<Uint8Array>;
  decisionHeader: string | null;
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
 * headers arrive (5xx/429/network). Once SSE starts, the stream is returned
 * as-is for the provider wrapper to consume.
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
      if (
        headersTimedOut &&
        err instanceof Error &&
        (err.name === "AbortError" || err.name === "TimeoutError")
      ) {
        const tcErr = new TensorCostTimeoutError(
          `tensorcost: proxy did not respond with headers within ${headersTimeoutMs}ms`,
          "headers",
          { cause: err, attempt: attempt + 1 },
        );
        opts.circuit.recordFailure();
        throw tcErr;
      }
      const tcErr = new TensorCostNetworkError(
        `tensorcost: network error reaching proxy: ${(err as Error).message}`,
        { cause: err as Error, attempt: attempt + 1 },
      );
      lastError = tcErr;
      opts.circuit.recordFailure();
      if (!opts.failOpenFast && attempt + 1 < retryConfig.maxAttempts) {
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
        throw new TensorCostProxyError("tensorcost: proxy returned empty stream body", {
          status: resp.status,
          requestId,
          attempt: attempt + 1,
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
      };
    }

    if (resp.status === 429 && attempt + 1 < retryConfig.maxAttempts) {
      await sleep(computeDelay(attempt, retryConfig, retryAfterMs(resp)));
      continue;
    }
    if (resp.status >= 500 && attempt + 1 < retryConfig.maxAttempts) {
      opts.circuit.recordFailure();
      await sleep(computeDelay(attempt, retryConfig));
      continue;
    }

    // Policy refusals and 4xx — reuse proxyRequest error mapping via body read.
    const errBody = await resp.text().catch(() => "");
    let detail: Record<string, unknown> | undefined;
    try {
      const parsed = JSON.parse(errBody) as Record<string, unknown>;
      const errObj = parsed?.["error"] as Record<string, unknown> | undefined;
      detail =
        errObj && typeof errObj["code"] === "string" ? errObj : undefined;
    } catch {
      detail = undefined;
    }
    const code = typeof detail?.["code"] === "string" ? detail["code"] : null;
    if (code === "RUN_BUDGET_EXCEEDED" || code === "PERIOD_BUDGET_EXCEEDED") {
      const numOrNull = (v: unknown): number | null =>
        typeof v === "number" ? v : null;
      if (code === "PERIOD_BUDGET_EXCEEDED") {
        throw new TensorCostPeriodBudgetExceededError(
          typeof detail?.["message"] === "string"
            ? `tensorcost: ${detail["message"]}`
            : "tensorcost: period budget exceeded",
          typeof detail?.["scope"] === "string" ? detail["scope"] : null,
          typeof detail?.["period"] === "string" ? detail["period"] : null,
          numOrNull(detail?.["cap_cents"]),
          numOrNull(detail?.["spent_cents"]),
          { status: 403, requestId, attempt: attempt + 1 },
        );
      }
      throw new TensorCostRunBudgetExceededError(
        typeof detail?.["message"] === "string"
          ? `tensorcost: ${detail["message"]}`
          : "tensorcost: run budget exceeded",
        typeof detail?.["workflow_id"] === "string" ? detail["workflow_id"] : null,
        numOrNull(detail?.["cap_cents"]),
        numOrNull(detail?.["spent_cents"]),
        { status: 403, requestId, attempt: attempt + 1 },
      );
    }
    if (code === "MODEL_GOVERNANCE_DENIED") {
      throw new TensorCostModelGovernanceDeniedError(
        "tensorcost: model denied by tenant governance",
        null,
        null,
        null,
        { status: 403, requestId, attempt: attempt + 1 },
      );
    }
    if (code === "GUARDRAIL_HARD_STOP") {
      throw new TensorCostGuardrailHardStopError(
        "tensorcost: request blocked by hard_stop guardrail",
        null,
        { status: 403, requestId, attempt: attempt + 1 },
      );
    }
    if (code === "COMPLIANCE_DENIED") {
      throw new TensorCostComplianceDeniedError(
        typeof detail?.["message"] === "string"
          ? `tensorcost: ${detail["message"]}`
          : "tensorcost: request refused by compliance policy",
        code,
        { status: 403, requestId, attempt: attempt + 1 },
      );
    }
    if (code === "COMPLIANCE_TEAM_MISMATCH") {
      throw new TensorCostComplianceTeamMismatchError(
        typeof detail?.["message"] === "string"
          ? `tensorcost: ${detail["message"]}`
          : "tensorcost: SDK team scope does not match token binding",
        { status: 403, requestId, attempt: attempt + 1 },
      );
    }
    throw new TensorCostProviderError(
      `tensorcost: upstream provider returned ${resp.status}`,
      { status: resp.status, requestId, attempt: attempt + 1 },
    );
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
