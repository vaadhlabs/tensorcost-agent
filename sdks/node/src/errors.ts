/**
 * Typed error hierarchy for the TensorCost Node SDK.
 *
 * Every error thrown by the SDK's applied-mode request path is one of
 * these classes. Customer code can `instanceof`-branch on any of them
 * without importing raw Error types from the underlying http layer.
 *
 * Hierarchy:
 *
 *   TensorCostError (base)
 *   ├── TensorCostNetworkError    couldn't reach the proxy at all
 *   ├── TensorCostTimeoutError    request or idle window exceeded limit
 *   ├── TensorCostProxyError      proxy responded with 5xx
 *   ├── TensorCostQuotaError      proxy responded with 429
 *   └── TensorCostProviderError   proxy forwarded an upstream provider error
 *
 * All subclasses carry:
 *   - `rootCause`   the underlying Error, if any (avoids clash with ES2022 Error.cause)
 *   - `status`      HTTP status code (null when the request never completed)
 *   - `requestId`   value of the X-TC-Request-ID header the proxy stamps
 *   - `attempt`     1-based attempt number at the time the error occurred
 */

export interface TensorCostErrorOptions {
  cause?: Error;
  status?: number | null;
  requestId?: string | null;
  attempt?: number;
}

export abstract class TensorCostError extends Error {
  /** The underlying Error that triggered this one, if any. */
  readonly rootCause: Error | undefined;
  readonly status: number | null;
  readonly requestId: string | null;
  readonly attempt: number;

  constructor(message: string, opts: TensorCostErrorOptions = {}) {
    super(message);
    // noImplicitOverride requires `override` for inherited props.
    // We set name on the *instance* without re-declaring the property type,
    // which is the correct approach for custom Error subclasses in strict TS.
    Object.defineProperty(this, "name", {
      value: this.constructor.name,
      writable: true,
      enumerable: false,
      configurable: true,
    });
    this.rootCause = opts.cause;
    this.status = opts.status ?? null;
    this.requestId = opts.requestId ?? null;
    this.attempt = opts.attempt ?? 1;

    // Keep stack traces clean in V8 environments.
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }
}

/** The SDK could not establish a connection to the proxy at all. */
export class TensorCostNetworkError extends TensorCostError {
  constructor(message: string, opts: TensorCostErrorOptions = {}) {
    super(message, opts);
  }
}

/** A request or idle-window timeout fired before the operation completed. */
export class TensorCostTimeoutError extends TensorCostError {
  /** "headers" — no response headers; "total" — wall-clock; "idle" — streaming idle. */
  readonly timeoutKind: "total" | "idle" | "headers";

  constructor(
    message: string,
    timeoutKind: "total" | "idle" | "headers",
    opts: TensorCostErrorOptions = {},
  ) {
    super(message, opts);
    this.timeoutKind = timeoutKind;
  }
}

/** The proxy returned a 5xx response (TensorCost's own infrastructure error). */
export class TensorCostProxyError extends TensorCostError {
  constructor(message: string, opts: TensorCostErrorOptions = {}) {
    super(message, opts);
  }
}

/**
 * The proxy returned 429 Too Many Requests.
 * `retryAfterMs` is set when the response included a `Retry-After` header.
 */
export class TensorCostQuotaError extends TensorCostError {
  readonly retryAfterMs: number | null;

  constructor(
    message: string,
    retryAfterMs: number | null,
    opts: TensorCostErrorOptions = {},
  ) {
    super(message, { ...opts, status: opts.status ?? 429 });
    this.retryAfterMs = retryAfterMs;
  }
}

/**
 * The proxy successfully reached the upstream provider, but the provider
 * itself returned an error (4xx/5xx from OpenAI, Anthropic, etc.). The
 * proxy's response still carries the provider's status in `status`.
 */
export class TensorCostProviderError extends TensorCostError {
  constructor(message: string, opts: TensorCostErrorOptions = {}) {
    super(message, opts);
  }
}

/**
 * The agent run this call belongs to has hit its spend cap, and the proxy
 * refused the request (HTTP 403, `code: "RUN_BUDGET_EXCEEDED"`).
 *
 * This error is TERMINAL and must never trigger the direct-to-provider
 * fallback. That distinction is the entire point: every other proxy error
 * means "TensorCost is unhealthy, go around us", and falling back is
 * correct. A budget refusal means "TensorCost is working exactly as
 * configured, and the answer is no" — falling back would call the provider
 * anyway, bill the customer, and reduce the cap to a suggestion.
 *
 * See `shouldFallBackToProvider()` in proxy-client.ts, which is the single
 * predicate every provider wrapper consults before falling back.
 */
export class TensorCostRunBudgetExceededError extends TensorCostError {
  constructor(
    message: string,
    readonly workflowId: string | null,
    readonly capCents: number | null,
    readonly spentCents: number | null,
    opts: TensorCostErrorOptions = {},
  ) {
    super(message, opts);
  }
}

/**
 * Daily or monthly agent/team spend cap exceeded (HTTP 403,
 * `code: "PERIOD_BUDGET_EXCEEDED"`). Terminal — must not fall back to the
 * provider or the cap becomes a suggestion.
 */
export class TensorCostPeriodBudgetExceededError extends TensorCostError {
  constructor(
    message: string,
    readonly scope: string | null,
    readonly period: string | null,
    readonly capCents: number | null,
    readonly spentCents: number | null,
    opts: TensorCostErrorOptions = {},
  ) {
    super(message, opts);
  }
}

/**
 * Tenant model governance refused this model (HTTP 403,
 * `code: "MODEL_GOVERNANCE_DENIED"`). Terminal — must not fall back to the
 * provider, or Enforce would become a suggestion.
 */
export class TensorCostModelGovernanceDeniedError extends TensorCostError {
  constructor(
    message: string,
    readonly provider: string | null,
    readonly model: string | null,
    readonly governanceStatus: string | null,
    opts: TensorCostErrorOptions = {},
  ) {
    super(message, opts);
  }
}

/**
 * Guardrail hard_stop refused this call (HTTP 403,
 * `code: "GUARDRAIL_HARD_STOP"`). Terminal — same fail-closed contract as
 * run-budget and model governance denials.
 */
export class TensorCostGuardrailHardStopError extends TensorCostError {
  constructor(
    message: string,
    readonly policyId: string | null,
    opts: TensorCostErrorOptions = {},
  ) {
    super(message, opts);
  }
}

/** Compliance policy refused this call (metadata or in-process DLP). */
export class TensorCostComplianceDeniedError extends TensorCostError {
  constructor(
    message: string,
    readonly reason: string | null,
    opts: TensorCostErrorOptions = {},
  ) {
    super(message, { ...opts, status: opts.status ?? 403 });
  }
}

/** Client team_id does not match token-bound team scope. */
export class TensorCostComplianceTeamMismatchError extends TensorCostError {
  constructor(message: string, opts: TensorCostErrorOptions = {}) {
    super(message, { ...opts, status: opts.status ?? 403 });
  }
}
