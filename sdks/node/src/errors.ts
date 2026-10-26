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
 *   ├── TensorCostNetworkError    couldn't reach the proxy at all          ┐
 *   ├── TensorCostTimeoutError    request or idle window exceeded limit    │ outages: the only
 *   ├── TensorCostProxyError      proxy responded with 5xx                 │ errors that may fall
 *   ├── TensorCostQuotaError      proxy 429, or 402 plan ceiling           ┘ back to the provider
 *   ├── TensorCostProviderError   proxy relayed an upstream provider 4xx
 *   └── TensorCostRefusedError    TensorCost refused the call (terminal)
 *       ├── TensorCostRunBudgetExceededError
 *       ├── TensorCostRunStoppedError
 *       ├── TensorCostPeriodBudgetExceededError
 *       │   └── TensorCostBudgetHardCapError
 *       ├── TensorCostModelGovernanceDeniedError
 *       ├── TensorCostGuardrailHardStopError
 *       ├── TensorCostComplianceDeniedError
 *       └── TensorCostComplianceTeamMismatchError
 *
 * A refusal code this SDK version doesn't model still arrives as a plain
 * TensorCostRefusedError with its `code`, `reason` and message intact.
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
 * The proxy returned 429 Too Many Requests (TensorCost's own rate limit),
 * or 402 with no refusal body (the plan's monthly proxy-request ceiling;
 * `status` says which). Neither carries a refusal, so both may fall back.
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
 * The proxy reached the upstream provider and relayed the provider's own
 * 4xx (OpenAI 400 invalid request, Anthropic 401, and so on). `status` is
 * the provider's status and `body` is what the provider sent, parsed as
 * JSON when it was JSON. Never falls back: the provider would only answer
 * the same way again.
 */
export class TensorCostProviderError extends TensorCostError {
  readonly body: unknown;

  constructor(message: string, opts: TensorCostErrorOptions & { body?: unknown } = {}) {
    super(message, opts);
    this.body = opts.body ?? null;
  }
}

/**
 * TensorCost answered and the answer is no: a budget, a stop, a policy.
 *
 * Every refusal is terminal. It never retries and never falls back to the
 * provider, because the proxy that sent it is healthy and the refusal is
 * the product. `code` is the server's refusal code (null when the server
 * refused without naming one), `reason` its optional sub-reason, and
 * `detail` the refusal object as the server sent it.
 *
 * Subclasses cover the codes this SDK version models. Anything else, a
 * code added on the server after this release included, is thrown as this
 * class directly so it is still terminal.
 */
export class TensorCostRefusedError extends TensorCostError {
  readonly code: string | null;
  readonly reason: string | null;
  readonly detail: Record<string, unknown> | null;

  constructor(
    message: string,
    refusal: {
      code?: string | null;
      reason?: string | null;
      detail?: Record<string, unknown> | null;
    } = {},
    opts: TensorCostErrorOptions = {},
  ) {
    super(message, opts);
    this.code = refusal.code ?? null;
    this.reason = refusal.reason ?? null;
    this.detail = refusal.detail ?? null;
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
 * predicate every provider wrapper consults before falling back. Since
 * 1.4.1 it is an allow-list of outages, so a refusal can't slip past it.
 */
export class TensorCostRunBudgetExceededError extends TensorCostRefusedError {
  constructor(
    message: string,
    readonly workflowId: string | null,
    readonly capCents: number | null,
    readonly spentCents: number | null,
    opts: TensorCostErrorOptions & { detail?: Record<string, unknown> | null } = {},
  ) {
    super(message, { code: "RUN_BUDGET_EXCEEDED", detail: opts.detail ?? null }, opts);
  }
}

/**
 * An operator stopped the agent run this call belongs to, and the proxy
 * refused the request (HTTP 403, `code: "RUN_STOPPED"`).
 *
 * Terminal for the same reason as the budget refusals: a stop is a decision
 * from a healthy proxy, so falling back to the provider would simply undo
 * it. Carries the who / when / why the proxy recorded.
 */
export class TensorCostRunStoppedError extends TensorCostRefusedError {
  /**
   * Build the error from the proxy's 403 `error` object. Shared by the
   * chat-completions client and the govern-layer admit client, which both
   * receive the same body.
   */
  static fromProxyBody(
    detail: Record<string, unknown> | undefined,
    opts: TensorCostErrorOptions = {},
  ): TensorCostRunStoppedError {
    const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
    return new TensorCostRunStoppedError(
      str(detail?.["message"]) ?? "tensorcost: agent run was stopped",
      {
        workflowId: str(detail?.["workflow_id"]),
        stoppedBy: str(detail?.["stopped_by"]),
        stoppedAt: str(detail?.["stopped_at"]),
        reason: str(detail?.["reason"]),
      },
      { ...opts, detail: detail ?? null },
    );
  }

  override readonly code = "RUN_STOPPED";
  readonly workflowId: string | null;
  readonly stoppedBy: string | null;
  readonly stoppedAt: string | null;

  constructor(
    message: string,
    detail: {
      workflowId?: string | null;
      stoppedBy?: string | null;
      stoppedAt?: string | null;
      reason?: string | null;
    } = {},
    opts: TensorCostErrorOptions & { detail?: Record<string, unknown> | null } = {},
  ) {
    super(
      message,
      { code: "RUN_STOPPED", reason: detail.reason ?? null, detail: opts.detail ?? null },
      { ...opts, status: opts.status ?? 403 },
    );
    this.workflowId = detail.workflowId ?? null;
    this.stoppedBy = detail.stoppedBy ?? null;
    this.stoppedAt = detail.stoppedAt ?? null;
  }
}

/**
 * Daily or monthly agent/team spend cap exceeded (HTTP 403,
 * `code: "PERIOD_BUDGET_EXCEEDED"`). Terminal — must not fall back to the
 * provider or the cap becomes a suggestion.
 */
export class TensorCostPeriodBudgetExceededError extends TensorCostRefusedError {
  constructor(
    message: string,
    readonly scope: string | null,
    readonly period: string | null,
    readonly capCents: number | null,
    readonly spentCents: number | null,
    opts: TensorCostErrorOptions & {
      code?: string | null;
      reason?: string | null;
      detail?: Record<string, unknown> | null;
    } = {},
  ) {
    super(
      message,
      {
        code: opts.code ?? "PERIOD_BUDGET_EXCEEDED",
        reason: opts.reason ?? null,
        detail: opts.detail ?? null,
      },
      opts,
    );
  }
}

/**
 * A money budget's hard cap refused the call (HTTP 403).
 *
 * The server sends `error.code: "PERIOD_BUDGET_EXCEEDED"` with
 * `error.budget_code: "BUDGET_HARD_CAP"` (that code is what made 1.4.0
 * treat it as terminal), and older servers sent `error.code:
 * "BUDGET_HARD_CAP"`. Both map here. `code` keeps whichever the server
 * sent; `budgetCode` is always "BUDGET_HARD_CAP".
 *
 * It extends TensorCostPeriodBudgetExceededError because 1.4.0 already
 * raised that class for the current wire shape, so an existing `catch` for
 * it keeps working, and scope / period / capCents / spentCents are filled
 * the same way.
 *
 * `reason: "spend_not_known_yet"` (with `retryable: true` on the wire)
 * means the budget hasn't been counted yet, usually for a few seconds after
 * it is created or its scope changes, so the cap holds calls until the
 * first count lands. That case is `retryable`, and `retryAfterSeconds`
 * comes from the Retry-After header or the body's `retry_after_seconds`.
 * The SDK does not retry on its own.
 */
export class TensorCostBudgetHardCapError extends TensorCostPeriodBudgetExceededError {
  readonly budgetCode = "BUDGET_HARD_CAP";
  readonly budgetId: string | null;
  readonly budgetName: string | null;
  readonly burnPct: number | null;
  readonly retryable: boolean;
  readonly retryAfterSeconds: number | null;

  constructor(
    message: string,
    fields: {
      code?: string | null;
      budgetId?: string | null;
      budgetName?: string | null;
      burnPct?: number | null;
      scope?: string | null;
      period?: string | null;
      capCents?: number | null;
      spentCents?: number | null;
      reason?: string | null;
      retryable?: boolean | null;
      retryAfterSeconds?: number | null;
      detail?: Record<string, unknown> | null;
    } = {},
    opts: TensorCostErrorOptions = {},
  ) {
    super(
      message,
      fields.scope ?? null,
      fields.period ?? null,
      fields.capCents ?? null,
      fields.spentCents ?? null,
      {
        ...opts,
        status: opts.status ?? 403,
        code: fields.code ?? "BUDGET_HARD_CAP",
        reason: fields.reason ?? null,
        detail: fields.detail ?? null,
      },
    );
    this.budgetId = fields.budgetId ?? null;
    this.budgetName = fields.budgetName ?? null;
    this.burnPct = fields.burnPct ?? null;
    this.retryable = fields.retryable === true || this.reason === "spend_not_known_yet";
    this.retryAfterSeconds = fields.retryAfterSeconds ?? null;
  }
}

/**
 * Tenant model governance refused this model (HTTP 403,
 * `code: "MODEL_GOVERNANCE_DENIED"`). Terminal — must not fall back to the
 * provider, or Enforce would become a suggestion.
 */
export class TensorCostModelGovernanceDeniedError extends TensorCostRefusedError {
  constructor(
    message: string,
    readonly provider: string | null,
    readonly model: string | null,
    readonly governanceStatus: string | null,
    opts: TensorCostErrorOptions & { detail?: Record<string, unknown> | null } = {},
  ) {
    super(message, { code: "MODEL_GOVERNANCE_DENIED", detail: opts.detail ?? null }, opts);
  }
}

/**
 * Guardrail hard_stop refused this call (HTTP 403,
 * `code: "GUARDRAIL_HARD_STOP"`). Terminal — same fail-closed contract as
 * run-budget and model governance denials.
 */
export class TensorCostGuardrailHardStopError extends TensorCostRefusedError {
  constructor(
    message: string,
    readonly policyId: string | null,
    opts: TensorCostErrorOptions & {
      reason?: string | null;
      detail?: Record<string, unknown> | null;
    } = {},
  ) {
    super(
      message,
      { code: "GUARDRAIL_HARD_STOP", reason: opts.reason ?? null, detail: opts.detail ?? null },
      opts,
    );
  }
}

/** Compliance policy refused this call (metadata or in-process DLP). */
export class TensorCostComplianceDeniedError extends TensorCostRefusedError {
  constructor(
    message: string,
    reason: string | null,
    opts: TensorCostErrorOptions & { detail?: Record<string, unknown> | null } = {},
  ) {
    super(
      message,
      { code: "COMPLIANCE_DENIED", reason, detail: opts.detail ?? null },
      { ...opts, status: opts.status ?? 403 },
    );
  }
}

/** Client team_id does not match token-bound team scope. */
export class TensorCostComplianceTeamMismatchError extends TensorCostRefusedError {
  constructor(
    message: string,
    opts: TensorCostErrorOptions & { detail?: Record<string, unknown> | null } = {},
  ) {
    super(
      message,
      { code: "COMPLIANCE_TEAM_MISMATCH", detail: opts.detail ?? null },
      { ...opts, status: opts.status ?? 403 },
    );
  }
}
