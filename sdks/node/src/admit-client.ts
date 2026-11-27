/**
 * Govern-layer admit — metadata-only POST /api/inference-proxy/v1/admit.
 *
 * Fail-open only when TensorCost gave no decision: a network error, a
 * timeout, a 5xx, or a 429 / 402 with no refusal body. The customer's call
 * then proceeds to the provider and the result says `unavailable` and why,
 * so the wrapper can mark that call's own observation.
 *
 * A bare 429 is TensorCost pacing this process, not an outage (1.9.0). It is
 * retried inside the admit budget: after Retry-After when that fits (and
 * every call to the same /admit holds off until then), else after a short
 * jittered backoff. Still throttled when the budget runs out, it takes the
 * same fail mode as an outage, with the reason `admit_throttled`. Through
 * 1.8.2 a 429 was not retried and was recorded as `admit_unavailable`.
 * A bare 402 (the plan's ceiling) is not retried: `admit_plan_limit`.
 *
 * Every other non-2xx is an answer, and the result carries it as
 * `refusal`, a TensorCostRefusedError subclass the wrapper throws as is:
 * RUN_STOPPED with its who / when / why, BUDGET_HARD_CAP as the budget
 * error, a code this SDK doesn't know as a plain TensorCostRefusedError.
 * /admit only answers 403 when a gate refused, so a 403 with no readable
 * code is still a refusal. A 401/400/404 means admission could not be
 * checked for this call; that is not an outage either, so it refuses too.
 */

import type { ControlLayer } from "./layer.js";
import type { Operation, Provider } from "./types.js";
import type { ObservationTransport } from "./transport.js";
import { ADMIT_PATH, ADMIT_TIMEOUT_MS } from "./layer.js";
import { SDK_VERSION, sdkCapabilityHeader } from "./version.js";
import {
  TensorCostGatewayRejectedError,
  TensorCostKeyRejectedError,
  TensorCostProxyStatusError,
  unexplained4xx,
  TensorCostNetworkError,
  TensorCostProxyError,
  TensorCostQuotaError,
  TensorCostRefusedError,
  TensorCostTimeoutError,
} from "./errors.js";
import { retryAfterMs } from "./retry.js";
import { decisionSaysRefuse, parseRefusalBody, refusalError } from "./refusal.js";

export interface AdmitMetadata {
  provider: Provider;
  model: string;
  operation: Operation;
  correlation_id: string;
  environment?: string | null;
  connection_id?: string | null;
  agent_id?: string;
  workflow_id?: string;
  customer?: string;
  feature?: string;
  application?: string | null;
  tags?: string[];
  team_id?: string | null;
}

export interface AdmitRequestOptions {
  baseUrl: string;
  bearerToken: string;
  maxLayer: ControlLayer;
  metadata: AdmitMetadata;
  /** No longer read: the admit client doesn't post observations of its own. */
  transport?: ObservationTransport;
  failOpen?: boolean;
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
}

export interface AdmitResult {
  allowed: boolean;
  decisionHeader: string | null;
  /**
   * Set whenever `allowed` is false: the refusal, typed by its code, or a
   * TensorCostGatewayRejectedError when something in front of TensorCost
   * answered with a non-JSON 4xx.
   */
  refusal?:
    | TensorCostRefusedError
    | TensorCostGatewayRejectedError
    | TensorCostProxyStatusError
    | TensorCostKeyRejectedError;
  /**
   * True when /admit could not be reached and the call was let through
   * (fail-open). The wrapper marks the call's own observation with
   * decision_reason "admit_unavailable". Through 1.7.0 the admit client
   * posted a second, empty observation for this, which the server stored
   * as a 0-token spend event: one phantom call per admit outage.
   */
  unavailable?: boolean;
  /**
   * With `unavailable`: why /admit gave no answer. `admit_throttled` is a
   * bare 429 still there when the admit budget ran out (TensorCost pacing
   * this process, retried within the budget, Retry-After honoured);
   * `admit_plan_limit` a bare 402; `admit_unavailable` an outage.
   */
  reason?: AdmitUncheckedReason;
}

export type AdmitUncheckedReason = "admit_unavailable" | "admit_throttled" | "admit_plan_limit";

export async function admitRequest(
  opts: AdmitRequestOptions,
): Promise<AdmitResult> {
  const fetchImpl = opts.fetchImpl ?? globalThis.fetch.bind(globalThis);
  const url = `${opts.baseUrl.replace(/\/+$/, "")}${ADMIT_PATH}`;
  const headers: Record<string, string> = {
    Authorization: `Bearer ${opts.bearerToken}`,
    "Content-Type": "application/json",
    "x-tc-sdk-version": SDK_VERSION,
    "x-tc-sdk-capabilities": sdkCapabilityHeader(),
    "x-tc-max-layer": opts.maxLayer,
  };

  const body = {
    provider: opts.metadata.provider,
    model: opts.metadata.model,
    operation: opts.metadata.operation,
    correlation_id: opts.metadata.correlation_id,
    ...(opts.metadata.environment ? { environment: opts.metadata.environment } : {}),
    ...(opts.metadata.connection_id ? { connection_id: opts.metadata.connection_id } : {}),
    ...(opts.metadata.agent_id ? { agent_id: opts.metadata.agent_id } : {}),
    ...(opts.metadata.workflow_id ? { workflow_id: opts.metadata.workflow_id } : {}),
    ...(opts.metadata.customer ? { customer: opts.metadata.customer } : {}),
    ...(opts.metadata.feature ? { feature: opts.metadata.feature } : {}),
    ...(opts.metadata.application ? { application: opts.metadata.application } : {}),
    ...(opts.metadata.tags && opts.metadata.tags.length > 0
      ? { tags: opts.metadata.tags.join(",") }
      : {}),
    ...(opts.metadata.team_id ? { team_id: opts.metadata.team_id } : {}),
  };

  const budgetMs = opts.timeoutMs ?? ADMIT_TIMEOUT_MS;
  const deadline = Date.now() + budgetMs;

  try {
    for (let attempt = 0; ; attempt += 1) {
      // A Retry-After another call got from this /admit holds every call:
      // wait it out when it ends inside this call's budget, else don't send.
      const heldUntil = throttledUntil.get(url) ?? 0;
      if (heldUntil > Date.now()) {
        if (heldUntil + MIN_ATTEMPT_MS > deadline) {
          throw throttled(429, heldUntil - Date.now(), null);
        }
        await sleep(heldUntil - Date.now());
      }

      const resp = await postOnce(fetchImpl, url, headers, body, deadline - Date.now());
      const decisionHeader = resp.headers.get("x-tc-decision");
      const requestId = resp.headers.get("x-tc-request-id");

      if (resp.status >= 400 && resp.status < 500) {
        const refusal = await refusalFrom4xx(resp, decisionHeader);
        if (refusal) return { allowed: false, decisionHeader, refusal };
        // Only a bare 429 or 402 gets here: TensorCost pacing this process,
        // or the plan's ceiling. Neither is an outage, and neither decided.
        const raMs = retryAfterMs(resp) ?? null;
        if (resp.status === 402) throw throttled(402, raMs, requestId);
        if (raMs !== null) throttledUntil.set(url, Date.now() + raMs);
        // Retry-After is the least we wait; the backoff is the floor, so
        // "Retry-After: 0" can't turn into a busy loop.
        const wait = Math.max(raMs ?? 0, backoffMs(attempt));
        if (Date.now() + wait + MIN_ATTEMPT_MS > deadline) throw throttled(429, raMs, requestId);
        await sleep(wait);
        continue;
      }

      if (!resp.ok) {
        throw new TensorCostProxyError(`admit POST failed status=${resp.status}`, {
          status: resp.status,
          requestId,
        });
      }

      return { allowed: true, decisionHeader };
    }
  } catch (err) {
    if (opts.failOpen !== false) {
      return { allowed: true, decisionHeader: null, unavailable: true, reason: uncheckedReasonOf(err) };
    }
    throw err;
  }
}

/** The shortest attempt worth sending: below this the budget is spent. */
const MIN_ATTEMPT_MS = 50;
/** Backoff for a 429 with no Retry-After: 25 ms, doubling, with jitter. */
const BACKOFF_BASE_MS = 25;

/** Per /admit URL: no call is sent before this (a 429's Retry-After). */
const throttledUntil = new Map<string, number>();

function backoffMs(attempt: number): number {
  const ceiling = BACKOFF_BASE_MS * 2 ** attempt;
  return ceiling * (0.5 + Math.random() * 0.5);
}

function throttled(status: 429 | 402, raMs: number | null, requestId: string | null): TensorCostQuotaError {
  return new TensorCostQuotaError(
    status === 429
      ? "tensorcost: admit is rate limiting this process (429) and the admit budget ran out"
      : "tensorcost: admit says the plan's request ceiling is reached (402)",
    raMs,
    { status, requestId },
  );
}

/** Why a call went ahead without an /admit answer. */
function uncheckedReasonOf(err: unknown): AdmitUncheckedReason {
  if (err instanceof TensorCostQuotaError) {
    return err.status === 402 ? "admit_plan_limit" : "admit_throttled";
  }
  return "admit_unavailable";
}

async function postOnce(
  fetchImpl: typeof fetch,
  url: string,
  headers: Record<string, string>,
  body: unknown,
  remainingMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Math.max(0, remainingMs));
  try {
    return await fetchImpl(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    const e = err as Error;
    throw e?.name === "AbortError" || e?.name === "TimeoutError"
      ? new TensorCostTimeoutError(
          `tensorcost: admit timed out after ${Math.round(remainingMs)}ms`,
          "total",
          { cause: e },
        )
      : new TensorCostNetworkError(
          `tensorcost: network error reaching admit: ${e?.message ?? String(err)}`,
          { cause: e },
        );
  } finally {
    clearTimeout(timer);
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, Math.max(0, ms)));
}

export const __test__ = {
  reset(): void {
    throttledUntil.clear();
  },
};

/**
 * The refusal in a 4xx from /admit, or undefined for the one 4xx that is
 * an outage: a 429 (the tenant throttler) or 402 (plan ceiling) with no
 * refusal body.
 */
async function refusalFrom4xx(
  resp: Response,
  decisionHeader: string | null,
): Promise<
  | TensorCostRefusedError
  | TensorCostGatewayRejectedError
  | TensorCostProxyStatusError
  | TensorCostKeyRejectedError
  | undefined
> {
  const text = await resp.text().catch(() => "");
  const parsed = parseRefusalBody(text);
  if (
    parsed === null &&
    (resp.status === 429 || resp.status === 402) &&
    !decisionSaysRefuse(decisionHeader)
  ) {
    return undefined;
  }
  const ctx = {
    status: resp.status,
    requestId: resp.headers.get("x-tc-request-id"),
    attempt: 1,
    retryAfterHeader: resp.headers.get("retry-after"),
  };
  if (parsed !== null) return refusalError(parsed, ctx);
  let body: unknown;
  try {
    body = JSON.parse(text) as unknown;
  } catch {
    body = undefined;
  }
  if (body === undefined || body === null || typeof body !== "object") {
    if (decisionSaysRefuse(decisionHeader)) return refusalError(null, ctx);
    // HTML or nothing: something in front of TensorCost said no, unless
    // x-tc-decision says TensorCost itself answered.
    return unexplained4xx(resp.status, decisionHeader, text ? text.slice(0, 500) : null, {
      requestId: ctx.requestId,
    });
  }
  if (resp.status === 403) return refusalError(null, ctx);
  if (resp.status === 401) {
    // /admit rejected the TensorCost token.
    return new TensorCostKeyRejectedError(401, { requestId: ctx.requestId });
  }
  const rec = body as Record<string, unknown>;
  const message = typeof rec["message"] === "string" ? rec["message"] : null;
  return new TensorCostRefusedError(
    `tensorcost: admission check returned ${resp.status}` +
      (message ? ` (${message})` : "") +
      "; the call was not sent",
    { code: null },
    { status: resp.status, requestId: ctx.requestId, attempt: 1 },
  );
}
