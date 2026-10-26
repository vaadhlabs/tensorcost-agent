/**
 * Govern-layer admit — metadata-only POST /api/inference-proxy/v1/admit.
 *
 * Fail-open only on an outage: network error, timeout, 5xx, or a 429 / 402
 * with no refusal body. The customer's call then proceeds to the provider and
 * an `admit_unavailable` observation is emitted when transport is set.
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
  TensorCostNetworkError,
  TensorCostProxyError,
  TensorCostQuotaError,
  TensorCostRefusedError,
  TensorCostTimeoutError,
} from "./errors.js";
import { retryAfterMs } from "./retry.js";
import { decisionSaysRefuse, parseRefusalBody, refusalError } from "./refusal.js";
import { observationScope } from "./wrap-overhead.js";

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
  refusal?: TensorCostRefusedError | TensorCostGatewayRejectedError | TensorCostKeyRejectedError;
}

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

  const controller = new AbortController();
  const timer = setTimeout(
    () => controller.abort(),
    opts.timeoutMs ?? ADMIT_TIMEOUT_MS,
  );

  try {
    let resp: Response;
    try {
      resp = await fetchImpl(url, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (err) {
      const e = err as Error;
      throw e?.name === "AbortError" || e?.name === "TimeoutError"
        ? new TensorCostTimeoutError(
            `tensorcost: admit timed out after ${opts.timeoutMs ?? ADMIT_TIMEOUT_MS}ms`,
            "total",
            { cause: e },
          )
        : new TensorCostNetworkError(
            `tensorcost: network error reaching admit: ${e?.message ?? String(err)}`,
            { cause: e },
          );
    }

    const decisionHeader = resp.headers.get("x-tc-decision");
    const requestId = resp.headers.get("x-tc-request-id");

    if (resp.status >= 400 && resp.status < 500) {
      const refusal = await refusalFrom4xx(resp, decisionHeader);
      if (refusal) return { allowed: false, decisionHeader, refusal };
      // Only a bare 429 or 402 gets here.
      throw new TensorCostQuotaError(`admit POST failed status=${resp.status}`, retryAfterMs(resp) ?? null, {
        status: resp.status,
        requestId,
      });
    }

    if (!resp.ok) {
      throw new TensorCostProxyError(`admit POST failed status=${resp.status}`, {
        status: resp.status,
        requestId,
      });
    }

    return { allowed: true, decisionHeader };
  } catch (err) {
    if (opts.failOpen !== false) {
      postAdmitFailureObservation(opts, err as Error);
      return { allowed: true, decisionHeader: null };
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * The refusal in a 4xx from /admit, or undefined for the one 4xx that is
 * an outage: a 429 (the tenant throttler) or 402 (plan ceiling) with no
 * refusal body.
 */
async function refusalFrom4xx(
  resp: Response,
  decisionHeader: string | null,
): Promise<
  TensorCostRefusedError | TensorCostGatewayRejectedError | TensorCostKeyRejectedError | undefined
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
    // HTML or nothing: something in front of TensorCost said no.
    return new TensorCostGatewayRejectedError(resp.status, text ? text.slice(0, 500) : null, {
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

function postAdmitFailureObservation(
  opts: AdmitRequestOptions,
  err: Error,
): void {
  if (!opts.transport) return;
  try {
    opts.transport.post({
      sdk_version: SDK_VERSION,
      provider: opts.metadata.provider,
      model: opts.metadata.model,
      operation: opts.metadata.operation,
      request_at: new Date().toISOString(),
      response_at: new Date().toISOString(),
      input_tokens: null,
      output_tokens: null,
      cost_usd_cents: null,
      status: "success",
      error_message: null,
      correlation_id: opts.metadata.correlation_id,
      ...(opts.metadata.environment ? { environment: opts.metadata.environment } : {}),
      ...(opts.metadata.connection_id ? { connection_id: opts.metadata.connection_id } : {}),
      ...observationScope(opts.metadata.application, opts.metadata.tags),
      metadata: {
        decision_action: "allow",
        decision_reason: "admit_unavailable",
      },
    });
  } catch {
    /* belt-and-suspenders */
  }
}
