/**
 * Govern-layer admit — metadata-only POST /api/inference-proxy/v1/admit.
 *
 * Fail-open on transport failure: the customer's call proceeds to the
 * provider and an error observation is emitted when transport is available.
 *
 * A 403 is a refusal, never a transport failure. When its body says
 * RUN_STOPPED the result carries a TensorCostRunStoppedError, so the
 * provider wrapper can throw it with the who / when / why intact instead of
 * a generic governance denial.
 */

import type { ControlLayer } from "./layer.js";
import type { Operation, Provider } from "./types.js";
import type { ObservationTransport } from "./transport.js";
import { ADMIT_PATH, ADMIT_TIMEOUT_MS } from "./layer.js";
import { SDK_VERSION, sdkCapabilityHeader } from "./version.js";
import { TensorCostRunStoppedError, type TensorCostError } from "./errors.js";

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
  /** The specific refusal, when a 403 body names one the SDK models. */
  refusal?: TensorCostError;
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
    const resp = await fetchImpl(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    const decisionHeader = resp.headers.get("x-tc-decision");

    if (resp.status === 403) {
      const refusal = await refusalFrom403(resp);
      return { allowed: false, decisionHeader, ...(refusal ? { refusal } : {}) };
    }

    if (!resp.ok) {
      throw new Error(`admit POST failed status=${resp.status}`);
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

/** The modelled refusal in a 403 body, or undefined for any other 403. */
async function refusalFrom403(resp: Response): Promise<TensorCostError | undefined> {
  let detail: Record<string, unknown> | undefined;
  try {
    const parsed = (await resp.json()) as Record<string, unknown>;
    const errObj = parsed?.["error"];
    detail = errObj && typeof errObj === "object" ? (errObj as Record<string, unknown>) : undefined;
  } catch {
    return undefined;
  }
  if (detail?.["code"] === "RUN_STOPPED") {
    return TensorCostRunStoppedError.fromProxyBody(detail, {
      status: 403,
      requestId: resp.headers.get("x-tc-request-id"),
      attempt: 1,
    });
  }
  return undefined;
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
      metadata: {
        decision_action: "allow",
        decision_reason: "admit_unavailable",
      },
    });
  } catch {
    /* belt-and-suspenders */
  }
}
