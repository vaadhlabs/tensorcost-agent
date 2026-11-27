/**
 * Shared by the OpenAI and Anthropic wrappers for the TensorCost hops a
 * call waits on.
 */

import { TensorCostQuotaError } from "../errors.js";
import { emitEvent, type LifecycleEventCallback, type Provider } from "../telemetry.js";
import type { ObservationTransport } from "../transport.js";
import { recordUnchecked, uncheckedMeta, type UncheckedReason } from "../unchecked.js";

/** Metadata on a call that went ahead because /admit couldn't be reached. */
export const ADMIT_UNAVAILABLE: Record<string, string> = Object.freeze({
  decision_action: "allow",
  decision_reason: "admit_unavailable",
}) as Record<string, string>;

/**
 * The TensorCost token for a hop on the customer's request path, waited on
 * for at most `budgetMs` (the hop's own budget). The `getToken` override is
 * for tests.
 */
export function callPathToken(
  opts: { transport: ObservationTransport; getToken?: () => Promise<string> },
  budgetMs: number,
): Promise<string> {
  return opts.getToken ? opts.getToken() : opts.transport.getToken({ budgetMs });
}

/**
 * A call is going to the provider without TensorCost's check (fail-open):
 * count it, warn at most once a minute, raise `on_fallback` with the reason,
 * and return the metadata its observation carries.
 */
export function goingUnchecked(
  reason: UncheckedReason,
  ctx: {
    onLifecycleEvent: LifecycleEventCallback | undefined;
    provider: Provider;
    model: string;
    operation: string;
    consecutiveFailures: number;
  },
): Record<string, string> {
  recordUnchecked(reason);
  emitEvent(ctx.onLifecycleEvent, {
    kind: "on_fallback",
    provider: ctx.provider,
    model: ctx.model,
    operation: ctx.operation,
    attemptNumber: 1,
    elapsedMs: 0,
    consecutiveFailures: ctx.consecutiveFailures,
    reason,
  });
  return uncheckedMeta(reason);
}

/** Why the proxy path fell back: null error means the circuit was open and the proxy wasn't tried. */
export function proxyFallbackReason(err: unknown): UncheckedReason {
  if (err === null || err === undefined) return "circuit_open";
  if (err instanceof TensorCostQuotaError) {
    return err.status === 402 ? "proxy_plan_limit" : "proxy_throttled";
  }
  return "proxy_unavailable";
}

/** Why the token exchange in front of /admit left the call undecided. */
export function admitErrorReason(err: unknown): UncheckedReason {
  if (err instanceof TensorCostQuotaError) {
    return err.status === 402 ? "admit_plan_limit" : "admit_throttled";
  }
  return "admit_unavailable";
}
