/**
 * Prefetch JWT + published control layer so the first model call is not
 * blocked on cold `/sdk-token/exchange` or `/sdk-layer`.
 *
 * Call once at process/Lambda init (module scope), before traffic arrives:
 *
 *   import { prewarm } from "@tensorcost/sdk";
 *   await prewarm({ apiKey, tenantId, baseUrl });
 */

import { resolveConfig } from "./config.js";
import { getSharedRuntime, sharedTransports } from "./runtime.js";
import type { TransportStats } from "./transport.js";
import type { WrapOptions } from "./types.js";

export async function prewarm(options: WrapOptions = {}): Promise<void> {
  const config = resolveConfig(options);
  if (config.maxLayer === "off") return;

  const { transport, sdkLayer } = getSharedRuntime(config);

  await Promise.allSettled([
    transport.getToken(),
    sdkLayer.effectiveLayer(config.maxLayer),
  ]);
}

/**
 * Best-effort drain before a Lambda freeze: sends what is queued and what
 * is waiting to retry. If TensorCost is still unreachable it returns after
 * one attempt rather than waiting out the timeout; the rest stays buffered
 * and keeps retrying. Resolves to how many observations are still
 * undelivered (0 when everything landed).
 */
export async function flushObservations(
  options: WrapOptions = {},
  timeoutMs = 5_000,
): Promise<number> {
  const config = resolveConfig(options);
  if (config.maxLayer === "off") return 0;
  return getSharedRuntime(config).transport.flush(timeoutMs);
}

/**
 * Delivery counters summed over every transport wrap() has created in this
 * process: what is in flight, what is waiting in memory, what went to the
 * disk spool, and what was lost for good (`dropped`).
 */
export function observationStats(): TransportStats {
  const total: TransportStats = {
    inFlight: 0,
    buffered: 0,
    bufferedBytes: 0,
    dropped: 0,
    spooled: 0,
    deliveredAfterRetry: 0,
    rejected: 0,
  };
  for (const t of sharedTransports()) {
    const s = t.stats();
    total.inFlight += s.inFlight;
    total.buffered += s.buffered;
    total.bufferedBytes += s.bufferedBytes;
    total.dropped += s.dropped;
    total.spooled += s.spooled;
    total.deliveredAfterRetry += s.deliveredAfterRetry;
    total.rejected += s.rejected;
  }
  return total;
}
