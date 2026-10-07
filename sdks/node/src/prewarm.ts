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
import { getSharedRuntime } from "./runtime.js";
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

/** Best-effort drain of fire-and-forget observation POSTs before Lambda freeze. */
export async function flushObservations(
  options: WrapOptions = {},
  timeoutMs = 5_000,
): Promise<void> {
  const config = resolveConfig(options);
  if (config.maxLayer === "off") return;
  await getSharedRuntime(config).transport.flush(timeoutMs);
}
