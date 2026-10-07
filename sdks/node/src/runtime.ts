/**
 * Process-wide shared transport + sdk-layer clients so `prewarm()` and
 * `wrap()` reuse the same JWT and published-layer cache.
 */

import { resolveConfig } from "./config.js";
import { SdkLayerClient } from "./sdk-layer.js";
import { ObservationTransport } from "./transport.js";
import type { ResolvedConfig, WrapOptions } from "./types.js";

export interface TensorCostRuntime {
  transport: ObservationTransport;
  sdkLayer: SdkLayerClient;
}

const runtimes = new Map<string, TensorCostRuntime>();

export function sharedRuntimeKey(config: {
  baseUrl: string;
  apiKey: string;
  tenantId: string | null;
  failOpen?: boolean;
}): string {
  return `${config.baseUrl}\0${config.tenantId ?? ""}\0${config.apiKey}\0${config.failOpen !== false ? "1" : "0"}`;
}

export function getSharedRuntime(
  options: WrapOptions | ResolvedConfig,
): TensorCostRuntime {
  const config =
    "maxLayer" in options && "apiKey" in options && "baseUrl" in options
      ? (options as ResolvedConfig)
      : resolveConfig(options as WrapOptions);

  const key = sharedRuntimeKey({
    baseUrl: config.baseUrl,
    apiKey: config.apiKey,
    tenantId: config.tenantId,
    failOpen: config.failOpen,
  });
  const existing = runtimes.get(key);
  if (existing) {
    return existing;
  }

  const transport = new ObservationTransport({
    baseUrl: config.baseUrl,
    apiKey: config.apiKey,
    tenantId: config.tenantId,
    failOpen: config.failOpen,
  });
  const sdkLayer = new SdkLayerClient({
    baseUrl: config.baseUrl,
    getToken: () => transport.getToken(),
    teamId: config.teamId,
  });

  const runtime = { transport, sdkLayer };
  runtimes.set(key, runtime);
  return runtime;
}

/** Fire-and-forget JWT + sdk-layer warmup on the shared runtime. */
export function warmupSharedRuntime(config: ResolvedConfig): void {
  if (config.maxLayer === "off") return;
  const { transport, sdkLayer } = getSharedRuntime(config);
  void (async () => {
    try {
      await transport.getToken();
    } catch {
      /* warmup */
    }
    try {
      await sdkLayer.effectiveLayer(config.maxLayer);
    } catch {
      /* warmup */
    }
  })();
}

export const __test__ = {
  clearRuntimes(): void {
    runtimes.clear();
  },
};
