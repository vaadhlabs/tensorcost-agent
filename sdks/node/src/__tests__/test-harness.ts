import type { ControlLayer } from "../layer.js";
import type { SdkLayerClient } from "../sdk-layer.js";
import type { AppliedModeHardeningOpts } from "../providers/openai.js";
import { CircuitBreaker } from "../circuit.js";
import { DEFAULT_RETRY_CONFIG } from "../retry.js";

export function mockSdkLayer(fallback: ControlLayer = "observe"): SdkLayerClient {
  return {
    effectiveLayer: async (code: ControlLayer) => code,
  } as SdkLayerClient;
}

export function defaultHardening(
  overrides: Partial<AppliedModeHardeningOpts> = {},
): AppliedModeHardeningOpts {
  return {
    maxLayer: "observe",
    appliedMode: false,
    proxyUrl: null,
    baseUrl: "http://localhost:1",
    sdkLayer: mockSdkLayer(),
    retry: DEFAULT_RETRY_CONFIG,
    timeoutMs: 60_000,
    headersTimeoutMs: 2_000,
    idleTimeoutMs: 30_000,
    onLifecycleEvent: undefined,
    failOpenEnabled: true,
    circuit: new CircuitBreaker(),
    ...overrides,
  };
}
