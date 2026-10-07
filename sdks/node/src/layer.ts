/**
 * Control-layer ladder — mirrors @tensorcost/contracts/control-layer for
 * the published SDK (no contracts runtime dependency).
 */

export const CONTROL_LAYERS = [
  "off",
  "observe",
  "govern",
  "steer",
  "route",
] as const;

export type ControlLayer = (typeof CONTROL_LAYERS)[number];

const LAYER_RANK: Record<ControlLayer, number> = {
  off: 0,
  observe: 1,
  govern: 2,
  steer: 3,
  route: 4,
};

export function layerRank(layer: ControlLayer): number {
  return LAYER_RANK[layer];
}

export function minLayer(a: ControlLayer, b: ControlLayer): ControlLayer {
  return layerRank(a) <= layerRank(b) ? a : b;
}

export function isControlLayer(value: string): value is ControlLayer {
  return (CONTROL_LAYERS as readonly string[]).includes(value);
}

/** steer and route require a proxy URL. */
export function needsProxyUrl(layer: ControlLayer): boolean {
  return layerRank(layer) >= layerRank("steer");
}

/** govern runs metadata-only admit before the direct provider call. */
export function needsAdmit(layer: ControlLayer): boolean {
  return layerRank(layer) >= layerRank("govern");
}

export function resolveMaxLayer(options: {
  maxLayer?: ControlLayer;
  appliedMode?: boolean;
}): ControlLayer {
  if (options.maxLayer !== undefined) {
    return options.maxLayer;
  }
  if (options.appliedMode) {
    // eslint-disable-next-line no-console
    console.warn(
      "tensorcost: appliedMode is deprecated; use maxLayer: 'route' instead",
    );
    return "route";
  }
  return "observe";
}

export const PROXY_OPERATION_PATHS: Record<string, string> = {
  "chat.completions": "/api/inference-proxy/v1/chat/completions",
  messages: "/api/inference-proxy/v1/messages",
  completions: "/api/inference-proxy/v1/completions",
  embeddings: "/api/inference-proxy/v1/embeddings",
  responses: "/api/inference-proxy/v1/responses",
};

export function proxyPathForOperation(operation: string): string | null {
  return PROXY_OPERATION_PATHS[operation] ?? null;
}

export const ADMIT_PATH = "/api/inference-proxy/v1/admit";
export const SDK_LAYER_PATH = "/api/inference-proxy/v1/sdk-layer";

/** Wire shape from GET /v1/sdk-layer — mirrors @tensorcost/contracts (no runtime dep). */
export type ComplianceEnforcementMode = "observe" | "enforce";
export type ComplianceFailMode = "open" | "closed";
export type DlpAction = "redact" | "refuse";

export interface CompiledComplianceSnapshot {
  frameworks: string[];
  mode: ComplianceEnforcementMode;
  fail_mode: ComplianceFailMode;
  allowed_providers?: string[];
  allowed_models?: string[];
  allowed_regions?: string[];
  max_grants: DataGrant[];
  dlp: { enabled: boolean; detectors: string[]; action: DlpAction };
  audit: { stamp: boolean };
}
export const SDK_LAYER_CACHE_TTL_MS = 30_000;
export const SDK_LAYER_DECAY_MS = 5 * 60_000;
/** Cap on the blocking sdk-layer GET so a hung TensorCost cannot stall the agent. */
export const SDK_LAYER_FETCH_TIMEOUT_MS = 250;
/** Govern admit must fail-open quickly; agents already wait on the model. */
export const ADMIT_TIMEOUT_MS = 500;
/** Steer proxy must send response headers quickly or fail-open to direct. */
export const PROXY_HEADERS_TIMEOUT_MS = 2_000;

export const DATA_GRANTS = [
  "telemetry",
  "retain_completions",
  "retain_prompts",
  "retain_tools",
] as const;

export type DataGrant = (typeof DATA_GRANTS)[number];

export function intersectGrants(
  codeMax: readonly DataGrant[] | null | undefined,
  consoleGrants: readonly DataGrant[],
): DataGrant[] {
  const consoleSet = new Set(consoleGrants);
  if (codeMax == null) {
    return [...consoleSet];
  }
  if (codeMax.length === 0) {
    return [];
  }
  return codeMax.filter((g) => consoleSet.has(g));
}
