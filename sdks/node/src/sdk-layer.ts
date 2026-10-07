/**
 * Fetches the console-published control layer with a 30s cache.
 * Decays to govern when the endpoint is unreachable for SDK_LAYER_DECAY_MS.
 *
 * Refresh never blocks an agent call once a snapshot exists: expired cache
 * is served immediately (stale-while-revalidate). Cold fetch is capped at
 * SDK_LAYER_FETCH_TIMEOUT_MS so a hung TensorCost cannot stall the model call.
 */

import type { CompiledComplianceSnapshot, ControlLayer } from "./layer.js";
import {
  intersectGrants,
  isControlLayer,
  minLayer,
  SDK_LAYER_CACHE_TTL_MS,
  SDK_LAYER_DECAY_MS,
  SDK_LAYER_FETCH_TIMEOUT_MS,
  SDK_LAYER_PATH,
  type DataGrant,
  type DlpAction,
} from "./layer.js";
import {
  TensorCostComplianceDeniedError,
  TensorCostComplianceTeamMismatchError,
} from "./errors.js";
import { SDK_VERSION, sdkCapabilityHeader } from "./version.js";

export interface SdkLayerClientOptions {
  baseUrl: string;
  getToken: () => Promise<string>;
  /** Optional team scope for compliance overlay (narrow-only). */
  teamId?: string | null;
  fetchImpl?: typeof fetch;
}

type LayerSnapshot = {
  layer: ControlLayer;
  dataGrants: DataGrant[];
  compliance: CompiledComplianceSnapshot | null;
};

function parseComplianceSnapshot(raw: unknown): CompiledComplianceSnapshot | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  if (!Array.isArray(o.frameworks)) return null;
  const mode = o.mode === "enforce" ? "enforce" : "observe";
  const fail_mode = o.fail_mode === "closed" ? "closed" : "open";
  const dlpRaw = o.dlp;
  const dlp =
    dlpRaw && typeof dlpRaw === "object"
      ? {
          enabled: Boolean((dlpRaw as { enabled?: unknown }).enabled),
          detectors: Array.isArray((dlpRaw as { detectors?: unknown }).detectors)
            ? ((dlpRaw as { detectors: unknown[] }).detectors.filter(
                (d): d is string => typeof d === "string",
              ))
            : [],
          action: ((dlpRaw as { action?: unknown }).action === "redact"
            ? "redact"
            : "refuse") as DlpAction,
        }
      : { enabled: false, detectors: [], action: "refuse" as const };
  const auditRaw = o.audit;
  const audit =
    auditRaw && typeof auditRaw === "object"
      ? { stamp: Boolean((auditRaw as { stamp?: unknown }).stamp) }
      : { stamp: false };
  const max_grants = Array.isArray(o.max_grants)
    ? (o.max_grants.filter(
        (g): g is DataGrant =>
          g === "telemetry" ||
          g === "retain_completions" ||
          g === "retain_prompts" ||
          g === "retain_tools",
      ) as DataGrant[])
    : [];
  return {
    frameworks: o.frameworks.filter((f): f is string => typeof f === "string"),
    mode,
    fail_mode,
    max_grants,
    dlp,
    audit,
    ...(Array.isArray(o.allowed_providers)
      ? { allowed_providers: o.allowed_providers.filter((p): p is string => typeof p === "string") }
      : {}),
    ...(Array.isArray(o.allowed_models)
      ? { allowed_models: o.allowed_models.filter((m): m is string => typeof m === "string") }
      : {}),
    ...(Array.isArray(o.allowed_regions)
      ? { allowed_regions: o.allowed_regions.filter((r): r is string => typeof r === "string") }
      : {}),
  };
}

export class SdkLayerClient {
  private readonly baseUrl: string;
  private readonly getToken: () => Promise<string>;
  private readonly teamId: string | null;
  private readonly fetchImpl: typeof fetch;

  private cache: {
    layer: ControlLayer;
    dataGrants: DataGrant[];
    compliance: CompiledComplianceSnapshot | null;
    fetchedAt: number;
  } | null = null;
  /** 0 until the first successful fetch — cold start must not skip decay-to-govern. */
  private lastSuccessAt = 0;
  private refreshInflight: Promise<void> | null = null;

  constructor(opts: SdkLayerClientOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/+$/, "");
    this.getToken = opts.getToken;
    this.teamId = opts.teamId?.trim() || null;
    this.fetchImpl = opts.fetchImpl ?? globalThis.fetch.bind(globalThis);
  }

  /** Compiled compliance snapshot (cached with sdk-layer). */
  async getComplianceSnapshot(
    codeMaxLayer: ControlLayer = "observe",
  ): Promise<CompiledComplianceSnapshot | null> {
    const snap = await this.fetchPublishedSnapshot(codeMaxLayer);
    if (snap.compliance) return snap.compliance;
    if (this.cache?.compliance?.fail_mode === "closed") {
      throw new TensorCostComplianceDeniedError(
        "tensorcost: compliance snapshot unavailable (fail-closed)",
        "snapshot_unavailable",
      );
    }
    return null;
  }

  /** effective = min(codeMaxLayer, published). */
  async effectiveLayer(codeMaxLayer: ControlLayer): Promise<ControlLayer> {
    const snap = await this.fetchPublishedSnapshot(codeMaxLayer);
    return minLayer(codeMaxLayer, snap.layer);
  }

  /** intersect(codeMaxGrants, console.data_grants). Absent codeMax = all console grants. */
  async effectiveGrants(
    codeMaxGrants?: readonly DataGrant[],
    codeMaxLayer: ControlLayer = "observe",
  ): Promise<DataGrant[]> {
    const snap = await this.fetchPublishedSnapshot(codeMaxLayer);
    const consoleGrants = intersectGrants(codeMaxGrants, snap.dataGrants);
    if (!snap.compliance?.max_grants?.length) return consoleGrants;
    return intersectGrants(snap.compliance.max_grants, consoleGrants);
  }

  private async fetchPublishedSnapshot(codeMaxLayer: ControlLayer): Promise<LayerSnapshot> {
    const now = Date.now();

    if (this.cache && now - this.cache.fetchedAt < SDK_LAYER_CACHE_TTL_MS) {
      return {
        layer: this.cache.layer,
        dataGrants: this.cache.dataGrants,
        compliance: this.cache.compliance,
      };
    }

    if (this.cache && now - this.lastSuccessAt < SDK_LAYER_DECAY_MS) {
      this.scheduleRefresh();
      return {
        layer: this.cache.layer,
        dataGrants: this.cache.dataGrants,
        compliance: this.cache.compliance,
      };
    }

    return this.refreshNow(codeMaxLayer);
  }

  private scheduleRefresh(): void {
    if (this.refreshInflight) return;
    this.refreshInflight = this.refreshNow("observe")
      .then(() => undefined)
      .catch(() => undefined)
      .finally(() => {
        this.refreshInflight = null;
      });
  }

  private async refreshNow(codeMaxLayer: ControlLayer): Promise<LayerSnapshot> {
    const now = Date.now();
    try {
      const token = await this.getToken();
      const qs = this.teamId ? `?teamId=${encodeURIComponent(this.teamId)}` : "";
      const url = `${this.baseUrl}${SDK_LAYER_PATH}${qs}`;
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), SDK_LAYER_FETCH_TIMEOUT_MS);
      try {
        const resp = await this.fetchImpl(url, {
          method: "GET",
          headers: {
            Authorization: `Bearer ${token}`,
            "x-tc-sdk-version": SDK_VERSION,
            "x-tc-sdk-capabilities": sdkCapabilityHeader(),
          },
          signal: controller.signal,
        });

        if (resp.status === 403) {
          throw new TensorCostComplianceTeamMismatchError(
            "tensorcost: SDK team scope does not match token binding",
          );
        }
        if (!resp.ok) {
          throw new Error(`sdk-layer GET failed status=${resp.status}`);
        }

        const body = (await resp.json()) as {
          published_layer?: string;
          layer?: string;
          routing_paused?: boolean;
          data_grants?: string[];
          compliance?: unknown;
        };
        const layerRaw = body.published_layer ?? body.layer ?? "observe";
        let layer: ControlLayer = isControlLayer(layerRaw) ? layerRaw : "observe";
        if (body.routing_paused) {
          layer = minLayer(layer, "govern");
        }
        const dataGrants = (body.data_grants ?? []).filter(
          (g): g is DataGrant =>
            g === "telemetry" ||
            g === "retain_completions" ||
            g === "retain_prompts" ||
            g === "retain_tools",
        );

        const compliance = parseComplianceSnapshot(body.compliance);

        this.cache = { layer, dataGrants, compliance, fetchedAt: Date.now() };
        this.lastSuccessAt = Date.now();
        return { layer, dataGrants, compliance };
      } finally {
        clearTimeout(timer);
      }
    } catch (err) {
      if (
        err instanceof TensorCostComplianceTeamMismatchError ||
        err instanceof TensorCostComplianceDeniedError
      ) {
        throw err;
      }
      if (now - this.lastSuccessAt >= SDK_LAYER_DECAY_MS) {
        if (this.cache?.compliance?.fail_mode === "closed") {
          throw new TensorCostComplianceDeniedError(
            "tensorcost: compliance snapshot unavailable (fail-closed)",
            "snapshot_unavailable",
          );
        }
        return { layer: "govern", dataGrants: [], compliance: null };
      }
      return {
        layer: this.cache?.layer ?? codeMaxLayer,
        dataGrants: this.cache?.dataGrants ?? [],
        compliance: this.cache?.compliance ?? null,
      };
    }
  }
}
