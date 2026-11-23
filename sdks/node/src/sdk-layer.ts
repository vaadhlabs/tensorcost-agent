/**
 * Fetches the console-published control layer with a 30s cache.
 * Decays to govern when the endpoint is unreachable for SDK_LAYER_DECAY_MS.
 *
 * A call waits on TensorCost here at most once, and for at most
 * SDK_LAYER_FETCH_TIMEOUT_MS: the first call of a cold process, before any
 * answer (or failure) is known, token exchange included. Every other call
 * is answered from what is known (fresh cache, stale cache within the decay
 * window, or the decay fallback after a failure) and refreshes in the
 * background. Through 1.7.0 the cold cap applied to the GET alone, after an
 * unbounded token exchange, and a failure was never remembered, so with
 * TensorCost unreachable every call paid the 2 s exchange timeout once for
 * the layer and again for the compliance snapshot.
 */

import type { CompiledComplianceSnapshot, ControlLayer } from "./layer.js";
import {
  intersectGrants,
  isControlLayer,
  minLayer,
  SDK_LAYER_CACHE_TTL_MS,
  SDK_LAYER_DECAY_MS,
  SDK_LAYER_FETCH_TIMEOUT_MS,
  SDK_LAYER_REFRESH_TIMEOUT_MS,
  SDK_LAYER_RETRY_MS,
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
  /** When the last refresh failed; 0 after a success. */
  private lastFailureAt = 0;
  /** The token is bound to another team (403): every call throws it until a refresh succeeds. */
  private teamMismatch: TensorCostComplianceTeamMismatchError | null = null;
  private refreshInflight: Promise<void> | null = null;
  private refreshStartedAt = 0;

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
    return this.complianceOf(await this.fetchPublishedSnapshot(codeMaxLayer));
  }

  /** effective = min(codeMaxLayer, published). */
  async effectiveLayer(codeMaxLayer: ControlLayer): Promise<ControlLayer> {
    const snap = await this.fetchPublishedSnapshot(codeMaxLayer);
    return minLayer(codeMaxLayer, snap.layer);
  }

  /**
   * What effectiveLayer() and getComplianceSnapshot() would answer, when
   * that is known without waiting; null for a cold client that has no
   * answer yet. Throws what they would throw. Lets a wrapper call the
   * provider in the same tick and hand back the provider's own APIPromise.
   */
  peek(
    codeMaxLayer: ControlLayer,
  ): { layer: ControlLayer; compliance: CompiledComplianceSnapshot | null } | null {
    const snap = this.snapshotNow(codeMaxLayer, Date.now());
    if (snap === null) return null;
    return { layer: minLayer(codeMaxLayer, snap.layer), compliance: this.complianceOf(snap) };
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

  private complianceOf(snap: LayerSnapshot): CompiledComplianceSnapshot | null {
    if (snap.compliance) return snap.compliance;
    if (this.cache?.compliance?.fail_mode === "closed") {
      throw new TensorCostComplianceDeniedError(
        "tensorcost: compliance snapshot unavailable (fail-closed)",
        "snapshot_unavailable",
      );
    }
    return null;
  }

  private async fetchPublishedSnapshot(codeMaxLayer: ControlLayer): Promise<LayerSnapshot> {
    const known = this.snapshotNow(codeMaxLayer, Date.now());
    if (known !== null) return known;

    // Cold: nothing known yet. Wait for the one refresh, at most the budget,
    // measured from when that refresh started.
    this.scheduleRefresh(Date.now(), true);
    const inflight = this.refreshInflight;
    const wait = Math.max(0, this.refreshStartedAt + SDK_LAYER_FETCH_TIMEOUT_MS - Date.now());
    if (inflight && wait > 0) {
      let timer: ReturnType<typeof setTimeout> | undefined;
      await Promise.race([
        inflight,
        new Promise<void>((resolve) => {
          timer = setTimeout(resolve, wait);
        }),
      ]).finally(() => clearTimeout(timer));
    }
    const now = Date.now();
    return this.snapshotNow(codeMaxLayer, now) ?? this.fallback(codeMaxLayer, now);
  }

  /** The snapshot a call can use right now, or null when the client is cold. */
  private snapshotNow(codeMaxLayer: ControlLayer, now: number): LayerSnapshot | null {
    if (this.teamMismatch) {
      this.scheduleRefresh(now);
      throw this.teamMismatch;
    }
    if (this.cache && now - this.cache.fetchedAt < SDK_LAYER_CACHE_TTL_MS) {
      return this.cachedSnapshot();
    }
    if (this.cache && now - this.lastSuccessAt < SDK_LAYER_DECAY_MS) {
      this.scheduleRefresh(now);
      return this.cachedSnapshot();
    }
    if (this.cache || this.lastFailureAt > 0) {
      this.scheduleRefresh(now);
      return this.fallback(codeMaxLayer, now);
    }
    return null;
  }

  private cachedSnapshot(): LayerSnapshot {
    const c = this.cache!;
    return { layer: c.layer, dataGrants: c.dataGrants, compliance: c.compliance };
  }

  /** What a call gets when TensorCost hasn't answered. */
  private fallback(codeMaxLayer: ControlLayer, now: number): LayerSnapshot {
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

  /** Start a background refresh unless one is running or one failed a moment ago. */
  private scheduleRefresh(now: number, cold = false): void {
    if (this.refreshInflight) return;
    if (!cold && this.lastFailureAt > 0 && now - this.lastFailureAt < SDK_LAYER_RETRY_MS) return;
    this.refreshStartedAt = now;
    this.refreshInflight = this.refresh().finally(() => {
      this.refreshInflight = null;
    });
  }

  /** One GET of the published layer. Never rejects; the outcome lands in the fields. */
  private async refresh(): Promise<void> {
    try {
      const token = await this.getToken();
      const qs = this.teamId ? `?teamId=${encodeURIComponent(this.teamId)}` : "";
      const url = `${this.baseUrl}${SDK_LAYER_PATH}${qs}`;
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), SDK_LAYER_REFRESH_TIMEOUT_MS);
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
          this.teamMismatch = new TensorCostComplianceTeamMismatchError(
            "tensorcost: SDK team scope does not match token binding",
          );
          this.lastFailureAt = Date.now();
          return;
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
        this.lastFailureAt = 0;
        this.teamMismatch = null;
      } finally {
        clearTimeout(timer);
      }
    } catch {
      this.lastFailureAt = Date.now();
    }
  }
}
