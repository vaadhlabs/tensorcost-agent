/**
 * Configuration resolution for the SDK.
 *
 * Resolution order:
 *   1. Explicit options to wrap()
 *   2. Environment variables (TENSORCOST_API_KEY, TENSORCOST_BASE_URL,
 *      TENSORCOST_TENANT_ID, TENSORCOST_PROXY_URL)
 *   3. Defaults (only for baseUrl)
 *
 * If apiKey is still missing after resolution, MissingConfigError is
 * raised. This is the *one* case the SDK does not fail open — it means
 * the SDK was never configured at all, and silently dropping every
 * observation would be worse than a loud import-time failure.
 *
 * If maxLayer is steer or route and proxyUrl is missing from both options
 * and env, TensorCostConfigError is thrown.
 */

import type { ResolvedConfig, WrapOptions } from "./types.js";
import { DEFAULT_RETRY_CONFIG } from "./retry.js";
import {
  DATA_GRANTS,
  needsProxyUrl,
  PROXY_HEADERS_TIMEOUT_MS,
  resolveMaxLayer,
  type DataGrant,
} from "./layer.js";

export const DEFAULT_BASE_URL = "https://api.tensorcost.com";

export class MissingConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MissingConfigError";
  }
}

/** Thrown when applied mode is enabled but required config is absent. */
export class TensorCostConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TensorCostConfigError";
  }
}

export function resolveConfig(options: WrapOptions = {}): ResolvedConfig {
  const apiKey = options.apiKey ?? process.env.TENSORCOST_API_KEY ?? "";
  const baseUrlRaw =
    options.baseUrl ?? process.env.TENSORCOST_BASE_URL ?? DEFAULT_BASE_URL;
  const tenantId =
    options.tenantId ?? process.env.TENSORCOST_TENANT_ID ?? null;
  // Phase A4 batch 3 — environment tag. Trimmed; capped at 64 chars per
  // project_environment_scoping.md. An empty / whitespace-only value
  // collapses to null so the SDK omits the field from the envelope and
  // the backend's column default ('production') applies.
  const environmentRaw =
    options.environment ?? process.env.TENSORCOST_ENVIRONMENT ?? null;
  let environment: string | null = null;
  if (environmentRaw !== null) {
    const trimmed = String(environmentRaw).trim();
    if (trimmed.length > 64) {
      throw new MissingConfigError(
        "TensorCost environment must be 64 characters or fewer.",
      );
    }
    environment = trimmed.length === 0 ? null : trimmed;
  }
  // Phase A4 batch 4 — provider-connection id. Same trim+cap rules as
  // environment; collapsing blank to null so the SDK omits the field
  // when not configured.
  const connectionIdRaw =
    options.connectionId ?? process.env.TENSORCOST_CONNECTION_ID ?? null;
  let connectionId: string | null = null;
  if (connectionIdRaw !== null) {
    const trimmed = String(connectionIdRaw).trim();
    if (trimmed.length > 64) {
      throw new MissingConfigError(
        "TensorCost connectionId must be 64 characters or fewer.",
      );
    }
    connectionId = trimmed.length === 0 ? null : trimmed;
  }
  const teamIdRaw = options.teamId ?? process.env.TENSORCOST_TEAM_ID ?? null;
  let teamId: string | null = null;
  if (teamIdRaw !== null) {
    const trimmed = String(teamIdRaw).trim();
    teamId = trimmed.length === 0 ? null : trimmed;
  }
  const failOpen = options.failOpen ?? true;

  const maxLayer = resolveMaxLayer({
    maxLayer: options.maxLayer,
    appliedMode: options.appliedMode,
  });
  const appliedMode = needsProxyUrl(maxLayer);
  const proxyUrlRaw =
    options.proxyUrl ?? process.env.TENSORCOST_PROXY_URL ?? null;
  const proxyUrl =
    proxyUrlRaw !== null ? proxyUrlRaw.replace(/\/+$/, "") : null;

  if (!apiKey) {
    throw new MissingConfigError(
      "TensorCost apiKey not found. Pass { apiKey } to wrap() or set the " +
        "TENSORCOST_API_KEY environment variable.",
    );
  }

  if (needsProxyUrl(maxLayer) && !proxyUrl) {
    throw new TensorCostConfigError(
      `TensorCost maxLayer '${maxLayer}' requires a proxy URL. ` +
        "Pass { proxyUrl } to wrap() or set the TENSORCOST_PROXY_URL " +
        "environment variable.",
    );
  }

  // Hardening defaults.
  const retry = {
    ...DEFAULT_RETRY_CONFIG,
    ...options.retry,
  };
  const timeoutMs = options.timeoutMs ?? 60_000;
  const headersTimeoutMs =
    options.headersTimeoutMs ?? PROXY_HEADERS_TIMEOUT_MS;
  const idleTimeoutMs = options.idleTimeoutMs ?? 30_000;
  const onLifecycleEvent = options.onLifecycleEvent;
  const failOpenEnabled = options.failOpenEnabled ?? true;
  const providerApiKey = options.providerApiKey ?? null;

  // SDK metadata defaults. Both trim and collapse blank-to-null.
  const deploymentRaw =
    options.deployment ?? process.env.TENSORCOST_DEPLOYMENT ?? null;
  const deployment =
    deploymentRaw !== null
      ? String(deploymentRaw).trim() || null
      : null;

  const promptTemplateIdRaw =
    options.promptTemplateId ?? process.env.TENSORCOST_PROMPT_TEMPLATE_ID ?? null;
  const promptTemplateId =
    promptTemplateIdRaw !== null
      ? String(promptTemplateIdRaw).trim() || null
      : null;

  // Per-agent-run attribution defaults. Same trim + blank-collapses-to-null
  // rule as deployment / promptTemplateId — no length cap here (the DB
  // column is unconstrained `text`); customers pass their own identifiers
  // (session ids, job ids) which can legitimately be longer than the
  // 64-char environment/connectionId caps.
  const agentIdRaw = options.agentId ?? process.env.TENSORCOST_AGENT_ID ?? null;
  const agentId = agentIdRaw !== null ? String(agentIdRaw).trim() || null : null;

  const workflowIdRaw =
    options.workflowId ?? process.env.TENSORCOST_WORKFLOW_ID ?? null;
  const workflowId =
    workflowIdRaw !== null ? String(workflowIdRaw).trim() || null : null;

  const applicationRaw =
    options.application ??
    process.env.TC_APPLICATION ??
    process.env.TENSORCOST_APPLICATION ??
    null;
  const application =
    applicationRaw !== null ? String(applicationRaw).trim() || null : null;

  const tagsRaw =
    options.tags ?? process.env.TC_TAGS ?? process.env.TENSORCOST_TAGS ?? null;
  let tags: string[] = [];
  if (Array.isArray(tagsRaw)) {
    tags = tagsRaw.map((x) => String(x).trim()).filter((x) => x.length > 0);
  } else if (tagsRaw !== null) {
    tags = String(tagsRaw)
      .split(",")
      .map((x) => x.trim())
      .filter((x) => x.length > 0);
  }

  const customer = resolveAttributionField(
    options.customer ?? process.env.TENSORCOST_CUSTOMER ?? null,
    "customer",
  );
  const feature = resolveAttributionField(
    options.feature ?? process.env.TENSORCOST_FEATURE ?? null,
    "feature",
  );

  const maxGrants = normalizeMaxGrants(options.maxGrants);

  return {
    apiKey,
    baseUrl: baseUrlRaw.replace(/\/+$/, ""),
    tenantId: tenantId || null,
    environment,
    connectionId,
    teamId,
    failOpen,
    maxLayer,
    appliedMode,
    proxyUrl,
    retry,
    timeoutMs,
    headersTimeoutMs,
    idleTimeoutMs,
    onLifecycleEvent,
    failOpenEnabled,
    providerApiKey,
    deployment,
    promptTemplateId,
    agentId,
    workflowId,
    application,
    tags,
    customer,
    feature,
    maxGrants,
  };
}

function normalizeMaxGrants(
  raw: readonly DataGrant[] | undefined,
): DataGrant[] {
  if (!raw || raw.length === 0) return [];
  const allowed = new Set<string>(DATA_GRANTS);
  return raw.filter((g): g is DataGrant => allowed.has(g));
}

function resolveAttributionField(
  raw: string | null,
  fieldName: string,
): string | null {
  if (raw === null) return null;
  const trimmed = String(raw).trim();
  if (trimmed.length === 0) return null;
  if (trimmed.length > 128) {
    throw new MissingConfigError(
      `TensorCost ${fieldName} must be 128 characters or fewer.`,
    );
  }
  return trimmed;
}
