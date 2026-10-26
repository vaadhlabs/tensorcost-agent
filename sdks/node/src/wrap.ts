/**
 * The single public entry point: `wrap(client, options?)`.
 *
 * Returns the SAME client reference (mutates in place), per the SDK
 * ergonomics non-negotiable: customer changes one line, no other code
 * paths shift.
 *
 * Control layers:
 *   off     — client returned unchanged (no transport, no hooks)
 *   observe — telemetry only
 *   govern  — admit (metadata) then direct provider call
 *   steer/route — proxy in request path (OpenAI / Anthropic only)
 */

import { MissingConfigError, resolveConfig } from "./config.js";
import { TensorCostConfigError } from "./config.js";
import { detectProvider } from "./providers/detect.js";
import { installAppliedMode } from "./applied.js";
import * as openaiProvider from "./providers/openai.js";
import * as anthropicProvider from "./providers/anthropic.js";
import * as bedrockProvider from "./providers/bedrock.js";
import * as vertexProvider from "./providers/vertex.js";
import { CircuitBreaker } from "./circuit.js";
import { bindCircuitObservations } from "./circuit-observations.js";
import { layerRank, needsAdmit, needsProxyUrl } from "./layer.js";
import { getSharedRuntime, warmupSharedRuntime } from "./runtime.js";
import type { WrapOptions } from "./types.js";

function steerRouteUnsupportedError(
  provider: string,
  maxLayer: string,
): TensorCostConfigError {
  return new TensorCostConfigError(
    `TensorCost maxLayer '${maxLayer}' is not supported for ${provider} clients. ` +
      "Bedrock and Vertex use provider-native signing that cannot be intercepted at " +
      "the proxy layer. Use maxLayer 'govern' or lower for telemetry, or wrap an " +
      "OpenAI / Anthropic client for steer/route.",
  );
}

const warned = new Set<string>();

function warnOnce(key: string, message: string): void {
  if (warned.has(key)) return;
  warned.add(key);
  // eslint-disable-next-line no-console
  console.warn(message);
}

/**
 * Every layer talks to TensorCost with a short-lived token, and the token
 * exchange needs the tenant id. Without one, no token: no /admit, no proxy
 * call, no observation, and every call goes straight to the provider.
 * Through 1.4.0 that happened silently, whatever maxLayer said.
 */
function noTenantError(layer: string): MissingConfigError {
  return new MissingConfigError(
    `TensorCost maxLayer '${layer}' needs your tenant id, and none is set, so the SDK ` +
      "could never check a call: stops, caps and budgets would not be enforced. Pass " +
      "{ tenantId } to wrap() or set TENSORCOST_TENANT_ID. The console shows it under " +
      "Developers › SDK setup, in the TENSORCOST_TENANT_ID line of the env block.",
  );
}

export function wrap<T>(client: T, options: WrapOptions = {}): T {
  const config = resolveConfig(options);

  // L0 off — no transport, no provider hooks.
  if (config.maxLayer === "off") {
    return client;
  }

  const provider = detectProvider(client);

  if (
    needsProxyUrl(config.maxLayer) &&
    (provider === "bedrock" || provider === "vertex")
  ) {
    throw steerRouteUnsupportedError(provider, config.maxLayer);
  }

  if (!config.tenantId) {
    if (needsAdmit(config.maxLayer)) throw noTenantError(config.maxLayer);
    warnOnce(
      "no-tenant",
      "tensorcost: no tenant id is set (tenantId or TENSORCOST_TENANT_ID), so the SDK " +
        "can't get a token and no observations reach TensorCost. Calls are not affected. " +
        "The console shows the id under Developers › SDK setup.",
    );
  }

  // Bedrock and Vertex have no /admit check yet: govern is accepted but only
  // observes. Said once, so nobody believes a cap is enforced there.
  if ((provider === "bedrock" || provider === "vertex") && needsAdmit(config.maxLayer)) {
    warnOnce(
      `govern-${provider}`,
      `tensorcost: govern isn't available for ${provider === "bedrock" ? "Bedrock" : "Vertex"} ` +
        "yet; calls are observed only. Stops, caps and budgets are not enforced on this client.",
    );
  }

  // Anthropic runs at govern at most until the proxy has a Messages route
  // (warns once). Decided here, before installAppliedMode, so the client's
  // baseURL is never pointed at a proxy route that doesn't exist.
  const maxLayer =
    provider === "anthropic" ? anthropicProvider.anthropicLayer(config.maxLayer) : config.maxLayer;

  const { transport, sdkLayer } = getSharedRuntime(config);
  warmupSharedRuntime(config);

  if (
    needsProxyUrl(maxLayer) &&
    config.proxyUrl !== null &&
    (provider === "openai" || provider === "anthropic")
  ) {
    installAppliedMode(client, config.proxyUrl, provider, {
      application: config.application,
      tags: config.tags,
      customer: config.customer,
      feature: config.feature,
      teamId: config.teamId,
    });
  }

  const circuit = new CircuitBreaker(
    undefined,
    bindCircuitObservations(transport, {
      provider,
      tenantId: config.tenantId,
      environment: config.environment,
    }),
  );

  const hardening = {
    maxLayer,
    appliedMode: needsProxyUrl(maxLayer) && config.appliedMode,
    proxyUrl: config.proxyUrl,
    baseUrl: config.baseUrl,
    sdkLayer,
    retry: config.retry,
    timeoutMs: config.timeoutMs,
    headersTimeoutMs: config.headersTimeoutMs,
    idleTimeoutMs: config.idleTimeoutMs,
    onLifecycleEvent: config.onLifecycleEvent,
    failOpenEnabled: config.failOpenEnabled,
    circuit,
    application: config.application,
    tags: config.tags,
    customer: config.customer,
    feature: config.feature,
    teamId: config.teamId,
  };

  if (provider === "openai") {
    openaiProvider.install(
      client,
      transport,
      config.tenantId,
      config.environment,
      config.connectionId,
      hardening,
      config.deployment,
      config.promptTemplateId,
      config.agentId,
      config.workflowId,
    );
  } else if (provider === "anthropic") {
    anthropicProvider.install(
      client,
      transport,
      config.tenantId,
      config.environment,
      config.connectionId,
      hardening,
      config.deployment,
      config.promptTemplateId,
      config.agentId,
      config.workflowId,
    );
  } else if (provider === "bedrock") {
    bedrockProvider.install(
      client,
      transport,
      config.tenantId,
      config.environment,
      config.connectionId,
      config.deployment,
      config.promptTemplateId,
      config.agentId,
      config.workflowId,
      config.customer,
      config.feature,
      { application: config.application, tags: config.tags },
    );
  } else {
    vertexProvider.install(
      client,
      transport,
      config.tenantId,
      config.environment,
      config.connectionId,
      config.deployment,
      config.promptTemplateId,
      config.agentId,
      config.workflowId,
      config.customer,
      config.feature,
      { application: config.application, tags: config.tags },
    );
  }
  return client;
}

export const __test__ = {
  layerRank,
  steerRouteUnsupportedError,
  resetWarnings(): void {
    warned.clear();
  },
};
