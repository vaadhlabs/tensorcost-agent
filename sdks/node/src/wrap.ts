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

import { resolveConfig } from "./config.js";
import { TensorCostConfigError } from "./config.js";
import { detectProvider } from "./providers/detect.js";
import { installAppliedMode } from "./applied.js";
import * as openaiProvider from "./providers/openai.js";
import * as anthropicProvider from "./providers/anthropic.js";
import * as bedrockProvider from "./providers/bedrock.js";
import * as vertexProvider from "./providers/vertex.js";
import { CircuitBreaker } from "./circuit.js";
import { bindCircuitObservations } from "./circuit-observations.js";
import { layerRank, needsProxyUrl } from "./layer.js";
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

  const { transport, sdkLayer } = getSharedRuntime(config);
  warmupSharedRuntime(config);

  if (
    needsProxyUrl(config.maxLayer) &&
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
    maxLayer: config.maxLayer,
    appliedMode: config.appliedMode,
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
    );
  }
  return client;
}

export const __test__ = { layerRank, steerRouteUnsupportedError };
