/**
 * Anthropic client wrapping.
 *
 * Intercepts `client.messages.create`. Same observation contract as the
 * OpenAI wrapper — only metadata is captured (model, input/output tokens),
 * never the message content itself.
 *
 * Applied-mode hardening (v0.4.0): messages.create routes through the
 * proxy-client module, which handles retries, timeouts, typed errors,
 * telemetry hooks, and the fail-open circuit breaker.
 */

import type { Observation } from "../types.js";
import type { ObservationTransport } from "../transport.js";
import {
  proxyRequest,
  proxyRequestStream,
  sseChunksFromStream,
  shouldFallBackToProvider,
} from "../proxy-client.js";
import { admitRequest } from "../admit-client.js";
import {
  assertTeamScope,
  complianceStampFrom,
  enforceInProcessDlp,
} from "../compliance-guard.js";
import { extractAnthropicMessagesText } from "../dlp.js";
import {
  TensorCostProxyError,
  TensorCostModelGovernanceDeniedError,
  TensorCostComplianceDeniedError,
  TensorCostComplianceTeamMismatchError,
} from "../errors.js";
import { TensorCostConfigError } from "../config.js";
import type { CircuitBreaker } from "../circuit.js";
import { emitEvent } from "../telemetry.js";
import type { LifecycleEventCallback } from "../telemetry.js";
import type { RetryConfig } from "../retry.js";
import type { ControlLayer } from "../layer.js";
import { needsProxyUrl } from "../layer.js";
import type { SdkLayerClient } from "../sdk-layer.js";
import { decisionMetadata } from "../decision.js";
import { SDK_VERSION } from "../version.js";
import { resolveModelCallLink } from "../trace-link.js";
import { WrapOverhead, postObservation, withAwaitExcluded } from "../wrap-overhead.js";
const MARKER = "__tensorcost_wrapped__";
const WITH_META_MARKER = "__tensorcost_with_meta__";

export interface AppliedModeHardeningOpts {
  maxLayer: ControlLayer;
  appliedMode: boolean;
  proxyUrl: string | null;
  baseUrl: string;
  sdkLayer: SdkLayerClient;
  retry: RetryConfig;
  timeoutMs: number;
  headersTimeoutMs: number;
  idleTimeoutMs: number;
  onLifecycleEvent: LifecycleEventCallback | undefined;
  failOpenEnabled: boolean;
  circuit: CircuitBreaker;
  application?: string | null;
  tags?: string[];
  customer?: string | null;
  feature?: string | null;
  teamId?: string | null;
}

function nowIso(): string {
  return new Date().toISOString();
}

function extractUsage(
  response: unknown,
): { input: number | null; output: number | null } {
  if (!response || typeof response !== "object") {
    return { input: null, output: null };
  }
  const usage = (response as Record<string, any>).usage;
  if (!usage || typeof usage !== "object") {
    return { input: null, output: null };
  }
  const u = usage as Record<string, unknown>;
  const input = typeof u.input_tokens === "number" ? u.input_tokens : null;
  const output =
    typeof u.output_tokens === "number" ? u.output_tokens : null;
  return { input, output };
}

/**
 * Build the metadata bag to attach to an observation. Same logic as the
 * OpenAI provider — kept local to avoid a shared-utils dep.
 */
function buildMetadata(
  deployment: string | null,
  promptTemplateId: string | null,
  perCallMeta?: Record<string, unknown>,
): Record<string, unknown> | undefined {
  const base: Record<string, unknown> = {};
  if (deployment !== null) base.deployment = deployment;
  if (promptTemplateId !== null) base.prompt_template_id = promptTemplateId;
  if (perCallMeta) Object.assign(base, perCallMeta);
  return Object.keys(base).length > 0 ? base : undefined;
}

function wrapMethod(
  bound: (...args: any[]) => any,
  opts: {
    transport: ObservationTransport;
    tenantId: string | null;
    environment: string | null;
    connectionId: string | null;
    hardening: AppliedModeHardeningOpts;
    clientRef: unknown;
    getToken?: () => Promise<string>;
    deployment: string | null;
    promptTemplateId: string | null;
    perCallMeta?: Record<string, unknown>;
    /** Wrap-time agent-id default (per-agent-run cost attribution). */
    agentId?: string | null;
    /** Wrap-time workflow(run)-id default (per-agent-run cost attribution). */
    workflowId?: string | null;
    /** Per-call agent-id override (set via withMeta proxy). */
    perCallAgentId?: string;
    /** Per-call workflow-id override (set via withMeta proxy). */
    perCallWorkflowId?: string;
    /** Per-call customer override (set via withMeta proxy). */
    perCallCustomer?: string;
    /** Per-call feature override (set via withMeta proxy). */
    perCallFeature?: string;
    /** Shared correlation id (agent-sdk + tensorcost wrap integration). */
    perCallCorrelationId?: string;
    /** Agent trace id when proxied under a sub_agent span. */
    perCallTraceId?: string;
    /** sub_agent span id — parent for proxied model_call ingest. */
    perCallParentSpanId?: string;
  },
): (...args: any[]) => any {
  const wrapped = async function (this: unknown, ...args: any[]) {
    const overhead = new WrapOverhead();
    const requestAt = nowIso();
    const firstArg = args[0];
    const model =
      (firstArg && typeof firstArg === "object" && firstArg.model) ||
      "unknown";
    const { correlationId, traceLink } = resolveModelCallLink({
      correlationId: opts.perCallCorrelationId,
      traceId: opts.perCallTraceId,
      parentSpanId: opts.perCallParentSpanId,
    });

    const meta = buildMetadata(opts.deployment, opts.promptTemplateId, opts.perCallMeta);
    const metaSpread = meta !== undefined ? { metadata: meta } : {};
    // Per-agent-run attribution. Dedicated top-level fields (NOT the
    // metadata bag — see Observation.agent_id doc in types.ts). Per-call
    // withMeta() override beats the wrap-time default.
    const resolvedAgentId = opts.perCallAgentId ?? opts.agentId ?? undefined;
    const resolvedWorkflowId = opts.perCallWorkflowId ?? opts.workflowId ?? undefined;
    const resolvedCustomer =
      opts.perCallCustomer ?? opts.hardening.customer ?? undefined;
    const resolvedFeature =
      opts.perCallFeature ?? opts.hardening.feature ?? undefined;

    const effectiveLayer = await withAwaitExcluded(overhead, () =>
      opts.hardening.sdkLayer.effectiveLayer(opts.hardening.maxLayer),
    );

    let complianceStamp: { compliance_frameworks?: string[]; dlp_action?: string } =
      {};
    try {
      assertTeamScope({ clientTeamId: opts.hardening.teamId ?? null });
      const complianceSnap = await withAwaitExcluded(overhead, () =>
        opts.hardening.sdkLayer.getComplianceSnapshot(effectiveLayer),
      );
      const dlpResult = enforceInProcessDlp(
        extractAnthropicMessagesText(firstArg),
        complianceSnap,
      );
      complianceStamp = complianceStampFrom(complianceSnap, dlpResult);
    } catch (err) {
      if (
        err instanceof TensorCostComplianceDeniedError ||
        err instanceof TensorCostComplianceTeamMismatchError
      ) {
        throw err;
      }
    }

    let decisionMeta: Record<string, string> | undefined;
    if (effectiveLayer === "govern") {
      try {
        const getToken =
          opts.getToken ?? (() => opts.transport.getToken());
        const admit = await withAwaitExcluded(overhead, async () => {
          const bearerToken = await getToken();
          return admitRequest({
            baseUrl: opts.hardening.baseUrl,
            bearerToken,
            maxLayer: effectiveLayer,
            metadata: {
              provider: "anthropic",
              model: String(model),
              operation: "messages",
              correlation_id: correlationId,
              environment: opts.environment,
              connection_id: opts.connectionId,
              agent_id: resolvedAgentId,
              workflow_id: resolvedWorkflowId,
              customer: resolvedCustomer,
              feature: resolvedFeature,
              application: opts.hardening.application ?? null,
              tags: opts.hardening.tags ?? [],
              team_id: opts.hardening.teamId ?? null,
            },
            transport: opts.transport,
            failOpen: true,
          });
        });
        if (!admit.allowed) {
          throw new TensorCostModelGovernanceDeniedError(
            "tensorcost: request refused at govern layer",
            "anthropic",
            String(model),
            null,
            { attempt: 1 },
          );
        }
        decisionMeta = decisionMetadata(admit.decisionHeader);
      } catch (err) {
        if (err instanceof TensorCostModelGovernanceDeniedError) throw err;
      }
    }

    const decisionSpread =
      decisionMeta !== undefined
        ? {
            metadata: {
              ...(meta !== undefined ? meta : {}),
              ...decisionMeta,
            },
          }
        : metaSpread;

    const stampObservation = (obs: Observation): Observation => ({
      ...obs,
      ...(complianceStamp.compliance_frameworks?.length
        ? { compliance_frameworks: complianceStamp.compliance_frameworks }
        : {}),
      ...(complianceStamp.dlp_action ? { dlp_action: complianceStamp.dlp_action } : {}),
    });

    const proxyUrl = opts.hardening.proxyUrl;
    if (needsProxyUrl(effectiveLayer) && proxyUrl) {
      // Read the original provider URL from defaultHeaders (set at installAppliedMode).
      const dh = opts.clientRef && typeof opts.clientRef === "object"
        ? ((opts.clientRef as Record<string, unknown>)["defaultHeaders"] as Record<string, string> | undefined)
        : undefined;
      const providerUrl = dh?.["x-tc-provider-url"] ?? "https://api.anthropic.com";
      const providerAuth = dh?.["x-tc-provider-auth"] ?? "";

      const body: Record<string, unknown> = {
        ...(firstArg && typeof firstArg === "object" ? firstArg : {}),
        model: String(model),
      };

      const isStreamingRequest = body.stream === true;

      let proxyResult: unknown;
      let proxyFailed = false;
      let proxyFailureError: Error | null = null;

      const circuitWasOpen = opts.hardening.circuit.shouldBypass();
      let shouldTryProxy = !circuitWasOpen;
      if (circuitWasOpen) {
        shouldTryProxy = opts.hardening.circuit.shouldProbe();
      }

      if (shouldTryProxy) {
        try {
          const getToken = opts.getToken ?? (() => (opts.transport as any).getToken());
          if (isStreamingRequest) {
            const proxyStreamResp = await withAwaitExcluded(overhead, async () => {
              const bearerToken = await getToken();
              return proxyRequestStream({
                proxyUrl,
                bearerToken,
                providerUrl,
                providerAuth,
                body,
                model: String(model),
                provider: "anthropic",
                operation: "messages",
                correlationId,
                environment: opts.environment,
                application: opts.hardening.application ?? null,
                tags: opts.hardening.tags ?? [],
                agentId: resolvedAgentId,
                workflowId: resolvedWorkflowId,
                customer: resolvedCustomer,
                feature: resolvedFeature,
                teamId: opts.hardening.teamId ?? undefined,
                retry: opts.hardening.retry,
                timeoutMs: opts.hardening.timeoutMs,
                headersTimeoutMs: opts.hardening.headersTimeoutMs,
                idleTimeoutMs: opts.hardening.idleTimeoutMs,
                onLifecycleEvent: opts.hardening.onLifecycleEvent,
                circuit: opts.hardening.circuit,
                maxLayer: effectiveLayer,
                failOpenFast: opts.hardening.failOpenEnabled,
              });
            });
            decisionMeta = decisionMetadata(proxyStreamResp.decisionHeader);
            proxyResult = sseChunksFromStream(proxyStreamResp.stream);
          } else {
            const proxyResp = await withAwaitExcluded(overhead, async () => {
              const bearerToken = await getToken();
              return proxyRequest({
                proxyUrl,
                bearerToken,
                providerUrl,
                providerAuth,
                body,
                model: String(model),
                provider: "anthropic",
                operation: "messages",
                correlationId,
                environment: opts.environment,
                application: opts.hardening.application ?? null,
                tags: opts.hardening.tags ?? [],
                agentId: resolvedAgentId,
                workflowId: resolvedWorkflowId,
                customer: resolvedCustomer,
                feature: resolvedFeature,
                retry: opts.hardening.retry,
                timeoutMs: opts.hardening.timeoutMs,
                headersTimeoutMs: opts.hardening.headersTimeoutMs,
                onLifecycleEvent: opts.hardening.onLifecycleEvent,
                circuit: opts.hardening.circuit,
                maxLayer: effectiveLayer,
                failOpenFast: opts.hardening.failOpenEnabled,
              });
            });
            proxyResult = proxyResp.body;
            decisionMeta = decisionMetadata(proxyResp.decisionHeader);
          }
        } catch (proxyErr) {
          if (!shouldFallBackToProvider(proxyErr)) throw proxyErr;
          proxyFailed = true;
          proxyFailureError = proxyErr as Error;
        } finally {
          opts.hardening.circuit.releaseStuckProbe();
        }
      } else {
        proxyFailed = true;
      }

      if (proxyFailed) {
        if (!opts.hardening.failOpenEnabled) {
          const tcErr =
            proxyFailureError instanceof TensorCostProxyError
              ? proxyFailureError
              : new TensorCostProxyError(
                  proxyFailureError?.message ?? "proxy unavailable",
                  { cause: proxyFailureError ?? undefined, attempt: 1 },
                );
          const obs: Observation = {
            sdk_version: SDK_VERSION,
            provider: "anthropic",
            model: String(model),
            operation: "messages",
            request_at: requestAt,
            response_at: nowIso(),
            input_tokens: null,
            output_tokens: null,
            cost_usd_cents: null,
            status: "error",
            error_message: `${tcErr.name}: ${tcErr.message}`,
            correlation_id: correlationId,
            ...traceLink,
            ...(opts.environment !== null ? { environment: opts.environment } : {}),
            ...(opts.connectionId !== null ? { connection_id: opts.connectionId } : {}),
            ...(resolvedAgentId !== undefined ? { agent_id: resolvedAgentId } : {}),
            ...(resolvedWorkflowId !== undefined ? { workflow_id: resolvedWorkflowId } : {}),
            ...(resolvedCustomer !== undefined ? { customer: resolvedCustomer } : {}),
            ...(resolvedFeature !== undefined ? { feature: resolvedFeature } : {}),
            ...decisionSpread,
          };
          postObservation(opts.transport, overhead, stampObservation(obs));
          throw tcErr;
        }

        emitEvent(opts.hardening.onLifecycleEvent, {
          kind: "on_fallback",
          provider: "anthropic",
          model: String(model),
          operation: "messages",
          attemptNumber: 1,
          elapsedMs: 0,
          consecutiveFailures: opts.hardening.circuit.failures,
        });

        try {
          const directResponse = await withAwaitExcluded(overhead, () =>
            bound.apply(this, args),
          );
          const { input, output } = extractUsage(directResponse);
          const obs: Observation = {
            sdk_version: SDK_VERSION,
            provider: "anthropic",
            model: String(model),
            operation: "messages",
            request_at: requestAt,
            response_at: nowIso(),
            input_tokens: input,
            output_tokens: output,
            cost_usd_cents: null,
            status: "success",
            error_message: null,
            correlation_id: correlationId,
            ...traceLink,
            ...(opts.environment !== null ? { environment: opts.environment } : {}),
            ...(opts.connectionId !== null ? { connection_id: opts.connectionId } : {}),
            ...(resolvedAgentId !== undefined ? { agent_id: resolvedAgentId } : {}),
            ...(resolvedWorkflowId !== undefined ? { workflow_id: resolvedWorkflowId } : {}),
            ...(resolvedCustomer !== undefined ? { customer: resolvedCustomer } : {}),
            ...(resolvedFeature !== undefined ? { feature: resolvedFeature } : {}),
            ...decisionSpread,
          };
          postObservation(opts.transport, overhead, stampObservation(obs));
          return directResponse;
        } catch (directErr) {
          const obs: Observation = {
            sdk_version: SDK_VERSION,
            provider: "anthropic",
            model: String(model),
            operation: "messages",
            request_at: requestAt,
            response_at: nowIso(),
            input_tokens: null,
            output_tokens: null,
            cost_usd_cents: null,
            status: "error",
            error_message: `${(directErr as Error).name}: ${(directErr as Error).message}`,
            correlation_id: correlationId,
            ...traceLink,
            ...(opts.environment !== null ? { environment: opts.environment } : {}),
            ...(opts.connectionId !== null ? { connection_id: opts.connectionId } : {}),
            ...(resolvedAgentId !== undefined ? { agent_id: resolvedAgentId } : {}),
            ...(resolvedWorkflowId !== undefined ? { workflow_id: resolvedWorkflowId } : {}),
            ...(resolvedCustomer !== undefined ? { customer: resolvedCustomer } : {}),
            ...(resolvedFeature !== undefined ? { feature: resolvedFeature } : {}),
            ...decisionSpread,
          };
          postObservation(opts.transport, overhead, stampObservation(obs));
          throw directErr;
        }
      }

      const proxyDecisionSpread =
        decisionMeta !== undefined
          ? {
              metadata: {
                ...(meta !== undefined ? meta : {}),
                ...decisionMeta,
              },
            }
          : metaSpread;

      // Proxy success.
      const respBody = proxyResult as Record<string, unknown>;
      const usageRaw = respBody?.usage as Record<string, unknown> | undefined;
      const input =
        typeof usageRaw?.input_tokens === "number" ? usageRaw.input_tokens : null;
      const output =
        typeof usageRaw?.output_tokens === "number" ? usageRaw.output_tokens : null;
      const obs: Observation = {
        sdk_version: SDK_VERSION,
        provider: "anthropic",
        model: String(model),
        operation: "messages",
        request_at: requestAt,
        response_at: nowIso(),
        input_tokens: input,
        output_tokens: output,
        cost_usd_cents: null,
        status: "success",
        error_message: null,
        correlation_id: correlationId,
        ...traceLink,
        ...(opts.environment !== null ? { environment: opts.environment } : {}),
        ...(opts.connectionId !== null ? { connection_id: opts.connectionId } : {}),
        ...(resolvedAgentId !== undefined ? { agent_id: resolvedAgentId } : {}),
        ...(resolvedWorkflowId !== undefined ? { workflow_id: resolvedWorkflowId } : {}),
        ...(resolvedCustomer !== undefined ? { customer: resolvedCustomer } : {}),
        ...(resolvedFeature !== undefined ? { feature: resolvedFeature } : {}),
        ...proxyDecisionSpread,
      };
      postObservation(opts.transport, overhead, stampObservation(obs));
      return proxyResult;
    }

    // Observe / govern direct path.
    try {
      const response = await withAwaitExcluded(overhead, () =>
        bound.apply(this, args),
      );
      const { input, output } = extractUsage(response);
      const obs: Observation = {
        sdk_version: SDK_VERSION,
        provider: "anthropic",
        model: String(model),
        operation: "messages",
        request_at: requestAt,
        response_at: nowIso(),
        input_tokens: input,
        output_tokens: output,
        cost_usd_cents: null,
        status: "success",
        error_message: null,
        correlation_id: correlationId,
        ...traceLink,
        ...(opts.environment !== null ? { environment: opts.environment } : {}),
        ...(opts.connectionId !== null ? { connection_id: opts.connectionId } : {}),
        ...(resolvedAgentId !== undefined ? { agent_id: resolvedAgentId } : {}),
        ...(resolvedWorkflowId !== undefined ? { workflow_id: resolvedWorkflowId } : {}),
        ...(resolvedCustomer !== undefined ? { customer: resolvedCustomer } : {}),
        ...(resolvedFeature !== undefined ? { feature: resolvedFeature } : {}),
        ...decisionSpread,
      };
      postObservation(opts.transport, overhead, stampObservation(obs));
      return response;
    } catch (err) {
      const obs: Observation = {
        sdk_version: SDK_VERSION,
        provider: "anthropic",
        model: String(model),
        operation: "messages",
        request_at: requestAt,
        response_at: nowIso(),
        input_tokens: null,
        output_tokens: null,
        cost_usd_cents: null,
        status: "error",
        error_message: `${(err as Error).name}: ${(err as Error).message}`,
        correlation_id: correlationId,
        ...traceLink,
        ...(opts.environment !== null ? { environment: opts.environment } : {}),
        ...(opts.connectionId !== null ? { connection_id: opts.connectionId } : {}),
        ...(resolvedAgentId !== undefined ? { agent_id: resolvedAgentId } : {}),
        ...(resolvedWorkflowId !== undefined ? { workflow_id: resolvedWorkflowId } : {}),
        ...(resolvedCustomer !== undefined ? { customer: resolvedCustomer } : {}),
        ...(resolvedFeature !== undefined ? { feature: resolvedFeature } : {}),
        ...decisionSpread,
      };
      postObservation(opts.transport, overhead, stampObservation(obs));
      throw err;
    }
  };

  Object.defineProperty(wrapped, MARKER, { value: true, enumerable: false });
  return wrapped;
}

export function install(
  client: unknown,
  transport: ObservationTransport,
  tenantId: string | null,
  environment: string | null,
  connectionId: string | null,
  hardening: AppliedModeHardeningOpts,
  deployment: string | null = null,
  promptTemplateId: string | null = null,
  agentId: string | null = null,
  workflowId: string | null = null,
): void {
  if (!client || typeof client !== "object") return;
  const c = client as Record<string, any>;
  const messages = c.messages;

  let originalMessagesBound: ((...args: any[]) => any) | undefined;

  if (
    messages &&
    typeof messages === "object" &&
    typeof messages.create === "function" &&
    !messages.create[MARKER]
  ) {
    originalMessagesBound = messages.create.bind(messages);
    messages.create = wrapMethod(originalMessagesBound!, {
      transport,
      tenantId,
      environment,
      connectionId,
      hardening,
      clientRef: client,
      deployment,
      promptTemplateId,
      agentId,
      workflowId,
    });
  }

  // withMeta() — builds a fresh wrapMethod against the original (pre-wrap)
  // bound so exactly one observation is emitted per call.
  if (typeof c[WITH_META_MARKER] === "undefined" && originalMessagesBound !== undefined) {
    c[WITH_META_MARKER] = true;

    (client as Record<string, unknown>)["withMeta"] = function (
      callMeta: {
        deployment?: string;
        promptTemplateId?: string;
        agentId?: string;
        workflowId?: string;
        customer?: string;
        feature?: string;
        correlationId?: string;
        traceId?: string;
        parentSpanId?: string;
      } & Record<string, unknown>,
    ): unknown {
      const perCallMeta: Record<string, unknown> = {};
      if (callMeta.deployment !== undefined) perCallMeta.deployment = callMeta.deployment;
      if (callMeta.promptTemplateId !== undefined) perCallMeta.prompt_template_id = callMeta.promptTemplateId;
      for (const [k, v] of Object.entries(callMeta)) {
        if (
          k !== "deployment" &&
          k !== "promptTemplateId" &&
          k !== "agentId" &&
          k !== "workflowId" &&
          k !== "customer" &&
          k !== "feature" &&
          k !== "correlationId" &&
          k !== "traceId" &&
          k !== "parentSpanId"
        ) {
          perCallMeta[k] = v;
        }
      }
      const perCallAgentId = callMeta.agentId;
      const perCallWorkflowId = callMeta.workflowId;
      const perCallCustomer = callMeta.customer;
      const perCallFeature = callMeta.feature;
      const perCallCorrelationId = callMeta.correlationId;
      const perCallTraceId = callMeta.traceId;
      const perCallParentSpanId = callMeta.parentSpanId;

      const proxyCreate =
        originalMessagesBound !== undefined
          ? wrapMethod(originalMessagesBound!, {
              transport,
              tenantId,
              environment,
              connectionId,
              hardening,
              clientRef: client,
              deployment,
              promptTemplateId,
              agentId,
              workflowId,
              perCallAgentId,
              perCallWorkflowId,
              perCallCustomer,
              perCallFeature,
              perCallCorrelationId,
              perCallTraceId,
              perCallParentSpanId,
              perCallMeta,
            })
          : undefined;

      return {
        ...c,
        messages:
          messages !== undefined && proxyCreate !== undefined
            ? { ...messages, create: proxyCreate }
            : messages,
      };
    };
  }
}

export const __test__ = { MARKER, WITH_META_MARKER };
