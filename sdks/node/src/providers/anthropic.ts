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
  TensorCostComplianceDeniedError,
  TensorCostComplianceTeamMismatchError,
  TensorCostRefusedError,
} from "../errors.js";
import { providerTarget } from "../applied.js";
import { providerNativeError } from "../provider-errors.js";
import type { CircuitBreaker } from "../circuit.js";
import { emitEvent } from "../telemetry.js";
import type { LifecycleEventCallback } from "../telemetry.js";
import type { RetryConfig } from "../retry.js";
import type { CompiledComplianceSnapshot, ControlLayer } from "../layer.js";
import { ADMIT_TIMEOUT_MS, needsProxyUrl } from "../layer.js";
import type { SdkLayerClient } from "../sdk-layer.js";
import { decisionMetadata } from "../decision.js";
import { SDK_VERSION } from "../version.js";
import {
  billingAccountField,
  billingAccountOfError,
  clientBaseUrl,
} from "../billing-account.js";
import {
  TensorCostAPIPromise,
  anthropicStreamTracker,
  observeApiCall,
  tapStream,
  type CallSource,
  type Usage,
} from "../api-promise.js";
import { resolveModelCallLink } from "../trace-link.js";
import {
  WrapOverhead,
  observationScope,
  postObservation,
  withAwaitExcluded,
} from "../wrap-overhead.js";
import { ADMIT_UNAVAILABLE, callPathToken } from "./call-path.js";
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
  // Not async: create() hands back an APIPromise, as the OpenAI wrapper
  // does (see api-promise.ts). @anthropic-ai/sdk's own messages.stream()
  // calls create(...).withResponse(), so through 1.7.0 it failed outright
  // on a wrapped client.
  const wrapped = function (this: unknown, ...args: any[]) {
    const self = this;
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
    // Per-agent-run attribution. Dedicated top-level fields (NOT the
    // metadata bag — see Observation.agent_id doc in types.ts). Per-call
    // withMeta() override beats the wrap-time default.
    const resolvedAgentId = opts.perCallAgentId ?? opts.agentId ?? undefined;
    const resolvedWorkflowId = opts.perCallWorkflowId ?? opts.workflowId ?? undefined;
    const resolvedCustomer =
      opts.perCallCustomer ?? opts.hardening.customer ?? undefined;
    const resolvedFeature =
      opts.perCallFeature ?? opts.hardening.feature ?? undefined;

    let complianceStamp: { compliance_frameworks?: string[]; dlp_action?: string } =
      {};

    const observation = (o: {
      status: "success" | "error";
      usage?: Usage;
      error?: unknown;
      billingAccount?: string | null;
      decisionMeta?: Record<string, string>;
    }): Observation => ({
      sdk_version: SDK_VERSION,
      provider: "anthropic",
      model: String(model),
      operation: "messages",
      request_at: requestAt,
      response_at: nowIso(),
      input_tokens: o.usage?.input ?? null,
      output_tokens: o.usage?.output ?? null,
      cost_usd_cents: null,
      status: o.status,
      error_message:
        o.error !== undefined ? `${(o.error as Error)?.name}: ${(o.error as Error)?.message}` : null,
      correlation_id: correlationId,
      ...traceLink,
      ...(opts.environment !== null ? { environment: opts.environment } : {}),
      ...(opts.connectionId !== null ? { connection_id: opts.connectionId } : {}),
      ...(resolvedAgentId !== undefined ? { agent_id: resolvedAgentId } : {}),
      ...(resolvedWorkflowId !== undefined ? { workflow_id: resolvedWorkflowId } : {}),
      ...(resolvedCustomer !== undefined ? { customer: resolvedCustomer } : {}),
      ...(resolvedFeature !== undefined ? { feature: resolvedFeature } : {}),
      ...(o.billingAccount !== undefined ? billingAccountField(o.billingAccount) : {}),
      ...(o.decisionMeta !== undefined
        ? { metadata: { ...(meta ?? {}), ...o.decisionMeta } }
        : meta !== undefined
          ? { metadata: meta }
          : {}),
    });

    const post = (obs: Observation): void =>
      postObservation(opts.transport, overhead, {
        ...obs,
        ...(complianceStamp.compliance_frameworks?.length
          ? { compliance_frameworks: complianceStamp.compliance_frameworks }
          : {}),
        ...(complianceStamp.dlp_action ? { dlp_action: complianceStamp.dlp_action } : {}),
        ...observationScope(opts.hardening.application, opts.hardening.tags),
      });

    const applyCompliance = (snap: CompiledComplianceSnapshot | null): void => {
      assertTeamScope({ clientTeamId: opts.hardening.teamId ?? null });
      const dlpResult = enforceInProcessDlp(extractAnthropicMessagesText(firstArg), snap);
      complianceStamp = complianceStampFrom(snap, dlpResult);
    };

    const callProvider = (decisionMeta?: Record<string, string>): unknown => {
      const url = clientBaseUrl(opts.clientRef);
      let call: unknown;
      try {
        call = bound.apply(self, args);
      } catch (err) {
        call = Promise.reject(err);
      }
      overhead.pauseForAwait();
      return observeApiCall(call, {
        url,
        usageOf: extractUsage,
        streamTracker: anthropicStreamTracker,
        success: (usage, billingAccount) => {
          overhead.resumeAfterAwait();
          post(observation({ status: "success", usage, billingAccount, decisionMeta }));
        },
        error: (err) => {
          overhead.resumeAfterAwait();
          post(
            observation({
              status: "error",
              error: err,
              billingAccount: billingAccountOfError(err, url),
              decisionMeta,
            }),
          );
        },
      });
    };

    const proxied = (layer: ControlLayer) => needsProxyUrl(layer) && !!opts.hardening.proxyUrl;

    let known: ReturnType<SdkLayerClient["peek"]> = null;
    try {
      const sl = opts.hardening.sdkLayer as Partial<SdkLayerClient>;
      known = typeof sl.peek === "function" ? sl.peek.call(sl, opts.hardening.maxLayer) : null;
    } catch (err) {
      return TensorCostAPIPromise.rejected(err);
    }
    if (known !== null && !proxied(known.layer) && known.layer !== "govern") {
      try {
        applyCompliance(known.compliance);
      } catch (err) {
        if (
          err instanceof TensorCostComplianceDeniedError ||
          err instanceof TensorCostComplianceTeamMismatchError
        ) {
          return TensorCostAPIPromise.rejected(err);
        }
      }
      return callProvider();
    }

    const run = async (): Promise<CallSource> => {
      const effectiveLayer = await withAwaitExcluded(overhead, () =>
        opts.hardening.sdkLayer.effectiveLayer(opts.hardening.maxLayer),
      );

      try {
        const complianceSnap = await withAwaitExcluded(overhead, () =>
          opts.hardening.sdkLayer.getComplianceSnapshot(effectiveLayer),
        );
        applyCompliance(complianceSnap);
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
          const admit = await withAwaitExcluded(overhead, async () => {
            const bearerToken = await callPathToken(opts, ADMIT_TIMEOUT_MS);
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
              failOpen: opts.hardening.failOpenEnabled,
            });
          });
          if (!admit.allowed) {
            // The admit client types every refusal by its code. The generic
            // error is only a guard; it used to say "model governance" for
            // every refusal, budget caps included.
            throw (
              admit.refusal ??
              new TensorCostRefusedError("tensorcost: request refused at govern layer", {}, {
                status: 403,
                attempt: 1,
              })
            );
          }
          decisionMeta = admit.unavailable ? ADMIT_UNAVAILABLE : decisionMetadata(admit.decisionHeader);
        } catch (err) {
          // Same rule as the proxy path: go around TensorCost only when it
          // is unavailable (token exchange or admit unreachable), and only
          // with failOpenEnabled. A refusal, known code or not, stops the call.
          if (!shouldFallBackToProvider(err) || !opts.hardening.failOpenEnabled) throw err;
          decisionMeta = ADMIT_UNAVAILABLE;
        }
      }

      const proxyUrl = opts.hardening.proxyUrl;
      if (proxied(effectiveLayer) && proxyUrl) {
        const { providerUrl, providerAuth } = providerTarget(
          opts.clientRef,
          "anthropic",
          "https://api.anthropic.com",
        );

        const body: Record<string, unknown> = {
          ...(firstArg && typeof firstArg === "object" ? firstArg : {}),
          model: String(model),
        };

        const isStreamingRequest = body.stream === true;
        const request = {
          proxyUrl,
          providerUrl,
          providerAuth,
          body,
          model: String(model),
          provider: "anthropic" as const,
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
        };

        let proxyResult: unknown;
        let proxyResponse: Response | null = null;
        let proxyFailed = false;
        let proxyFailureError: Error | null = null;

        const circuitWasOpen = opts.hardening.circuit.shouldBypass();
        let shouldTryProxy = !circuitWasOpen;
        if (circuitWasOpen) {
          shouldTryProxy = opts.hardening.circuit.shouldProbe();
        }

        if (shouldTryProxy) {
          try {
            if (isStreamingRequest) {
              const proxyStreamResp = await withAwaitExcluded(overhead, async () => {
                const bearerToken = await callPathToken(opts, opts.hardening.headersTimeoutMs);
                return proxyRequestStream({ ...request, bearerToken });
              });
              decisionMeta = decisionMetadata(proxyStreamResp.decisionHeader);
              proxyResult = sseChunksFromStream(proxyStreamResp.stream);
              proxyResponse = proxyStreamResp.response ?? null;
            } else {
              const proxyResp = await withAwaitExcluded(overhead, async () => {
                const bearerToken = await callPathToken(opts, opts.hardening.headersTimeoutMs);
                return proxyRequest({ ...request, bearerToken });
              });
              proxyResult = proxyResp.body;
              proxyResponse = proxyResp.response ?? null;
              decisionMeta = decisionMetadata(proxyResp.decisionHeader);
            }
          } catch (proxyErr) {
            if (!shouldFallBackToProvider(proxyErr)) throw providerNativeError(opts.clientRef, proxyErr);
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
            post(observation({ status: "error", error: tcErr, decisionMeta }));
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
          return { kind: "call", call: callProvider(decisionMeta) };
        }

        const proxyMeta = decisionMeta;
        if (isStreamingRequest) {
          overhead.pauseForAwait();
          const stream = tapStream(
            proxyResult,
            anthropicStreamTracker(),
            (usage) => {
              overhead.resumeAfterAwait();
              post(observation({ status: "success", usage, decisionMeta: proxyMeta }));
            },
            (err) => {
              overhead.resumeAfterAwait();
              post(observation({ status: "error", error: err, decisionMeta: proxyMeta }));
            },
          );
          return { kind: "proxy", data: stream, response: proxyResponse };
        }
        post(observation({ status: "success", usage: extractUsage(proxyResult), decisionMeta: proxyMeta }));
        return { kind: "proxy", data: proxyResult, response: proxyResponse };
      }

      // Observe / govern direct path.
      return { kind: "call", call: callProvider(decisionMeta) };
    };

    return new TensorCostAPIPromise(run());
  };

  Object.defineProperty(wrapped, MARKER, { value: true, enumerable: false });
  return wrapped;
}

let pinWarned = false;

/**
 * The layer an Anthropic client actually runs at.
 *
 * The proxy has no Anthropic Messages route yet, so steer/route can't put
 * the proxy in the request path for Claude. Before 1.5.0 that proxy call
 * failed and fail-open sent the call straight to Anthropic, which meant
 * nothing could stop a Claude run. Govern runs the pre-call /admit check
 * instead, so stops, caps and budgets still apply. One warning per
 * process says so.
 */
export function anthropicLayer(requested: ControlLayer): ControlLayer {
  if (!needsProxyUrl(requested)) return requested;
  if (!pinWarned) {
    pinWarned = true;
    // eslint-disable-next-line no-console
    console.warn(
      `tensorcost: maxLayer '${requested}' isn't available for Anthropic clients yet ` +
        "(the TensorCost proxy has no Anthropic Messages route), so this client runs " +
        "in 'govern': every call is checked with /admit before it goes to Anthropic, " +
        "which is what lets a run be stopped or capped. Set maxLayer: 'govern' to " +
        "silence this warning.",
    );
  }
  return "govern";
}

export function install(
  client: unknown,
  transport: ObservationTransport,
  tenantId: string | null,
  environment: string | null,
  connectionId: string | null,
  hardeningIn: AppliedModeHardeningOpts,
  deployment: string | null = null,
  promptTemplateId: string | null = null,
  agentId: string | null = null,
  workflowId: string | null = null,
): void {
  if (!client || typeof client !== "object") return;
  const pinned = anthropicLayer(hardeningIn.maxLayer);
  const hardening: AppliedModeHardeningOpts =
    pinned === hardeningIn.maxLayer
      ? hardeningIn
      : { ...hardeningIn, maxLayer: pinned, appliedMode: false };
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

export const __test__ = {
  MARKER,
  WITH_META_MARKER,
  resetPinWarning(): void {
    pinWarned = false;
  },
};
