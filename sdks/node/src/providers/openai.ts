/**
 * OpenAI client wrapping.
 *
 * Intercepts:
 *   * `client.chat.completions.create`
 *   * `client.completions.create` (legacy text completions; v0 still ships)
 *
 * We capture only metadata (model, token usage) — never prompt content.
 * Customer prompts stay on their machine.
 *
 * Applied-mode hardening (v0.4.0): chat.completions.create routes through
 * the proxy-client module, which handles retries, timeouts, typed errors,
 * telemetry hooks, and the fail-open circuit breaker. When the circuit is
 * open or the proxy errors after retries, and `failOpenEnabled=true`, the
 * call falls back to the customer's direct provider path.
 */

import type { Observation, Operation } from "../types.js";
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
import { extractOpenAiMessagesText } from "../dlp.js";
import {
  TensorCostProxyError,
  TensorCostComplianceDeniedError,
  TensorCostComplianceTeamMismatchError,
  TensorCostRefusedError,
} from "../errors.js";
import { openAiProviderFor, providerTarget } from "../applied.js";
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
  observeApiCall,
  openAiStreamTracker,
  tapStream,
  type CallSource,
  type Usage,
} from "../api-promise.js";
import {
  WrapOverhead,
  observationScope,
  postObservation,
  withAwaitExcluded,
} from "../wrap-overhead.js";
import { ADMIT_UNAVAILABLE, callPathToken } from "./call-path.js";
import { resolveModelCallLink } from "../trace-link.js";
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
  const input = typeof u.prompt_tokens === "number" ? u.prompt_tokens : null;
  const output =
    typeof u.completion_tokens === "number" ? u.completion_tokens : null;
  return { input, output };
}

/**
 * Build the metadata bag to attach to an observation. Merges the wrap-time
 * defaults (deployment + promptTemplateId) with any per-call overrides
 * supplied via withMeta(). Returns undefined when nothing is configured so
 * the field is absent from the envelope and the wire is not polluted by an
 * empty object.
 */
function buildMetadata(
  deployment: string | null,
  promptTemplateId: string | null,
  perCallMeta?: Record<string, unknown>,
): Record<string, unknown> | undefined {
  const base: Record<string, unknown> = {};
  if (deployment !== null) base.deployment = deployment;
  if (promptTemplateId !== null) base.prompt_template_id = promptTemplateId;
  // Per-call meta wins over wrap-time defaults for overlapping keys.
  if (perCallMeta) Object.assign(base, perCallMeta);
  return Object.keys(base).length > 0 ? base : undefined;
}

function wrapMethod(
  bound: (...args: any[]) => any,
  opts: {
    operation: Operation;
    transport: ObservationTransport;
    tenantId: string | null;
    environment: string | null;
    connectionId: string | null;
    hardening: AppliedModeHardeningOpts;
    /** Wrap-time deployment tag (stamped on every observation). */
    deployment: string | null;
    /** Wrap-time prompt-template id default. */
    promptTemplateId: string | null;
    /** The wrapped client (for reading base URL / key in fallback). */
    clientRef: unknown;
    /** Override token getter for tests. */
    getToken?: () => Promise<string>;
    /** Per-call metadata override (set via withMeta proxy). */
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
  // azure_openai for an AzureOpenAI client: on observations and the govern
  // check. The proxy request keeps "openai", which picks the proxy's route.
  const providerName = openAiProviderFor(opts.clientRef);

  // Not async: create() hands back an APIPromise (see api-promise.ts). The
  // provider's own when nothing has to be checked first, a
  // TensorCostAPIPromise when /admit, the proxy or a cold layer fetch must
  // finish before the call is made.
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

    // Build metadata once per call. Per-call overrides beat wrap-time defaults.
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
      provider: providerName,
      model: String(model),
      operation: opts.operation,
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

    /** In-process DLP and the team check, from a compliance snapshot. */
    const applyCompliance = (snap: CompiledComplianceSnapshot | null): void => {
      assertTeamScope({ clientTeamId: opts.hardening.teamId ?? null });
      const dlpResult = enforceInProcessDlp(extractOpenAiMessagesText(args), snap, args);
      complianceStamp = complianceStampFrom(snap, dlpResult);
    };

    /**
     * Call the provider and record the call when its answer (or its stream)
     * is in. Returns what the client's create() returned, observed.
     */
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
        streamTracker: openAiStreamTracker,
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

    const proxied = (layer: ControlLayer) =>
      needsProxyUrl(layer) && !!opts.hardening.proxyUrl && opts.operation === "chat.completions";
    const admitted = (layer: ControlLayer) =>
      layer === "govern" && opts.operation === "chat.completions";

    // Same tick when the layer is known and nothing needs checking first:
    // the caller gets the client's own APIPromise.
    let known: ReturnType<SdkLayerClient["peek"]> = null;
    try {
      const sl = opts.hardening.sdkLayer as Partial<SdkLayerClient>;
      known = typeof sl.peek === "function" ? sl.peek.call(sl, opts.hardening.maxLayer) : null;
    } catch (err) {
      return TensorCostAPIPromise.rejected(err);
    }
    if (known !== null && !proxied(known.layer) && !admitted(known.layer)) {
      if (opts.operation === "chat.completions") {
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
      }
      return callProvider();
    }

    const run = async (): Promise<CallSource> => {
      const effectiveLayer = await withAwaitExcluded(overhead, () =>
        opts.hardening.sdkLayer.effectiveLayer(opts.hardening.maxLayer),
      );

      if (opts.operation === "chat.completions") {
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
          // Fail-open on snapshot fetch errors unless HIPAA/PCI closed (server enforces on admit).
        }
      }

      let decisionMeta: Record<string, string> | undefined;
      if (admitted(effectiveLayer)) {
        try {
          const admit = await withAwaitExcluded(overhead, async () => {
            // The token wait counts against admit's own budget: with
            // TensorCost unreachable the call waits at most that, once.
            const bearerToken = await callPathToken(opts, ADMIT_TIMEOUT_MS);
            return admitRequest({
              baseUrl: opts.hardening.baseUrl,
              bearerToken,
              maxLayer: effectiveLayer,
              metadata: {
                provider: providerName,
                model: String(model),
                operation: opts.operation,
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

      // steer/route path: route through proxy-client with full hardening.
      const proxyUrl = opts.hardening.proxyUrl;
      if (proxied(effectiveLayer) && proxyUrl) {
        const { providerUrl, providerAuth } = providerTarget(
          opts.clientRef,
          "openai",
          "https://api.openai.com/v1",
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
          provider: "openai" as const,
          operation: String(opts.operation),
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

        // Snapshot circuit state before the call to detect fallback events.
        const circuitWasOpen = opts.hardening.circuit.shouldBypass();
        let shouldTryProxy = !circuitWasOpen;
        if (circuitWasOpen) {
          shouldTryProxy = opts.hardening.circuit.shouldProbe();
        }

        if (shouldTryProxy) {
          try {
            // The token wait counts against the proxy's headers budget.
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
            // A run-budget refusal is a decision, not a failure. Rethrow it
            // immediately so it never reaches the fail-open fallback below —
            // otherwise the cap would be enforced by the proxy and then
            // silently bypassed by us calling OpenAI directly.
            if (!shouldFallBackToProvider(proxyErr)) throw providerNativeError(opts.clientRef, proxyErr);
            proxyFailed = true;
            proxyFailureError = proxyErr as Error;
          } finally {
            opts.hardening.circuit.releaseStuckProbe();
          }
        } else {
          proxyFailed = true;
        }

        // Fail-open: direct provider only when the proxy path failed — not
        // after a successful HALF_OPEN probe (circuitWasOpen must not trigger
        // a second provider call).
        if (proxyFailed) {
          if (!opts.hardening.failOpenEnabled) {
            // Customer opted out of fail-open. Throw the proxy error.
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

          // Emit fallback event and call the provider directly.
          emitEvent(opts.hardening.onLifecycleEvent, {
            kind: "on_fallback",
            provider: "openai",
            model: String(model),
            operation: String(opts.operation),
            attemptNumber: 1,
            elapsedMs: 0,
            consecutiveFailures: opts.hardening.circuit.failures,
          });
          return { kind: "call", call: callProvider(decisionMeta) };
        }

        // Proxy call succeeded. A stream is recorded when it ends, with the
        // usage chunk the proxy relayed.
        const proxyMeta = decisionMeta;
        if (isStreamingRequest) {
          overhead.pauseForAwait();
          const stream = tapStream(
            proxyResult,
            openAiStreamTracker(),
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

      // Observe / govern direct path (or legacy completions.create).
      return { kind: "call", call: callProvider(decisionMeta) };
    };

    return new TensorCostAPIPromise(run());
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

  const chat = c.chat;
  const chatCompletions =
    chat && typeof chat === "object" ? chat.completions : undefined;

  // Capture the original (pre-wrap) bound functions so withMeta() can build
  // fresh wrappers from the real provider method — not from an already-wrapped
  // one that would post a second observation.
  let originalChatBound: ((...args: any[]) => any) | undefined;
  let originalLegacyBound: ((...args: any[]) => any) | undefined;

  if (
    chatCompletions &&
    typeof chatCompletions.create === "function" &&
    !chatCompletions.create[MARKER]
  ) {
    originalChatBound = chatCompletions.create.bind(chatCompletions);
    chatCompletions.create = wrapMethod(originalChatBound!, {
      operation: "chat.completions",
      transport,
      tenantId,
      environment,
      connectionId,
      hardening,
      deployment,
      promptTemplateId,
      agentId,
      workflowId,
      clientRef: client,
    });
  } else if (chatCompletions && chatCompletions.create?.[MARKER]) {
    // Already wrapped — store the underlying original for withMeta.
    // We can't easily unwrap; withMeta on a double-wrapped client is a no-op
    // for the double-post concern (the original is not recoverable without
    // a stronger bookkeeping mechanism). This is an edge case (wrap called
    // twice); the MARKER guard below prevents the second wrap anyway.
    originalChatBound = chatCompletions.create;
  }

  const legacy = c.completions;
  if (
    legacy &&
    typeof legacy === "object" &&
    typeof legacy.create === "function" &&
    !legacy.create[MARKER]
  ) {
    originalLegacyBound = legacy.create.bind(legacy);
    legacy.create = wrapMethod(originalLegacyBound!, {
      operation: "completions",
      transport,
      tenantId,
      environment,
      connectionId,
      hardening: { ...hardening, appliedMode: false }, // legacy completions not proxied
      deployment,
      promptTemplateId,
      agentId,
      workflowId,
      clientRef: client,
    });
  }

  // withMeta() — returns a shallow proxy of the client with per-call metadata
  // overrides. Builds fresh wrapMethod() calls against the ORIGINAL (pre-wrap)
  // bound function so exactly one observation is emitted per call.
  //
  // Usage:
  //   (client as any).withMeta({ promptTemplateId: 'rerank-v3' })
  //     .chat.completions.create({ model: 'gpt-4o', messages })
  if (typeof c[WITH_META_MARKER] === "undefined" && originalChatBound !== undefined) {
    c[WITH_META_MARKER] = true; // mark so we don't add it twice

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
      // Any extra caller-supplied keys also land in the metadata bag.
      // agentId / workflowId / customer / feature are extracted separately
      // below (they are dedicated envelope fields, not metadata bag members)
      // so they are excluded from this freeform pass-through.
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

      const proxyChat =
        originalChatBound !== undefined
          ? wrapMethod(originalChatBound!, {
              operation: "chat.completions",
              transport,
              tenantId,
              environment,
              connectionId,
              hardening,
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
              clientRef: client,
            })
          : undefined;

      const proxyLegacy =
        originalLegacyBound !== undefined
          ? wrapMethod(originalLegacyBound!, {
              operation: "completions",
              transport,
              tenantId,
              environment,
              connectionId,
              hardening: { ...hardening, appliedMode: false },
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
              clientRef: client,
            })
          : undefined;

      return {
        ...c,
        chat: chatCompletions !== undefined && proxyChat !== undefined
          ? { ...chat, completions: { ...chatCompletions, create: proxyChat } }
          : chat,
        completions: legacy !== undefined && proxyLegacy !== undefined
          ? { ...legacy, create: proxyLegacy }
          : legacy,
      };
    };
  }
}

export const __test__ = { MARKER, WITH_META_MARKER };
