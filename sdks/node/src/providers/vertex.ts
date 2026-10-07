/**
 * Google Vertex AI (google-genai Client, vertexai=true) — observe-only.
 *
 * Patches `client.models.generate_content` and `generate_content_stream`
 * on the unified `@google/genai` Client when constructed in Vertex mode.
 * Mirrors the Python `_vertex.py` provider and the Bedrock observe-only
 * contract: no applied mode, fail-open transport, streaming placeholders.
 */

import type { Observation, Operation } from "../types.js";
import type { ObservationTransport } from "../transport.js";

import { SDK_VERSION } from "../version.js";
import { resolveModelCallLink } from "../trace-link.js";
import { WrapOverhead, postObservation, withAwaitExcluded } from "../wrap-overhead.js";
const MARKER = "__tensorcost_wrapped__";

function nowIso(): string {
  return new Date().toISOString();
}

function resolveModel(args: unknown[]): string {
  if (args.length === 0) return "unknown";
  const first = args[0];
  if (typeof first === "string") return first;
  if (first && typeof first === "object") {
    const model = (first as Record<string, unknown>).model;
    if (typeof model === "string") return model;
  }
  return "unknown";
}

function extractUsage(
  response: unknown,
): { input: number | null; output: number | null } {
  if (!response || typeof response !== "object") {
    return { input: null, output: null };
  }
  const usage = (response as Record<string, unknown>).usageMetadata
    ?? (response as Record<string, unknown>).usage_metadata;
  if (!usage || typeof usage !== "object") {
    return { input: null, output: null };
  }
  const u = usage as Record<string, unknown>;
  const inputRaw = u.promptTokenCount ?? u.prompt_token_count;
  const outputRaw = u.candidatesTokenCount ?? u.candidates_token_count;
  return {
    input: typeof inputRaw === "number" ? inputRaw : null,
    output: typeof outputRaw === "number" ? outputRaw : null,
  };
}

type MethodSpec = {
  name: "generateContent" | "generateContentStream";
  operation: Operation;
  streaming: boolean;
};

const METHODS: MethodSpec[] = [
  { name: "generateContent", operation: "vertex.generate_content", streaming: false },
  {
    name: "generateContentStream",
    operation: "vertex.generate_content_stream",
    streaming: true,
  },
];

function wrapMethod(
  bound: (...args: unknown[]) => unknown,
  opts: {
    operation: Operation;
    streaming: boolean;
    transport: ObservationTransport;
    environment: string | null;
    connectionId: string | null;
    deployment: string | null;
    promptTemplateId: string | null;
    agentId: string | null;
    workflowId: string | null;
    customer: string | null;
    feature: string | null;
  },
): (...args: unknown[]) => unknown {
  const wrapped = async (...args: unknown[]) => {
    const overhead = new WrapOverhead();
    const model = resolveModel(args);
    const requestAt = nowIso();
    // No withMeta() here; an agent-sdk wrap-* shim publishes the link
    // around this call, which ties the observation to its model_call span.
    const { correlationId, traceLink } = resolveModelCallLink({});

    const meta: Record<string, unknown> = {};
    if (opts.deployment !== null) meta.deployment = opts.deployment;
    if (opts.promptTemplateId !== null) meta.prompt_template_id = opts.promptTemplateId;
    const metaSpread = Object.keys(meta).length > 0 ? { metadata: meta } : {};

    const commonFields = {
      sdk_version: SDK_VERSION,
      provider: "vertex" as const,
      model,
      operation: opts.operation,
      request_at: requestAt,
      correlation_id: correlationId,
      ...traceLink,
      ...(opts.environment !== null ? { environment: opts.environment } : {}),
      ...(opts.connectionId !== null ? { connection_id: opts.connectionId } : {}),
      ...(opts.agentId !== null ? { agent_id: opts.agentId } : {}),
      ...(opts.workflowId !== null ? { workflow_id: opts.workflowId } : {}),
      ...(opts.customer !== null ? { customer: opts.customer } : {}),
      ...(opts.feature !== null ? { feature: opts.feature } : {}),
      ...metaSpread,
    };

    try {
      const response = await withAwaitExcluded(overhead, async () => bound(...args));

      if (opts.streaming) {
        postObservation(opts.transport, overhead, {
          ...commonFields,
          response_at: nowIso(),
          input_tokens: null,
          output_tokens: null,
          cost_usd_cents: null,
          status: "success",
          error_message:
            "streaming: per-chunk token accounting not yet implemented",
        });
        return response;
      }

      const { input, output } = extractUsage(response);
      postObservation(opts.transport, overhead, {
        ...commonFields,
        response_at: nowIso(),
        input_tokens: input,
        output_tokens: output,
        cost_usd_cents: null,
        status: "success",
        error_message: null,
      });
      return response;
    } catch (err) {
      postObservation(opts.transport, overhead, {
        ...commonFields,
        response_at: nowIso(),
        input_tokens: null,
        output_tokens: null,
        cost_usd_cents: null,
        status: "error",
        error_message: `${(err as Error).name}: ${(err as Error).message}`,
      });
      throw err;
    }
  };

  // Functions are objects at runtime, so stamping the marker works — but TS
  // won't widen a call signature straight to an index-signature type, hence
  // the hop through unknown.
  (wrapped as unknown as Record<string, unknown>)[MARKER] = true;
  return wrapped;
}

export function install(
  client: unknown,
  transport: ObservationTransport,
  tenantId: string | null,
  environment: string | null,
  connectionId: string | null,
  deployment: string | null = null,
  promptTemplateId: string | null = null,
  agentId: string | null = null,
  workflowId: string | null = null,
  customer: string | null = null,
  feature: string | null = null,
): void {
  void tenantId;
  if (!client || typeof client !== "object") return;
  const c = client as Record<string, unknown>;
  const models = c.models;
  if (!models || typeof models !== "object") return;
  const m = models as Record<string, unknown>;

  for (const spec of METHODS) {
    const original = m[spec.name];
    if (typeof original !== "function") continue;
    const fn = original as (...args: unknown[]) => unknown;
    if ((fn as unknown as Record<string, unknown>)[MARKER]) continue;

    m[spec.name] = wrapMethod(fn.bind(models), {
      operation: spec.operation,
      streaming: spec.streaming,
      transport,
      environment,
      connectionId,
      deployment,
      promptTemplateId,
      agentId,
      workflowId,
      customer,
      feature,
    });
  }
}

export const __test__ = { MARKER };
