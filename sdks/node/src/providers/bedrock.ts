/**
 * AWS Bedrock observe-only client wrapping.
 *
 * Bedrock differs fundamentally from OpenAI and Anthropic: requests go through
 * the AWS Smithy client's `send()` method rather than named sub-resource methods
 * like `chat.completions.create`. The Smithy client also handles SigV4 signing
 * internally, so we cannot intercept the HTTP layer with a baseURL redirect the
 * way applied mode does for the HTTP-Bearer providers.
 *
 * What we CAN do: register a Smithy middleware on `client.middlewareStack` that
 * wraps the inner `next()` handler. Every `client.send(command)` call passes
 * through the middleware stack, so this gives us a clean before/after hook with
 * access to the command input and the resolved output.
 *
 * Supported commands (non-streaming):
 *   - InvokeModelCommand         → operation "bedrock.invoke_model"
 *   - ConverseCommand            → operation "bedrock.converse"
 *
 * Streaming commands (token accounting deferred):
 *   - InvokeModelWithResponseStreamCommand → operation "bedrock.invoke_model_stream"
 *   - ConverseStreamCommand                → operation "bedrock.converse_stream"
 *
 * For streaming commands we emit an observation with input_tokens=null,
 * output_tokens=null, and a note in error_message that per-chunk token
 * accounting is not yet implemented. The observation still records latency
 * and success/error status — enough for cost attribution once per-model
 * pricing and a token estimator are wired in.
 *
 * Token-count extraction:
 *   InvokeModel: the x-amzn-bedrock-input/output-token-count response
 *   headers first (set for every model family). Without them, body shapes:
 *     1. Anthropic-on-Bedrock: usage.input_tokens / output_tokens
 *     2. Amazon Nova / Titan: usage.inputTokens / outputTokens
 *     3. Meta Llama: prompt_token_count / generation_token_count
 *   Converse API (any model): response.usage.inputTokens / outputTokens.
 *   If none match we emit null/null and log a warning so operators can report gaps.
 *
 * Applied mode:
 *   Not supported. SigV4 signing happens inside the AWS SDK before our
 *   middleware sees the request, and the signed URL/headers are region-specific.
 *   wrap(bedrockClient, { appliedMode: true }) throws TensorCostConfigError
 *   immediately in wrap.ts (before this module is called).
 */

import type { Observation, Operation } from "../types.js";
import type { ObservationTransport } from "../transport.js";

import { SDK_VERSION } from "../version.js";
import { resolveModelCallLink } from "../trace-link.js";
import { WrapOverhead, postObservation, withAwaitExcluded } from "../wrap-overhead.js";

// Marker placed on the middleware to prevent double-registration when
// wrap() is called twice on the same client.
const MIDDLEWARE_NAME = "__tensorcost_bedrock_observe__";

// ── Smithy middleware type stubs ──────────────────────────────────────────────
// We don't import @aws-sdk/middleware-stack or @smithy/* to avoid a hard dep.
// The shapes below are structurally compatible with what the SDK passes at
// runtime; TypeScript can verify this without importing the real types.

interface SmithyArgs {
  input: Record<string, unknown>;
}

interface SmithyOutput {
  output: Record<string, unknown>;
  /** Raw HTTP response, passed back up the middleware stack. */
  response?: { headers?: Record<string, string | undefined> };
}

/**
 * Bedrock reports token counts on every InvokeModel response in these
 * headers, whatever the model family. They are the primary source; body
 * shapes are the fallback for when the raw response is not available.
 */
function tokensFromHeaders(
  response: SmithyOutput["response"],
): { input: number | null; output: number | null } | null {
  const h = response?.headers;
  if (!h) return null;
  const read = (name: string): number | null => {
    const raw = h[name] ?? h[name.toLowerCase()];
    const n = raw === undefined ? NaN : Number(raw);
    return Number.isFinite(n) && n >= 0 ? n : null;
  };
  const input = read("x-amzn-bedrock-input-token-count");
  const output = read("x-amzn-bedrock-output-token-count");
  return input === null && output === null ? null : { input, output };
}

type SmithyNext = (args: SmithyArgs) => Promise<SmithyOutput>;

type SmithyMiddleware = (
  next: SmithyNext,
  context: { commandName?: string },
) => (args: SmithyArgs) => Promise<SmithyOutput>;

interface MiddlewareStack {
  add: (
    middleware: SmithyMiddleware,
    opts: {
      step: "initialize" | "serialize" | "build" | "finalize" | "deserialize";
      name: string;
      priority?: "high" | "normal" | "low";
    },
  ) => void;
  /** Available from AWS SDK v3.x — lets us check for existing middleware. */
  identify?: () => Array<[string, unknown]>;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function nowIso(): string {
  return new Date().toISOString();
}

/**
 * Map a Smithy commandName string to the Operation we record in the
 * observation envelope. Falls through to "bedrock.invoke_model" for
 * unknown command names so we always record something.
 */
function commandToOperation(commandName: string | undefined): {
  operation: Operation;
  isStreaming: boolean;
} {
  switch (commandName) {
    case "InvokeModelCommand":
      return { operation: "bedrock.invoke_model", isStreaming: false };
    case "ConverseCommand":
      return { operation: "bedrock.converse", isStreaming: false };
    case "InvokeModelWithResponseStreamCommand":
      return { operation: "bedrock.invoke_model_stream", isStreaming: true };
    case "ConverseStreamCommand":
      return { operation: "bedrock.converse_stream", isStreaming: true };
    default:
      // Unknown Bedrock command — record it under invoke_model so it still
      // shows up in the dashboard rather than being silently dropped.
      return { operation: "bedrock.invoke_model", isStreaming: false };
  }
}

/**
 * Extract token counts from an InvokeModel response body.
 *
 * The response body is a Uint8Array containing a JSON document whose
 * shape depends on the model family. We try two common shapes and fall
 * back to null/null rather than throwing.
 */
function extractInvokeModelTokens(
  output: Record<string, unknown>,
): { input: number | null; output: number | null } {
  // The `body` field is a Uint8Array (or Buffer in Node).
  const rawBody = output.body;
  if (!rawBody) return { input: null, output: null };

  let parsed: Record<string, unknown>;
  try {
    const text =
      typeof rawBody === "string"
        ? rawBody
        : Buffer.from(rawBody as Uint8Array).toString("utf-8");
    parsed = JSON.parse(text) as Record<string, unknown>;
  } catch {
    console.warn("tensorcost: could not parse Bedrock InvokeModel response body for token counts");
    return { input: null, output: null };
  }

  // Meta Llama: top-level prompt_token_count / generation_token_count.
  if (
    typeof parsed.prompt_token_count === "number" ||
    typeof parsed.generation_token_count === "number"
  ) {
    return {
      input: typeof parsed.prompt_token_count === "number" ? parsed.prompt_token_count : null,
      output:
        typeof parsed.generation_token_count === "number" ? parsed.generation_token_count : null,
    };
  }

  const usage = parsed.usage as Record<string, unknown> | undefined;
  if (!usage || typeof usage !== "object") {
    console.warn(
      "tensorcost: Bedrock InvokeModel response has no usage field — " +
        "token counts will be missing. If this is a new model family, " +
        "please open an issue at https://github.com/vaadhlabs/tensorcost.",
    );
    return { input: null, output: null };
  }

  // Anthropic-on-Bedrock: usage.input_tokens / output_tokens
  if (
    typeof usage.input_tokens === "number" ||
    typeof usage.output_tokens === "number"
  ) {
    return {
      input: typeof usage.input_tokens === "number" ? usage.input_tokens : null,
      output: typeof usage.output_tokens === "number" ? usage.output_tokens : null,
    };
  }

  // Amazon Nova / Titan: usage.inputTokens / outputTokens
  if (
    typeof usage.inputTokens === "number" ||
    typeof usage.outputTokens === "number"
  ) {
    return {
      input: typeof usage.inputTokens === "number" ? usage.inputTokens : null,
      output: typeof usage.outputTokens === "number" ? usage.outputTokens : null,
    };
  }

  console.warn(
    "tensorcost: Bedrock InvokeModel usage shape not recognised — " +
      "token counts will be missing. Known shapes: " +
      "Anthropic (usage.input_tokens/output_tokens), " +
      "Nova/Titan (usage.inputTokens/outputTokens), " +
      "Llama (prompt_token_count/generation_token_count).",
  );
  return { input: null, output: null };
}

/**
 * Extract token counts from a Converse API response.
 *
 * The Converse API has a standardised usage block regardless of model
 * family: response.usage.inputTokens / outputTokens.
 */
function extractConverseTokens(
  output: Record<string, unknown>,
): { input: number | null; output: number | null } {
  const usage = output.usage as Record<string, unknown> | undefined;
  if (!usage || typeof usage !== "object") {
    return { input: null, output: null };
  }
  return {
    input: typeof usage.inputTokens === "number" ? usage.inputTokens : null,
    output: typeof usage.outputTokens === "number" ? usage.outputTokens : null,
  };
}

// ── Public install() ──────────────────────────────────────────────────────────

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
  if (!client || typeof client !== "object") return;
  const c = client as Record<string, unknown>;

  const middlewareStack = c.middlewareStack as MiddlewareStack | undefined;
  if (!middlewareStack || typeof middlewareStack.add !== "function") {
    console.warn(
      "tensorcost: BedrockRuntimeClient does not expose a middlewareStack — " +
        "observations will not be recorded. Is @aws-sdk/client-bedrock-runtime >= 3.x installed?",
    );
    return;
  }

  // Guard against double-registration when wrap() is called twice on the
  // same client. The identify() API is available from SDK v3 onwards.
  if (typeof middlewareStack.identify === "function") {
    const existing = middlewareStack.identify();
    const alreadyInstalled = existing.some(([name]) => name === MIDDLEWARE_NAME);
    if (alreadyInstalled) return;
  }

  const middleware: SmithyMiddleware = (next, context) => async (args) => {
    const overhead = new WrapOverhead();
    const requestAt = nowIso();
    // No withMeta() here; an agent-sdk wrap-* shim publishes the link
    // around this call, which ties the observation to its model_call span.
    const { correlationId, traceLink } = resolveModelCallLink({});

    // modelId lives in the command input for both InvokeModel and Converse.
    const modelId =
      typeof args.input.modelId === "string" ? args.input.modelId : "unknown";

    const { operation, isStreaming } = commandToOperation(context.commandName);

    // Build wrap-time metadata bag (deployment + promptTemplateId).
    const bedrockMeta: Record<string, unknown> = {};
    if (deployment !== null) bedrockMeta.deployment = deployment;
    if (promptTemplateId !== null) bedrockMeta.prompt_template_id = promptTemplateId;
    const metaSpread = Object.keys(bedrockMeta).length > 0 ? { metadata: bedrockMeta } : {};

    // Per-agent-run attribution — wrap-time only. Bedrock has no per-call
    // hook (the Smithy middleware is registered once at wrap() time, and
    // there is no withMeta() proxy for Bedrock clients), so agent_id /
    // workflow_id are constant for the lifetime of this wrapped client.
    // A customer who needs a distinct workflow_id per run must construct
    // a new BedrockRuntimeClient (and re-wrap) per run — see
    // Observation.workflow_id doc in types.ts.
    const commonFields = {
      sdk_version: SDK_VERSION,
      provider: "bedrock" as const,
      model: modelId,
      operation,
      request_at: requestAt,
      correlation_id: correlationId,
      ...traceLink,
      ...(environment !== null ? { environment } : {}),
      ...(connectionId !== null ? { connection_id: connectionId } : {}),
      ...(agentId !== null ? { agent_id: agentId } : {}),
      ...(workflowId !== null ? { workflow_id: workflowId } : {}),
      ...(customer !== null ? { customer } : {}),
      ...(feature !== null ? { feature } : {}),
      ...metaSpread,
    };

    // Streaming commands: we cannot aggregate tokens across chunks without
    // collecting the full stream. Emit a placeholder observation now —
    // per-chunk token accounting is a follow-up workstream.
    if (isStreaming) {
      let status: "success" | "error" = "success";
      let errorMessage: string | null = null;
      let result: SmithyOutput;
      try {
        result = await withAwaitExcluded(overhead, () => next(args));
      } catch (err) {
        status = "error";
        errorMessage = `${(err as Error).name}: ${(err as Error).message}`;
        postObservation(transport, overhead, {
          ...commonFields,
          response_at: nowIso(),
          input_tokens: null,
          output_tokens: null,
          cost_usd_cents: null,
          status: "error",
          error_message: errorMessage,
        });
        throw err;
      }
      postObservation(transport, overhead, {
        ...commonFields,
        response_at: nowIso(),
        input_tokens: null,
        output_tokens: null,
        cost_usd_cents: null,
        status,
        // Signal to operators that this is a streaming placeholder, not a gap.
        error_message:
          "streaming: per-chunk token accounting not yet implemented",
      });
      return result;
    }

    // Non-streaming path: InvokeModel + Converse.
    try {
      const result = await withAwaitExcluded(overhead, () => next(args));

      let input: number | null = null;
      let output: number | null = null;

      if (operation === "bedrock.converse") {
        ({ input, output } = extractConverseTokens(result.output));
      } else {
        ({ input, output } =
          tokensFromHeaders(result.response) ?? extractInvokeModelTokens(result.output));
      }

      postObservation(transport, overhead, {
        ...commonFields,
        response_at: nowIso(),
        input_tokens: input,
        output_tokens: output,
        cost_usd_cents: null,
        status: "success",
        error_message: null,
      });
      return result;
    } catch (err) {
      postObservation(transport, overhead, {
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

  middlewareStack.add(middleware, {
    step: "build",
    name: MIDDLEWARE_NAME,
    priority: "normal",
  });
}

export const __test__ = { MIDDLEWARE_NAME };
