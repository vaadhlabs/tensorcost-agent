/**
 * Provider detection — duck-typed.
 *
 * We avoid importing `openai`, `@anthropic-ai/sdk`, or
 * `@aws-sdk/client-bedrock-runtime` at runtime so that the SDK has zero
 * peer-dep cost when the customer only uses one provider. Detection
 * inspects the shape of the client object instead.
 *
 * Bedrock detection uses the Smithy-generated `config.serviceId` field,
 * which is the most precise available discriminator — it's set by the
 * AWS SDK generator and does not depend on the constructor name (which
 * minifiers can change).
 */

import type { Provider } from "../types.js";

export class UnsupportedClientError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UnsupportedClientError";
  }
}

export function detectProvider(client: unknown): Provider {
  if (client === null || typeof client !== "object") {
    throw new UnsupportedClientError(
      "tensorcost: wrap() requires an object (an OpenAI, Anthropic, or BedrockRuntimeClient instance).",
    );
  }
  const c = client as Record<string, any>;

  // BedrockRuntimeClient (AWS SDK v3 / Smithy): has a `send()` method AND
  // config.serviceId === "Bedrock Runtime". We check this first because
  // Smithy clients have a generic enough shape that the OpenAI/Anthropic
  // checks could theoretically collide on oddly-named properties.
  const hasBedrockShape =
    typeof c.send === "function" &&
    c.config != null &&
    typeof c.config === "object" &&
    (c.config as Record<string, unknown>).serviceId === "Bedrock Runtime";

  if (hasBedrockShape) {
    return "bedrock";
  }

  // google-genai Client in Vertex mode: vertexai=true + models.generate_content.
  const apiClient = c._apiClient;
  const vertexaiFlag =
    c.vertexai === true ||
    (apiClient != null &&
      typeof apiClient === "object" &&
      (apiClient as Record<string, unknown>).vertexai === true);
  const models = c.models;
  const hasVertexShape =
    vertexaiFlag &&
    models != null &&
    typeof models === "object" &&
    typeof (models as Record<string, unknown>).generateContent === "function";

  if (hasVertexShape) {
    return "vertex";
  }

  // OpenAI: has chat.completions.create OR completions.create
  const chat = c.chat;
  const chatCompletions =
    chat && typeof chat === "object" ? chat.completions : undefined;
  const hasOpenAIShape =
    (chatCompletions && typeof chatCompletions.create === "function") ||
    (c.completions && typeof c.completions.create === "function");

  // Anthropic: has messages.create
  const messages = c.messages;
  const hasAnthropicShape =
    messages && typeof messages.create === "function";

  // Disambiguation: OpenAI also has a top-level `completions` AND `chat`.
  // Anthropic only exposes `messages`. If both shapes match (unusual),
  // prefer the more specific Anthropic match only when there's no chat.
  if (hasOpenAIShape) {
    return "openai";
  }
  if (hasAnthropicShape) {
    return "anthropic";
  }
  throw new UnsupportedClientError(
    "tensorcost: client does not look like an OpenAI, Anthropic, BedrockRuntimeClient, " +
      "or Vertex AI Client instance " +
      "(missing chat.completions.create / completions.create / messages.create / " +
      "send+config.serviceId='Bedrock Runtime' / vertexai=true+models.generateContent).",
  );
}
