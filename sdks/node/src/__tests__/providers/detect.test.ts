import { describe, expect, it } from "vitest";
import { UnsupportedClientError, detectProvider } from "../../providers/detect.js";

describe("detectProvider", () => {
  it("identifies an OpenAI-shaped client via chat.completions.create", () => {
    const c = { chat: { completions: { create: () => {} } } };
    expect(detectProvider(c)).toBe("openai");
  });

  it("identifies an OpenAI-shaped client via legacy completions.create", () => {
    const c = { completions: { create: () => {} } };
    expect(detectProvider(c)).toBe("openai");
  });

  it("identifies an Anthropic-shaped client via messages.create", () => {
    const c = { messages: { create: () => {} } };
    expect(detectProvider(c)).toBe("anthropic");
  });

  // ── Bedrock detection ─────────────────────────────────────────────────────

  it("identifies a BedrockRuntimeClient via send + config.serviceId", () => {
    const c = {
      send: async () => {},
      config: { serviceId: "Bedrock Runtime" },
      middlewareStack: { add: () => {}, identify: () => [] },
    };
    expect(detectProvider(c)).toBe("bedrock");
  });

  it("identifies a Vertex Client via vertexai + models.generateContent", () => {
    const c = {
      vertexai: true,
      models: { generateContent: () => {} },
    };
    expect(detectProvider(c)).toBe("vertex");
  });

  it("does NOT treat Gemini Developer API client as Vertex", () => {
    const c = {
      vertexai: false,
      models: { generateContent: () => {} },
    };
    expect(() => detectProvider(c)).toThrow(UnsupportedClientError);
  });

  it("does NOT treat a generic Smithy client as Bedrock when serviceId differs", () => {
    // DynamoDB or S3 have a different serviceId — they should fall through to
    // the UnsupportedClientError path (they're not AI providers).
    const dynamoLike = {
      send: async () => {},
      config: { serviceId: "DynamoDB" },
    };
    expect(() => detectProvider(dynamoLike)).toThrow(UnsupportedClientError);
  });

  it("does NOT treat an object with only send() and no config as Bedrock", () => {
    const notBedrock = { send: async () => {} };
    expect(() => detectProvider(notBedrock)).toThrow(UnsupportedClientError);
  });

  it("still detects OpenAI correctly after Bedrock check is added (no regression)", () => {
    const c = { chat: { completions: { create: () => {} } } };
    expect(detectProvider(c)).toBe("openai");
  });

  // ── Edge cases ────────────────────────────────────────────────────────────

  it("throws on null / non-object inputs", () => {
    expect(() => detectProvider(null)).toThrow(UnsupportedClientError);
    expect(() => detectProvider(42 as unknown)).toThrow(UnsupportedClientError);
  });

  it("throws on objects with neither shape", () => {
    expect(() => detectProvider({ foo: "bar" })).toThrow(UnsupportedClientError);
  });
});
