/**
 * Tests for the Vertex observe-only provider wrapper.
 */

import { describe, expect, it, vi, beforeEach } from "vitest";
import { ObservationTransport } from "../transport.js";
import * as vertexProvider from "../providers/vertex.js";
import { TensorCostConfigError } from "../config.js";
import { wrap } from "../wrap.js";

function fakeTransport(): { transport: ObservationTransport; observations: any[] } {
  const observations: any[] = [];
  const fetchImpl: typeof fetch = async (input, init = {}) => {
    const url = typeof input === "string" ? input : (input as URL).toString();
    if (url.includes("sdk-token/exchange")) {
      return new Response(JSON.stringify({ token: "tok", expires_in: 900 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    if (url.includes("/api/inference-proxy/observation")) {
      observations.push(JSON.parse(init?.body as string));
      return new Response(null, { status: 202 });
    }
    return new Response(null, { status: 404 });
  };
  const transport = new ObservationTransport({
    baseUrl: "http://api.local",
    apiKey: "k",
    tenantId: "t-1",
    fetchImpl,
  });
  return { transport, observations };
}

async function flush(times = 30) {
  for (let i = 0; i < times; i++) await Promise.resolve();
}

function makeFakeVertexClient() {
  const usageMetadata = { promptTokenCount: 42, candidatesTokenCount: 17 };
  const generateContent = vi.fn(async () => ({ text: "4", usageMetadata }));
  const generateContentStream = vi.fn(async () => (async function* () {})());

  const client = {
    vertexai: true,
    models: {
      generateContent,
      generateContentStream,
    },
  };

  return { client, generateContent, generateContentStream, usageMetadata };
}

describe("vertex provider wrapper", () => {
  it("records generate_content observation with token counts", async () => {
    const { transport, observations } = fakeTransport();
    const { client } = makeFakeVertexClient();

    vertexProvider.install(client, transport, null, null, null);

    await client.models.generateContent({
      model: "gemini-2.0-flash",
      contents: "hello",
    });
    await flush();

    expect(observations).toHaveLength(1);
    expect(observations[0].provider).toBe("vertex");
    expect(observations[0].operation).toBe("vertex.generate_content");
    expect(observations[0].model).toBe("gemini-2.0-flash");
    expect(observations[0].input_tokens).toBe(42);
    expect(observations[0].output_tokens).toBe(17);
    expect(observations[0].status).toBe("success");
  });

  it("records streaming placeholder observation", async () => {
    const { transport, observations } = fakeTransport();
    const { client } = makeFakeVertexClient();

    vertexProvider.install(client, transport, null, null, "conn-vtx");

    await client.models.generateContentStream({
      model: "gemini-2.0-flash",
      contents: "stream",
    });
    await flush();

    expect(observations).toHaveLength(1);
    expect(observations[0].operation).toBe("vertex.generate_content_stream");
    expect(observations[0].input_tokens).toBeNull();
    expect(observations[0].output_tokens).toBeNull();
    expect(observations[0].connection_id).toBe("conn-vtx");
    expect(observations[0].error_message).toContain("streaming");
  });

  it("records error and rethrows", async () => {
    const { transport, observations } = fakeTransport();
    const { client, generateContent } = makeFakeVertexClient();
    generateContent.mockRejectedValueOnce(new Error("quota exceeded"));

    vertexProvider.install(client, transport, null, null, null);

    await expect(
      client.models.generateContent({ model: "gemini-2.0-flash", contents: "x" }),
    ).rejects.toThrow("quota exceeded");
    await flush();

    expect(observations[0].status).toBe("error");
    expect(observations[0].error_message).toContain("quota exceeded");
  });

  it("is idempotent on double install", async () => {
    const { transport } = fakeTransport();
    const { client } = makeFakeVertexClient();

    vertexProvider.install(client, transport, null, null, null);
    const afterFirst = client.models.generateContent;
    vertexProvider.install(client, transport, null, null, null);
    expect(client.models.generateContent).toBe(afterFirst);
  });
});

describe("wrap() + Vertex applied mode guard", () => {
  it("throws TensorCostConfigError when maxLayer route", () => {
    const { client } = makeFakeVertexClient();
    expect(() =>
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
        maxLayer: "route",
        proxyUrl: "http://proxy.local",
      }),
    ).toThrow(TensorCostConfigError);
  });

  it("wrap() succeeds without appliedMode", () => {
    const { client } = makeFakeVertexClient();
    expect(() =>
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
      }),
    ).not.toThrow();
    expect((client.models.generateContent as any).__tensorcost_wrapped__).toBe(true);
  });
});
