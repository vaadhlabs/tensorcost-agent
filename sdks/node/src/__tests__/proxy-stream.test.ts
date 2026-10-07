import { describe, expect, it } from "vitest";
import { proxyRequestStream, sseChunksFromStream } from "../proxy-client.js";
import { CircuitBreaker } from "../circuit.js";

describe("proxyRequestStream", () => {
  it("returns SSE body on 200", async () => {
    const sseBody = "data: {\"id\":\"1\"}\n\ndata: [DONE]\n\n";
    const fetchImpl = async () =>
      new Response(sseBody, {
        status: 200,
        headers: {
          "content-type": "text/event-stream",
          "x-tc-decision": "routed",
        },
      });

    const result = await proxyRequestStream({
      proxyUrl: "https://proxy.local",
      bearerToken: "jwt",
      providerUrl: "https://api.openai.com/v1",
      providerAuth: "Bearer sk-test",
      body: { model: "gpt-4o-mini", messages: [], stream: true },
      model: "gpt-4o-mini",
      provider: "openai",
      operation: "chat.completions",
      correlationId: "corr-1",
      environment: null,
      circuit: new CircuitBreaker(),
      fetchImpl,
    });

    expect(result.decisionHeader).toBe("routed");
    const chunks: unknown[] = [];
    for await (const chunk of sseChunksFromStream(result.stream)) {
      chunks.push(chunk);
    }
    expect(chunks).toEqual([{ id: "1" }]);
  });
});
