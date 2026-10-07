import { describe, expect, it, vi } from "vitest";
import { ObservationTransport } from "../transport.js";
import {
  WrapOverhead,
  postObservation,
  withAwaitExcluded,
} from "../wrap-overhead.js";
import type { Observation } from "../types.js";

function makeObs(): Observation {
  return {
    sdk_version: "tensorcost-node/0.1.0",
    provider: "bedrock",
    model: "anthropic.claude-3-haiku",
    operation: "bedrock.converse",
    request_at: new Date().toISOString(),
    response_at: new Date().toISOString(),
    input_tokens: 1,
    output_tokens: 2,
    cost_usd_cents: null,
    status: "success",
    error_message: null,
    correlation_id: "corr-1",
  };
}

describe("WrapOverhead", () => {
  it("excludes awaited I/O from sdk_added_ms", async () => {
    const overhead = new WrapOverhead();
    await new Promise((r) => setTimeout(r, 5));
    await withAwaitExcluded(overhead, () => new Promise((r) => setTimeout(r, 30)));
    await new Promise((r) => setTimeout(r, 2));
    const sdkMs = overhead.sdkAddedMs();
    expect(sdkMs).toBeGreaterThanOrEqual(5);
    expect(sdkMs).toBeLessThan(15);
  });

  it("includes token-extraction sync work before postObservation", async () => {
    const overhead = new WrapOverhead();
    await withAwaitExcluded(overhead, async () => {
      await new Promise((r) => setTimeout(r, 20));
    });
    await new Promise((r) => setTimeout(r, 3));
    const beforePost = overhead.sdkAddedMs();
    postObservation({ post: vi.fn() } as unknown as ObservationTransport, overhead, makeObs());
    expect(beforePost).toBeGreaterThanOrEqual(2);
    expect(beforePost).toBeLessThan(10);
  });

  it("postObservation returns immediately without awaiting transport delivery", async () => {
    let resolveSlow: ((r: Response) => void) | null = null;
    const fetchImpl: typeof fetch = () =>
      new Promise((resolve) => {
        resolveSlow = resolve;
      });
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: async (input, init = {}) => {
        const url = typeof input === "string" ? input : (input as URL).toString();
        if (url.includes("sdk-token/exchange")) {
          return new Response(JSON.stringify({ token: "tok", expires_in: 900 }), {
            status: 200,
            headers: { "content-type": "application/json" },
          });
        }
        return fetchImpl(input, init);
      },
    });

    const overhead = new WrapOverhead();
    const t0 = performance.now();
    postObservation(transport, overhead, makeObs());
    expect(performance.now() - t0).toBeLessThan(5);
    expect(resolveSlow).toBeNull();
  });

  it("postObservation is fail-open when transport.post throws", () => {
    const transport = {
      post() {
        throw new Error("boom");
      },
    } as unknown as ObservationTransport;
    const overhead = new WrapOverhead();
    expect(() => postObservation(transport, overhead, makeObs())).not.toThrow();
  });
});
