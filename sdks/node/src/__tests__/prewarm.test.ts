import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { prewarm } from "../prewarm.js";
import { getSharedRuntime, __test__ as runtimeTest } from "../runtime.js";

describe("prewarm", () => {
  beforeEach(() => {
    runtimeTest.clearRuntimes();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shares JWT cache with getSharedRuntime after prewarm", async () => {
    let exchangeCalls = 0;
    const fetchImpl = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/sdk-token/exchange")) {
        exchangeCalls += 1;
        return new Response(JSON.stringify({ token: "jwt-1", expires_in: 900 }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.includes("/sdk-layer")) {
        return new Response(JSON.stringify({ published_layer: "observe" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      return new Response("{}", { status: 202 });
    });
    vi.stubGlobal("fetch", fetchImpl);

    await prewarm({
      apiKey: "tc_test_key",
      tenantId: "tenant-1",
      baseUrl: "https://api.tensorcost.com",
    });

    expect(exchangeCalls).toBe(1);

    const { transport } = getSharedRuntime({
      apiKey: "tc_test_key",
      tenantId: "tenant-1",
      baseUrl: "https://api.tensorcost.com",
    });
    await transport.getToken();
    expect(exchangeCalls).toBe(1);
  });
});
