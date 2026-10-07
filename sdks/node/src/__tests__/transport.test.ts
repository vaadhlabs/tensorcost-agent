import { describe, expect, it, vi } from "vitest";
import { ObservationTransport } from "../transport.js";
import type { Observation } from "../types.js";

function makeObservation(): Observation {
  return {
    sdk_version: "tensorcost-node/0.1.0",
    provider: "openai",
    model: "gpt-4o-mini",
    operation: "chat.completions",
    request_at: new Date().toISOString(),
    response_at: new Date().toISOString(),
    input_tokens: 10,
    output_tokens: 20,
    cost_usd_cents: null,
    status: "success",
    error_message: null,
    correlation_id: "00000000-0000-0000-0000-000000000000",
  };
}

function makeFakeFetch(
  handlers: Record<string, (init: RequestInit) => Response | Promise<Response>>,
) {
  const calls: { url: string; init: RequestInit }[] = [];
  const fetchImpl: typeof fetch = async (input, init = {}) => {
    const url = typeof input === "string" ? input : (input as URL).toString();
    calls.push({ url, init });
    for (const [pattern, handler] of Object.entries(handlers)) {
      if (url.includes(pattern)) return handler(init);
    }
    return new Response("not found", { status: 404 });
  };
  return { fetchImpl, calls };
}

async function flushMicrotasks(times = 5) {
  for (let i = 0; i < times; i += 1) {
    await Promise.resolve();
  }
}

describe("ObservationTransport", () => {
  it("posts observations to /api/inference-proxy/observation with bearer token", async () => {
    const { fetchImpl, calls } = makeFakeFetch({
      "/api/inference-proxy/sdk-token/exchange": () =>
        new Response(
          JSON.stringify({ token: "tok-123", expires_in: 900 }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      "/api/inference-proxy/observation": () => new Response(null, { status: 202 }),
    });
    const t = new ObservationTransport({
      baseUrl: "http://api.local",
      apiKey: "long-lived-key",
      tenantId: "e82770de-06d6-415c-911f-c0d65cf45190",
      fetchImpl,
    });
    t.post(makeObservation());
    await flushMicrotasks(20);

    const exchange = calls.find((c) => c.url.includes("sdk-token/exchange"));
    const post = calls.find((c) => c.url.includes("/observation"));
    expect(exchange).toBeDefined();
    expect(exchange!.url).toBe("http://api.local/api/inference-proxy/sdk-token/exchange");
    expect(JSON.parse(exchange!.init.body as string)).toEqual({
      tenant_id: "e82770de-06d6-415c-911f-c0d65cf45190",
      sdk_long_lived_token: "long-lived-key",
    });
    expect(post).toBeDefined();
    const headers = post!.init.headers as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer tok-123");
    expect(headers["Content-Type"]).toBe("application/json");
    const body = JSON.parse(post!.init.body as string);
    expect(body.provider).toBe("openai");
    expect(body.model).toBe("gpt-4o-mini");
  });

  it("returns synchronously even when delivery is slow (fire-and-forget)", async () => {
    let resolveSlow: ((r: Response) => void) | null = null;
    const slow = new Promise<Response>((res) => {
      resolveSlow = res;
    });
    const fetchImpl: typeof fetch = (input) => {
      const url = typeof input === "string" ? input : (input as URL).toString();
      if (url.includes("sdk-token/exchange")) {
        return Promise.resolve(
          new Response(JSON.stringify({ token: "t", expires_in: 900 }), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        );
      }
      return slow;
    };
    const t = new ObservationTransport({
      baseUrl: "http://api.local",
      apiKey: "k",
      tenantId: "t-1",
      fetchImpl,
    });
    const t0 = Date.now();
    t.post(makeObservation());
    expect(Date.now() - t0).toBeLessThan(50);
    resolveSlow!(new Response(null, { status: 202 }));
    await flushMicrotasks(20);
  });

  it("caches the token across multiple posts (single exchange call)", async () => {
    let exchanges = 0;
    const fetchImpl: typeof fetch = async (input) => {
      const url = typeof input === "string" ? input : (input as URL).toString();
      if (url.includes("sdk-token/exchange")) {
        exchanges += 1;
        return new Response(
          JSON.stringify({ token: "tok", expires_in: 900 }),
          { status: 200, headers: { "content-type": "application/json" } },
        );
      }
      return new Response(null, { status: 202 });
    };
    const t = new ObservationTransport({
      baseUrl: "http://api.local",
      apiKey: "k",
      tenantId: "t-1",
      fetchImpl,
    });
    t.post(makeObservation());
    t.post(makeObservation());
    t.post(makeObservation());
    await flushMicrotasks(50);
    expect(exchanges).toBe(1);
  });

  it("swallows network errors when failOpen=true", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const fetchImpl: typeof fetch = () => Promise.reject(new Error("ECONNREFUSED"));
    const t = new ObservationTransport({
      baseUrl: "http://api.local",
      apiKey: "k",
      tenantId: "t-1",
      fetchImpl,
      failOpen: true,
    });
    expect(() => t.post(makeObservation())).not.toThrow();
    await flushMicrotasks(20);
    expect(warnSpy).toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it("drops observations when the queue is saturated", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    // never-resolving fetch keeps everything pending
    const fetchImpl: typeof fetch = () => new Promise(() => {});
    const t = new ObservationTransport({
      baseUrl: "http://api.local",
      apiKey: "k",
      tenantId: "t-1",
      fetchImpl,
    });
    // Queue depth is 1024; push 1100.
    for (let i = 0; i < 1100; i += 1) t.post(makeObservation());
    await flushMicrotasks(5);
    expect(warnSpy).toHaveBeenCalled();
    warnSpy.mockRestore();
  });
});
