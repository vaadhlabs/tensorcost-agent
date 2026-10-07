import { describe, expect, it, vi } from "vitest";
import { ObservationTransport } from "../../transport.js";
import * as openaiProvider from "../../providers/openai.js";
import { CircuitBreaker } from "../../circuit.js";
import { TensorCostProxyError } from "../../errors.js";
import { defaultHardening } from "../test-harness.js";

function fakeFetch(
  log: { observations: any[] },
): typeof fetch {
  return async (input, init = {}) => {
    const url = typeof input === "string" ? input : (input as URL).toString();
    if (url.includes("sdk-token/exchange")) {
      return new Response(
        JSON.stringify({ token: "tok", expires_in: 900 }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }
    if (url.includes("/api/inference-proxy/observation")) {
      log.observations.push(JSON.parse(init.body as string));
      return new Response(null, { status: 202 });
    }
    return new Response(null, { status: 404 });
  };
}

async function flush(times = 30) {
  for (let i = 0; i < times; i += 1) await Promise.resolve();
}

describe("openai provider wrapper", () => {
  it("posts an observation with the right shape on success", async () => {
    const log = { observations: [] as any[] };
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: fakeFetch(log),
    });
    const client = {
      chat: {
        completions: {
          create: async (_args: unknown) => ({
            id: "chatcmpl-1",
            usage: { prompt_tokens: 11, completion_tokens: 22 },
          }),
        },
      },
    };
    openaiProvider.install(client, transport, "tenant-x", null, null, defaultHardening());

    const resp = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(resp.id).toBe("chatcmpl-1");
    await flush();

    expect(log.observations).toHaveLength(1);
    const obs = log.observations[0];
    expect(obs.provider).toBe("openai");
    expect(obs.model).toBe("gpt-4o-mini");
    expect(obs.operation).toBe("chat.completions");
    expect(obs.input_tokens).toBe(11);
    expect(obs.output_tokens).toBe(22);
    expect(obs.status).toBe("success");
    expect(obs.error_message).toBeNull();
    expect(obs.tenant_id).toBeUndefined();
    expect(typeof obs.correlation_id).toBe("string");
    expect(obs.sdk_version).toBe("tensorcost-node/1.3.0");
    // Phase A4 batch 4 — connection_id absent when wrap() got no
    // connectionId. Backend then resolves env via the SDK envelope's
    // `environment` field or falls back to 'production'.
    expect(obs.connection_id).toBeUndefined();
  });

  it("stamps connection_id on the envelope when configured", async () => {
    const log = { observations: [] as any[] };
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: fakeFetch(log),
    });
    const client = {
      chat: {
        completions: {
          create: async (_a: unknown) => ({
            id: "chatcmpl-2",
            usage: { prompt_tokens: 1, completion_tokens: 2 },
          }),
        },
      },
    };
    openaiProvider.install(
      client,
      transport,
      "tenant-x",
      null,
      "conn-abc-123",
      defaultHardening(),
    );
    await client.chat.completions.create({ model: "gpt-4o-mini" });
    await flush();
    expect(log.observations).toHaveLength(1);
    expect(log.observations[0].connection_id).toBe("conn-abc-123");
  });

  it("still posts an observation when the underlying call throws (error path) and re-throws", async () => {
    const log = { observations: [] as any[] };
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: fakeFetch(log),
    });
    const client = {
      chat: {
        completions: {
          create: async (_args: unknown) => {
            throw new Error("upstream 500");
          },
        },
      },
    };
    openaiProvider.install(client, transport, null, null, null, defaultHardening());

    await expect(
      client.chat.completions.create({ model: "gpt-4o" }),
    ).rejects.toThrow("upstream 500");
    await flush();
    expect(log.observations).toHaveLength(1);
    expect(log.observations[0].status).toBe("error");
    expect(log.observations[0].error_message).toContain("upstream 500");
  });

  it("does NOT throw when TensorCost is unreachable (fail-open)", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: () => Promise.reject(new Error("ECONNREFUSED")),
      failOpen: true,
    });
    const client = {
      chat: {
        completions: {
          create: async (_args: unknown) => ({
            id: "ok",
            usage: { prompt_tokens: 1, completion_tokens: 1 },
          }),
        },
      },
    };
    openaiProvider.install(client, transport, null, null, null, defaultHardening());
    const resp = await client.chat.completions.create({ model: "gpt-4o" });
    expect(resp.id).toBe("ok");
    await flush();
    warnSpy.mockRestore();
  });

  it("wraps legacy completions.create as well", async () => {
    const log = { observations: [] as any[] };
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: fakeFetch(log),
    });
    const client = {
      completions: {
        create: async (_a: unknown) => ({
          usage: { prompt_tokens: 3, completion_tokens: 4 },
        }),
      },
    };
    openaiProvider.install(client, transport, null, null, null, defaultHardening());
    await client.completions.create({ model: "text-davinci-003" });
    await flush();
    expect(log.observations[0].operation).toBe("completions");
  });
});

// ---------------------------------------------------------------------------
// Applied-mode path through the provider wrapper

describe("openai provider — applied mode via wrapMethod", () => {
  it("routes through proxy and posts observation on proxy success", async () => {
    const log = { observations: [] as any[] };
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: fakeFetch(log),
    });

    // Proxy fetch: returns 200 with a response body.
    const proxyFetch: typeof fetch = async (input, _init) => {
      const url = typeof input === "string" ? input : (input as URL).toString();
      if (url.includes("inference-proxy/v1")) {
        return new Response(
          JSON.stringify({ id: "proxy-resp", usage: { prompt_tokens: 7, completion_tokens: 8 } }),
          { status: 200, headers: { "content-type": "application/json" } },
        );
      }
      return new Response(null, { status: 404 });
    };

    const circuit = new CircuitBreaker();
    const hardening = defaultHardening({
      maxLayer: "route",
      appliedMode: true,
      proxyUrl: "https://proxy.local",
      retry: { maxAttempts: 1, baseDelayMs: 1, maxDelayMs: 2 },
      timeoutMs: 5000,
      circuit,
    });

    const client = {
      baseURL: "https://api.openai.com/v1",
      apiKey: "sk-test",
      defaultHeaders: {
        "x-tc-provider-url": "https://api.openai.com/v1",
        "x-tc-provider-auth": "Bearer sk-test",
      } as Record<string, string>,
      chat: {
        completions: {
          create: vi.fn().mockResolvedValue({ id: "direct", usage: {} }),
        },
      },
    };

    openaiProvider.install(client, transport, "tenant-a", null, null, hardening);

    // Inject a custom getToken via the internal path: we patch the transport's
    // private getToken so the provider wrapper can get a token.
    // The provider wrapper reads `(opts.transport as any).getToken()`.
    // We patch it on the transport instance.
    (transport as any).getToken = async () => "test-jwt";
    // Also inject the proxy fetch.
    (transport as any).fetchImpl = proxyFetch;

    // The provider wrapper uses: getToken ?? (() => (opts.transport as any).getToken())
    // So we just need to make transport.getToken() work.
    // But proxyRequest uses its own opts.fetchImpl, which falls back to globalThis.fetch.
    // We need to inject the fetch into the proxyRequest call too.
    // The cleanest way without modifying signatures: we install a global fetch mock.
    const savedFetch = globalThis.fetch;
    (globalThis as any).fetch = proxyFetch;
    let resp: unknown;
    try {
      resp = await client.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [],
      });
    } finally {
      (globalThis as any).fetch = savedFetch;
    }
    // Proxy response should be returned.
    expect((resp as any).id).toBe("proxy-resp");
  });

  it("falls back to direct provider on proxy error with failOpenEnabled=true", async () => {
    const log = { observations: [] as any[] };
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: fakeFetch(log),
    });

    const directResponse = { id: "direct-fallback", usage: { prompt_tokens: 1, completion_tokens: 1 } };
    const client = {
      baseURL: "https://api.openai.com/v1",
      apiKey: "sk-test",
      defaultHeaders: {
        "x-tc-provider-url": "https://api.openai.com/v1",
        "x-tc-provider-auth": "Bearer sk-test",
      } as Record<string, string>,
      chat: {
        completions: {
          create: vi.fn().mockResolvedValue(directResponse),
        },
      },
    };

    const circuit = new CircuitBreaker();
    const hardening = defaultHardening({
      maxLayer: "route",
      appliedMode: true,
      proxyUrl: "https://proxy.local",
      retry: { maxAttempts: 1, baseDelayMs: 1, maxDelayMs: 2 },
      timeoutMs: 5000,
      circuit,
    });

    openaiProvider.install(client, transport, null, null, null, hardening);
    (transport as any).getToken = async () => "test-jwt";

    // Proxy returns 503.
    const proxyFetch: typeof fetch = async () =>
      new Response("{}", { status: 503 });
    const savedFetch = globalThis.fetch;
    (globalThis as any).fetch = proxyFetch;
    try {
      const resp = await client.chat.completions.create({ model: "gpt-4o-mini" });
      expect((resp as any).id).toBe("direct-fallback");
    } finally {
      (globalThis as any).fetch = savedFetch;
    }
    await flush();
  });

  it("throws TensorCostProxyError on proxy error with failOpenEnabled=false", async () => {
    const log = { observations: [] as any[] };
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: fakeFetch(log),
    });

    const client = {
      baseURL: "https://api.openai.com/v1",
      apiKey: "sk-test",
      defaultHeaders: {
        "x-tc-provider-url": "https://api.openai.com/v1",
        "x-tc-provider-auth": "Bearer sk-test",
      } as Record<string, string>,
      chat: {
        completions: {
          create: vi.fn().mockResolvedValue({ id: "direct", usage: {} }),
        },
      },
    };

    const circuit = new CircuitBreaker();
    const hardening = defaultHardening({
      maxLayer: "route",
      appliedMode: true,
      proxyUrl: "https://proxy.local",
      retry: { maxAttempts: 1, baseDelayMs: 1, maxDelayMs: 2 },
      timeoutMs: 5000,
      failOpenEnabled: false,
      circuit,
    });

    openaiProvider.install(client, transport, null, null, null, hardening);
    (transport as any).getToken = async () => "test-jwt";

    const proxyFetch: typeof fetch = async () =>
      new Response("{}", { status: 503 });
    const savedFetch = globalThis.fetch;
    (globalThis as any).fetch = proxyFetch;
    try {
      await expect(
        client.chat.completions.create({ model: "gpt-4o-mini" }),
      ).rejects.toBeInstanceOf(TensorCostProxyError);
    } finally {
      (globalThis as any).fetch = savedFetch;
    }
    await flush();
  });

  it("probes the proxy when circuit is open, then falls back on failure", async () => {
    const log = { observations: [] as any[] };
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: fakeFetch(log),
    });

    const directResponse = { id: "bypass-direct", usage: { prompt_tokens: 1, completion_tokens: 1 } };
    const client = {
      baseURL: "https://api.openai.com/v1",
      apiKey: "sk-test",
      defaultHeaders: {
        "x-tc-provider-url": "https://api.openai.com/v1",
        "x-tc-provider-auth": "Bearer sk-test",
      } as Record<string, string>,
      chat: {
        completions: {
          create: vi.fn().mockResolvedValue(directResponse),
        },
      },
    };

    const circuit = new CircuitBreaker({ openThreshold: 1, closeThreshold: 5 });
    circuit.recordFailure(); // Trip the circuit.
    expect(circuit.currentState).toBe("OPEN");

    const hardening = defaultHardening({
      maxLayer: "route",
      appliedMode: true,
      proxyUrl: "https://proxy.local",
      retry: { maxAttempts: 1, baseDelayMs: 1, maxDelayMs: 2 },
      timeoutMs: 5000,
      circuit,
    });

    openaiProvider.install(client, transport, null, null, null, hardening);
    (transport as any).getToken = async () => "test-jwt";

    let proxyFetchCalled = false;
    const savedFetch = globalThis.fetch;
    (globalThis as any).fetch = async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("proxy.local") || url.includes("inference-proxy/v1")) {
        proxyFetchCalled = true;
        throw new Error("ECONNREFUSED");
      }
      return new Response("{}", { status: 404 });
    };
    try {
      const resp = await client.chat.completions.create({ model: "gpt-4o-mini" });
      expect((resp as any).id).toBe("bypass-direct");
      expect(circuit.currentState).toBe("OPEN"); // probe failure re-opens
      expect(proxyFetchCalled).toBe(true);
    } finally {
      (globalThis as any).fetch = savedFetch;
    }
    await flush();
  });

  it("does not double-call provider when HALF_OPEN probe succeeds", async () => {
    const log = { observations: [] as any[] };
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: fakeFetch(log),
    });

    let directCalls = 0;
    const directCreate = async () => {
      directCalls += 1;
      return { id: "must-not-run", usage: {} };
    };
    const client = {
      baseURL: "https://api.openai.com/v1",
      apiKey: "sk-test",
      defaultHeaders: {
        "x-tc-provider-url": "https://api.openai.com/v1",
        "x-tc-provider-auth": "Bearer sk-test",
      } as Record<string, string>,
      chat: {
        completions: {
          create: directCreate,
        },
      },
    };

    const circuit = new CircuitBreaker({ openThreshold: 1, closeThreshold: 5 });
    circuit.recordFailure();

    const hardening = defaultHardening({
      maxLayer: "route",
      appliedMode: true,
      proxyUrl: "https://proxy.local",
      retry: { maxAttempts: 1, baseDelayMs: 1, maxDelayMs: 2 },
      timeoutMs: 5000,
      circuit,
    });

    openaiProvider.install(client, transport, null, null, null, hardening);
    (transport as any).getToken = async () => "test-jwt";

    const savedFetch = globalThis.fetch;
    (globalThis as any).fetch = async () =>
      new Response(JSON.stringify({ id: "via-proxy", usage: { prompt_tokens: 1, completion_tokens: 1 } }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    try {
      const resp = await client.chat.completions.create({ model: "gpt-4o-mini" });
      expect((resp as any).id).toBe("via-proxy");
      expect(directCalls).toBe(0);
    } finally {
      (globalThis as any).fetch = savedFetch;
    }
    await flush();
  });
});
