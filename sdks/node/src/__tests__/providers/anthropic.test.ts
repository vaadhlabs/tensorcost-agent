import { describe, expect, it, vi } from "vitest";
import { ObservationTransport } from "../../transport.js";
import * as anthropicProvider from "../../providers/anthropic.js";
import { CircuitBreaker } from "../../circuit.js";
import {
  TensorCostBudgetHardCapError,
  TensorCostModelGovernanceDeniedError,
  TensorCostProxyError,
} from "../../errors.js";
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

describe("anthropic provider wrapper", () => {
  it("posts an observation with the right shape on success", async () => {
    const log = { observations: [] as any[] };
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: fakeFetch(log),
    });
    const client = {
      messages: {
        create: async (_args: unknown) => ({
          id: "msg_1",
          usage: { input_tokens: 33, output_tokens: 44 },
        }),
      },
    };
    anthropicProvider.install(client, transport, "tenant-y", null, null, defaultHardening());

    const resp = await client.messages.create({
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 100,
      messages: [{ role: "user", content: "hello" }],
    });
    expect(resp.id).toBe("msg_1");
    await flush();

    expect(log.observations).toHaveLength(1);
    const obs = log.observations[0];
    expect(obs.provider).toBe("anthropic");
    expect(obs.model).toBe("claude-3-5-sonnet-20241022");
    expect(obs.operation).toBe("messages");
    expect(obs.input_tokens).toBe(33);
    expect(obs.output_tokens).toBe(44);
    expect(obs.status).toBe("success");
    expect(obs.tenant_id).toBeUndefined();
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
      messages: {
        create: async (_a: unknown) => ({
          id: "msg_3",
          usage: { input_tokens: 1, output_tokens: 1 },
        }),
      },
    };
    anthropicProvider.install(
      client,
      transport,
      "tenant-y",
      null,
      "conn-anth-9",
      defaultHardening(),
    );
    await client.messages.create({ model: "claude-3-5-sonnet-20241022" });
    await flush();
    expect(log.observations).toHaveLength(1);
    expect(log.observations[0].connection_id).toBe("conn-anth-9");
  });

  it("re-throws + posts an error observation on failure", async () => {
    const log = { observations: [] as any[] };
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: fakeFetch(log),
    });
    const client = {
      messages: {
        create: async (_args: unknown) => {
          throw new Error("rate limited");
        },
      },
    };
    anthropicProvider.install(client, transport, null, null, null, defaultHardening());
    await expect(
      client.messages.create({ model: "claude-3-haiku-20240307" }),
    ).rejects.toThrow("rate limited");
    await flush();
    expect(log.observations).toHaveLength(1);
    expect(log.observations[0].status).toBe("error");
    expect(log.observations[0].error_message).toContain("rate limited");
  });

  it("fails open when TensorCost is unreachable", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: () => Promise.reject(new Error("ECONNREFUSED")),
      failOpen: true,
    });
    const client = {
      messages: {
        create: async (_args: unknown) => ({
          id: "msg_2",
          usage: { input_tokens: 5, output_tokens: 6 },
        }),
      },
    };
    anthropicProvider.install(client, transport, null, null, null, defaultHardening());
    const resp = await client.messages.create({
      model: "claude-3-haiku-20240307",
    });
    expect(resp.id).toBe("msg_2");
    await flush();
    warnSpy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// Applied-mode path through the Anthropic provider wrapper

describe("anthropic provider — applied mode via wrapMethod", () => {
  it("pins route to govern: admits, never calls the proxy, calls Anthropic directly, warns once", async () => {
    // 1.5.0: the proxy has no Anthropic Messages route, so steer/route for
    // a Claude client would only ever fail open around TensorCost. Govern
    // keeps the pre-call /admit check, which is what can stop a run.
    anthropicProvider.__test__.resetPinWarning();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const log = { observations: [] as any[] };
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: fakeFetch(log),
    });
    const hardening = defaultHardening({
      maxLayer: "route",
      appliedMode: true,
      proxyUrl: "https://proxy.local",
      baseUrl: "https://api.local",
      retry: { maxAttempts: 1, baseDelayMs: 1, maxDelayMs: 2 },
      timeoutMs: 5000,
    });
    const direct = vi.fn().mockResolvedValue({ id: "direct", usage: { input_tokens: 2, output_tokens: 3 } });
    const client = {
      baseURL: "https://api.anthropic.com",
      apiKey: "anth-test",
      messages: { create: direct },
    };
    anthropicProvider.install(client, transport, "t1", null, null, hardening);
    // A second client in the same process doesn't warn again.
    anthropicProvider.install(
      { messages: { create: vi.fn() } },
      transport,
      "t1",
      null,
      null,
      hardening,
    );
    (transport as any).getToken = async () => "test-jwt";

    const urls: string[] = [];
    const savedFetch = globalThis.fetch;
    (globalThis as any).fetch = async (input: RequestInfo | URL) => {
      urls.push(String(input));
      return new Response(null, {
        status: 204,
        headers: { "x-tc-decision": "layer=govern; action=allow" },
      });
    };
    try {
      const resp = await client.messages.create({ model: "claude-3-5-sonnet-20241022" });
      expect((resp as any).id).toBe("direct");
    } finally {
      (globalThis as any).fetch = savedFetch;
    }
    expect(urls).toEqual(["https://api.local/api/inference-proxy/v1/admit"]);
    expect(direct).toHaveBeenCalledTimes(1);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0]![0])).toMatch(/runs in 'govern'.*\/admit/);
    warn.mockRestore();
    await flush();
    expect(log.observations).toHaveLength(1);
  });

  it("a budget refusal on the pinned govern path is terminal: one admit, no Anthropic call", async () => {
    anthropicProvider.__test__.resetPinWarning();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: fakeFetch({ observations: [] }),
    });
    const direct = vi.fn().mockResolvedValue({ id: "must-not-run", usage: {} });
    const client = { messages: { create: direct } };
    anthropicProvider.install(
      client,
      transport,
      "t1",
      null,
      null,
      defaultHardening({ maxLayer: "route", proxyUrl: "https://proxy.local" }),
    );
    (transport as any).getToken = async () => "test-jwt";
    const fetchImpl = vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            error: { code: "BUDGET_HARD_CAP", budget_id: "b1", reason: "spend_not_known_yet", burn_pct: 0 },
          }),
          { status: 403, headers: { "content-type": "application/json", "retry-after": "3" } },
        ),
    );
    const savedFetch = globalThis.fetch;
    (globalThis as any).fetch = fetchImpl;
    let err: unknown;
    try {
      err = await client.messages.create({ model: "claude-3-5-sonnet-20241022" }).catch((e: unknown) => e);
    } finally {
      (globalThis as any).fetch = savedFetch;
      warn.mockRestore();
    }
    expect(err).toBeInstanceOf(TensorCostBudgetHardCapError);
    expect(err).not.toBeInstanceOf(TensorCostModelGovernanceDeniedError);
    expect(err).toMatchObject({
      code: "BUDGET_HARD_CAP",
      reason: "spend_not_known_yet",
      budgetId: "b1",
      retryable: true,
      retryAfterSeconds: 3,
    });
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(direct).not.toHaveBeenCalled();
  });

  it("falls back to direct on proxy 5xx with failOpenEnabled=true", async () => {
    const log = { observations: [] as any[] };
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: fakeFetch(log),
    });

    const directResponse = { id: "direct-anth", usage: { input_tokens: 1, output_tokens: 1 } };
    const client = {
      baseURL: "https://api.anthropic.com",
      apiKey: "anth-test",
      defaultHeaders: {
        "x-tc-provider-url": "https://api.anthropic.com",
        "x-tc-provider-auth": "x-api-key anth-test",
      } as Record<string, string>,
      messages: {
        create: vi.fn().mockResolvedValue(directResponse),
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

    anthropicProvider.install(client, transport, null, null, null, hardening);
    (transport as any).getToken = async () => "test-jwt";

    const proxyFetch: typeof fetch = async () => new Response("{}", { status: 503 });
    const savedFetch = globalThis.fetch;
    (globalThis as any).fetch = proxyFetch;
    try {
      const resp = await client.messages.create({ model: "claude-3-5-sonnet-20241022" });
      expect((resp as any).id).toBe("direct-anth");
    } finally {
      (globalThis as any).fetch = savedFetch;
    }
    await flush();
  });

  it("throws TensorCostProxyError on a TensorCost 503 with failOpenEnabled=false (govern honours it since 1.5.0)", async () => {
    const log = { observations: [] as any[] };
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: fakeFetch(log),
    });

    const client = {
      baseURL: "https://api.anthropic.com",
      apiKey: "anth-test",
      defaultHeaders: {
        "x-tc-provider-url": "https://api.anthropic.com",
        "x-tc-provider-auth": "x-api-key anth-test",
      } as Record<string, string>,
      messages: {
        create: vi.fn().mockResolvedValue({ id: "direct", usage: {} }),
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

    anthropicProvider.install(client, transport, null, null, null, hardening);
    (transport as any).getToken = async () => "test-jwt";

    const proxyFetch: typeof fetch = async () => new Response("{}", { status: 503 });
    const savedFetch = globalThis.fetch;
    (globalThis as any).fetch = proxyFetch;
    try {
      await expect(
        client.messages.create({ model: "claude-3-5-sonnet-20241022" }),
      ).rejects.toBeInstanceOf(TensorCostProxyError);
    } finally {
      (globalThis as any).fetch = savedFetch;
    }
    await flush();
  });
});
