/**
 * 1.4.1 wiring: the Anthropic govern pin through wrap(), the tenant-id
 * requirement, the Bedrock/Vertex govern warning, observation `application`
 * / `tags` (held back behind an internal flag), and the token exchange
 * typing what it gets back.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { wrap, __test__ as wrapTest } from "../wrap.js";
import { __test__ as anthropicTest } from "../providers/anthropic.js";
import { __test__ as runtimeTest } from "../runtime.js";
import { __test__ as overheadTest } from "../wrap-overhead.js";
import { ObservationTransport } from "../transport.js";
import { MissingConfigError } from "../config.js";
import {
  TensorCostKeyRejectedError,
  TensorCostNetworkError,
  TensorCostProxyError,
  TensorCostTimeoutError,
} from "../errors.js";
import { shouldFallBackToProvider } from "../proxy-client.js";

const savedFetch = globalThis.fetch;
const savedTenant = process.env.TENSORCOST_TENANT_ID;
let warn: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  anthropicTest.resetPinWarning();
  wrapTest.resetWarnings();
  runtimeTest.clearRuntimes();
  delete process.env.TENSORCOST_TENANT_ID;
  warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
});
afterEach(() => {
  globalThis.fetch = savedFetch;
  overheadTest.setObservationScope(false);
  if (savedTenant === undefined) delete process.env.TENSORCOST_TENANT_ID;
  else process.env.TENSORCOST_TENANT_ID = savedTenant;
  warn.mockRestore();
});

async function flush(times = 50) {
  for (let i = 0; i < times; i += 1) await Promise.resolve();
}

const warnings = (needle: string) =>
  warn.mock.calls.map((a) => String(a[0])).filter((m) => m.includes(needle));

/** Answers token exchange, sdk-layer and admit; records everything. */
function stubApi() {
  const calls: { url: string; body: unknown }[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : null;
    calls.push({ url, body });
    if (url.includes("sdk-token/exchange")) {
      return new Response(JSON.stringify({ token: "jwt", expires_in: 900 }), { status: 200 });
    }
    if (url.includes("sdk-layer")) {
      return new Response(JSON.stringify({ published_layer: "route" }), { status: 200 });
    }
    if (url.includes("/v1/admit")) {
      return new Response(null, {
        status: 204,
        headers: { "x-tc-decision": "layer=govern; action=allow" },
      });
    }
    if (url.includes("/observation")) return new Response(null, { status: 202 });
    return new Response("{}", { status: 404 });
  }) as typeof fetch;
  return calls;
}

const openaiFake = () => ({
  baseURL: "https://api.openai.com/v1",
  apiKey: "sk",
  chat: { completions: { create: vi.fn(async () => ({ id: "c", usage: {} })) } },
});

describe("Anthropic is pinned to govern", () => {
  it("wrap() with maxLayer 'route' admits, never touches the proxy or the client's baseURL, warns once", async () => {
    const calls = stubApi();
    const create = vi.fn(async () => ({ id: "msg", usage: { input_tokens: 1, output_tokens: 1 } }));
    const client = { baseURL: "https://api.anthropic.com", apiKey: "anth", messages: { create } };
    const opts = {
      apiKey: "tc",
      tenantId: "t-1",
      baseUrl: "https://tc.test",
      proxyUrl: "https://proxy.test",
      maxLayer: "route" as const,
    };
    wrap(client, opts);
    wrap({ baseURL: "https://api.anthropic.com", apiKey: "anth", messages: { create: vi.fn() } }, opts);

    expect(client.baseURL).toBe("https://api.anthropic.com");
    const resp = await client.messages.create({ model: "claude-sonnet-4-5" });
    expect((resp as { id: string }).id).toBe("msg");
    expect(create).toHaveBeenCalledTimes(1);
    expect(calls.some((c) => c.url.includes("/v1/admit"))).toBe(true);
    expect(calls.some((c) => c.url.includes("proxy.test"))).toBe(false);
    const pin = warnings("Anthropic");
    expect(pin).toHaveLength(1);
    expect(pin[0]).toContain("runs in 'govern'");
  });

  it("does not warn for an Anthropic client already at govern", () => {
    stubApi();
    wrap(
      { baseURL: "https://api.anthropic.com", apiKey: "a", messages: { create: vi.fn() } },
      { apiKey: "tc", tenantId: "t-1", baseUrl: "https://tc.test", maxLayer: "govern" },
    );
    expect(warnings("Anthropic")).toHaveLength(0);
  });
});

describe("the tenant id is required at govern and above", () => {
  it.each(["govern", "steer", "route"] as const)("wrap() at %s without one throws MissingConfigError", (layer) => {
    stubApi();
    let err: unknown;
    try {
      wrap(openaiFake(), { apiKey: "tc", baseUrl: "https://tc.test", proxyUrl: "https://proxy.test", maxLayer: layer });
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(MissingConfigError);
    const msg = (err as Error).message;
    expect(msg).toContain("TENSORCOST_TENANT_ID");
    expect(msg).toContain("tenantId");
    expect(msg).toContain("Developers › SDK setup");
  });

  it("failOpenEnabled doesn't change that", () => {
    expect(() =>
      wrap(openaiFake(), { apiKey: "tc", baseUrl: "https://tc.test", maxLayer: "govern", failOpenEnabled: true }),
    ).toThrow(MissingConfigError);
  });

  it("TENSORCOST_TENANT_ID satisfies it", () => {
    stubApi();
    process.env.TENSORCOST_TENANT_ID = "t-env";
    expect(() => wrap(openaiFake(), { apiKey: "tc", baseUrl: "https://tc.test", maxLayer: "govern" })).not.toThrow();
  });

  it("observe without one keeps working and warns once", async () => {
    stubApi();
    const a = openaiFake();
    wrap(a, { apiKey: "tc", baseUrl: "https://tc.test" });
    wrap(openaiFake(), { apiKey: "tc", baseUrl: "https://tc.test" });
    const resp = await a.chat.completions.create({ model: "gpt-4o-mini" });
    expect((resp as { id: string }).id).toBe("c");
    const w = warnings("no tenant id");
    expect(w).toHaveLength(1);
    expect(w[0]).toContain("no observations reach TensorCost");
  });
});

describe("Bedrock and Vertex at govern warn that nothing is enforced", () => {
  it.each([
    ["Bedrock", () => ({ send: vi.fn(), config: { serviceId: "Bedrock Runtime" } })],
    ["Vertex", () => ({ vertexai: true, models: { generateContent: vi.fn() } })],
  ] as const)("%s: one warning per process", (name, make) => {
    stubApi();
    const opts = { apiKey: "tc", tenantId: "t-1", baseUrl: "https://tc.test", maxLayer: "govern" as const };
    wrap(make(), opts);
    wrap(make(), opts);
    const w = warnings("govern isn't available");
    expect(w).toEqual([
      `tensorcost: govern isn't available for ${name} yet; calls are observed only. ` +
        "Stops, caps and budgets are not enforced on this client.",
    ]);
  });

  it("no warning at observe", () => {
    stubApi();
    wrap({ send: vi.fn(), config: { serviceId: "Bedrock Runtime" } }, { apiKey: "tc", tenantId: "t-1", baseUrl: "https://tc.test" });
    expect(warnings("govern isn't available")).toHaveLength(0);
  });
});

describe("observations: application and tags are held back in 1.4.1", () => {
  const run = async () => {
    const calls = stubApi();
    const client = openaiFake();
    wrap(client, {
      apiKey: "tc",
      tenantId: "t-1",
      baseUrl: "https://tc.test",
      application: "checkout",
      tags: ["team=pay", "tier=gold"],
    });
    await client.chat.completions.create({ model: "gpt-4o-mini" });
    await flush();
    return calls.filter((c) => c.url.includes("/observation")).map((c) => c.body as Record<string, unknown>);
  };

  it("are not sent by default (byoc-v0.1.40 rejects them with a 400)", async () => {
    const obs = await run();
    expect(obs).toHaveLength(1);
    expect(obs[0]).not.toHaveProperty("application");
    expect(obs[0]).not.toHaveProperty("tags");
  });

  it("are sent when the internal flag is on (1.5)", async () => {
    overheadTest.setObservationScope(true);
    const obs = await run();
    expect(obs[0]).toMatchObject({ application: "checkout", tags: ["team=pay", "tier=gold"] });
  });
});

describe("token exchange types what it gets back", () => {
  const mk = (fetchImpl: typeof fetch) =>
    new ObservationTransport({ tenantId: "t-1", baseUrl: "https://tc.test", apiKey: "k", fetchImpl });
  const tokenErr = (f: () => Promise<Response>) =>
    mk(f as unknown as typeof fetch)
      .getToken()
      .catch((e: unknown) => e);

  it("network failure → TensorCostNetworkError (falls back)", async () => {
    const err = await tokenErr(async () => {
      throw new TypeError("fetch failed");
    });
    expect(err).toBeInstanceOf(TensorCostNetworkError);
    expect(shouldFallBackToProvider(err)).toBe(true);
  });

  it("abort → TensorCostTimeoutError (falls back)", async () => {
    const err = await tokenErr(async () => {
      throw Object.assign(new Error("aborted"), { name: "AbortError" });
    });
    expect(err).toBeInstanceOf(TensorCostTimeoutError);
  });

  it("5xx → TensorCostProxyError (falls back)", async () => {
    const err = await tokenErr(async () => new Response("", { status: 502 }));
    expect(err).toBeInstanceOf(TensorCostProxyError);
    expect((err as Error).message).toBe("token-exchange failed status=502");
  });

  it.each([401, 403])("%s → TensorCostKeyRejectedError (does not fall back)", async (status) => {
    const err = await tokenErr(async () => new Response("", { status }));
    expect(err).toBeInstanceOf(TensorCostKeyRejectedError);
    expect((err as Error).message).toContain("the TensorCost key was rejected");
    expect((err as TensorCostKeyRejectedError).status).toBe(status);
    expect(shouldFallBackToProvider(err)).toBe(false);
  });
});
