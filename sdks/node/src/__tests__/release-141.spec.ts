/**
 * 1.4.1 wiring: the Anthropic govern pin through wrap(), observation
 * `application` / `tags`, and the token exchange typing its outages.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { wrap } from "../wrap.js";
import { __test__ as anthropicTest } from "../providers/anthropic.js";
import { __test__ as runtimeTest } from "../runtime.js";
import { ObservationTransport, __test__ as transportTest } from "../transport.js";
import {
  TensorCostNetworkError,
  TensorCostProxyError,
  TensorCostTimeoutError,
} from "../errors.js";
import { shouldFallBackToProvider } from "../proxy-client.js";

const savedFetch = globalThis.fetch;
let warn: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  anthropicTest.resetPinWarning();
  transportTest.resetNoTenantWarning();
  runtimeTest.clearRuntimes();
  warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
});
afterEach(() => {
  globalThis.fetch = savedFetch;
  warn.mockRestore();
});

async function flush(times = 50) {
  for (let i = 0; i < times; i += 1) await Promise.resolve();
}

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

describe("Anthropic is pinned to govern", () => {
  it("wrap() with maxLayer 'route' admits, never touches the proxy or the client's baseURL, warns once", async () => {
    const calls = stubApi();
    const create = vi.fn(async () => ({ id: "msg", usage: { input_tokens: 1, output_tokens: 1 } }));
    const client = {
      baseURL: "https://api.anthropic.com",
      apiKey: "anth",
      messages: { create },
    };
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

    const pinWarnings = warn.mock.calls.filter((a) => String(a[0]).includes("Anthropic"));
    expect(pinWarnings).toHaveLength(1);
    expect(String(pinWarnings[0]![0])).toContain("runs in 'govern'");
  });

  it("does not warn for an Anthropic client already at govern", () => {
    stubApi();
    wrap(
      { baseURL: "https://api.anthropic.com", apiKey: "a", messages: { create: vi.fn() } },
      { apiKey: "tc", tenantId: "t-1", baseUrl: "https://tc.test", maxLayer: "govern" },
    );
    expect(warn.mock.calls.filter((a) => String(a[0]).includes("Anthropic"))).toHaveLength(0);
  });
});

describe("observations carry application and tags", () => {
  it("are stamped from wrap({ application, tags })", async () => {
    const calls = stubApi();
    const client = {
      baseURL: "https://api.openai.com/v1",
      apiKey: "sk",
      chat: { completions: { create: vi.fn(async () => ({ id: "c", usage: {} })) } },
    };
    wrap(client, {
      apiKey: "tc",
      tenantId: "t-1",
      baseUrl: "https://tc.test",
      application: "checkout",
      tags: ["team=pay", "tier=gold"],
    });
    await client.chat.completions.create({ model: "gpt-4o-mini" });
    await flush();
    const obs = calls.filter((c) => c.url.includes("/observation")).map((c) => c.body);
    expect(obs).toHaveLength(1);
    expect(obs[0]).toMatchObject({ application: "checkout", tags: ["team=pay", "tier=gold"] });
  });

  it("are omitted when not configured", async () => {
    const calls = stubApi();
    const client = {
      baseURL: "https://api.openai.com/v1",
      apiKey: "sk",
      chat: { completions: { create: vi.fn(async () => ({ id: "c", usage: {} })) } },
    };
    wrap(client, { apiKey: "tc", tenantId: "t-1", baseUrl: "https://tc.test" });
    await client.chat.completions.create({ model: "gpt-4o-mini" });
    await flush();
    const obs = calls.filter((c) => c.url.includes("/observation")).map((c) => c.body);
    expect(obs[0]).not.toHaveProperty("application");
    expect(obs[0]).not.toHaveProperty("tags");
  });
});

describe("no tenant id at govern/route", () => {
  it("warns once and keeps 1.4.0's outcome (straight to the provider)", async () => {
    const calls = stubApi();
    const create = vi.fn(async () => ({ id: "c", usage: {} }));
    const client = { baseURL: "https://api.openai.com/v1", apiKey: "sk", chat: { completions: { create } } };
    wrap(client, { apiKey: "tc", baseUrl: "https://tc.test", maxLayer: "govern" });
    await client.chat.completions.create({ model: "gpt-4o-mini" });
    await client.chat.completions.create({ model: "gpt-4o-mini" });
    expect(create).toHaveBeenCalledTimes(2);
    expect(calls.some((c) => c.url.includes("/v1/admit"))).toBe(false);
    const w = warn.mock.calls.filter((a) => String(a[0]).includes("tenant id"));
    expect(w).toHaveLength(1);
    expect(String(w[0]![0])).toContain("NOT enforced");
  });
});

describe("token exchange types its outages", () => {
  const mk = (fetchImpl: typeof fetch) =>
    new ObservationTransport({ tenantId: "t-1", baseUrl: "https://tc.test", apiKey: "k", fetchImpl });

  it("network failure → TensorCostNetworkError (falls back)", async () => {
    const err = await mk((async () => {
      throw new TypeError("fetch failed");
    }) as unknown as typeof fetch)
      .getToken()
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(TensorCostNetworkError);
    expect(shouldFallBackToProvider(err)).toBe(true);
  });

  it("abort → TensorCostTimeoutError (falls back)", async () => {
    const err = await mk((async () => {
      throw Object.assign(new Error("aborted"), { name: "AbortError" });
    }) as unknown as typeof fetch)
      .getToken()
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(TensorCostTimeoutError);
  });

  it("5xx → TensorCostProxyError (falls back); 401 → plain Error (does not)", async () => {
    const e5 = await mk((async () => new Response("", { status: 502 })) as unknown as typeof fetch)
      .getToken()
      .catch((e: unknown) => e);
    expect(e5).toBeInstanceOf(TensorCostProxyError);
    expect((e5 as Error).message).toBe("token-exchange failed status=502");
    const e4 = await mk((async () => new Response("", { status: 401 })) as unknown as typeof fetch)
      .getToken()
      .catch((e: unknown) => e);
    expect((e4 as Error).message).toBe("token-exchange failed status=401");
    expect(shouldFallBackToProvider(e4)).toBe(false);
  });
});
