/**
 * Tests for applied-mode wiring (Layer 2).
 *
 * Uses the same duck-typed fake client pattern as the other provider
 * tests — no real openai / @anthropic-ai/sdk import needed.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { TensorCostConfigError } from "../config.js";
import { ObservationTransport } from "../transport.js";
import * as openaiProvider from "../providers/openai.js";
import { wrap } from "../wrap.js";
import { installAppliedMode, providerTarget, __test__ } from "../applied.js";
import { defaultHardening } from "./test-harness.js";

const { readBaseUrl, readApiKey, isAzureOpenAI, bearerAuth, anthropicAuth, azureAuth } =
  __test__;

// ---------------------------------------------------------------------------
// Helper factories
// ---------------------------------------------------------------------------

function fakeOpenAIClient(overrides: Record<string, unknown> = {}) {
  return {
    baseURL: "https://api.openai.com/v1",
    apiKey: "sk-test-key",
    defaultHeaders: {} as Record<string, string>,
    chat: {
      completions: {
        create: async (_args: unknown) => ({
          id: "chatcmpl-1",
          usage: { prompt_tokens: 10, completion_tokens: 20 },
        }),
      },
    },
    ...overrides,
  };
}

function fakeAzureClient() {
  return {
    baseURL: "https://my-resource.openai.azure.com/openai/deployments/my-model",
    apiKey: "azure-api-key",
    azure: true,
    defaultHeaders: {} as Record<string, string>,
    chat: {
      completions: {
        create: async (_args: unknown) => ({
          id: "chatcmpl-az-1",
          usage: { prompt_tokens: 5, completion_tokens: 10 },
        }),
      },
    },
  };
}

function fakeAnthropicClient(overrides: Record<string, unknown> = {}) {
  return {
    baseURL: "https://api.anthropic.com",
    apiKey: "anth-test-key",
    defaultHeaders: {} as Record<string, string>,
    messages: {
      create: async (_args: unknown) => ({
        id: "msg_1",
        usage: { input_tokens: 15, output_tokens: 25 },
      }),
    },
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Unit tests for internal helpers
// ---------------------------------------------------------------------------

describe("readBaseUrl", () => {
  it("returns the baseURL string", () => {
    expect(readBaseUrl({ baseURL: "https://api.openai.com/v1" })).toBe(
      "https://api.openai.com/v1",
    );
  });

  it("strips trailing slashes", () => {
    expect(readBaseUrl({ baseURL: "https://api.openai.com/v1/" })).toBe(
      "https://api.openai.com/v1",
    );
  });

  it("returns null when missing", () => {
    expect(readBaseUrl({})).toBeNull();
  });

  it("returns null when not a string", () => {
    expect(readBaseUrl({ baseURL: 42 })).toBeNull();
  });
});

describe("readApiKey", () => {
  it("returns the apiKey string", () => {
    expect(readApiKey({ apiKey: "sk-abc" })).toBe("sk-abc");
  });

  it("returns null when missing", () => {
    expect(readApiKey({})).toBeNull();
  });
});

describe("isAzureOpenAI", () => {
  it("detects azure=true flag", () => {
    expect(isAzureOpenAI({ azure: true, baseURL: "https://whatever" })).toBe(true);
  });

  it("detects .openai.azure.com in baseURL", () => {
    expect(
      isAzureOpenAI({
        baseURL: "https://my-resource.openai.azure.com/openai/deployments/x",
      }),
    ).toBe(true);
  });

  it("returns false for vanilla OpenAI", () => {
    expect(isAzureOpenAI({ baseURL: "https://api.openai.com/v1" })).toBe(false);
  });
});

describe("auth formatters", () => {
  it("bearerAuth formats correctly", () => {
    expect(bearerAuth("sk-abc")).toBe("Bearer sk-abc");
  });

  it("anthropicAuth formats correctly", () => {
    expect(anthropicAuth("anth-abc")).toBe("x-api-key anth-abc");
  });

  it("azureAuth formats correctly", () => {
    expect(azureAuth("azure-key")).toBe("api-key azure-key");
  });
});

// ---------------------------------------------------------------------------
// installAppliedMode — per-provider header injection
// ---------------------------------------------------------------------------

describe("installAppliedMode leaves the client alone (1.4.1)", () => {
  // Through 1.4.0 this rewrote baseURL to the proxy and replaced
  // defaultHeaders with a plain object, which broke openai@4 (its
  // defaultHeaders is a method) and sent the "direct" fallback to the proxy.
  it.each([
    ["openai", () => fakeOpenAIClient()],
    ["anthropic", () => fakeAnthropicClient()],
    ["openai", () => fakeAzureClient()],
  ] as const)("%s: baseURL and defaultHeaders are untouched", (provider, make) => {
    const c = make();
    const before = { baseURL: c.baseURL, defaultHeaders: c.defaultHeaders, keys: Object.keys(c.defaultHeaders) };
    installAppliedMode(c, "https://proxy.tensorcost.com/", provider, {
      application: "a",
      tags: ["t=1"],
      customer: "c",
    });
    expect(c.baseURL).toBe(before.baseURL);
    expect(c.defaultHeaders).toBe(before.defaultHeaders);
    expect(Object.keys(c.defaultHeaders)).toEqual(before.keys);
  });

  it("keeps a defaultHeaders method intact", () => {
    const defaultHeaders = function () {
      return { "x-custom": "hello" };
    };
    const c = { ...fakeOpenAIClient(), defaultHeaders };
    installAppliedMode(c, "https://proxy.tensorcost.com", "openai");
    expect(c.defaultHeaders).toBe(defaultHeaders);
  });

  it("throws when the client has no baseURL", () => {
    const c = fakeOpenAIClient({ baseURL: undefined });
    expect(() => installAppliedMode(c, "https://proxy.tensorcost.com", "openai")).toThrow(
      TensorCostConfigError,
    );
  });

  it("throws when the client has no apiKey", () => {
    const c = fakeOpenAIClient({ apiKey: undefined });
    expect(() => installAppliedMode(c, "https://proxy.tensorcost.com", "openai")).toThrow(
      TensorCostConfigError,
    );
  });
});

describe("providerTarget — what the proxy forwards to", () => {
  it("OpenAI: the client's base URL and a Bearer key", () => {
    expect(providerTarget(fakeOpenAIClient(), "openai", "x")).toEqual({
      providerUrl: "https://api.openai.com/v1",
      providerAuth: "Bearer sk-test-key",
    });
  });

  it("Anthropic: x-api-key format", () => {
    expect(providerTarget(fakeAnthropicClient(), "anthropic", "x")).toEqual({
      providerUrl: "https://api.anthropic.com",
      providerAuth: "x-api-key anth-test-key",
    });
  });

  it("Azure: the deployment URL and api-key format", () => {
    expect(providerTarget(fakeAzureClient(), "openai", "x")).toEqual({
      providerUrl: "https://my-resource.openai.azure.com/openai/deployments/my-model",
      providerAuth: "api-key azure-api-key",
    });
  });

  it("reads the key per call, so a rotated key is used", () => {
    const c = fakeOpenAIClient();
    c.apiKey = "sk-rotated";
    expect(providerTarget(c, "openai", "x").providerAuth).toBe("Bearer sk-rotated");
  });

  it("honours x-tc-provider-* set by hand on plain defaultHeaders", () => {
    const c = fakeOpenAIClient({
      defaultHeaders: { "x-tc-provider-url": "https://gw.example", "x-tc-provider-auth": "Bearer gw" },
    });
    expect(providerTarget(c, "openai", "x")).toEqual({
      providerUrl: "https://gw.example",
      providerAuth: "Bearer gw",
    });
  });
});

// ---------------------------------------------------------------------------
// wrap() — config-level applied-mode validation
// ---------------------------------------------------------------------------

describe("wrap() applied-mode config", () => {
  let savedProxy: string | undefined;

  beforeEach(() => {
    savedProxy = process.env.TENSORCOST_PROXY_URL;
    delete process.env.TENSORCOST_PROXY_URL;
  });

  afterEach(() => {
    if (savedProxy === undefined) delete process.env.TENSORCOST_PROXY_URL;
    else process.env.TENSORCOST_PROXY_URL = savedProxy;
  });

  it("appliedMode: false — no proxy URL required; behavior unchanged", () => {
    const c = fakeOpenAIClient();
    const original = c.chat.completions.create;
    wrap(c, {
      apiKey: "tc-key",
      baseUrl: "http://localhost:1",
      appliedMode: false,
    });
    // baseURL untouched
    expect(c.baseURL).toBe("https://api.openai.com/v1");
    // method is still wrapped (telemetry) but baseURL is the provider's
    expect(c.chat.completions.create).not.toBe(original); // wrapped for telemetry
  });

  it("appliedMode: true + proxyUrl set — wraps the client without changing it", () => {
    const c = fakeOpenAIClient();
    const original = c.chat.completions.create;
    wrap(c, {
      apiKey: "tc-key",
      tenantId: "t-1",
      baseUrl: "http://localhost:1",
      appliedMode: true,
      proxyUrl: "https://proxy.tensorcost.com",
    });
    expect(c.baseURL).toBe("https://api.openai.com/v1");
    expect(c.defaultHeaders).toEqual({});
    expect(c.chat.completions.create).not.toBe(original);
  });

  it("appliedMode: true + TENSORCOST_PROXY_URL env — accepted, client unchanged", () => {
    process.env.TENSORCOST_PROXY_URL = "https://proxy-env.tensorcost.com";
    const c = fakeOpenAIClient();
    wrap(c, {
      apiKey: "tc-key",
      tenantId: "t-1",
      baseUrl: "http://localhost:1",
      appliedMode: true,
    });
    expect(c.baseURL).toBe("https://api.openai.com/v1");
  });

  it("appliedMode: true + neither proxyUrl nor env — throws TensorCostConfigError at wrap time", () => {
    const c = fakeOpenAIClient();
    expect(() =>
      wrap(c, {
        apiKey: "tc-key",
        tenantId: "t-1",
        baseUrl: "http://localhost:1",
        appliedMode: true,
        // no proxyUrl; TENSORCOST_PROXY_URL deleted in beforeEach
      }),
    ).toThrow(TensorCostConfigError);
  });

  it("appliedMode: true + neither — error message is clear", () => {
    const c = fakeOpenAIClient();
    expect(() =>
      wrap(c, { apiKey: "tc-key", appliedMode: true }),
    ).toThrow(/proxy URL/i);
  });
});

// ---------------------------------------------------------------------------
// Applied-mode + observe-only telemetry coexistence
// ---------------------------------------------------------------------------

describe("applied mode + telemetry coexistence", () => {
  it("telemetry still fires post-call when appliedMode=true", async () => {
    const observations: unknown[] = [];

    // Build a fake fetch that handles both the token exchange and the
    // observation endpoint. The proxy itself is not exercised here —
    // the fake client just returns a canned response.
    const fakeFetch: typeof fetch = async (input, init = {}) => {
      const url = typeof input === "string" ? input : (input as URL).toString();
      if (url.includes("sdk-token/exchange")) {
        return new Response(
          JSON.stringify({ token: "tok", expires_in: 900 }),
          { status: 200, headers: { "content-type": "application/json" } },
        );
      }
      if (url.includes("/observation")) {
        observations.push(JSON.parse((init as RequestInit).body as string));
        return new Response(null, { status: 202 });
      }
      return new Response(null, { status: 404 });
    };

    // Directly use the OpenAI provider wrapper (not wrap()) so we can
    // inject the fake fetch into the transport without setting up an
    // actual TensorCost backend. We also call installAppliedMode to
    // verify it doesn't break telemetry.
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "tc-key",
      fetchImpl: fakeFetch,
    });

    const c = fakeOpenAIClient();
    installAppliedMode(c, "https://proxy.tensorcost.com", "openai");
    openaiProvider.install(c, transport, null, null, null, defaultHardening());

    const resp = await c.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(resp.id).toBe("chatcmpl-1");

    // Flush microtasks so the fire-and-forget observation delivery runs.
    for (let i = 0; i < 30; i++) await Promise.resolve();

    expect(observations).toHaveLength(1);
    expect((observations[0] as Record<string, unknown>).provider).toBe("openai");
    expect((observations[0] as Record<string, unknown>).status).toBe("success");
  });
});

// ---------------------------------------------------------------------------
// Response verbatim — no SDK-side rewriting
// ---------------------------------------------------------------------------

describe("response body passthrough", () => {
  it("returns the provider response unchanged regardless of applied mode", async () => {
    const c = fakeOpenAIClient();
    wrap(c, {
      apiKey: "tc-key",
      tenantId: "t-1",
      baseUrl: "http://localhost:1",
      appliedMode: true,
      proxyUrl: "https://proxy.tensorcost.com",
    });
    // The fake create() returns a canned response; applied mode does not
    // transform it.
    const resp = await c.chat.completions.create({ model: "gpt-4o-mini" });
    expect(resp).toEqual({
      id: "chatcmpl-1",
      usage: { prompt_tokens: 10, completion_tokens: 20 },
    });
  });
});
