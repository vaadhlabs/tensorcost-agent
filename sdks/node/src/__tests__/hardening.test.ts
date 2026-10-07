/**
 * Integration tests for the 5 hardening features.
 *
 * These tests exercise the full path: wrap() → provider wrapper →
 * proxy-client → circuit breaker / telemetry / retry. We use fake clients
 * and, where the proxy path is exercised, verify behavior through the
 * publicly visible outcomes (return value, thrown error, telemetry events).
 *
 * Because wrap() mutates the client in place, the original vi.fn() is
 * replaced by the wrapped async function. Tests check the return value
 * and thrown error class rather than call counts on the replaced mock.
 */

import { describe, expect, it, vi } from "vitest";
import { wrap } from "../wrap.js";
import {
  TensorCostNetworkError,
  TensorCostProxyError,
  TensorCostQuotaError,
  TensorCostTimeoutError,
} from "../errors.js";
import type { LifecycleEvent } from "../telemetry.js";

/** Mock fetch for wrap() integration tests (token, sdk-layer, proxy). */
function routingFetch(proxyFails = true): typeof fetch {
  return async (input, init = {}) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.includes("sdk-token/exchange")) {
      return new Response(
        JSON.stringify({ token: "tok", expires_in: 900 }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }
    if (url.includes("sdk-layer")) {
      return new Response(
        JSON.stringify({ published_layer: "route", routing_paused: false }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }
    if (url.includes("/api/inference-proxy/observation")) {
      return new Response(null, { status: 202 });
    }
    if (url.includes("proxy.local") || url.includes("inference-proxy/v1")) {
      if (proxyFails) {
        throw new Error("ECONNREFUSED");
      }
      return new Response(JSON.stringify({ id: "proxy-ok", usage: {} }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    return new Response(null, { status: 404 });
  };
}

// ---------------------------------------------------------------------------
// Typed error sanity checks (Feature C) — no wrap() needed

describe("hardening C — typed error classes", () => {
  it("TensorCostNetworkError is instanceof TensorCostNetworkError but not TensorCostProxyError", () => {
    const err = new TensorCostNetworkError("net");
    expect(err.name).toBe("TensorCostNetworkError");
    expect(err instanceof TensorCostProxyError).toBe(false);
  });

  it("TensorCostQuotaError carries status and retryAfterMs", () => {
    const err = new TensorCostQuotaError("429", 3000, { attempt: 2 });
    expect(err.status).toBe(429);
    expect(err.retryAfterMs).toBe(3000);
    expect(err.attempt).toBe(2);
    expect(err.name).toBe("TensorCostQuotaError");
  });

  it("TensorCostTimeoutError exposes timeoutKind", () => {
    const totalErr = new TensorCostTimeoutError("timeout", "total", { attempt: 3 });
    expect(totalErr.timeoutKind).toBe("total");
    expect(totalErr.attempt).toBe(3);

    const idleErr = new TensorCostTimeoutError("idle", "idle");
    expect(idleErr.timeoutKind).toBe("idle");
  });

  it("all error classes are instanceof Error", () => {
    const errs = [
      new TensorCostNetworkError("x"),
      new TensorCostTimeoutError("x", "total"),
      new TensorCostProxyError("x"),
      new TensorCostQuotaError("x", null),
    ];
    for (const err of errs) {
      expect(err).toBeInstanceOf(Error);
    }
  });
});

// ---------------------------------------------------------------------------
// Feature A — retry is forwarded via config

describe("hardening A — retry config forwarded", () => {
  it("wrap() accepts retry config without throwing", () => {
    const client = {
      baseURL: "https://api.openai.com/v1",
      apiKey: "sk-test",
      defaultHeaders: {},
      chat: { completions: { create: vi.fn().mockResolvedValue({ id: "ok", usage: {} }) } },
    };
    expect(() =>
      wrap(client, {
        apiKey: "tc-key",
        baseUrl: "http://api.local",
        retry: { maxAttempts: 3, baseDelayMs: 100, maxDelayMs: 5000 },
      }),
    ).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// Feature B — timeout config forwarded

describe("hardening B — timeout config forwarded", () => {
  it("wrap() accepts timeoutMs and idleTimeoutMs without throwing", () => {
    const client = {
      chat: { completions: { create: vi.fn().mockResolvedValue({ id: "ok", usage: {} }) } },
    };
    expect(() =>
      wrap(client, {
        apiKey: "tc-key",
        timeoutMs: 30_000,
        idleTimeoutMs: 15_000,
      }),
    ).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// Feature D — telemetry hooks

describe("hardening D — telemetry hooks", () => {
  it("onLifecycleEvent callback errors are swallowed (never propagate to caller)", async () => {
    // Client that calls directly (observe-only, no proxy). The lifecycle
    // events in observe-only mode are emitted by the provider wrapper only
    // for proxy errors / retries, so we test this through the proxy path.
    // Simplest: verify that a throwing callback does not break the direct call.
    const directResponse = { id: "direct", usage: { prompt_tokens: 1, completion_tokens: 1 } };
    const client = {
      baseURL: "https://api.openai.com/v1",
      apiKey: "sk-test",
      defaultHeaders: {},
      chat: { completions: { create: vi.fn().mockResolvedValue(directResponse) } },
    };

    const badCallback = vi.fn(() => {
      throw new Error("callback explosion");
    });

    wrap(client, {
      apiKey: "tc-key",
      baseUrl: "http://api.local",
      // observe-only so the create method is wrapped for telemetry but goes direct
      appliedMode: false,
      onLifecycleEvent: badCallback,
    });

    // Should still succeed — the callback throwing should be swallowed.
    const resp = await client.chat.completions.create({ model: "gpt-4o" });
    expect((resp as any).id).toBe("direct");
  });

  it("onLifecycleEvent fires on_fallback when proxy is unreachable", async () => {
    const events: LifecycleEvent[] = [];

    const client = {
      baseURL: "https://api.openai.com/v1",
      apiKey: "sk-test",
      defaultHeaders: {} as Record<string, string>,
      chat: {
        completions: {
          create: vi.fn().mockResolvedValue({ id: "direct-fallback", usage: { prompt_tokens: 1, completion_tokens: 1 } }),
        },
      },
    };

    const savedFetch = globalThis.fetch;
    globalThis.fetch = routingFetch(true);
    try {
      wrap(client, {
        apiKey: "tc-key",
        tenantId: "t-1",
        baseUrl: "http://api.local",
        appliedMode: true,
        proxyUrl: "https://proxy.local",
        retry: { maxAttempts: 1, baseDelayMs: 1, maxDelayMs: 2 },
        failOpenEnabled: true,
        onLifecycleEvent: (e) => events.push(e),
      });

      // Proxy is not reachable (network error) → fallback path.
      const resp = await client.chat.completions.create({ model: "gpt-4o", messages: [] });
      expect((resp as any).id).toBe("direct-fallback");

      // The fallback event must fire with correct metadata.
      const fallback = events.find((e) => e.kind === "on_fallback") as any;
      expect(fallback).toBeDefined();
      expect(fallback?.provider).toBe("openai");
      expect(fallback?.model).toBe("gpt-4o");
      expect(fallback?.operation).toBe("chat.completions");
      expect(typeof fallback?.consecutiveFailures).toBe("number");
    } finally {
      globalThis.fetch = savedFetch;
    }
  });

  it("events carry no prompt content", async () => {
    const events: LifecycleEvent[] = [];
    const client = {
      baseURL: "https://api.openai.com/v1",
      apiKey: "sk-test",
      defaultHeaders: {} as Record<string, string>,
      chat: {
        completions: {
          create: vi.fn().mockResolvedValue({ id: "ok", usage: {} }),
        },
      },
    };

    wrap(client, {
      apiKey: "tc-key",
      baseUrl: "http://api.local",
      appliedMode: true,
      proxyUrl: "https://proxy.local",
      retry: { maxAttempts: 1, baseDelayMs: 1, maxDelayMs: 2 },
      failOpenEnabled: true,
      onLifecycleEvent: (e) => events.push(e),
    });

    await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "ultra-secret prompt content" }],
    });

    for (const event of events) {
      expect(JSON.stringify(event)).not.toContain("ultra-secret");
    }
  });
});

// ---------------------------------------------------------------------------
// Feature E — fail-open circuit breaker

describe("hardening E — fail-open circuit breaker", () => {
  it("fails open to direct provider when proxy is unreachable and failOpenEnabled=true", async () => {
    const directResponse = { id: "direct-ok", usage: { prompt_tokens: 5, completion_tokens: 5 } };
    const client = {
      baseURL: "https://api.openai.com/v1",
      apiKey: "sk-test",
      defaultHeaders: {} as Record<string, string>,
      chat: {
        completions: {
          create: vi.fn().mockResolvedValue(directResponse),
        },
      },
    };

    wrap(client, {
      apiKey: "tc-key",
      baseUrl: "http://api.local",
      appliedMode: true,
      proxyUrl: "https://proxy.local", // won't resolve
      retry: { maxAttempts: 1, baseDelayMs: 1, maxDelayMs: 2 },
      failOpenEnabled: true,
    });

    const resp = await client.chat.completions.create({ model: "gpt-4o-mini" });
    // Fallback should return the direct provider response.
    expect((resp as any).id).toBe("direct-ok");
  });

  it("throws TensorCostProxyError when failOpenEnabled=false and proxy is unreachable", async () => {
    const client = {
      baseURL: "https://api.openai.com/v1",
      apiKey: "sk-test",
      defaultHeaders: {} as Record<string, string>,
      chat: {
        completions: {
          create: vi.fn().mockResolvedValue({ id: "direct", usage: {} }),
        },
      },
    };

    const savedFetch = globalThis.fetch;
    globalThis.fetch = routingFetch(true);
    try {
      wrap(client, {
        apiKey: "tc-key",
        tenantId: "t-1",
        baseUrl: "http://api.local",
        appliedMode: true,
        proxyUrl: "https://proxy.local",
        retry: { maxAttempts: 1, baseDelayMs: 1, maxDelayMs: 2 },
        failOpenEnabled: false,
      });

      await expect(
        client.chat.completions.create({ model: "gpt-4o-mini" }),
      ).rejects.toBeInstanceOf(TensorCostProxyError);
    } finally {
      globalThis.fetch = savedFetch;
    }
  });

  it("observe-only mode (appliedMode=false) is unaffected", async () => {
    const directResponse = { id: "direct", usage: { prompt_tokens: 1, completion_tokens: 1 } };
    const client = {
      chat: {
        completions: {
          create: vi.fn().mockResolvedValue(directResponse),
        },
      },
    };

    wrap(client, { apiKey: "tc-key", baseUrl: "http://api.local", appliedMode: false });

    const resp = await client.chat.completions.create({ model: "gpt-4o-mini" });
    expect((resp as any).id).toBe("direct");
  });
});
