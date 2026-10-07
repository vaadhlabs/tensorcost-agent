/**
 * Tests for the proxy-client module — retries, timeouts, typed errors,
 * telemetry hooks, and circuit breaker integration.
 */

import { describe, expect, it, vi } from "vitest";
import { proxyRequest } from "../proxy-client.js";
import {
  TensorCostNetworkError,
  TensorCostTimeoutError,
  TensorCostProxyError,
  TensorCostQuotaError,
  TensorCostProviderError,
} from "../errors.js";
import { CircuitBreaker } from "../circuit.js";
import type { LifecycleEvent } from "../telemetry.js";
import type { RetryConfig } from "../retry.js";

// Minimal retry config to keep tests fast.
const FAST_RETRY: RetryConfig = {
  maxAttempts: 3,
  baseDelayMs: 1,
  maxDelayMs: 2,
};

const NO_RETRY: RetryConfig = {
  maxAttempts: 1,
  baseDelayMs: 1,
  maxDelayMs: 2,
};

function makeOpts(
  fetchImpl: typeof fetch,
  overrides: {
    retry?: RetryConfig;
    timeoutMs?: number;
    headersTimeoutMs?: number;
    body?: Record<string, unknown>;
    onLifecycleEvent?: (e: LifecycleEvent) => void;
    circuit?: CircuitBreaker;
    failOpenFast?: boolean;
  } = {},
) {
  return {
    proxyUrl: "https://proxy.local",
    bearerToken: "jwt",
    providerUrl: "https://api.openai.com/v1",
    providerAuth: "Bearer sk-test",
    body: overrides.body ?? { model: "gpt-4o-mini", messages: [] },
    model: "gpt-4o-mini",
    provider: "openai" as const,
    operation: "chat.completions",
    correlationId: "corr-1",
    environment: null,
    retry: overrides.retry ?? NO_RETRY,
    timeoutMs: overrides.timeoutMs ?? 5000,
    headersTimeoutMs: overrides.headersTimeoutMs,
    onLifecycleEvent: overrides.onLifecycleEvent,
    circuit: overrides.circuit ?? new CircuitBreaker(),
    failOpenFast: overrides.failOpenFast,
    fetchImpl,
  };
}

function okFetch(body: Record<string, unknown> = { id: "chatcmpl-1", usage: { prompt_tokens: 10, completion_tokens: 20 } }): typeof fetch {
  return async () =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json", "x-tc-request-id": "req-1" },
    });
}

function statusFetch(status: number, headers: Record<string, string> = {}): typeof fetch {
  return async () =>
    new Response(JSON.stringify({ error: "err" }), {
      status,
      headers: { "content-type": "application/json", ...headers },
    });
}

function networkErrorFetch(): typeof fetch {
  return async () => {
    throw new Error("ECONNREFUSED");
  };
}

// ---------------------------------------------------------------------------
// Happy path

describe("proxyRequest — success", () => {
  it("returns the parsed response body on 200", async () => {
    const result = await proxyRequest(makeOpts(okFetch()));
    expect((result.body as any).id).toBe("chatcmpl-1");
  });

  it("fires before_request and after_response telemetry events", async () => {
    const events: LifecycleEvent[] = [];
    await proxyRequest(makeOpts(okFetch(), { onLifecycleEvent: (e) => events.push(e) }));
    expect(events.some((e) => e.kind === "before_request")).toBe(true);
    expect(events.some((e) => e.kind === "after_response")).toBe(true);
    const after = events.find((e) => e.kind === "after_response") as any;
    expect(after?.status).toBe(200);
    expect(after?.requestId).toBe("req-1");
  });

  it("records circuit success on 200", async () => {
    const circuit = new CircuitBreaker({ openThreshold: 2, closeThreshold: 2 });
    // Pre-open: one failure.
    circuit.recordFailure();
    // Probe to enter HALF_OPEN.
    circuit.shouldProbe();
    await proxyRequest(makeOpts(okFetch(), { circuit }));
    // After success the circuit should still be HALF_OPEN (needs 2 successes to close).
    // We just verify it did NOT go back to OPEN.
    expect(circuit.currentState).not.toBe("OPEN");
  });
});

// ---------------------------------------------------------------------------
// Network errors

describe("proxyRequest — network errors", () => {
  it("throws TensorCostNetworkError on connection failure", async () => {
    await expect(
      proxyRequest(makeOpts(networkErrorFetch())),
    ).rejects.toBeInstanceOf(TensorCostNetworkError);
  });

  it("retries on network error and eventually throws after maxAttempts", async () => {
    let calls = 0;
    const fetchImpl: typeof fetch = async () => {
      calls++;
      throw new Error("ECONNREFUSED");
    };
    const events: LifecycleEvent[] = [];
    await expect(
      proxyRequest(makeOpts(fetchImpl, {
        retry: FAST_RETRY,
        onLifecycleEvent: (e) => events.push(e),
      })),
    ).rejects.toBeInstanceOf(TensorCostNetworkError);
    expect(calls).toBe(3); // maxAttempts
    const retries = events.filter((e) => e.kind === "on_retry");
    expect(retries).toHaveLength(2);
    const errors = events.filter((e) => e.kind === "on_error");
    expect(errors).toHaveLength(1);
  });

  it("does not retry network errors when failOpenFast is true", async () => {
    let calls = 0;
    const fetchImpl: typeof fetch = async () => {
      calls++;
      throw new Error("ECONNREFUSED");
    };
    await expect(
      proxyRequest(makeOpts(fetchImpl, {
        retry: FAST_RETRY,
        failOpenFast: true,
      })),
    ).rejects.toBeInstanceOf(TensorCostNetworkError);
    expect(calls).toBe(1);
  });

  it("records circuit failure on network error", async () => {
    const circuit = new CircuitBreaker({ openThreshold: 1, closeThreshold: 5 });
    await expect(
      proxyRequest(makeOpts(networkErrorFetch(), { circuit })),
    ).rejects.toBeInstanceOf(TensorCostNetworkError);
    expect(circuit.currentState).toBe("OPEN");
  });
});

// ---------------------------------------------------------------------------
// Timeout

describe("proxyRequest — timeout", () => {
  it("throws TensorCostTimeoutError (headers) when headers never arrive (streaming body)", async () => {
    const neverResolve: typeof fetch = (_url, init) => {
      return new Promise((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          const err = new Error("The operation was aborted");
          err.name = "AbortError";
          reject(err);
        });
      });
    };
    const err = await proxyRequest(
      makeOpts(neverResolve, {
        headersTimeoutMs: 20,
        timeoutMs: 60_000,
        retry: NO_RETRY,
        body: { model: "gpt-4o-mini", stream: true },
      }),
    ).catch((e) => e);
    expect(err).toBeInstanceOf(TensorCostTimeoutError);
    expect((err as TensorCostTimeoutError).timeoutKind).toBe("headers");
  });

  it("throws TensorCostTimeoutError when the request times out", async () => {
    const neverResolve: typeof fetch = (_url, init) => {
      // Actually we need to listen for the signal abort.
      return new Promise((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          const err = new Error("The operation was aborted");
          err.name = "AbortError";
          reject(err);
        });
      });
    };
    // Very short timeout so the test runs fast.
    const err = await proxyRequest(
      makeOpts(neverResolve, { timeoutMs: 20, retry: NO_RETRY }),
    ).catch((e) => e);
    expect(err).toBeInstanceOf(TensorCostTimeoutError);
    expect((err as TensorCostTimeoutError).timeoutKind).toBe("total");
  });
});

// ---------------------------------------------------------------------------
// 5xx proxy errors

describe("proxyRequest — 5xx", () => {
  it("throws TensorCostProxyError on 503", async () => {
    await expect(
      proxyRequest(makeOpts(statusFetch(503))),
    ).rejects.toBeInstanceOf(TensorCostProxyError);
  });

  it("retries on 5xx until maxAttempts then throws", async () => {
    let calls = 0;
    const fetchImpl: typeof fetch = async () => {
      calls++;
      return new Response("{}", { status: 502 });
    };
    await expect(
      proxyRequest(makeOpts(fetchImpl, { retry: FAST_RETRY })),
    ).rejects.toBeInstanceOf(TensorCostProxyError);
    expect(calls).toBe(3);
  });

  it("fires on_retry with reason='5xx' on each retry", async () => {
    const events: LifecycleEvent[] = [];
    const fetchImpl: typeof fetch = async () => new Response("{}", { status: 503 });
    await expect(
      proxyRequest(makeOpts(fetchImpl, {
        retry: FAST_RETRY,
        onLifecycleEvent: (e) => events.push(e),
      })),
    ).rejects.toBeDefined();
    const retries = events.filter((e) => e.kind === "on_retry") as any[];
    expect(retries.length).toBeGreaterThan(0);
    expect(retries.every((e) => e.reason === "5xx")).toBe(true);
  });

  it("records circuit failure on 5xx", async () => {
    const circuit = new CircuitBreaker({ openThreshold: 2, closeThreshold: 5 });
    const fetchImpl: typeof fetch = async () => new Response("{}", { status: 500 });
    await expect(proxyRequest(makeOpts(fetchImpl, { retry: NO_RETRY, circuit }))).rejects.toBeDefined();
    await expect(proxyRequest(makeOpts(fetchImpl, { retry: NO_RETRY, circuit }))).rejects.toBeDefined();
    expect(circuit.currentState).toBe("OPEN");
  });
});

// ---------------------------------------------------------------------------
// 429 quota errors

describe("proxyRequest — 429", () => {
  it("throws TensorCostQuotaError on 429", async () => {
    await expect(
      proxyRequest(makeOpts(statusFetch(429))),
    ).rejects.toBeInstanceOf(TensorCostQuotaError);
  });

  it("honours Retry-After header on 429", async () => {
    const events: LifecycleEvent[] = [];
    let calls = 0;
    const fetchImpl: typeof fetch = async () => {
      calls++;
      return new Response("{}", {
        status: 429,
        headers: { "retry-after": "0" }, // immediate retry
      });
    };
    await expect(
      proxyRequest(makeOpts(fetchImpl, {
        retry: FAST_RETRY,
        onLifecycleEvent: (e) => events.push(e),
      })),
    ).rejects.toBeInstanceOf(TensorCostQuotaError);
    const retries = events.filter((e) => e.kind === "on_retry") as any[];
    expect(retries.every((e) => e.reason === "quota_429")).toBe(true);
    expect(calls).toBe(3);
  });

  it("does not record circuit failure on 429 (proxy is healthy)", async () => {
    const circuit = new CircuitBreaker({ openThreshold: 1, closeThreshold: 5 });
    await expect(
      proxyRequest(makeOpts(statusFetch(429), { retry: NO_RETRY, circuit })),
    ).rejects.toBeDefined();
    // Circuit should still be CLOSED since 429 is not a proxy infrastructure error.
    expect(circuit.currentState).toBe("CLOSED");
  });
});

// ---------------------------------------------------------------------------
// 4xx provider errors (non-retriable)

describe("proxyRequest — 4xx provider errors", () => {
  it("throws TensorCostProviderError on 401 and does NOT retry", async () => {
    let calls = 0;
    const fetchImpl: typeof fetch = async () => {
      calls++;
      return new Response("{}", { status: 401 });
    };
    await expect(
      proxyRequest(makeOpts(fetchImpl, { retry: FAST_RETRY })),
    ).rejects.toBeInstanceOf(TensorCostProviderError);
    expect(calls).toBe(1); // no retry on 4xx
  });

  it("throws TensorCostProviderError on 422", async () => {
    await expect(
      proxyRequest(makeOpts(statusFetch(422))),
    ).rejects.toBeInstanceOf(TensorCostProviderError);
  });
});

// ---------------------------------------------------------------------------
// Circuit breaker bypass

describe("proxyRequest — circuit breaker", () => {
  it("still fetches when HALF_OPEN probe is in flight (bypass is provider-side)", async () => {
    const circuit = new CircuitBreaker({ openThreshold: 1, closeThreshold: 5 });
    circuit.recordFailure();
    circuit.shouldProbe();
    expect(circuit.currentState).toBe("HALF_OPEN");

    let fetchCalled = false;
    const fetchImpl: typeof fetch = async () => {
      fetchCalled = true;
      return new Response("{}", { status: 200 });
    };
    await proxyRequest(makeOpts(fetchImpl, { circuit }));
    expect(fetchCalled).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Telemetry — attempt numbers

describe("proxyRequest — telemetry attempt numbers", () => {
  it("increments attemptNumber across retries", async () => {
    const events: LifecycleEvent[] = [];
    const fetchImpl: typeof fetch = async () => new Response("{}", { status: 503 });
    await expect(
      proxyRequest(makeOpts(fetchImpl, {
        retry: FAST_RETRY,
        onLifecycleEvent: (e) => events.push(e),
      })),
    ).rejects.toBeDefined();
    const before = events.filter((e) => e.kind === "before_request");
    expect(before.map((e) => e.attemptNumber)).toEqual([1, 2, 3]);
  });
});
