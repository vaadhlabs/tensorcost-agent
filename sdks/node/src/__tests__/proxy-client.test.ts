/**
 * Tests for the proxy-client module — retries, timeouts, typed errors,
 * telemetry hooks, and circuit breaker integration.
 */

import { describe, expect, it, vi } from "vitest";
import { proxyRequest, proxyRequestStream, shouldFallBackToProvider } from "../proxy-client.js";
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

  it("retries the proxy's own 503 until maxAttempts then throws", async () => {
    let calls = 0;
    const fetchImpl: typeof fetch = async () => {
      calls++;
      return new Response("{}", { status: 503, headers: { "x-tc-error-origin": "proxy" } });
    };
    await expect(
      proxyRequest(makeOpts(fetchImpl, { retry: FAST_RETRY })),
    ).rejects.toBeInstanceOf(TensorCostProxyError);
    expect(calls).toBe(3);
  });

  it("does not retry a 502 with no origin header", async () => {
    let calls = 0;
    const fetchImpl: typeof fetch = async () => {
      calls++;
      return new Response("{}", { status: 502 });
    };
    await expect(
      proxyRequest(makeOpts(fetchImpl, { retry: FAST_RETRY })),
    ).rejects.toBeInstanceOf(TensorCostProxyError);
    expect(calls).toBe(1);
  });

  it("fires on_retry with reason='5xx' on each retry", async () => {
    const events: LifecycleEvent[] = [];
    const fetchImpl: typeof fetch = async () =>
      new Response("{}", { status: 503, headers: { "x-tc-error-origin": "gateway" } });
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
    const fetchImpl: typeof fetch = async () =>
      new Response("{}", { status: 503, headers: { "x-tc-error-origin": "proxy" } });
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

// ---------------------------------------------------------------------------
// Never resend a call that may have reached the provider (1.8.0).
//
// Through 1.7.0 every 5xx, timeout and network error was retried, then fell
// back to the provider, so one call the provider had already billed could
// run six times. Retry and fallback now need proof the attempt never left.

function counting(respond: () => Promise<Response>): { fetchImpl: typeof fetch; calls: () => number } {
  let n = 0;
  return {
    fetchImpl: (async () => {
      n++;
      return respond();
    }) as typeof fetch,
    calls: () => n,
  };
}

const errOf = async (p: Promise<unknown>): Promise<unknown> => p.then(() => null, (e: unknown) => e);

describe.each([
  { name: "proxyRequest", send: proxyRequest },
  { name: "proxyRequestStream", send: proxyRequestStream },
])("$name — where did the call get to", ({ send }) => {
  it("proxy-origin 503: retried with backoff, then allowed to fall back", async () => {
    const f = counting(async () =>
      new Response(JSON.stringify({ statusCode: 503, message: "TensorCost could not check the SDK token" }), {
        status: 503,
        headers: { "x-tc-error-origin": "proxy" },
      }),
    );
    const err = await errOf(send(makeOpts(f.fetchImpl, { retry: FAST_RETRY })));
    expect(err).toBeInstanceOf(TensorCostProxyError);
    expect(f.calls()).toBe(3);
    expect((err as TensorCostProxyError).mayHaveReachedProvider).toBe(false);
    expect(shouldFallBackToProvider(err)).toBe(true);
  });

  it("gateway 503 (its breaker is open, nothing forwarded): retried, then allowed to fall back", async () => {
    const f = counting(async () =>
      new Response(JSON.stringify({ error: { code: "UPSTREAM_UNAVAILABLE" } }), {
        status: 503,
        headers: { "x-tc-error-origin": "gateway", "retry-after": "1" },
      }),
    );
    const err = await errOf(send(makeOpts(f.fetchImpl, { retry: FAST_RETRY })));
    expect(f.calls()).toBe(3);
    expect(shouldFallBackToProvider(err)).toBe(true);
  });

  it("gateway 502 for its own connect failure to the proxy: retried, then allowed to fall back", async () => {
    const f = counting(async () =>
      new Response(
        JSON.stringify({ error: { code: "UPSTREAM_PROXY_ERROR", message: "connect ECONNREFUSED 10.0.3.7:4115" } }),
        { status: 502, headers: { "x-tc-error-origin": "gateway" } },
      ),
    );
    const err = await errOf(send(makeOpts(f.fetchImpl, { retry: FAST_RETRY })));
    expect(f.calls()).toBe(3);
    expect(shouldFallBackToProvider(err)).toBe(true);
  });

  it("503 with x-should-retry: true: retried", async () => {
    const f = counting(async () => new Response("{}", { status: 503, headers: { "x-should-retry": "true" } }));
    const err = await errOf(send(makeOpts(f.fetchImpl, { retry: FAST_RETRY })));
    expect(f.calls()).toBe(3);
    expect(shouldFallBackToProvider(err)).toBe(true);
  });

  it("provider-origin 502: not retried, surfaced as the provider's error, not a circuit failure", async () => {
    const circuit = new CircuitBreaker({ openThreshold: 1, closeThreshold: 5 });
    const f = counting(async () =>
      new Response(JSON.stringify({ error: { message: "Bad gateway", type: "server_error" } }), {
        status: 502,
        headers: { "content-type": "application/json", "x-tc-error-origin": "provider" },
      }),
    );
    const err = await errOf(send(makeOpts(f.fetchImpl, { retry: FAST_RETRY, circuit })));
    expect(f.calls()).toBe(1);
    expect(err).toBeInstanceOf(TensorCostProviderError);
    expect((err as TensorCostProviderError).status).toBe(502);
    expect((err as TensorCostProviderError).body).toEqual({ error: { message: "Bad gateway", type: "server_error" } });
    expect((err as TensorCostProviderError).mayHaveReachedProvider).toBe(true);
    expect(shouldFallBackToProvider(err)).toBe(false);
    expect(circuit.currentState).toBe("CLOSED");
  });

  it("provider-origin 504 (the provider did not answer in time): not retried, no fallback", async () => {
    const f = counting(async () =>
      new Response(JSON.stringify({ error: { message: "the provider did not answer in time" } }), {
        status: 504,
        headers: { "x-tc-error-origin": "provider", "x-should-retry": "false" },
      }),
    );
    const err = await errOf(send(makeOpts(f.fetchImpl, { retry: FAST_RETRY })));
    expect(f.calls()).toBe(1);
    expect(shouldFallBackToProvider(err)).toBe(false);
  });

  it("provider-origin 429: the provider's rate limit, surfaced as its error", async () => {
    const f = counting(async () =>
      new Response(JSON.stringify({ error: { message: "Rate limit reached", type: "requests" } }), {
        status: 429,
        headers: { "x-tc-error-origin": "provider", "retry-after": "0" },
      }),
    );
    const err = await errOf(send(makeOpts(f.fetchImpl, { retry: FAST_RETRY })));
    expect(f.calls()).toBe(1);
    expect(err).toBeInstanceOf(TensorCostProviderError);
    expect(shouldFallBackToProvider(err)).toBe(false);
  });

  it("x-should-retry: false: not retried and no fallback, even on the proxy's own 503", async () => {
    const f = counting(async () =>
      new Response("{}", { status: 503, headers: { "x-tc-error-origin": "proxy", "x-should-retry": "false" } }),
    );
    const err = await errOf(send(makeOpts(f.fetchImpl, { retry: FAST_RETRY })));
    expect(f.calls()).toBe(1);
    expect((err as TensorCostProxyError).mayHaveReachedProvider).toBe(true);
    expect(shouldFallBackToProvider(err)).toBe(false);
  });

  it("x-should-retry: false on a 429: not retried and no fallback", async () => {
    const f = counting(async () =>
      new Response("{}", { status: 429, headers: { "x-tc-error-origin": "proxy", "x-should-retry": "false" } }),
    );
    const err = await errOf(send(makeOpts(f.fetchImpl, { retry: FAST_RETRY })));
    expect(f.calls()).toBe(1);
    expect(err).toBeInstanceOf(TensorCostQuotaError);
    expect(shouldFallBackToProvider(err)).toBe(false);
  });

  it("proxy-origin 500 (an error after the forward): not retried, no fallback", async () => {
    const f = counting(async () => new Response("{}", { status: 500, headers: { "x-tc-error-origin": "proxy" } }));
    const err = await errOf(send(makeOpts(f.fetchImpl, { retry: FAST_RETRY })));
    expect(f.calls()).toBe(1);
    expect(shouldFallBackToProvider(err)).toBe(false);
  });

  it("gateway 502 for a proxy that reset mid-call: not retried, no fallback", async () => {
    const f = counting(async () =>
      new Response(JSON.stringify({ error: { code: "UPSTREAM_PROXY_ERROR", message: "socket hang up" } }), {
        status: 502,
        headers: { "x-tc-error-origin": "gateway" },
      }),
    );
    const err = await errOf(send(makeOpts(f.fetchImpl, { retry: FAST_RETRY })));
    expect(f.calls()).toBe(1);
    expect(shouldFallBackToProvider(err)).toBe(false);
  });

  it("5xx with no origin header: not retried, no fallback", async () => {
    const f = counting(async () => new Response("{}", { status: 503 }));
    const err = await errOf(send(makeOpts(f.fetchImpl, { retry: FAST_RETRY })));
    expect(f.calls()).toBe(1);
    expect(err).toBeInstanceOf(TensorCostProxyError);
    expect(shouldFallBackToProvider(err)).toBe(false);
  });

  it("timeout after the request was sent: not retried, no fallback, still a circuit failure", async () => {
    const circuit = new CircuitBreaker({ openThreshold: 1, closeThreshold: 5 });
    let calls = 0;
    const fetchImpl: typeof fetch = (_url, init) =>
      new Promise((_resolve, reject) => {
        calls++;
        init?.signal?.addEventListener("abort", () =>
          reject(Object.assign(new Error("aborted"), { name: "AbortError" })),
        );
      });
    const err = await errOf(
      send(makeOpts(fetchImpl, { retry: FAST_RETRY, timeoutMs: 20, headersTimeoutMs: 20, circuit })),
    );
    expect(calls).toBe(1);
    expect(err).toBeInstanceOf(TensorCostTimeoutError);
    expect((err as TensorCostTimeoutError).mayHaveReachedProvider).toBe(true);
    expect(shouldFallBackToProvider(err)).toBe(false);
    expect(circuit.currentState).toBe("OPEN");
  });

  it("connection reset: not retried, no fallback", async () => {
    let calls = 0;
    const fetchImpl: typeof fetch = async () => {
      calls++;
      throw new TypeError("fetch failed", {
        cause: Object.assign(new Error("read ECONNRESET"), { code: "ECONNRESET", syscall: "read" }),
      });
    };
    const err = await errOf(send(makeOpts(fetchImpl, { retry: FAST_RETRY })));
    expect(calls).toBe(1);
    expect(err).toBeInstanceOf(TensorCostNetworkError);
    expect((err as TensorCostNetworkError).mayHaveReachedProvider).toBe(true);
    expect(shouldFallBackToProvider(err)).toBe(false);
  });

  it("connection refused before send: retried with backoff, then allowed to fall back", async () => {
    let calls = 0;
    const fetchImpl: typeof fetch = async () => {
      calls++;
      throw new TypeError("fetch failed", {
        cause: Object.assign(new Error("connect ECONNREFUSED 10.0.3.7:443"), { code: "ECONNREFUSED", syscall: "connect" }),
      });
    };
    const err = await errOf(send(makeOpts(fetchImpl, { retry: FAST_RETRY })));
    expect(calls).toBe(3);
    expect((err as TensorCostNetworkError).mayHaveReachedProvider).toBe(false);
    expect(shouldFallBackToProvider(err)).toBe(true);
  });

  it("connection refused with fail-open: falls back at once, without retrying", async () => {
    let calls = 0;
    const fetchImpl: typeof fetch = async () => {
      calls++;
      throw new TypeError("fetch failed", {
        cause: Object.assign(new Error("getaddrinfo ENOTFOUND proxy.local"), { code: "ENOTFOUND", syscall: "getaddrinfo" }),
      });
    };
    const err = await errOf(send(makeOpts(fetchImpl, { retry: FAST_RETRY, failOpenFast: true })));
    expect(calls).toBe(1);
    expect(shouldFallBackToProvider(err)).toBe(true);
  });
});

describe("proxyRequest — a 2xx whose body doesn't arrive", () => {
  it("is never resent: the provider has already answered", async () => {
    let calls = 0;
    const fetchImpl: typeof fetch = async () => {
      calls++;
      const body = new ReadableStream<Uint8Array>({ start() {} }); // never sends a byte
      return new Response(body, { status: 200, headers: { "content-type": "application/json" } });
    };
    const err = await errOf(proxyRequest(makeOpts(fetchImpl, { retry: FAST_RETRY, timeoutMs: 30 })));
    expect(calls).toBe(1);
    expect(err).toBeInstanceOf(TensorCostTimeoutError);
    expect(shouldFallBackToProvider(err)).toBe(false);
  });
});
