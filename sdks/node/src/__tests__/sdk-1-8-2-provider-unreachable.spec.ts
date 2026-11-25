/**
 * 1.8.2: the proxy's answer for a provider it could not connect to.
 *
 * inference-proxy-service answers a provider send that got no answer in one
 * of three ways (unanswered-send.ts), each with `x-tc-upstream-error`:
 *
 *   provider_unreachable      502, no x-should-retry. Connect, DNS or TLS
 *                             failed before anything was written, after the
 *                             proxy's own one retry. Nothing was billed.
 *   provider_connection_lost  502, x-should-retry: false. May have run.
 *   provider_timeout          504, x-should-retry: false. May have run.
 *
 * All three carry `x-tc-error-origin: provider`. Through 1.8.1 the SDK
 * surfaced all three as the provider's error and never fell open. The first
 * is proven unsent, so going to the provider directly is safe: 1.8.2 does,
 * when fail-open is on, without retrying through the proxy or counting a
 * circuit failure (the proxy is healthy).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { classifyErrorAnswer } from "../delivery.js";
import { proxyRequest, proxyRequestStream, shouldFallBackToProvider } from "../proxy-client.js";
import { CircuitBreaker } from "../circuit.js";
import { TensorCostError, TensorCostProxyError } from "../errors.js";
import type { LifecycleEvent } from "../telemetry.js";
import type { RetryConfig } from "../retry.js";
import { wrap, __test__ as wrapTest } from "../wrap.js";
import { __test__ as runtimeTest } from "../runtime.js";
import { __test__ as exitHooks } from "../exit-flush.js";
import { fakeApiClient, jsonResponse } from "./fake-api-promise.js";

const FAST_RETRY: RetryConfig = { maxAttempts: 3, baseDelayMs: 1, maxDelayMs: 2 };

type Kind = "provider_unreachable" | "provider_connection_lost" | "provider_timeout";

/** The proxy's answer for `kind`, headers as unanswered-send.ts and error-origin.ts write them. */
function unanswered(kind: Kind, extra: Record<string, string> = {}): Response {
  const status = kind === "provider_timeout" ? 504 : 502;
  const headers: Record<string, string> = {
    "content-type": "application/json",
    "x-tc-error-origin": "provider",
    "x-tc-upstream-error": kind,
    ...(kind === "provider_unreachable" ? {} : { "x-should-retry": "false" }),
    ...extra,
  };
  const type = kind === "provider_timeout" ? "upstream_timeout" : "upstream_unavailable";
  return new Response(
    JSON.stringify({ error: { type, code: kind, message: "TensorCost couldn't reach OpenAI (api.openai.com)" } }),
    { status, headers },
  );
}

describe("classifyErrorAnswer: x-tc-upstream-error", () => {
  const h = (o: Record<string, string>) => new Headers(o);

  it("502 provider_unreachable with no x-should-retry is unsent, origin provider or not", () => {
    for (const origin of ["provider", "proxy", null]) {
      const headers = h({
        "x-tc-upstream-error": "provider_unreachable",
        ...(origin ? { "x-tc-error-origin": origin } : {}),
      });
      expect(classifyErrorAnswer(502, headers, null)).toBe("unsent");
    }
  });

  it("x-should-retry: false still wins over provider_unreachable", () => {
    const headers = h({
      "x-tc-error-origin": "provider",
      "x-tc-upstream-error": "provider_unreachable",
      "x-should-retry": "false",
    });
    expect(classifyErrorAnswer(502, headers, null)).toBe("provider");
    headers.delete("x-tc-error-origin");
    expect(classifyErrorAnswer(502, headers, null)).toBe("maybe_sent");
  });

  it("provider_unreachable on anything but a 502 is classified as before", () => {
    const headers = h({ "x-tc-error-origin": "provider", "x-tc-upstream-error": "provider_unreachable" });
    expect(classifyErrorAnswer(504, headers, null)).toBe("provider");
    expect(classifyErrorAnswer(500, headers, null)).toBe("provider");
  });

  it("connection_lost, timeout and unknown kinds are classified as before", () => {
    for (const [status, kind] of [
      [502, "provider_connection_lost"],
      [504, "provider_timeout"],
      [502, "something_new"],
    ] as const) {
      expect(classifyErrorAnswer(status, h({ "x-tc-error-origin": "provider", "x-tc-upstream-error": kind }), null)).toBe(
        "provider",
      );
      expect(classifyErrorAnswer(status, h({ "x-tc-error-origin": "proxy", "x-tc-upstream-error": kind }), null)).toBe(
        "maybe_sent",
      );
    }
  });
});

function counting(respond: () => Response): { fetchImpl: typeof fetch; calls: () => number } {
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

function opts(fetchImpl: typeof fetch, circuit: CircuitBreaker, events: LifecycleEvent[]) {
  return {
    proxyUrl: "https://proxy.local",
    bearerToken: "jwt",
    providerUrl: "https://api.openai.com/v1",
    providerAuth: "Bearer sk-test",
    body: { model: "gpt-4o-mini", messages: [] },
    model: "gpt-4o-mini",
    provider: "openai" as const,
    operation: "chat.completions",
    correlationId: "corr-1",
    environment: null,
    retry: FAST_RETRY,
    timeoutMs: 5000,
    onLifecycleEvent: (e: LifecycleEvent) => events.push(e),
    circuit,
    fetchImpl,
  };
}

describe.each([
  { name: "proxyRequest", send: proxyRequest },
  { name: "proxyRequestStream", send: proxyRequestStream },
])("$name: the proxy couldn't reach the provider", ({ send }) => {
  it("provider_unreachable: one attempt, no circuit failure, unsent so it may fall back", async () => {
    const f = counting(() => unanswered("provider_unreachable"));
    const circuit = new CircuitBreaker({ openThreshold: 1, closeThreshold: 5 });
    const events: LifecycleEvent[] = [];
    const err = await errOf(send(opts(f.fetchImpl, circuit, events)));

    expect(err).toBeInstanceOf(TensorCostProxyError);
    expect((err as TensorCostProxyError).status).toBe(502);
    expect((err as TensorCostProxyError).mayHaveReachedProvider).toBe(false);
    expect(shouldFallBackToProvider(err)).toBe(true);
    expect(f.calls()).toBe(1);
    expect(circuit.failures).toBe(0);
    expect(circuit.currentState).toBe("CLOSED");
    expect(events.filter((e) => e.kind === "on_retry")).toHaveLength(0);
    expect(events.find((e) => e.kind === "on_error")).toMatchObject({
      errorClass: "TensorCostProxyError",
      status: 502,
      reason: "provider_unreachable",
    });
  });

  it.each(["provider_connection_lost", "provider_timeout"] as const)(
    "%s: as in 1.8.1, never retried and never a fallback",
    async (kind) => {
      const f = counting(() => unanswered(kind));
      const circuit = new CircuitBreaker({ openThreshold: 1, closeThreshold: 5 });
      const err = await errOf(send(opts(f.fetchImpl, circuit, [])));
      expect(err).toBeInstanceOf(TensorCostError);
      expect(err).not.toBeInstanceOf(TensorCostProxyError);
      expect((err as TensorCostError).mayHaveReachedProvider).toBe(true);
      expect(shouldFallBackToProvider(err)).toBe(false);
      expect(f.calls()).toBe(1);
      expect(circuit.failures).toBe(0);
    },
  );

  it("provider_unreachable with x-should-retry: false (a routed call): may have been sent, no fallback", async () => {
    const f = counting(() => unanswered("provider_unreachable", { "x-should-retry": "false" }));
    const err = await errOf(send(opts(f.fetchImpl, new CircuitBreaker(), [])));
    expect((err as TensorCostError).mayHaveReachedProvider).toBe(true);
    expect(shouldFallBackToProvider(err)).toBe(false);
    expect(f.calls()).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Through wrap(): the real transport, circuit, proxy client and fallback.

const TENANT = "11111111-1111-1111-1111-111111111111";
const TC = "http://api.tc.test";
const PROXY = "http://proxy.test";

const chatOk = {
  id: "chatcmpl-1",
  object: "chat.completion",
  choices: [{ index: 0, message: { role: "assistant", content: "Ok." }, finish_reason: "stop" }],
  usage: { prompt_tokens: 10, completion_tokens: 2 },
};
const ask = { model: "gpt-4.1-mini", max_tokens: 20, messages: [{ role: "user", content: "Say ok." }] };

/** TensorCost up; the proxy answers every call with `kind`. */
function tensorcost(kind: Kind) {
  const state = { proxyRequests: 0 };
  const f = (async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : (input as Request).url ?? String(input);
    if (url.includes("/sdk-token/exchange")) return jsonResponse(200, { token: "jwt", expires_in: 900 });
    if (url.includes("/v1/sdk-layer")) return jsonResponse(200, { published_layer: "route" });
    if (url.includes("/inference-proxy/observation")) return new Response(null, { status: 202 });
    if (url.startsWith(PROXY)) {
      state.proxyRequests += 1;
      return unanswered(kind);
    }
    throw new TypeError(`fetch failed: unrouted ${url}`);
  }) as typeof fetch;
  return { f, state };
}

const savedFetch = globalThis.fetch;
let warn: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  wrapTest.resetWarnings();
  runtimeTest.clearRuntimes();
  warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
});
afterEach(() => {
  globalThis.fetch = savedFetch;
  warn.mockRestore();
  exitHooks.reset();
});

describe("wrap(): provider_unreachable", () => {
  const base = {
    apiKey: "tc",
    tenantId: TENANT,
    baseUrl: TC,
    maxLayer: "route" as const,
    proxyUrl: PROXY,
    flushOnExit: false,
    // The shipped defaults: 5 attempts, circuit opens at 3 failures.
  };

  it("fail-open on: one proxy request, then the provider directly; the circuit counts nothing", async () => {
    const tc = tensorcost("provider_unreachable");
    globalThis.fetch = tc.f;
    const events: LifecycleEvent[] = [];
    const fake = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(fake.client, { ...base, onLifecycleEvent: (e) => events.push(e) });

    for (let i = 1; i <= 4; i++) {
      const out = (await fake.client.chat.completions.create(ask)) as { id: string };
      expect(out.id).toBe("chatcmpl-1");
      expect(tc.state.proxyRequests).toBe(i);
      expect(fake.requests).toHaveLength(i);
    }
    // Four calls, more than the three failures that open the circuit: every
    // one still went through the proxy first, and no fallback saw a failure.
    const fallbacks = events.filter((e) => e.kind === "on_fallback");
    expect(fallbacks).toHaveLength(4);
    for (const e of fallbacks) expect(e).toMatchObject({ consecutiveFailures: 0 });
    expect(events.filter((e) => e.kind === "on_retry")).toHaveLength(0);
    expect(events.filter((e) => e.kind === "on_circuit_state")).toHaveLength(0);
  });

  it("fail-open off: throws the proxy error with mayHaveReachedProvider false, provider never called", async () => {
    const tc = tensorcost("provider_unreachable");
    globalThis.fetch = tc.f;
    const fake = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(fake.client, { ...base, failOpenEnabled: false });

    const err = await errOf(fake.client.chat.completions.create(ask));
    expect(err).toBeInstanceOf(TensorCostProxyError);
    expect((err as TensorCostProxyError).status).toBe(502);
    expect((err as TensorCostProxyError).mayHaveReachedProvider).toBe(false);
    expect(tc.state.proxyRequests).toBe(1);
    expect(fake.requests).toHaveLength(0);
  });

  it.each(["provider_connection_lost", "provider_timeout"] as const)(
    "%s with fail-open on: surfaced after one proxy request, provider never called",
    async (kind) => {
      const tc = tensorcost(kind);
      globalThis.fetch = tc.f;
      const fake = fakeApiClient(() => jsonResponse(200, chatOk));
      wrap(fake.client, base);

      const err = await errOf(fake.client.chat.completions.create(ask));
      expect(err).not.toBeNull();
      expect(err).not.toBeInstanceOf(TensorCostProxyError);
      expect(tc.state.proxyRequests).toBe(1);
      expect(fake.requests).toHaveLength(0);
    },
  );
});
