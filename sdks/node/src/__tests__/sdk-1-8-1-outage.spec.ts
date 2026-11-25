/**
 * Findings from a self-hosted run of 1.8.0 with inference-proxy scaled to 0,
 * so the gateway answered 502 and then 503 itself (x-tc-error-origin:
 * gateway):
 *
 *   F1. Calls the provider answered while TensorCost was down left no
 *       spend row. Each observation was posted once; a failed token
 *       exchange or POST dropped it, and flushObservations() after
 *       recovery had nothing to send.
 *   F2. With the circuit open, the call that became the half-open probe
 *       retried up to five times through the dead proxy before falling
 *       back: 5.0-7.6 s for that one call.
 *
 * Everything goes through wrap() with fetch routed per host, so the real
 * transport, circuit and proxy client run. Timers and Date are faked, so
 * the timing assertions are exact.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { wrap, __test__ as wrapTest } from "../wrap.js";
import { flushObservations, observationStats } from "../prewarm.js";
import { __test__ as runtimeTest } from "../runtime.js";
import { __test__ as exitHooks } from "../exit-flush.js";
import { fakeApiClient, jsonResponse } from "./fake-api-promise.js";

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

/** The gateway's own answer for an upstream with no endpoints. */
const gatewayDown = (status: number) =>
  new Response("upstream unavailable", { status, headers: { "x-tc-error-origin": "gateway" } });

function tensorcost(opts: { layer: string }) {
  const state = { down: true, exchanges: 0, proxyRequests: 0 };
  const accepted: Record<string, any>[] = [];
  const f = (async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = typeof input === "string" ? input : (input as Request).url ?? String(input);
    if (url.includes("/sdk-token/exchange")) {
      state.exchanges += 1;
      if (state.down) return gatewayDown(state.exchanges === 1 ? 502 : 503);
      return jsonResponse(200, { token: "jwt", expires_in: 900 });
    }
    if (url.includes("/v1/sdk-layer")) {
      if (state.down) return gatewayDown(503);
      return jsonResponse(200, { published_layer: opts.layer });
    }
    if (url.includes("/inference-proxy/observation")) {
      if (state.down) return gatewayDown(503);
      accepted.push(JSON.parse(String(init.body)));
      return new Response(null, { status: 202 });
    }
    if (url.startsWith(PROXY)) {
      state.proxyRequests += 1;
      return gatewayDown(503);
    }
    throw new TypeError(`fetch failed: unrouted ${url}`);
  }) as typeof fetch;
  return { f, state, accepted };
}

const savedFetch = globalThis.fetch;
let warn: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
  wrapTest.resetWarnings();
  runtimeTest.clearRuntimes();
  warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
});
afterEach(() => {
  vi.useRealTimers();
  globalThis.fetch = savedFetch;
  warn.mockRestore();
  exitHooks.reset();
});

/** Start a call and report how much fake time passed before it settled. */
async function timed(call: () => Promise<unknown>, budgetMs: number): Promise<number | null> {
  const started = Date.now();
  let settledAt: number | null = null;
  void call().then(
    () => (settledAt = Date.now()),
    () => (settledAt = Date.now()),
  );
  for (let elapsed = 0; elapsed <= budgetMs && settledAt === null; elapsed += 10) {
    await vi.advanceTimersByTimeAsync(10);
  }
  return settledAt === null ? null : settledAt - started;
}

describe("F2: the half-open probe is one attempt, then the call falls back at once", () => {
  const options = {
    apiKey: "tc",
    tenantId: TENANT,
    baseUrl: TC,
    maxLayer: "route" as const,
    proxyUrl: PROXY,
    flushOnExit: false,
    // The shipped retry defaults: 5 attempts, 500 ms doubling.
  };

  it("a call after the probe window sends one proxy request and goes direct without waiting", async () => {
    const tc = tensorcost({ layer: "route" });
    tc.state.down = false; // TensorCost up, inference-proxy scaled to 0
    globalThis.fetch = tc.f;
    const fake = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(fake.client, options);

    // Call 1, circuit CLOSED: retried until three failures open the
    // circuit, and not after that.
    expect(await timed(() => fake.client.chat.completions.create(ask), 10_000)).not.toBeNull();
    expect(tc.state.proxyRequests).toBe(3);
    expect(fake.requests).toHaveLength(1);

    // Call 2, inside the 5 s window: straight to the provider.
    expect(await timed(() => fake.client.chat.completions.create(ask), 50)).toBeLessThanOrEqual(10);
    expect(tc.state.proxyRequests).toBe(3);

    // Call 3, past the window: the probe. One proxy request, no backoff.
    // In 1.8.0 this was five requests and at least 3.75 s of backoff.
    vi.advanceTimersByTime(5_001);
    const probeMs = await timed(() => fake.client.chat.completions.create(ask), 10_000);
    expect(probeMs).not.toBeNull();
    expect(probeMs!).toBeLessThanOrEqual(10);
    expect(tc.state.proxyRequests).toBe(4);
    expect(fake.requests).toHaveLength(3);

    // The circuit is open again for a fresh window: call 4 doesn't probe.
    expect(await timed(() => fake.client.chat.completions.create(ask), 50)).toBeLessThanOrEqual(10);
    expect(tc.state.proxyRequests).toBe(4);
    vi.advanceTimersByTime(4_000);
    await timed(() => fake.client.chat.completions.create(ask), 50);
    expect(tc.state.proxyRequests).toBe(4);

    // And the next window gets exactly one probe again.
    vi.advanceTimersByTime(1_100);
    expect(await timed(() => fake.client.chat.completions.create(ask), 50)).toBeLessThanOrEqual(10);
    expect(tc.state.proxyRequests).toBe(5);
    expect(fake.requests).toHaveLength(6);
  });
});

describe("F1: observations made while TensorCost is down are delivered once it is back", () => {
  const options = { apiKey: "tc", tenantId: TENANT, baseUrl: TC, flushOnExit: false };

  it("12 calls during a 502/503 outage: all 12 delivered once after recovery, no flush needed", async () => {
    const tc = tensorcost({ layer: "observe" });
    globalThis.fetch = tc.f;
    const fake = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(fake.client, options);

    for (let i = 0; i < 12; i += 1) {
      expect(await timed(() => fake.client.chat.completions.create(ask), 1_000)).not.toBeNull();
    }
    expect(fake.requests).toHaveLength(12);
    await vi.advanceTimersByTimeAsync(30_000);
    expect(tc.accepted).toHaveLength(0);
    expect(observationStats()).toMatchObject({ buffered: 12, dropped: 0 });

    tc.state.down = false;
    await vi.advanceTimersByTimeAsync(120_000);
    const ids = tc.accepted.map((o) => o.correlation_id);
    expect(ids).toHaveLength(12);
    expect(new Set(ids).size).toBe(12);
    expect(tc.accepted.every((o) => o.input_tokens === 10 && o.output_tokens === 2)).toBe(true);
    expect(observationStats()).toMatchObject({ buffered: 0, inFlight: 0, dropped: 0, deliveredAfterRetry: 12 });

    const left = flushObservations(options, 1_000);
    await vi.advanceTimersByTimeAsync(20);
    expect(await left).toBe(0);
    expect(tc.accepted).toHaveLength(12);
  });

  it("flushObservations() during the outage returns how many are still waiting", async () => {
    const tc = tensorcost({ layer: "observe" });
    globalThis.fetch = tc.f;
    const fake = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(fake.client, options);
    for (let i = 0; i < 3; i += 1) await timed(() => fake.client.chat.completions.create(ask), 1_000);

    const left = flushObservations(options, 2_000);
    await vi.advanceTimersByTimeAsync(2_100);
    expect(await left).toBe(3);

    tc.state.down = false;
    const after = flushObservations(options, 2_000);
    await vi.advanceTimersByTimeAsync(100);
    expect(await after).toBe(0);
    expect(tc.accepted).toHaveLength(3);
  });
});
