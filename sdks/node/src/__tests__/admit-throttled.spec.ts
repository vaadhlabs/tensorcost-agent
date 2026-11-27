/**
 * A bare 429 from TensorCost is pacing, not an outage.
 *
 * Found by a self-hosted load test: the proxy's admit limit was 600/min for
 * the install's one tenant, and every call over it went to the provider with
 * no budget or governance check, marked `admit_unavailable` like an outage
 * and logged nowhere. Now /admit retries a 429 inside its own budget,
 * honouring Retry-After (also for the calls after it), and when it is still
 * throttled the configured fail mode applies exactly as for an outage, with
 * its own reason on the observation and the fall-back event, a warning at
 * most once a minute, and a per-process count in uncheckedCalls().
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { admitRequest, __test__ as admitTest } from "../admit-client.js";
import { TensorCostProxyError, TensorCostQuotaError } from "../errors.js";
import { ObservationTransport } from "../transport.js";
import * as openaiProvider from "../providers/openai.js";
import * as anthropicProvider from "../providers/anthropic.js";
import { uncheckedCalls, __test__ as uncheckedTest } from "../unchecked.js";
import type { LifecycleEvent } from "../telemetry.js";
import { defaultHardening } from "./test-harness.js";

const META = {
  provider: "openai" as const,
  model: "gpt-4o-mini",
  operation: "chat.completions" as const,
  correlation_id: "c-1",
};

const bare = (status: number, headers: Record<string, string> = {}) =>
  new Response(status === 204 ? null : "", { status, headers });

/** /v1/admit answers each call from `script` in turn; the last one repeats. */
function admitScript(script: Response[]) {
  const at: number[] = [];
  const impl = vi.fn(async () => {
    at.push(Date.now());
    const next = script.length > 1 ? script.shift()! : script[0]!;
    return next.clone();
  });
  return { impl: impl as unknown as typeof fetch, at, count: () => impl.mock.calls.length };
}

const admit = (fetchImpl: typeof fetch, extra: Record<string, unknown> = {}) =>
  admitRequest({
    baseUrl: "http://tc.local",
    bearerToken: "tok",
    maxLayer: "govern",
    metadata: META,
    fetchImpl,
    ...extra,
  });

let warn: ReturnType<typeof vi.spyOn>;
beforeEach(() => {
  admitTest.reset();
  uncheckedTest.reset();
  warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
});
afterEach(() => {
  warn.mockRestore();
  vi.unstubAllGlobals();
});

describe("admitRequest on a 429", () => {
  it("retries a 429 without Retry-After and is admitted when the retry succeeds", async () => {
    const s = admitScript([bare(429), bare(204)]);
    const res = await admit(s.impl);
    expect(res).toEqual({ allowed: true, decisionHeader: null });
    expect(s.count()).toBe(2);
  });

  it("waits out a Retry-After that fits in the budget, then is admitted", async () => {
    const s = admitScript([bare(429, { "retry-after": "0.15" }), bare(204)]);
    const res = await admit(s.impl);
    expect(res.allowed).toBe(true);
    expect(res.unavailable).toBeUndefined();
    expect(s.count()).toBe(2);
    expect(s.at[1]! - s.at[0]!).toBeGreaterThanOrEqual(145);
  });

  it("gives up at once when Retry-After is past the budget, and says throttled", async () => {
    const s = admitScript([bare(429, { "retry-after": "2" })]);
    const t0 = Date.now();
    const res = await admit(s.impl);
    expect(Date.now() - t0).toBeLessThan(100);
    expect(res).toEqual({
      allowed: true,
      decisionHeader: null,
      unavailable: true,
      reason: "admit_throttled",
    });
    expect(s.count()).toBe(1);
  });

  it("honours that Retry-After on the next calls too, without sending them", async () => {
    const s = admitScript([bare(429, { "retry-after": "2" })]);
    await admit(s.impl);
    const again = await admit(s.impl);
    expect(again.reason).toBe("admit_throttled");
    expect(s.count()).toBe(1);
  });

  it("keeps retrying inside the budget when every answer is a bare 429", async () => {
    const s = admitScript([bare(429)]);
    const t0 = Date.now();
    const res = await admit(s.impl, { timeoutMs: 300 });
    expect(Date.now() - t0).toBeLessThanOrEqual(330);
    expect(s.count()).toBeGreaterThan(2);
    expect(res.reason).toBe("admit_throttled");
  });

  it("does not spin on Retry-After: 0", async () => {
    const s = admitScript([bare(429, { "retry-after": "0" })]);
    await admit(s.impl, { timeoutMs: 300 });
    expect(s.count()).toBeLessThan(10);
  });

  it("raises the 429 when fail-open is off", async () => {
    const s = admitScript([bare(429, { "retry-after": "5" })]);
    const err = await admit(s.impl, { failOpen: false }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(TensorCostQuotaError);
    expect((err as TensorCostQuotaError).status).toBe(429);
    expect((err as TensorCostQuotaError).retryAfterMs).toBe(5_000);
  });

  it("does not retry a bare 402 (the plan ceiling) and names it", async () => {
    const s = admitScript([bare(402)]);
    const res = await admit(s.impl);
    expect(res.reason).toBe("admit_plan_limit");
    expect(s.count()).toBe(1);
  });

  it("a 429 with a refusal code is still a refusal, not pacing", async () => {
    const s = admitScript([
      new Response(JSON.stringify({ error: { code: "RATE_LIMITED_BY_POLICY", message: "no" } }), {
        status: 429,
        headers: { "content-type": "application/json" },
      }),
    ]);
    const res = await admit(s.impl);
    expect(res.allowed).toBe(false);
    expect(s.count()).toBe(1);
  });

  it("an outage still says admit_unavailable", async () => {
    const res = await admit((async () => {
      throw new Error("ECONNREFUSED");
    }) as unknown as typeof fetch);
    expect(res.reason).toBe("admit_unavailable");
  });
});

/** Token exchange, observations, /v1/admit and the proxy, routed by URL. */
function tensorcost(opts: { admit?: () => Response; proxy?: () => Response }) {
  const posted: Record<string, any>[] = [];
  const impl = vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.includes("sdk-token/exchange")) {
      return new Response(JSON.stringify({ token: "tok", expires_in: 900 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    if (url.includes("/v1/admit")) return (opts.admit ?? (() => bare(204)))();
    if (url.includes("/inference-proxy/observation")) {
      posted.push(JSON.parse(String(init.body)));
      return new Response(null, { status: 202 });
    }
    if (url.startsWith("http://proxy.local")) return (opts.proxy ?? (() => bare(503)))();
    throw new TypeError(`unrouted ${url}`);
  });
  vi.stubGlobal("fetch", impl);
  const transport = new ObservationTransport({
    tenantId: "t-1",
    baseUrl: "http://api.local",
    apiKey: "k",
    fetchImpl: impl as unknown as typeof fetch,
  });
  return { transport, posted };
}

const layer = (l: "govern" | "route") =>
  ({ effectiveLayer: async () => l, getComplianceSnapshot: async () => null }) as never;

function openaiClient() {
  const create = vi.fn(async () => ({ id: "x", usage: { prompt_tokens: 3, completion_tokens: 1 } }));
  return { client: { chat: { completions: { create } } }, create };
}

const ask = { model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] };

async function settled(transport: ObservationTransport) {
  await transport.flush(1_000);
}

describe("govern: admit still throttled when its budget runs out", () => {
  it("fail-open: the call goes ahead, marked admit_throttled, counted, warned once", async () => {
    const tc = tensorcost({ admit: () => bare(429, { "retry-after": "30" }) });
    const events: LifecycleEvent[] = [];
    const { client, create } = openaiClient();
    openaiProvider.install(
      client, tc.transport, "t-1", null, null,
      defaultHardening({ maxLayer: "govern", sdkLayer: layer("govern"), onLifecycleEvent: (e) => events.push(e) }),
    );
    for (let i = 0; i < 5; i += 1) await client.chat.completions.create(ask);
    await settled(tc.transport);

    expect(create).toHaveBeenCalledTimes(5);
    expect(tc.posted).toHaveLength(5);
    for (const o of tc.posted) {
      expect(o.metadata).toMatchObject({ decision_action: "allow", decision_reason: "admit_throttled" });
    }
    expect(uncheckedCalls()).toMatchObject({ admit_throttled: 5 });
    const fallbacks = events.filter((e) => e.kind === "on_fallback");
    expect(fallbacks).toHaveLength(5);
    expect(fallbacks.every((e) => (e as { reason?: string }).reason === "admit_throttled")).toBe(true);
    const throttleWarnings = warn.mock.calls.filter((c) => String(c[0]).includes("rate limiting"));
    expect(throttleWarnings).toHaveLength(1);
    expect(String(throttleWarnings[0]![0])).toContain("without a budget or governance check");
  });

  it("fail-open off: the call is refused with the 429 and the provider is never called", async () => {
    const tc = tensorcost({ admit: () => bare(429) });
    const { client, create } = openaiClient();
    openaiProvider.install(
      client, tc.transport, "t-1", null, null,
      defaultHardening({ maxLayer: "govern", sdkLayer: layer("govern"), failOpenEnabled: false }),
    );
    const err = await client.chat.completions.create(ask).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(TensorCostQuotaError);
    expect(create).not.toHaveBeenCalled();
    expect(uncheckedCalls().admit_throttled).toBe(0);
  });

  it("anthropic behaves the same", async () => {
    const tc = tensorcost({ admit: () => bare(429, { "retry-after": "30" }) });
    const create = vi.fn(async () => ({ id: "x", usage: { input_tokens: 1, output_tokens: 1 } }));
    const client = { messages: { create } };
    anthropicProvider.install(
      client, tc.transport, "t-1", null, null,
      defaultHardening({ maxLayer: "govern", sdkLayer: layer("govern") }),
    );
    await client.messages.create({ model: "claude-3-5-haiku-latest", max_tokens: 5, messages: [] });
    await settled(tc.transport);
    expect(create).toHaveBeenCalledTimes(1);
    expect(tc.posted[0]!.metadata).toMatchObject({ decision_reason: "admit_throttled" });
    expect(uncheckedCalls().admit_throttled).toBe(1);
  });

  it("an admit outage is counted and recorded as admit_unavailable, not throttled", async () => {
    const tc = tensorcost({ admit: () => bare(503) });
    const { client } = openaiClient();
    openaiProvider.install(
      client, tc.transport, "t-1", null, null,
      defaultHardening({ maxLayer: "govern", sdkLayer: layer("govern") }),
    );
    await client.chat.completions.create(ask);
    await settled(tc.transport);
    expect(tc.posted[0]!.metadata).toMatchObject({ decision_reason: "admit_unavailable" });
    expect(uncheckedCalls()).toMatchObject({ admit_unavailable: 1, admit_throttled: 0 });
  });
});

describe("route: the proxy still throttled after its retries", () => {
  const routeHardening = (extra: Record<string, unknown> = {}) =>
    defaultHardening({
      maxLayer: "route",
      proxyUrl: "http://proxy.local",
      sdkLayer: layer("route"),
      retry: { maxAttempts: 2, baseDelayMs: 1, maxDelayMs: 5 },
      ...extra,
    });

  it("fail-open: goes direct, marked proxy_throttled on the observation and the event", async () => {
    const tc = tensorcost({ proxy: () => bare(429, { "retry-after": "0" }) });
    const events: LifecycleEvent[] = [];
    const { client, create } = openaiClient();
    openaiProvider.install(
      client, tc.transport, "t-1", null, null,
      routeHardening({ onLifecycleEvent: (e: LifecycleEvent) => events.push(e) }),
    );
    await client.chat.completions.create(ask);
    await settled(tc.transport);
    expect(create).toHaveBeenCalledTimes(1);
    expect(tc.posted[0]!.metadata).toMatchObject({
      decision_action: "allow",
      decision_reason: "proxy_throttled",
    });
    const fb = events.find((e) => e.kind === "on_fallback") as { reason?: string } | undefined;
    expect(fb?.reason).toBe("proxy_throttled");
    expect(uncheckedCalls().proxy_throttled).toBe(1);
    expect(warn.mock.calls.some((c) => String(c[0]).includes("proxy is rate limiting"))).toBe(true);
  });

  it("other proxy fall-backs are named too", async () => {
    const tc = tensorcost({ proxy: () => bare(503, { "x-tc-error-origin": "gateway" }) });
    const { client } = openaiClient();
    openaiProvider.install(client, tc.transport, "t-1", null, null, routeHardening());
    await client.chat.completions.create(ask);
    await settled(tc.transport);
    expect(tc.posted[0]!.metadata).toMatchObject({ decision_reason: "proxy_unavailable" });
    expect(uncheckedCalls().proxy_unavailable).toBe(1);
  });

  it("fail-open off: the throttled call raises and is not sent direct", async () => {
    const tc = tensorcost({ proxy: () => bare(429, { "retry-after": "0" }) });
    const { client, create } = openaiClient();
    openaiProvider.install(client, tc.transport, "t-1", null, null, routeHardening({ failOpenEnabled: false }));
    const err = await client.chat.completions.create(ask).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(TensorCostProxyError);
    expect(String((err as Error).message)).toContain("429");
    expect(create).not.toHaveBeenCalled();
    expect(uncheckedCalls().proxy_throttled).toBe(0);
  });
});
