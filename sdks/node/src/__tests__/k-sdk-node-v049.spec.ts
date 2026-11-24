/**
 * Findings from the Zelvora BYOC run of 1.7.0 (matrix evidence K, Nov 23):
 *
 *   1. Route mode sent openai-node's default baseURL ("…/v1") as
 *      x-tc-provider-url and the proxy 404'd every call; the 404 carried
 *      x-tc-decision but was reported as a firewall rejection.
 *   2. The wrapped create() returned a plain Promise, so withResponse() and
 *      asResponse() threw.
 *   3. Streams were recorded with 0 tokens, usage in the stream or not.
 *   4. Circuit-breaker transitions were posted as observations and stored
 *      as spend events.
 *   5. With TensorCost unreachable every call paid two 2 s token exchanges.
 *
 * Every test goes through wrap(), with fetch routed per host, so the real
 * transport, sdk-layer client and circuit run.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { wrap, __test__ as wrapTest } from "../wrap.js";
import { prewarm } from "../prewarm.js";
import { __test__ as runtimeTest } from "../runtime.js";
import { __test__ as anthropicTest } from "../providers/anthropic.js";
import { providerTarget } from "../applied.js";
import { TensorCostTimeoutError } from "../errors.js";
import {
  TensorCostGatewayRejectedError,
  TensorCostProxyStatusError,
} from "../errors.js";
import { FakeAPIPromise, fakeApiClient, jsonResponse, sseResponse } from "./fake-api-promise.js";

const TENANT = "11111111-1111-1111-1111-111111111111";
const TC = "http://api.tc.test";
const PROXY = "http://proxy.test";

const chatOk = {
  id: "chatcmpl-1",
  object: "chat.completion",
  choices: [{ index: 0, message: { role: "assistant", content: "Ok." }, finish_reason: "stop" }],
  usage: { prompt_tokens: 10, completion_tokens: 2 },
};
const msgOk = {
  id: "msg_1",
  type: "message",
  role: "assistant",
  content: [{ type: "text", text: "Ok." }],
  model: "claude-haiku-4-5",
  stop_reason: "end_turn",
  usage: { input_tokens: 10, output_tokens: 5 },
};

interface Scenario {
  layer?: string;
  proxy?: (init: RequestInit) => Response | Promise<Response>;
  admit?: () => Response | Promise<Response>;
  /** TensorCost answers nothing: every request to it hangs until aborted. */
  blackhole?: boolean;
}

type Obs = Record<string, any>;

function route(scn: Scenario) {
  const observations: Obs[] = [];
  const proxyHeaders: Headers[] = [];
  const f = (async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = typeof input === "string" ? input : (input as Request).url ?? String(input);
    if (url.startsWith(TC) || url.startsWith(PROXY)) {
      if (scn.blackhole) {
        return new Promise<Response>((_resolve, reject) => {
          init.signal?.addEventListener("abort", () => {
            const e = new Error("aborted");
            e.name = "AbortError";
            reject(e);
          });
        });
      }
    }
    if (url.includes("/sdk-token/exchange")) return jsonResponse(200, { token: "jwt", expires_in: 900 });
    if (url.includes("/v1/sdk-layer")) return jsonResponse(200, { published_layer: scn.layer ?? "route" });
    if (url.includes("/inference-proxy/observation")) {
      observations.push(JSON.parse(String(init.body)) as Obs);
      return new Response(null, { status: 202 });
    }
    if (url.includes("/v1/admit")) {
      return scn.admit
        ? scn.admit()
        : new Response(null, { status: 204, headers: { "x-tc-decision": "layer=govern; action=allow" } });
    }
    if (url.startsWith(PROXY)) {
      proxyHeaders.push(new Headers(init.headers ?? {}));
      return scn.proxy!(init);
    }
    throw new TypeError(`fetch failed: unrouted ${url}`);
  }) as typeof fetch;
  return { f, observations, proxyHeaders };
}

const savedFetch = globalThis.fetch;
let warn: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  wrapTest.resetWarnings();
  anthropicTest.resetPinWarning();
  runtimeTest.clearRuntimes();
  warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
});
afterEach(() => {
  globalThis.fetch = savedFetch;
  warn.mockRestore();
});

async function settle(ms = 20) {
  await new Promise((r) => setTimeout(r, ms));
}

function options(extra: Record<string, unknown> = {}) {
  return {
    apiKey: "tc",
    tenantId: TENANT,
    baseUrl: TC,
    retry: { maxAttempts: 1, baseDelayMs: 1, maxDelayMs: 1 },
    ...extra,
  };
}

function routeOptions(extra: Record<string, unknown> = {}) {
  return options({ maxLayer: "route", proxyUrl: PROXY, ...extra });
}

const ask = { model: "gpt-4.1-mini", max_tokens: 20, messages: [{ role: "user", content: "Say ok." }] };
const askClaude = { model: "claude-haiku-4-5", max_tokens: 20, messages: [{ role: "user", content: "Say ok." }] };

// ---------------------------------------------------------------------------
// 1. Route mode: provider URL and the 404 the proxy answered
// ---------------------------------------------------------------------------

describe("route mode: x-tc-provider-url is the provider's base without the API version", () => {
  it.each([
    ["https://api.openai.com/v1", "https://api.openai.com"],
    ["https://api.openai.com/v1/", "https://api.openai.com"],
    ["https://my-res.openai.azure.com/openai/v1", "https://my-res.openai.azure.com/openai"],
    [
      "https://my-res.openai.azure.com/openai/deployments/gpt-4o",
      "https://my-res.openai.azure.com/openai/deployments/gpt-4o",
    ],
    ["https://gateway.internal/llm", "https://gateway.internal/llm"],
  ])("client baseURL %s -> %s", (baseURL, want) => {
    expect(providerTarget({ baseURL, apiKey: "k" }, "openai", "x").providerUrl).toBe(want);
  });

  it("a hand-set x-tc-provider-url ending in /v1 is sent without it too", () => {
    const c = { baseURL: "https://api.openai.com/v1", apiKey: "k", defaultHeaders: { "x-tc-provider-url": "https://gw.example/v1" } };
    expect(providerTarget(c, "openai", "x").providerUrl).toBe("https://gw.example");
  });

  it("a default OpenAI client's call reaches the proxy with the bare host", async () => {
    const { f, proxyHeaders } = route({ proxy: () => jsonResponse(200, chatOk) });
    globalThis.fetch = f;
    const { client } = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(client, routeOptions());
    await client.chat.completions.create(ask);
    expect(proxyHeaders).toHaveLength(1);
    expect(proxyHeaders[0]!.get("x-tc-provider-url")).toBe("https://api.openai.com");
  });
});

describe("a 4xx that carries x-tc-decision came from TensorCost, not a firewall", () => {
  const decision = "layer=route; action=pass_through; reason=no_policy_match";

  it("proxy 404 with body null and x-tc-decision: TensorCostProxyStatusError naming the proxy", async () => {
    const { f } = route({
      proxy: () =>
        new Response("null", { status: 404, headers: { "content-type": "application/json", "x-tc-decision": decision } }),
    });
    globalThis.fetch = f;
    const { client } = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(client, routeOptions());
    const err = (await client.chat.completions.create(ask).catch((e: unknown) => e)) as Error;
    expect(err).toBeInstanceOf(TensorCostProxyStatusError);
    expect(err).not.toBeInstanceOf(TensorCostGatewayRejectedError);
    expect(err.message).toContain("TensorCost proxy");
    expect(err.message).toContain(decision);
    expect(err.message).not.toMatch(/firewall|load balancer/);
    expect((err as TensorCostProxyStatusError).status).toBe(404);
    expect((err as TensorCostProxyStatusError).decision).toBe(decision);
  });

  it("the same for a streamed call", async () => {
    const { f } = route({
      proxy: () => new Response(null, { status: 404, headers: { "x-tc-decision": decision } }),
    });
    globalThis.fetch = f;
    const { client } = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(client, routeOptions());
    const err = await client.chat.completions.create({ ...ask, stream: true }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(TensorCostProxyStatusError);
  });

  it("an HTML 403 with no x-tc-decision is still a gateway rejection", async () => {
    const { f } = route({
      proxy: () => new Response("<html>403</html>", { status: 403, headers: { "content-type": "text/html" } }),
    });
    globalThis.fetch = f;
    const { client } = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(client, routeOptions());
    await expect(client.chat.completions.create(ask)).rejects.toBeInstanceOf(TensorCostGatewayRejectedError);
  });

  it("/admit 404 with no body and x-tc-decision: TensorCostProxyStatusError too", async () => {
    const { f } = route({
      layer: "govern",
      admit: () => new Response(null, { status: 404, headers: { "x-tc-decision": "layer=govern; action=allow" } }),
    });
    globalThis.fetch = f;
    const { client } = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(client, options({ maxLayer: "govern" }));
    await expect(client.chat.completions.create(ask)).rejects.toBeInstanceOf(TensorCostProxyStatusError);
  });
});

// ---------------------------------------------------------------------------
// 2. create() returns an APIPromise: withResponse / asResponse work and record
// ---------------------------------------------------------------------------

describe("observe: the wrapped create() returns the provider's own APIPromise", () => {
  async function setup(shape: "openai" | "anthropic" = "openai") {
    const r = route({ layer: "observe" });
    globalThis.fetch = r.f;
    const body = shape === "openai" ? chatOk : msgOk;
    const fake = fakeApiClient(() => jsonResponse(200, body, { "x-request-id": "req_1" }), shape);
    wrap(fake.client, options());
    await prewarm(options());
    const create =
      shape === "openai"
        ? () => fake.client.chat.completions.create(ask)
        : () => fake.client.messages.create(askClaude);
    return { ...r, ...fake, create };
  }

  it("is an instance of the client's APIPromise class", async () => {
    const { create } = await setup();
    const p = create();
    expect(p).toBeInstanceOf(FakeAPIPromise);
    expect(typeof p.withResponse).toBe("function");
    await p;
  });

  it("await: the body, recorded once with its tokens", async () => {
    const { create, observations } = await setup();
    const r = await create();
    expect(r.id).toBe("chatcmpl-1");
    await settle();
    expect(observations).toHaveLength(1);
    expect(observations[0]).toMatchObject({ status: "success", input_tokens: 10, output_tokens: 2 });
  });

  it("withResponse(): data, response and request id, recorded once", async () => {
    const { create, observations } = await setup();
    const { data, response, request_id } = await create().withResponse();
    expect(data.id).toBe("chatcmpl-1");
    expect(response.status).toBe(200);
    expect(request_id).toBe("req_1");
    await settle();
    expect(observations).toHaveLength(1);
    expect(observations[0]).toMatchObject({ input_tokens: 10, output_tokens: 2 });
  });

  it("asResponse(): the raw Response, still recorded with its tokens", async () => {
    const { create, observations } = await setup();
    const response = await create().asResponse();
    expect(response.status).toBe(200);
    expect((await response.json()).id).toBe("chatcmpl-1");
    await settle();
    expect(observations).toHaveLength(1);
    expect(observations[0]).toMatchObject({ input_tokens: 10, output_tokens: 2 });
  });

  it("a provider error is recorded once and reaches the caller", async () => {
    const r = route({ layer: "observe" });
    globalThis.fetch = r.f;
    const fake = fakeApiClient(() => jsonResponse(400, { error: { message: "bad" } }));
    wrap(fake.client, options());
    await prewarm(options());
    const err = await fake.client.chat.completions.create(ask).withResponse().catch((e: unknown) => e);
    expect((err as { status?: number }).status).toBe(400);
    await settle();
    expect(r.observations).toHaveLength(1);
    expect(r.observations[0]).toMatchObject({ status: "error" });
  });

  it("Anthropic: messages.create() keeps its APIPromise, withResponse works and records", async () => {
    const { create, observations } = await setup("anthropic");
    const p = create();
    expect(p).toBeInstanceOf(FakeAPIPromise);
    const { data } = await p.withResponse();
    expect(data.id).toBe("msg_1");
    await settle();
    expect(observations).toHaveLength(1);
    expect(observations[0]).toMatchObject({ provider: "anthropic", input_tokens: 10, output_tokens: 5 });
  });

  it("the first call, before the SDK has the published layer, still has withResponse", async () => {
    const r = route({ layer: "observe" });
    globalThis.fetch = r.f;
    const fake = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(fake.client, options());
    const { data, response } = await fake.client.chat.completions.create(ask).withResponse();
    expect(data.id).toBe("chatcmpl-1");
    expect(response.status).toBe(200);
    await settle();
    expect(r.observations).toHaveLength(1);
  });
});

describe("govern and route: create() still has withResponse() and asResponse()", () => {
  it("govern: withResponse() is the provider's response, recorded once", async () => {
    const r = route({ layer: "govern" });
    globalThis.fetch = r.f;
    const fake = fakeApiClient(() => jsonResponse(200, chatOk, { "x-request-id": "req_g" }));
    wrap(fake.client, options({ maxLayer: "govern" }));
    const { data, request_id } = await fake.client.chat.completions.create(ask).withResponse();
    expect(data.id).toBe("chatcmpl-1");
    expect(request_id).toBe("req_g");
    await settle();
    expect(r.observations).toHaveLength(1);
    expect(r.observations[0]).toMatchObject({ input_tokens: 10, output_tokens: 2 });
  });

  it("route: withResponse() is the proxy's response with x-tc-decision; asResponse() reads", async () => {
    const r = route({
      proxy: () => jsonResponse(200, chatOk, { "x-tc-decision": "layer=route; action=pass_through", "x-request-id": "req_p" }),
    });
    globalThis.fetch = r.f;
    const fake = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(fake.client, routeOptions());
    const { data, response, request_id } = await fake.client.chat.completions.create(ask).withResponse();
    expect(data.id).toBe("chatcmpl-1");
    expect(response.headers.get("x-tc-decision")).toContain("pass_through");
    expect(request_id).toBe("req_p");
    const raw = await fake.client.chat.completions.create(ask).asResponse();
    expect((await raw.json()).id).toBe("chatcmpl-1");
    await settle();
    expect(r.observations).toHaveLength(2);
    expect(fake.requests).toHaveLength(0);
  });

  it("route fail-open: the direct call's withResponse() works", async () => {
    // The gateway's own 503 (its breaker is open): provably never forwarded.
    // An unmarked 503 (a load balancer's HTML page, say) no longer falls
    // back: something in front of TensorCost may have reset mid-call.
    const r = route({
      proxy: () => new Response("{}", { status: 503, headers: { "x-tc-error-origin": "gateway" } }),
    });
    globalThis.fetch = r.f;
    const fake = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(fake.client, routeOptions());
    const { data } = await fake.client.chat.completions.create(ask).withResponse();
    expect(data.id).toBe("chatcmpl-1");
    expect(fake.requests).toHaveLength(1);
  });

  it("_thenUnwrap() works on the TensorCost APIPromise (openai's own helpers call it)", async () => {
    const r = route({ proxy: () => jsonResponse(200, chatOk) });
    globalThis.fetch = r.f;
    const fake = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(fake.client, routeOptions());
    const p = fake.client.chat.completions.create(ask);
    const ids = await p._thenUnwrap((d: { id: string }) => d.id.toUpperCase());
    expect(ids).toBe("CHATCMPL-1");
  });
});

// ---------------------------------------------------------------------------
// 3. Streaming usage
// ---------------------------------------------------------------------------

describe("streams are recorded with the usage they carry, when they end", () => {
  const chunks = [
    { id: "c", object: "chat.completion.chunk", choices: [{ index: 0, delta: { role: "assistant", content: "" } }], usage: null },
    { id: "c", object: "chat.completion.chunk", choices: [{ index: 0, delta: { content: "Ok" } }], usage: null },
    { id: "c", object: "chat.completion.chunk", choices: [{ index: 0, delta: { content: "." } }], usage: null },
    { id: "c", object: "chat.completion.chunk", choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: null },
    { id: "c", object: "chat.completion.chunk", choices: [], usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 } },
  ];

  it("observe, OpenAI stream_options.include_usage: 10/2 after the last chunk, nothing before", async () => {
    const r = route({ layer: "observe" });
    globalThis.fetch = r.f;
    const fake = fakeApiClient(() => sseResponse(chunks));
    wrap(fake.client, options());
    await prewarm(options());
    const stream = await fake.client.chat.completions.create({ ...ask, stream: true, stream_options: { include_usage: true } });
    await settle();
    expect(r.observations).toHaveLength(0);
    let n = 0;
    let text = "";
    for await (const c of stream) {
      n += 1;
      text += c.choices?.[0]?.delta?.content ?? "";
    }
    expect(n).toBe(5);
    expect(text).toBe("Ok.");
    await settle();
    expect(r.observations).toHaveLength(1);
    expect(r.observations[0]).toMatchObject({ status: "success", input_tokens: 10, output_tokens: 2 });
  });

  it("observe, stream read through withResponse(): recorded once with usage", async () => {
    const r = route({ layer: "observe" });
    globalThis.fetch = r.f;
    const fake = fakeApiClient(() => sseResponse(chunks));
    wrap(fake.client, options());
    await prewarm(options());
    const { data } = await fake.client.chat.completions.create({ ...ask, stream: true }).withResponse();
    for await (const _ of data) void _;
    await settle();
    expect(r.observations).toHaveLength(1);
    expect(r.observations[0]).toMatchObject({ input_tokens: 10, output_tokens: 2 });
  });

  it("a stream the caller stops reading is recorded once, when it stops", async () => {
    const r = route({ layer: "observe" });
    globalThis.fetch = r.f;
    const fake = fakeApiClient(() => sseResponse(chunks));
    wrap(fake.client, options());
    await prewarm(options());
    const stream = await fake.client.chat.completions.create({ ...ask, stream: true });
    for await (const _ of stream) break;
    await settle();
    expect(r.observations).toHaveLength(1);
    expect(r.observations[0]).toMatchObject({ status: "success" });
  });

  it("observe, Anthropic: input from message_start, output from message_delta", async () => {
    const r = route({ layer: "observe" });
    globalThis.fetch = r.f;
    const events = [
      { type: "message_start", message: { id: "msg_1", usage: { input_tokens: 12, output_tokens: 1 } } },
      { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
      { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Ok." } },
      { type: "content_block_stop", index: 0 },
      { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 4 } },
      { type: "message_stop" },
    ];
    const fake = fakeApiClient(() => sseResponse(events), "anthropic");
    wrap(fake.client, options());
    await prewarm(options());
    const stream = await fake.client.messages.create({ ...askClaude, stream: true });
    for await (const _ of stream) void _;
    await settle();
    expect(r.observations).toHaveLength(1);
    expect(r.observations[0]).toMatchObject({ provider: "anthropic", input_tokens: 12, output_tokens: 4 });
  });

  it("route, OpenAI stream through the proxy: the usage chunk is recorded", async () => {
    const r = route({ proxy: () => sseResponse(chunks) });
    globalThis.fetch = r.f;
    const fake = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(fake.client, routeOptions());
    const stream = await fake.client.chat.completions.create({ ...ask, stream: true, stream_options: { include_usage: true } });
    for await (const _ of stream) void _;
    await settle();
    expect(r.observations).toHaveLength(1);
    expect(r.observations[0]).toMatchObject({ input_tokens: 10, output_tokens: 2 });
  });
});

// ---------------------------------------------------------------------------
// 4. Circuit transitions are not spend
// ---------------------------------------------------------------------------

describe("circuit-breaker transitions are not observations", () => {
  it("an outage that opens the circuit posts only the calls, and tells onLifecycleEvent", async () => {
    const r = route({ proxy: () => new Response(null, { status: 503, headers: { "x-tc-error-origin": "proxy" } }) });
    globalThis.fetch = r.f;
    const events: Record<string, unknown>[] = [];
    const fake = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(fake.client, routeOptions({ onLifecycleEvent: (e: Record<string, unknown>) => events.push(e) }));
    for (let i = 0; i < 5; i += 1) await fake.client.chat.completions.create(ask);
    await settle();
    expect(r.observations.filter((o) => o.model === "circuit-breaker")).toHaveLength(0);
    expect(r.observations.filter((o) => o.metadata?.circuit_state !== undefined)).toHaveLength(0);
    expect(r.observations).toHaveLength(5);
    const transitions = events.filter((e) => e.kind === "on_circuit_state");
    expect(transitions[0]).toMatchObject({ from: "CLOSED", to: "OPEN", provider: "openai" });
  });

  it("a govern admit that fails open posts one observation for the call, marked admit_unavailable", async () => {
    const r = route({ layer: "govern", admit: () => new Response(null, { status: 503 }) });
    globalThis.fetch = r.f;
    const fake = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(fake.client, options({ maxLayer: "govern" }));
    await fake.client.chat.completions.create(ask);
    await settle();
    expect(r.observations).toHaveLength(1);
    expect(r.observations[0]).toMatchObject({
      input_tokens: 10,
      output_tokens: 2,
      metadata: { decision_action: "allow", decision_reason: "admit_unavailable" },
    });
  });
});

// ---------------------------------------------------------------------------
// 5. TensorCost unreachable costs the call path its small budget, once
// ---------------------------------------------------------------------------

describe("TensorCost blackholed: the call path pays a bounded wait, once", () => {
  async function time(fn: () => Promise<unknown>): Promise<number> {
    const t = Date.now();
    await fn();
    return Date.now() - t;
  }

  it("observe: first call within the 250 ms layer budget, later calls without waiting", async () => {
    const r = route({ blackhole: true });
    globalThis.fetch = r.f;
    const fake = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(fake.client, options());
    const first = await time(() => fake.client.chat.completions.create(ask));
    const second = await time(() => fake.client.chat.completions.create(ask));
    const third = await time(() => fake.client.chat.completions.create(ask));
    expect(first).toBeLessThan(450);
    expect(second).toBeLessThan(60);
    expect(third).toBeLessThan(60);
    expect(fake.requests).toHaveLength(3);
  });

  it("govern: the token exchange counts against the 500 ms admit budget, and isn't retried per call", async () => {
    const r = route({ blackhole: true, layer: "govern" });
    globalThis.fetch = r.f;
    const fake = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(fake.client, options({ maxLayer: "govern" }));
    const first = await time(() => fake.client.chat.completions.create(ask));
    const second = await time(() => fake.client.chat.completions.create(ask));
    expect(first).toBeLessThan(950);
    expect(second).toBeLessThan(60);
    expect(fake.requests).toHaveLength(2);
  });

  it("route: falls open within its budgets, and the next call doesn't wait", async () => {
    const r = route({ blackhole: true });
    globalThis.fetch = r.f;
    const fake = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(fake.client, routeOptions({ headersTimeoutMs: 400 }));
    const first = await time(() => fake.client.chat.completions.create(ask));
    const second = await time(() => fake.client.chat.completions.create(ask));
    expect(first).toBeLessThan(950);
    expect(second).toBeLessThan(60);
    expect(fake.requests).toHaveLength(2);
  });

  it("route, proxy hung but TensorCost up: once the circuit opens, calls stop waiting on the proxy", async () => {
    const hang = (init: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init.signal?.addEventListener("abort", () => {
          const e = new Error("aborted");
          e.name = "AbortError";
          reject(e);
        });
      });
    const r = route({ proxy: hang });
    globalThis.fetch = r.f;
    const fake = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(fake.client, routeOptions({ headersTimeoutMs: 150, timeoutMs: 150 }));
    await prewarm(routeOptions());
    // Three timeouts open the circuit. Each surfaces: the proxy took the
    // request and may have forwarded it, so going direct could bill it twice.
    for (let i = 0; i < 3; i += 1) {
      await expect(fake.client.chat.completions.create(ask)).rejects.toBeInstanceOf(TensorCostTimeoutError);
    }
    expect(fake.requests).toHaveLength(0);
    // Once open, calls go direct before anything is sent to the proxy.
    const times: number[] = [];
    for (let i = 0; i < 4; i += 1) times.push(await time(() => fake.client.chat.completions.create(ask)));
    expect(Math.max(...times)).toBeLessThan(60);
    expect(fake.requests).toHaveLength(4);
  });
});
