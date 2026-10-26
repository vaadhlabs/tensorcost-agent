/**
 * wrap() against the real provider SDKs, not duck-typed fakes.
 *
 * The fakes have `defaultHeaders` as a plain object. openai@4 has it as a
 * method, so through 1.4.0 the applied-mode wiring (which replaced it with
 * an object) broke every call that reached the client's own request path:
 * the fail-open fallback, and every call below steer. And the real
 * @anthropic-ai/sdk client has a legacy `completions.create`, which made
 * the detector read it as OpenAI. Neither showed up with fakes.
 *
 * The SDKs aren't dependencies of this package. Each suite runs when its
 * variable points at a local copy's `index.mjs`, and skips otherwise:
 *
 *   TENSORCOST_TEST_OPENAI=/path/to/node_modules/openai/index.mjs
 *   TENSORCOST_TEST_ANTHROPIC=/path/to/node_modules/@anthropic-ai/sdk/index.mjs
 *
 * 1.4.1 was checked against openai@4.104.0 and @anthropic-ai/sdk@0.65.0.
 */

import { existsSync } from "node:fs";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { wrap, __test__ as wrapTest } from "../wrap.js";
import { __test__ as anthropicTest } from "../providers/anthropic.js";
import { __test__ as runtimeTest } from "../runtime.js";
import {
  TensorCostBudgetHardCapError,
  TensorCostGatewayRejectedError,
  TensorCostKeyRejectedError,
  TensorCostProxyError,
} from "../errors.js";

const OPENAI = process.env.TENSORCOST_TEST_OPENAI ?? "";
const ANTHROPIC = process.env.TENSORCOST_TEST_ANTHROPIC ?? "";

const TENANT = "11111111-1111-1111-1111-111111111111";
const chatOk = {
  id: "chatcmpl-direct",
  object: "chat.completion",
  choices: [{ index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 1, completion_tokens: 1 },
};
const msgOk = {
  id: "msg_direct",
  type: "message",
  role: "assistant",
  content: [{ type: "text", text: "hi" }],
  model: "claude-sonnet-4-5",
  stop_reason: "end_turn",
  usage: { input_tokens: 1, output_tokens: 1 },
};
const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(body === null ? null : JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });

interface Scenario {
  layer?: string;
  proxy?: () => Response | Promise<Response>;
  admit?: () => Response;
  exchange?: () => Response;
}

function router(scn: Scenario) {
  const calls: { url: string; auth: string | null }[] = [];
  const f = (async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = typeof input === "string" ? input : (input as Request).url ?? String(input);
    const h = new Headers(init.headers ?? {});
    if (!/sdk-layer|observation|exchange/.test(url)) calls.push({ url, auth: h.get("authorization") });
    if (url.includes("/sdk-token/exchange")) return scn.exchange ? scn.exchange() : json(200, { token: "jwt", expires_in: 900 });
    if (url.includes("/v1/sdk-layer")) return json(200, { published_layer: scn.layer ?? "route" });
    if (url.includes("/observation")) return new Response(null, { status: 202 });
    if (url.includes("/v1/admit")) {
      return scn.admit ? scn.admit() : new Response(null, { status: 204, headers: { "x-tc-decision": "layer=govern; action=allow" } });
    }
    if (url.startsWith("http://proxy.test")) return scn.proxy!();
    if (url.startsWith("https://api.openai.com")) return json(200, chatOk);
    if (url.startsWith("https://api.anthropic.com")) return json(200, msgOk);
    throw new TypeError(`fetch failed: unrouted ${url}`);
  }) as typeof fetch;
  return { f, calls };
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

describe.skipIf(!OPENAI || !existsSync(OPENAI))("a real openai@4 client", () => {
  async function client(scn: Scenario, opts: Record<string, unknown> = {}) {
    const { default: OpenAI } = (await import(OPENAI)) as { default: new (o: unknown) => any };
    const { f, calls } = router(scn);
    globalThis.fetch = f;
    const c = new OpenAI({ apiKey: "sk-provider", fetch: f, maxRetries: 0 });
    wrap(c, {
      apiKey: "tc",
      tenantId: TENANT,
      baseUrl: "http://api.tc.test",
      proxyUrl: "http://proxy.test",
      maxLayer: "route",
      retry: { maxAttempts: 1, baseDelayMs: 1, maxDelayMs: 1 },
      ...opts,
    });
    const create = () =>
      c.chat.completions.create({ model: "gpt-4o", messages: [{ role: "user", content: "x" }] });
    return { c, calls, create, OpenAI: OpenAI as any };
  }

  it("wrap() leaves baseURL and the defaultHeaders method alone", async () => {
    const { c } = await client({ proxy: () => json(200, chatOk) });
    expect(c.baseURL).toBe("https://api.openai.com/v1");
    expect(typeof c.defaultHeaders).toBe("function");
  });

  it.each([
    ["proxy 503", () => new Response("<html>503</html>", { status: 503 })],
    ["proxy unreachable", () => Promise.reject(new TypeError("fetch failed"))],
    ["proxy 429", () => new Response(null, { status: 429 })],
    ["proxy 402 (plan ceiling)", () => json(402, { kind: "scale-cogs-exceeded", error: "Payment Required" })],
  ])("%s with fail-open on: falls back to api.openai.com with the provider key", async (_n, proxy) => {
    const { create, calls } = await client({ proxy });
    const r = await create();
    expect(r.choices[0].message.content).toBe("hi");
    const direct = calls.filter((c) => c.url.startsWith("https://api.openai.com"));
    expect(direct).toHaveLength(1);
    expect(direct[0]!.url).toBe("https://api.openai.com/v1/chat/completions");
    expect(direct[0]!.auth).toBe("Bearer sk-provider");
  });

  it("token exchange 503 with fail-open on: falls back too", async () => {
    const { create, calls } = await client({ proxy: () => json(200, chatOk), exchange: () => json(503, {}) });
    await create();
    expect(calls.map((c) => c.url)).toEqual(["https://api.openai.com/v1/chat/completions"]);
  });

  it("a proxy 503 with fail-open off throws TensorCostProxyError and never calls OpenAI", async () => {
    const { create, calls } = await client({ proxy: () => json(503, {}) }, { failOpenEnabled: false });
    await expect(create()).rejects.toBeInstanceOf(TensorCostProxyError);
    expect(calls.some((c) => c.url.startsWith("https://api.openai.com"))).toBe(false);
  });

  it.each(["govern", "observe"])("route-configured, published layer %s: the call works", async (layer) => {
    const { create, calls } = await client({ layer, proxy: () => json(200, chatOk) });
    const r = await create();
    expect(r.id).toBe("chatcmpl-direct");
    expect(calls.some((c) => c.url.startsWith("http://proxy.test"))).toBe(false);
  });

  it("a relayed provider 400 is openai's BadRequestError, built by the client", async () => {
    const body = { error: { message: "bad", type: "invalid_request_error", param: "messages", code: "invalid_value" } };
    const { create, calls, OpenAI } = await client({ proxy: () => json(400, body, { "x-request-id": "req_1" }) });
    const err = await create().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(OpenAI.BadRequestError);
    expect(err).toBeInstanceOf(OpenAI.APIError);
    expect(err).toMatchObject({ status: 400, code: "invalid_value", type: "invalid_request_error", param: "messages" });
    expect((err as { tensorcost?: unknown }).tensorcost).toBeDefined();
    expect(calls.filter((c) => c.url.startsWith("https://api.openai.com"))).toHaveLength(0);
  });

  it("a relayed provider 401 is openai's AuthenticationError", async () => {
    const { create, OpenAI } = await client({
      proxy: () => json(401, { error: { message: "Incorrect API key", type: "invalid_request_error", code: "invalid_api_key" } }),
    });
    await expect(create()).rejects.toBeInstanceOf(OpenAI.AuthenticationError);
  });

  it("a money-budget hard cap is terminal: one proxy call, no OpenAI call", async () => {
    const { create, calls } = await client({
      proxy: () =>
        json(403, { error: { code: "PERIOD_BUDGET_EXCEEDED", budget_code: "BUDGET_HARD_CAP", budget_id: "b-1", retryable: false } }),
    });
    await expect(create()).rejects.toBeInstanceOf(TensorCostBudgetHardCapError);
    expect(calls.map((c) => c.url)).toEqual(["http://proxy.test/api/inference-proxy/v1/chat/completions"]);
  });

  it("a WAF's HTML 403 is TensorCostGatewayRejectedError, not an OpenAI error", async () => {
    const { create, OpenAI } = await client({
      proxy: () => new Response("<html><body>403 Forbidden</body></html>", { status: 403, headers: { "content-type": "text/html" } }),
    });
    const err = await create().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(TensorCostGatewayRejectedError);
    expect(err).not.toBeInstanceOf(OpenAI.APIError);
  });

  it("a revoked TensorCost key is TensorCostKeyRejectedError, terminal", async () => {
    const { create, calls } = await client({ proxy: () => json(200, chatOk), exchange: () => json(401, { message: "revoked" }) });
    await expect(create()).rejects.toBeInstanceOf(TensorCostKeyRejectedError);
    expect(calls).toHaveLength(0);
  });
});

describe.skipIf(!ANTHROPIC || !existsSync(ANTHROPIC))("a real @anthropic-ai/sdk client", () => {
  async function client(scn: Scenario, opts: Record<string, unknown> = {}) {
    const { default: Anthropic } = (await import(ANTHROPIC)) as { default: new (o: unknown) => any };
    const { f, calls } = router(scn);
    globalThis.fetch = f;
    const c = new Anthropic({ apiKey: "sk-ant", fetch: f, maxRetries: 0 });
    wrap(c, {
      apiKey: "tc",
      tenantId: TENANT,
      baseUrl: "http://api.tc.test",
      proxyUrl: "http://proxy.test",
      maxLayer: "route",
      ...opts,
    });
    const create = () =>
      c.messages.create({ model: "claude-sonnet-4-5", max_tokens: 8, messages: [{ role: "user", content: "x" }] });
    return { c, calls, create };
  }

  it("is detected as Anthropic, pinned to govern, admitted, then sent to api.anthropic.com", async () => {
    const { c, create, calls } = await client({ proxy: () => json(500, {}) });
    expect(c.baseURL).toBe("https://api.anthropic.com");
    const r = await create();
    expect(r.id).toBe("msg_direct");
    expect(calls.map((x) => x.url)).toEqual([
      "http://api.tc.test/api/inference-proxy/v1/admit",
      "https://api.anthropic.com/v1/messages",
    ]);
    expect(warn.mock.calls.map((a) => String(a[0])).filter((m) => m.includes("Anthropic"))).toHaveLength(1);
  });

  it("a money-budget hard cap on /admit stops the call before Anthropic", async () => {
    const { create, calls } = await client({
      admit: () =>
        json(
          403,
          { error: { code: "PERIOD_BUDGET_EXCEEDED", budget_code: "BUDGET_HARD_CAP", reason: "spend_not_known_yet", retryable: true, budget_id: "b-2" } },
          { "retry-after": "5" },
        ),
    });
    const err = await create().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(TensorCostBudgetHardCapError);
    expect(err).toMatchObject({ retryable: true, retryAfterSeconds: 5 });
    expect(calls.map((x) => x.url)).toEqual(["http://api.tc.test/api/inference-proxy/v1/admit"]);
  });
});
