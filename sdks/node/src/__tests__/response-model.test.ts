/**
 * Every observation carries the model the provider's response named.
 *
 * Field test (Zelvora N6): an Azure OpenAI deployment named for its job,
 * "support-bot", runs gpt-5.4-mini 2026-03-17. The SDK sent only the
 * request's model, the deployment name, which has no price, so the call was
 * stored at $0. The server now prices from `response_model` and keeps the
 * deployment beside it (ObservationDto.response_model, at most 200
 * characters). A stream sends the last chunk's model: Azure's first chunk
 * (prompt filter results) names none.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { wrap, __test__ as wrapTest } from "../wrap.js";
import { prewarm } from "../prewarm.js";
import { __test__ as runtimeTest } from "../runtime.js";
import { __test__ as anthropicTest } from "../providers/anthropic.js";
import { responseModel, responseModelField } from "../api-promise.js";
import { fakeApiClient, jsonResponse, sseResponse } from "./fake-api-promise.js";

const TENANT = "11111111-1111-1111-1111-111111111111";
const TC = "http://api.tc.test";
const SERVED = "gpt-5.4-mini-2026-03-17";

type Obs = Record<string, any>;

const chatOk = {
  id: "chatcmpl-azure",
  object: "chat.completion",
  model: SERVED,
  choices: [{ index: 0, message: { role: "assistant", content: "Ok." }, finish_reason: "stop" }],
  usage: { prompt_tokens: 9, completion_tokens: 4 },
};
const chunks = [
  // Azure's first chunk: prompt filter results, no model, no choices.
  { id: "", object: "", model: "", choices: [], prompt_filter_results: [{ prompt_index: 0 }] },
  { id: "chatcmpl-azure-stream", object: "chat.completion.chunk", model: SERVED, choices: [{ index: 0, delta: { content: "Ok." } }], usage: null },
  { id: "chatcmpl-azure-stream", object: "chat.completion.chunk", model: SERVED, choices: [], usage: { prompt_tokens: 9, completion_tokens: 4 } },
];
const msgOk = {
  id: "msg_direct",
  type: "message",
  role: "assistant",
  content: [{ type: "text", text: "Ok." }],
  model: "claude-haiku-4-5-20251001",
  stop_reason: "end_turn",
  usage: { input_tokens: 10, output_tokens: 5 },
};
const events = [
  { type: "message_start", message: { id: "msg_stream", model: "claude-haiku-4-5-20251001", usage: { input_tokens: 12, output_tokens: 1 } } },
  { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Ok." } },
  { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 4 } },
  { type: "message_stop" },
];
const ask = { model: "support-bot", messages: [{ role: "user", content: "Say ok." }] };
const askClaude = { model: "claude-haiku-4-5", max_tokens: 20, messages: [{ role: "user", content: "Say ok." }] };

function tc() {
  const observations: Obs[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = typeof input === "string" ? input : (input as Request).url ?? String(input);
    if (url.includes("/sdk-token/exchange")) return jsonResponse(200, { token: "jwt", expires_in: 900 });
    if (url.includes("/v1/sdk-layer")) return jsonResponse(200, { published_layer: "observe" });
    if (url.includes("/inference-proxy/observation")) {
      observations.push(JSON.parse(String(init.body)) as Obs);
      return new Response(null, { status: 202 });
    }
    throw new TypeError(`fetch failed: unrouted ${url}`);
  }) as typeof fetch;
  return observations;
}

const options = () => ({
  apiKey: "tc",
  tenantId: TENANT,
  baseUrl: TC,
  retry: { maxAttempts: 1, baseDelayMs: 1, maxDelayMs: 1 },
});

async function observed(respond: () => Response, shape: "openai" | "anthropic" = "openai") {
  const observations = tc();
  const fake = fakeApiClient(respond, shape);
  wrap(fake.client, options());
  await prewarm(options());
  return { observations, client: fake.client };
}

const settle = () => new Promise((r) => setTimeout(r, 20));

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

describe("OpenAI: the model the response named", () => {
  it("await create(): the served model beside the deployment the request named", async () => {
    const { observations, client } = await observed(() => jsonResponse(200, chatOk));
    await client.chat.completions.create(ask);
    await settle();
    expect(observations).toHaveLength(1);
    expect(observations[0]).toMatchObject({ model: "support-bot", response_model: SERVED });
  });

  it("withResponse()", async () => {
    const { observations, client } = await observed(() => jsonResponse(200, chatOk));
    await client.chat.completions.create(ask).withResponse();
    await settle();
    expect(observations[0]!.response_model).toBe(SERVED);
  });

  it("a stream: the last chunk that names a model", async () => {
    const { observations, client } = await observed(() => sseResponse(chunks));
    const stream = await client.chat.completions.create({ ...ask, stream: true, stream_options: { include_usage: true } });
    for await (const _ of stream) void _;
    await settle();
    expect(observations).toHaveLength(1);
    expect(observations[0]).toMatchObject({ model: "support-bot", response_model: SERVED, input_tokens: 9, output_tokens: 4 });
  });

  it("an error has none", async () => {
    const { observations, client } = await observed(() => jsonResponse(400, { error: { message: "bad" } }));
    await client.chat.completions.create(ask).catch(() => undefined);
    await settle();
    expect(observations[0]).not.toHaveProperty("response_model");
  });

  it("a model the server would refuse is left off, and the call still recorded", async () => {
    const { observations, client } = await observed(() => jsonResponse(200, { ...chatOk, model: "x".repeat(201) }));
    await client.chat.completions.create(ask);
    await settle();
    expect(observations[0]).not.toHaveProperty("response_model");
    expect(observations[0]!.input_tokens).toBe(9);
  });
});

describe("Anthropic: the model the response named", () => {
  it("await create()", async () => {
    const { observations, client } = await observed(() => jsonResponse(200, msgOk), "anthropic");
    await client.messages.create(askClaude);
    await settle();
    expect(observations[0]!.response_model).toBe("claude-haiku-4-5-20251001");
  });

  it("a stream: message_start's model", async () => {
    const { observations, client } = await observed(() => sseResponse(events), "anthropic");
    const stream = await client.messages.create({ ...askClaude, stream: true });
    for await (const _ of stream) void _;
    await settle();
    expect(observations[0]!.response_model).toBe("claude-haiku-4-5-20251001");
  });
});

describe("responseModel", () => {
  it("takes only a model id the server will", () => {
    expect(responseModel(" gpt-4o ")).toBe("gpt-4o");
    for (const bad of [null, undefined, "", "   ", 7, "x".repeat(201), "gpt\u00004o"]) expect(responseModel(bad)).toBeNull();
    expect(responseModelField(null)).toEqual({});
    expect(responseModelField(SERVED)).toEqual({ response_model: SERVED });
  });
});
