/**
 * Every observation carries the provider's id for the call when it has one.
 *
 * Field test against a self-hosted install: observe-mode SDK rows had no
 * provider_response_id, so the same call seen by the proxy, a gateway or a
 * cost import that knows the provider's id couldn't be matched to them
 * (same-call rule, packages/contracts/src/same-call.ts). The server reads
 * `provider_response_id` off the observation (ObservationDto, at most 200
 * characters, and an over-long one fails the whole observation).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { wrap, __test__ as wrapTest } from "../wrap.js";
import { prewarm } from "../prewarm.js";
import { __test__ as runtimeTest } from "../runtime.js";
import { __test__ as anthropicTest } from "../providers/anthropic.js";
import { ObservationTransport } from "../transport.js";
import * as bedrockProvider from "../providers/bedrock.js";
import * as vertexProvider from "../providers/vertex.js";
import { FakeAPIPromise, fakeApiClient, jsonResponse, sseResponse } from "./fake-api-promise.js";

const TENANT = "11111111-1111-1111-1111-111111111111";
const TC = "http://api.tc.test";
const PROXY = "http://proxy.test";

type Obs = Record<string, any>;

const chatOk = {
  id: "chatcmpl-direct",
  object: "chat.completion",
  choices: [{ index: 0, message: { role: "assistant", content: "Ok." }, finish_reason: "stop" }],
  usage: { prompt_tokens: 10, completion_tokens: 2 },
};
const legacyOk = {
  id: "cmpl-legacy",
  object: "text_completion",
  choices: [{ index: 0, text: "Ok.", finish_reason: "stop" }],
  usage: { prompt_tokens: 4, completion_tokens: 1 },
};
const msgOk = {
  id: "msg_direct",
  type: "message",
  role: "assistant",
  content: [{ type: "text", text: "Ok." }],
  model: "claude-haiku-4-5",
  stop_reason: "end_turn",
  usage: { input_tokens: 10, output_tokens: 5 },
};
const chunks = [
  { id: "chatcmpl-stream", object: "chat.completion.chunk", choices: [{ index: 0, delta: { content: "Ok." } }], usage: null },
  { id: "chatcmpl-stream", object: "chat.completion.chunk", choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: null },
  { id: "chatcmpl-stream", object: "chat.completion.chunk", choices: [], usage: { prompt_tokens: 9, completion_tokens: 3 } },
];
const events = [
  { type: "message_start", message: { id: "msg_stream", usage: { input_tokens: 12, output_tokens: 1 } } },
  { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
  { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Ok." } },
  { type: "content_block_stop", index: 0 },
  { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 4 } },
  { type: "message_stop" },
];
const ask = { model: "gpt-4.1-mini", messages: [{ role: "user", content: "Say ok." }] };
const askClaude = { model: "claude-haiku-4-5", max_tokens: 20, messages: [{ role: "user", content: "Say ok." }] };

function tc(layer: string, proxy?: () => Response) {
  const observations: Obs[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = typeof input === "string" ? input : (input as Request).url ?? String(input);
    if (url.includes("/sdk-token/exchange")) return jsonResponse(200, { token: "jwt", expires_in: 900 });
    if (url.includes("/v1/sdk-layer")) return jsonResponse(200, { published_layer: layer });
    if (url.includes("/inference-proxy/observation")) {
      observations.push(JSON.parse(String(init.body)) as Obs);
      return new Response(null, { status: 202 });
    }
    if (url.startsWith(PROXY) && proxy) return proxy();
    throw new TypeError(`fetch failed: unrouted ${url}`);
  }) as typeof fetch;
  return observations;
}

const options = (extra: Record<string, unknown> = {}) => ({
  apiKey: "tc",
  tenantId: TENANT,
  baseUrl: TC,
  retry: { maxAttempts: 1, baseDelayMs: 1, maxDelayMs: 1 },
  ...extra,
});

async function settle(ms = 20) {
  await new Promise((r) => setTimeout(r, ms));
}

async function observed(respond: () => Response, shape: "openai" | "anthropic" = "openai") {
  const observations = tc("observe");
  const fake = fakeApiClient(respond, shape);
  wrap(fake.client, options());
  await prewarm(options());
  return { observations, client: fake.client };
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

describe("OpenAI: the completion's id", () => {
  it("await create()", async () => {
    const { observations, client } = await observed(() => jsonResponse(200, chatOk));
    await client.chat.completions.create(ask);
    await settle();
    expect(observations).toHaveLength(1);
    expect(observations[0]!.provider_response_id).toBe("chatcmpl-direct");
  });

  it("withResponse()", async () => {
    const { observations, client } = await observed(() => jsonResponse(200, chatOk));
    const { data } = await client.chat.completions.create(ask).withResponse();
    expect(data.id).toBe("chatcmpl-direct");
    await settle();
    expect(observations[0]!.provider_response_id).toBe("chatcmpl-direct");
  });

  it("asResponse(), read off a copy of the body", async () => {
    const { observations, client } = await observed(() => jsonResponse(200, chatOk));
    const resp = await client.chat.completions.create(ask).asResponse();
    expect((await resp.json()).id).toBe("chatcmpl-direct");
    await settle();
    expect(observations).toHaveLength(1);
    expect(observations[0]!.provider_response_id).toBe("chatcmpl-direct");
  });

  it("a stream: the chunks' id, recorded at its end with its usage", async () => {
    const { observations, client } = await observed(() => sseResponse(chunks));
    const stream = await client.chat.completions.create({ ...ask, stream: true, stream_options: { include_usage: true } });
    for await (const _ of stream) void _;
    await settle();
    expect(observations).toHaveLength(1);
    expect(observations[0]).toMatchObject({
      provider_response_id: "chatcmpl-stream",
      input_tokens: 9,
      output_tokens: 3,
    });
  });

  it("a stream the caller stops reading after one chunk", async () => {
    const { observations, client } = await observed(() => sseResponse(chunks));
    const stream = await client.chat.completions.create({ ...ask, stream: true });
    for await (const _ of stream) break;
    await settle();
    expect(observations).toHaveLength(1);
    expect(observations[0]!.provider_response_id).toBe("chatcmpl-stream");
  });

  it("legacy completions.create()", async () => {
    const observations = tc("observe");
    const fake = fakeApiClient(() => jsonResponse(200, legacyOk));
    fake.client.completions = { create: fake.client.chat.completions.create };
    wrap(fake.client, options());
    await prewarm(options());
    await fake.client.completions.create({ model: "gpt-3.5-turbo-instruct", prompt: "x" });
    await settle();
    const legacy = observations.find((o) => o.operation === "completions");
    expect(legacy?.provider_response_id).toBe("cmpl-legacy");
  });

  it("an error has none", async () => {
    const { observations, client } = await observed(() => jsonResponse(400, { error: { message: "bad" } }));
    await client.chat.completions.create(ask).catch(() => undefined);
    await settle();
    expect(observations).toHaveLength(1);
    expect(observations[0]!.status).toBe("error");
    expect(observations[0]).not.toHaveProperty("provider_response_id");
  });

  it("an id over the server's 200 characters is left off, and the call still recorded", async () => {
    const { observations, client } = await observed(() => jsonResponse(200, { ...chatOk, id: "x".repeat(201) }));
    await client.chat.completions.create(ask);
    await settle();
    expect(observations).toHaveLength(1);
    expect(observations[0]).not.toHaveProperty("provider_response_id");
    expect(observations[0]!.input_tokens).toBe(10);
  });
});

describe("OpenAI route mode: the id the proxy relayed", () => {
  const routeOptions = () => options({ maxLayer: "route", proxyUrl: PROXY });

  it("a buffered answer", async () => {
    const observations = tc("route", () => jsonResponse(200, { ...chatOk, id: "chatcmpl-proxy" }));
    const fake = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(fake.client, routeOptions());
    await fake.client.chat.completions.create(ask);
    await settle();
    expect(observations).toHaveLength(1);
    expect(observations[0]!.provider_response_id).toBe("chatcmpl-proxy");
  });

  it("a stream", async () => {
    const observations = tc("route", () => sseResponse(chunks));
    const fake = fakeApiClient(() => jsonResponse(200, chatOk));
    wrap(fake.client, routeOptions());
    const stream = await fake.client.chat.completions.create({ ...ask, stream: true });
    for await (const _ of stream) void _;
    await settle();
    expect(observations).toHaveLength(1);
    expect(observations[0]!.provider_response_id).toBe("chatcmpl-stream");
  });
});

describe("Anthropic: the message's id", () => {
  it("await create()", async () => {
    const { observations, client } = await observed(() => jsonResponse(200, msgOk), "anthropic");
    const p = client.messages.create(askClaude);
    expect(p).toBeInstanceOf(FakeAPIPromise);
    await p;
    await settle();
    expect(observations[0]!.provider_response_id).toBe("msg_direct");
  });

  it("withResponse()", async () => {
    const { observations, client } = await observed(() => jsonResponse(200, msgOk), "anthropic");
    await client.messages.create(askClaude).withResponse();
    await settle();
    expect(observations[0]!.provider_response_id).toBe("msg_direct");
  });

  it("a stream: message_start's message.id", async () => {
    const { observations, client } = await observed(() => sseResponse(events), "anthropic");
    const stream = await client.messages.create({ ...askClaude, stream: true });
    for await (const _ of stream) void _;
    await settle();
    expect(observations).toHaveLength(1);
    expect(observations[0]).toMatchObject({
      provider_response_id: "msg_stream",
      input_tokens: 12,
      output_tokens: 4,
    });
  });
});

// ── Bedrock and Vertex: driven through their install() with a stub transport ──

function stubTransport() {
  const observations: Obs[] = [];
  const fetchImpl: typeof fetch = async (input, init = {}) => {
    const url = typeof input === "string" ? input : (input as URL).toString();
    if (url.includes("sdk-token/exchange")) return jsonResponse(200, { token: "tok", expires_in: 900 });
    if (url.includes("/api/inference-proxy/observation")) {
      observations.push(JSON.parse(init?.body as string));
      return new Response(null, { status: 202 });
    }
    return new Response(null, { status: 404 });
  };
  const transport = new ObservationTransport({ tenantId: "t-1", baseUrl: "http://api.local", apiKey: "k", fetchImpl });
  return { transport, observations };
}

async function flush(times = 30) {
  for (let i = 0; i < times; i++) await Promise.resolve();
}

/** A Smithy-shaped client: send() runs the registered middleware around `terminal`. */
function bedrockClient(terminal: () => Promise<{ output: Record<string, unknown>; response?: unknown }>) {
  const middlewares: Array<(next: any, ctx: any) => any> = [];
  return {
    config: { serviceId: "Bedrock Runtime" },
    middlewareStack: {
      add(mw: (next: any, ctx: any) => any) {
        middlewares.push(mw);
      },
      identify() {
        return [];
      },
    },
    async send(command: { constructor: { name: string }; input: Record<string, unknown> }) {
      let chain: (args: any) => Promise<any> = () => terminal();
      for (let i = middlewares.length - 1; i >= 0; i--) chain = middlewares[i]!(chain, { commandName: command.constructor.name });
      return chain({ input: command.input });
    },
  };
}

describe("Bedrock: the request id (x-amzn-requestid), which the proxy, invocation log and CUR share", () => {
  async function send(commandName: string, result: { output: Record<string, unknown>; response?: unknown }) {
    const { transport, observations } = stubTransport();
    const client = bedrockClient(async () => result);
    bedrockProvider.install(client, transport, null, null, null);
    await client.send({ constructor: { name: commandName }, input: { modelId: "amazon.nova-lite-v1:0" } });
    await flush();
    expect(observations).toHaveLength(1);
    return observations[0]!;
  }

  it("Converse: $metadata.requestId", async () => {
    const obs = await send("ConverseCommand", {
      output: { usage: { inputTokens: 3, outputTokens: 4 }, $metadata: { requestId: "req-converse" } },
    });
    expect(obs.provider_response_id).toBe("req-converse");
  });

  it("InvokeModel: the request id, not Anthropic on Bedrock's msg_bdrk_ body id", async () => {
    const obs = await send("InvokeModelCommand", {
      output: {
        body: Buffer.from(JSON.stringify({ id: "msg_bdrk_1", usage: { input_tokens: 5, output_tokens: 6 } })),
        $metadata: { requestId: "req-invoke" },
      },
    });
    expect(obs.provider_response_id).toBe("req-invoke");
  });

  it("from the response header when $metadata has none", async () => {
    const obs = await send("ConverseCommand", {
      output: { usage: { inputTokens: 3, outputTokens: 4 } },
      response: { headers: { "x-amzn-requestid": "req-header" } },
    });
    expect(obs.provider_response_id).toBe("req-header");
  });

  it("a stream's placeholder observation", async () => {
    const obs = await send("ConverseStreamCommand", {
      output: { stream: (async function* () {})(), $metadata: { requestId: "req-stream" } },
    });
    expect(obs.provider_response_id).toBe("req-stream");
  });

  it("none when the output has no request id", async () => {
    const obs = await send("ConverseCommand", { output: { usage: { inputTokens: 3, outputTokens: 4 } } });
    expect(obs).not.toHaveProperty("provider_response_id");
  });
});

describe("Vertex: GenerateContentResponse.responseId", () => {
  it("generateContent", async () => {
    const { transport, observations } = stubTransport();
    const client = {
      vertexai: true,
      models: {
        generateContent: vi.fn(async () => ({
          responseId: "vertex-resp-1",
          usageMetadata: { promptTokenCount: 4, candidatesTokenCount: 2 },
        })),
        generateContentStream: vi.fn(async () => (async function* () {})()),
      },
    };
    vertexProvider.install(client, transport, null, null, null);
    await client.models.generateContent({ model: "gemini-2.5-flash", contents: "x" });
    await flush();
    expect(observations).toHaveLength(1);
    expect(observations[0]!.provider_response_id).toBe("vertex-resp-1");
  });
});
