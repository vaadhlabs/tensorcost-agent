/**
 * billing_account against the real provider SDKs, not fakes.
 *
 * The fakes in billing-account.test.ts model `asResponse()` and the
 * clients' URL fields; this checks the real ones have them and that reading
 * them adds no request and hands back the same result. Each suite runs when
 * its variable points at a local copy's entry point, and skips otherwise:
 *
 *   TENSORCOST_TEST_OPENAI=/path/to/node_modules/openai/index.mjs        (openai@4)
 *   TENSORCOST_TEST_OPENAI5=/path/to/node_modules/openai/index.mjs       (openai@5)
 *   TENSORCOST_TEST_ANTHROPIC=/path/to/node_modules/@anthropic-ai/sdk/index.mjs
 *   TENSORCOST_TEST_GENAI=/path/to/node_modules/@google/genai/dist/node/index.mjs
 *
 * Bedrock uses this package's own @aws-sdk/client-bedrock-runtime dev copy.
 *
 * 1.7.0 was checked against openai@4.104.0, openai@5.23.2,
 * @anthropic-ai/sdk@0.65.0, @google/genai@1.30.0 and
 * @aws-sdk/client-bedrock-runtime@3.1147.0.
 */

import { existsSync } from "node:fs";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { wrap, __test__ as wrapTest } from "../wrap.js";
import { __test__ as anthropicTest } from "../providers/anthropic.js";
import { __test__ as runtimeTest } from "../runtime.js";

const OPENAI4 = process.env.TENSORCOST_TEST_OPENAI ?? "";
const OPENAI5 = process.env.TENSORCOST_TEST_OPENAI5 ?? "";
const ANTHROPIC = process.env.TENSORCOST_TEST_ANTHROPIC ?? "";
const GENAI = process.env.TENSORCOST_TEST_GENAI ?? "";
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
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });

/** Routes TensorCost's own endpoints, records observations, and answers provider calls with `provider`. */
function router(provider: (url: string) => Response) {
  const providerCalls: string[] = [];
  const observations: any[] = [];
  const f = (async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = typeof input === "string" ? input : (input as Request).url ?? String(input);
    if (url.includes("/sdk-token/exchange")) return json(200, { token: "jwt", expires_in: 900 });
    if (url.includes("/v1/sdk-layer")) return json(200, { published_layer: "observe" });
    if (url.includes("/observation")) {
      observations.push(JSON.parse(String(init.body)));
      return new Response(null, { status: 202 });
    }
    if (url.includes("/v1/admit")) return new Response(null, { status: 204 });
    providerCalls.push(url);
    return provider(url);
  }) as typeof fetch;
  return { f, providerCalls, observations };
}

async function settle() {
  for (let i = 0; i < 5; i += 1) await new Promise((r) => setTimeout(r, 0));
}

const tc = { apiKey: "tc", tenantId: TENANT, baseUrl: "http://api.tc.test", maxLayer: "observe" as const };

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

for (const [label, path] of [
  ["openai@4", OPENAI4],
  ["openai@5", OPENAI5],
] as const) {
  describe.skipIf(!path || !existsSync(path))(`a real ${label} client`, () => {
    async function load() {
      return ((await import(path)) as { default: any; AzureOpenAI: any });
    }

    it("labels a call with openai-organization, one provider request, same body", async () => {
      const { default: OpenAI } = await load();
      const r = router(() => json(200, chatOk, { "openai-organization": "org-real", "openai-project": "proj_1" }));
      globalThis.fetch = r.f;
      const c = new OpenAI({ apiKey: "sk", fetch: r.f, maxRetries: 0 });
      wrap(c, tc);
      const out = await c.chat.completions.create({ model: "gpt-4o", messages: [{ role: "user", content: "x" }] });
      expect(out.id).toBe("chatcmpl-direct");
      await settle();
      expect(r.providerCalls).toEqual(["https://api.openai.com/v1/chat/completions"]);
      expect(r.observations.at(-1).billing_account).toBe("org-real");
    });

    it("no header, no field", async () => {
      const { default: OpenAI } = await load();
      const r = router(() => json(200, chatOk));
      globalThis.fetch = r.f;
      const c = new OpenAI({ apiKey: "sk", fetch: r.f, maxRetries: 0 });
      wrap(c, tc);
      await c.chat.completions.create({ model: "gpt-4o", messages: [{ role: "user", content: "x" }] });
      await settle();
      expect(r.observations.at(-1)).not.toHaveProperty("billing_account");
    });

    it("a 429 is labelled from the APIError's headers", async () => {
      const { default: OpenAI } = await load();
      const r = router(() => json(429, { error: { message: "slow down" } }, { "openai-organization": "org-real" }));
      globalThis.fetch = r.f;
      const c = new OpenAI({ apiKey: "sk", fetch: r.f, maxRetries: 0 });
      wrap(c, tc);
      await expect(
        c.chat.completions.create({ model: "gpt-4o", messages: [{ role: "user", content: "x" }] }),
      ).rejects.toMatchObject({ status: 429 });
      await settle();
      expect(r.observations.at(-1).status).toBe("error");
      expect(r.observations.at(-1).billing_account).toBe("org-real");
    });

    it("a stream is labelled at stream start and still iterates", async () => {
      const { default: OpenAI } = await load();
      const sse =
        `data: ${JSON.stringify({ id: "c1", object: "chat.completion.chunk", choices: [{ index: 0, delta: { content: "hi" } }] })}\n\n` +
        "data: [DONE]\n\n";
      const r = router(
        () =>
          new Response(sse, {
            status: 200,
            headers: { "content-type": "text/event-stream", "openai-organization": "org-real" },
          }),
      );
      globalThis.fetch = r.f;
      const c = new OpenAI({ apiKey: "sk", fetch: r.f, maxRetries: 0 });
      wrap(c, tc);
      const stream = await c.chat.completions.create({
        model: "gpt-4o",
        stream: true,
        messages: [{ role: "user", content: "x" }],
      });
      const parts: string[] = [];
      for await (const chunk of stream) parts.push(chunk.choices[0]?.delta?.content ?? "");
      expect(parts.join("")).toBe("hi");
      await settle();
      expect(r.providerCalls).toHaveLength(1);
      expect(r.observations.at(-1).billing_account).toBe("org-real");
    });

    it("an AzureOpenAI client is labelled with its resource", async () => {
      const { AzureOpenAI } = await load();
      const r = router(() => json(200, chatOk));
      globalThis.fetch = r.f;
      const c = new AzureOpenAI({
        apiKey: "az",
        endpoint: "https://Contoso.openai.azure.com",
        apiVersion: "2024-10-21",
        deployment: "gpt-4o",
        fetch: r.f,
        maxRetries: 0,
      });
      wrap(c, tc);
      await c.chat.completions.create({ model: "gpt-4o", messages: [{ role: "user", content: "x" }] });
      await settle();
      expect(r.observations.at(-1).provider).toBe("azure_openai");
      expect(r.observations.at(-1).billing_account).toBe("contoso");
    });
  });
}

describe.skipIf(!ANTHROPIC || !existsSync(ANTHROPIC))("a real @anthropic-ai/sdk client", () => {
  it("labels a call with anthropic-organization-id, one provider request, same body", async () => {
    const { default: Anthropic } = (await import(ANTHROPIC)) as { default: any };
    const r = router(() => json(200, msgOk, { "anthropic-organization-id": "org-anth-real" }));
    globalThis.fetch = r.f;
    const c = new Anthropic({ apiKey: "sk-ant", fetch: r.f, maxRetries: 0 });
    wrap(c, tc);
    const out = await c.messages.create({
      model: "claude-sonnet-4-5",
      max_tokens: 8,
      messages: [{ role: "user", content: "x" }],
    });
    expect(out.id).toBe("msg_direct");
    await settle();
    expect(r.providerCalls).toEqual(["https://api.anthropic.com/v1/messages"]);
    expect(r.observations.at(-1).billing_account).toBe("org-anth-real");
  });

  it("no header, no field", async () => {
    const { default: Anthropic } = (await import(ANTHROPIC)) as { default: any };
    const r = router(() => json(200, msgOk));
    globalThis.fetch = r.f;
    const c = new Anthropic({ apiKey: "sk-ant", fetch: r.f, maxRetries: 0 });
    wrap(c, tc);
    await c.messages.create({ model: "claude-sonnet-4-5", max_tokens: 8, messages: [{ role: "user", content: "x" }] });
    await settle();
    expect(r.observations.at(-1)).not.toHaveProperty("billing_account");
  });
});

describe.skipIf(!GENAI || !existsSync(GENAI))("a real @google/genai Vertex client", () => {
  it("labels a call with the project in the URL it was sent to", async () => {
    const { GoogleGenAI } = (await import(GENAI)) as { GoogleGenAI: any };
    const c = new GoogleGenAI({ vertexai: true, project: "my-proj", location: "us-central1" });
    const { vertexCallUrl, billingAccountOfCall } = await import("../billing-account.js");
    // Same project and host as the URL the client's own ApiClient builds.
    const real = c.apiClient
      .constructUrl("publishers/google/models/gemini-2.0-flash:generateContent", c.apiClient.clientOptions.httpOptions, true)
      .toString();
    expect(billingAccountOfCall(real, null)).toBe("my-proj");
    expect(billingAccountOfCall(vertexCallUrl(c), null)).toBe("my-proj");

    const r = router(() => json(200, {}));
    globalThis.fetch = r.f;
    c.models.generateContent = async () => ({ usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 1 } });
    wrap(c, tc);
    await c.models.generateContent({ model: "gemini-2.0-flash", contents: "x" });
    await settle();
    expect(r.observations.at(-1).billing_account).toBe("my-proj");
  });

  it("express mode (API key) has no project, so no field", async () => {
    const { GoogleGenAI } = (await import(GENAI)) as { GoogleGenAI: any };
    const { vertexCallUrl } = await import("../billing-account.js");
    expect(vertexCallUrl(new GoogleGenAI({ vertexai: true, apiKey: "k" }))).toBeNull();
    expect(vertexCallUrl(new GoogleGenAI({ apiKey: "k" }))).toBeNull();
  });
});

describe("a real BedrockRuntimeClient", () => {
  async function run(env: Record<string, string | undefined>) {
    const { BedrockRuntimeClient, ConverseCommand } = await import("@aws-sdk/client-bedrock-runtime");
    const saved = { ...process.env };
    Object.assign(process.env, { AWS_ACCESS_KEY_ID: "AKIATEST", AWS_SECRET_ACCESS_KEY: "secret", ...env });
    for (const [k, v] of Object.entries(env)) if (v === undefined) delete process.env[k];
    try {
      const r = router(() => json(200, {}));
      globalThis.fetch = r.f;
      const sent: string[] = [];
      const c = new BedrockRuntimeClient({
        region: "us-east-1",
        requestHandler: {
          handle: async (req: any) => {
            sent.push(req.hostname);
            const body = JSON.stringify({
              output: { message: { role: "assistant", content: [{ text: "hi" }] } },
              stopReason: "end_turn",
              usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
              metrics: { latencyMs: 1 },
            });
            const { Readable } = await import("node:stream");
            return {
              // HttpResponse's shape; @smithy/protocol-http isn't hoisted here.
              response: {
                statusCode: 200,
                headers: { "content-type": "application/json" },
                body: Readable.from([Buffer.from(body)]),
              },
            };
          },
        } as any,
      });
      wrap(c, tc);
      const out = await c.send(
        new ConverseCommand({ modelId: "anthropic.claude-3-haiku", messages: [{ role: "user", content: [{ text: "x" }] }] }),
      );
      expect(out.usage?.inputTokens).toBe(1);
      await settle();
      expect(sent).toEqual(["bedrock-runtime.us-east-1.amazonaws.com"]);
      return r.observations.at(-1);
    } finally {
      process.env = saved;
    }
  }

  it("labels a call with the account the credentials carry (AWS_ACCOUNT_ID), no STS call", async () => {
    expect((await run({ AWS_ACCOUNT_ID: "123456789012" })).billing_account).toBe("123456789012");
  });

  it("plain access keys without an account send no field", async () => {
    expect(await run({ AWS_ACCOUNT_ID: undefined })).not.toHaveProperty("billing_account");
  });
});
