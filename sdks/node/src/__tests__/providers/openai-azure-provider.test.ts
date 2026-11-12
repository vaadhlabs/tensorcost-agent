/**
 * An AzureOpenAI client reports its calls as azure_openai, not openai.
 * Before, Azure spend landed under OpenAI: a budget scoped to azure_openai
 * never saw it, and it couldn't be paired with the Azure bill line for the
 * same call.
 */
import { describe, expect, it } from "vitest";
import { ObservationTransport } from "../../transport.js";
import * as openaiProvider from "../../providers/openai.js";
import { openAiProviderFor } from "../../applied.js";
import { defaultHardening } from "../test-harness.js";

function fakeFetch(log: { observations: any[] }): typeof fetch {
  return async (input, init = {}) => {
    const url = typeof input === "string" ? input : (input as URL).toString();
    if (url.includes("sdk-token/exchange")) {
      return new Response(JSON.stringify({ token: "tok", expires_in: 900 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    if (url.includes("/api/inference-proxy/observation")) {
      log.observations.push(JSON.parse(init.body as string));
      return new Response(null, { status: 202 });
    }
    return new Response(null, { status: 404 });
  };
}

async function flush(times = 30) {
  for (let i = 0; i < times; i += 1) await Promise.resolve();
}

function chatClient(extra: Record<string, unknown> = {}) {
  return {
    ...extra,
    chat: {
      completions: {
        create: async () => ({ id: "c-1", usage: { prompt_tokens: 20, completion_tokens: 57 } }),
      },
    },
  };
}

describe("openAiProviderFor", () => {
  it("says azure_openai for every Azure host and for the AzureOpenAI class", () => {
    expect(openAiProviderFor({ baseURL: "https://tc-zelvora-aoai-06a901.openai.azure.com/openai" })).toBe("azure_openai");
    expect(openAiProviderFor({ baseURL: "https://acme.cognitiveservices.azure.com/openai" })).toBe("azure_openai");
    expect(openAiProviderFor({ baseURL: "https://acme.services.ai.azure.com/openai/v1" })).toBe("azure_openai");
    class AzureOpenAI {
      baseURL = "https://gateway.example.com/openai";
    }
    expect(openAiProviderFor(new AzureOpenAI())).toBe("azure_openai");
  });

  it("says openai for OpenAI itself and for other OpenAI-compatible endpoints", () => {
    expect(openAiProviderFor({ baseURL: "https://api.openai.com/v1" })).toBe("openai");
    expect(openAiProviderFor({ baseURL: "https://openrouter.ai/api/v1" })).toBe("openai");
    expect(openAiProviderFor({})).toBe("openai");
    expect(openAiProviderFor(null)).toBe("openai");
  });
});

describe("observations from an Azure OpenAI client", () => {
  it("carry provider azure_openai", async () => {
    const log = { observations: [] as any[] };
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: fakeFetch(log),
    });
    const client = chatClient({ baseURL: "https://tc-zelvora-aoai-06a901.openai.azure.com/openai" });
    openaiProvider.install(client, transport, "tenant-x", null, null, defaultHardening());
    await client.chat.completions.create({ model: "gpt-5.4-mini", messages: [{ role: "user", content: "hi" }] });
    await transport.flush?.();
    await flush();
    expect(log.observations.length).toBeGreaterThan(0);
    expect(log.observations[0].provider).toBe("azure_openai");
  });

  it("an OpenAI client still says openai", async () => {
    const log = { observations: [] as any[] };
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: fakeFetch(log),
    });
    const client = chatClient({ baseURL: "https://api.openai.com/v1" });
    openaiProvider.install(client, transport, "tenant-x", null, null, defaultHardening());
    await client.chat.completions.create({ model: "gpt-5.4-mini", messages: [{ role: "user", content: "hi" }] });
    await transport.flush?.();
    await flush();
    expect(log.observations[0].provider).toBe("openai");
  });
});
