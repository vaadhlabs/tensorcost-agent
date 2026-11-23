/**
 * billing_account on SDK observations (1.7.0).
 *
 * Per provider: the header (or URL, or credentials) is there → the field is
 * sent; it isn't → no field; reading it throws → no field and the caller
 * gets exactly what the provider returned.
 */

import { describe, expect, it } from "vitest";
import { ObservationTransport } from "../transport.js";
import * as anthropicProvider from "../providers/anthropic.js";
import * as openaiProvider from "../providers/openai.js";
import * as vertexProvider from "../providers/vertex.js";
import * as bedrockProvider from "../providers/bedrock.js";
import {
  billingAccountOfCall,
  normalizeBillingAccount,
  vertexCallUrl,
} from "../billing-account.js";
import { defaultHardening } from "./test-harness.js";

function capture() {
  const observations: any[] = [];
  const fetchImpl: typeof fetch = async (input, init = {}) => {
    const url = typeof input === "string" ? input : (input as URL).toString();
    if (url.includes("sdk-token/exchange")) {
      return new Response(JSON.stringify({ token: "tok", expires_in: 900 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    if (url.includes("/api/inference-proxy/observation")) {
      observations.push(JSON.parse(init.body as string));
      return new Response(null, { status: 202 });
    }
    return new Response(null, { status: 404 });
  };
  const transport = new ObservationTransport({
    tenantId: "t-1",
    baseUrl: "http://api.local",
    apiKey: "k",
    fetchImpl,
  });
  return { transport, observations };
}

async function flush(times = 30) {
  for (let i = 0; i < times; i += 1) await Promise.resolve();
  await new Promise((r) => setTimeout(r, 0));
}

/** What openai / @anthropic-ai/sdk return from create(): a promise of the body with asResponse(). */
function apiPromise<T>(data: T, headers: unknown): Promise<T> & { asResponse(): Promise<unknown> } {
  const p = Promise.resolve(data) as Promise<T> & { asResponse(): Promise<unknown> };
  p.asResponse = () => Promise.resolve({ headers });
  return p;
}

const throwingHeaders = {
  get() {
    throw new Error("header read blew up");
  },
};

describe("billingAccountOfCall (same rules as @tensorcost/contracts)", () => {
  it("reads each provider's format", () => {
    expect(billingAccountOfCall("https://api.anthropic.com", new Headers({ "anthropic-organization-id": " org-a " }))).toBe("org-a");
    expect(
      billingAccountOfCall(
        "https://api.openai.com/v1",
        new Headers({ "openai-organization": "org-o", "openai-project": "proj_x" }),
      ),
    ).toBe("org-o");
    expect(billingAccountOfCall("https://api.openai.com/v1", { "OpenAI-Organization": "org-rec" })).toBe("org-rec");
    expect(billingAccountOfCall("https://MyRes.openai.azure.com/openai", null)).toBe("myres");
    expect(billingAccountOfCall("https://r2.cognitiveservices.azure.com/", null)).toBe("r2");
    expect(billingAccountOfCall("https://r3.services.ai.azure.com/models", null)).toBe("r3");
    expect(
      billingAccountOfCall("https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/x", null),
    ).toBe("my-proj");
    expect(billingAccountOfCall("https://aiplatform.googleapis.com/v1/projects/p-global/locations/global/x", null)).toBe(
      "p-global",
    );
  });

  it("names nothing for a host it doesn't know, or a bad value", () => {
    // A gateway in front of OpenAI may pass the header through; the backend
    // doesn't trust it there, so neither does the SDK.
    expect(billingAccountOfCall("https://gateway.corp/v1", new Headers({ "openai-organization": "org-o" }))).toBeNull();
    expect(billingAccountOfCall("not a url", null)).toBeNull();
    expect(billingAccountOfCall(null, null)).toBeNull();
    expect(billingAccountOfCall("https://api.anthropic.com", throwingHeaders)).toBeNull();
    expect(normalizeBillingAccount("x".repeat(129))).toBeNull();
    expect(normalizeBillingAccount("   ")).toBeNull();
  });
});

describe("anthropic", () => {
  const body = { id: "msg_1", usage: { input_tokens: 3, output_tokens: 4 } };
  const req = { model: "claude-sonnet-4-5", max_tokens: 8, messages: [{ role: "user", content: "x" }] };

  function client(create: (...a: unknown[]) => unknown) {
    return { baseURL: "https://api.anthropic.com", messages: { create } };
  }

  it("sends the organization header as billing_account", async () => {
    const { transport, observations } = capture();
    const c = client(() => apiPromise(body, new Headers({ "anthropic-organization-id": "org-anth" })));
    anthropicProvider.install(c, transport, null, null, null, defaultHardening());
    const r = await (c.messages.create as any)(req);
    expect(r).toBe(body);
    await flush();
    expect(observations).toHaveLength(1);
    expect(observations[0].billing_account).toBe("org-anth");
  });

  it("sends no field when the header is absent", async () => {
    const { transport, observations } = capture();
    const c = client(() => apiPromise(body, new Headers({ "request-id": "req_1" })));
    anthropicProvider.install(c, transport, null, null, null, defaultHardening());
    await (c.messages.create as any)(req);
    await flush();
    expect(observations[0]).not.toHaveProperty("billing_account");
  });

  it("a header read that throws costs the field, not the call", async () => {
    const { transport, observations } = capture();
    const c = client(() => apiPromise(body, throwingHeaders));
    anthropicProvider.install(c, transport, null, null, null, defaultHardening());
    const r = await (c.messages.create as any)(req);
    expect(r).toBe(body);
    await flush();
    expect(observations[0].status).toBe("success");
    expect(observations[0]).not.toHaveProperty("billing_account");
  });

  it("asResponse() itself throwing or never settling is the same", async () => {
    for (const asResponse of [
      () => {
        throw new Error("no");
      },
      () => new Promise(() => undefined),
      () => Promise.reject(new Error("no")),
    ]) {
      const { transport, observations } = capture();
      const c = client(() => Object.assign(Promise.resolve(body), { asResponse }));
      anthropicProvider.install(c, transport, null, null, null, defaultHardening());
      expect(await (c.messages.create as any)(req)).toBe(body);
      await flush();
      expect(observations[0]).not.toHaveProperty("billing_account");
    }
  });

  it("a provider error keeps its headers, so the error observation is labelled too", async () => {
    const { transport, observations } = capture();
    const err = Object.assign(new Error("overloaded"), {
      status: 529,
      headers: new Headers({ "anthropic-organization-id": "org-anth" }),
    });
    const c = client(() => Promise.reject(err));
    anthropicProvider.install(c, transport, null, null, null, defaultHardening());
    await expect((c.messages.create as any)(req)).rejects.toBe(err);
    await flush();
    expect(observations[0].status).toBe("error");
    expect(observations[0].billing_account).toBe("org-anth");
  });

  it("a client pointed somewhere else (a gateway) sends no field", async () => {
    const { transport, observations } = capture();
    const c = {
      baseURL: "https://llm-gateway.corp",
      messages: { create: () => apiPromise(body, new Headers({ "anthropic-organization-id": "org-anth" })) },
    };
    anthropicProvider.install(c, transport, null, null, null, defaultHardening());
    await (c.messages.create as any)(req);
    await flush();
    expect(observations[0]).not.toHaveProperty("billing_account");
  });
});

describe("openai", () => {
  const body = { id: "chatcmpl-1", usage: { prompt_tokens: 1, completion_tokens: 2 } };
  const req = { model: "gpt-4o", messages: [{ role: "user", content: "x" }] };

  function client(baseURL: string, create: (...a: unknown[]) => unknown, extra: Record<string, unknown> = {}) {
    return { baseURL, chat: { completions: { create } }, ...extra };
  }

  it("sends openai-organization, not the project", async () => {
    const { transport, observations } = capture();
    const c = client("https://api.openai.com/v1", () =>
      apiPromise(body, new Headers({ "openai-organization": "org-oai", "openai-project": "proj_1" })),
    );
    openaiProvider.install(c, transport, null, null, null, defaultHardening());
    expect(await (c.chat.completions.create as any)(req)).toBe(body);
    await flush();
    expect(observations[0].billing_account).toBe("org-oai");
  });

  it("reads openai@4's plain-record headers on an error", async () => {
    const { transport, observations } = capture();
    const err = Object.assign(new Error("rate limited"), { status: 429, headers: { "openai-organization": "org-oai" } });
    const c = client("https://api.openai.com/v1", () => Promise.reject(err));
    openaiProvider.install(c, transport, null, null, null, defaultHardening());
    await expect((c.chat.completions.create as any)(req)).rejects.toBe(err);
    await flush();
    expect(observations[0].billing_account).toBe("org-oai");
  });

  it("sends no field when the header is absent", async () => {
    const { transport, observations } = capture();
    const c = client("https://api.openai.com/v1", () => apiPromise(body, new Headers()));
    openaiProvider.install(c, transport, null, null, null, defaultHardening());
    await (c.chat.completions.create as any)(req);
    await flush();
    expect(observations[0]).not.toHaveProperty("billing_account");
  });

  it("a header read that throws costs the field, not the call", async () => {
    const { transport, observations } = capture();
    const c = client("https://api.openai.com/v1", () => apiPromise(body, throwingHeaders));
    openaiProvider.install(c, transport, null, null, null, defaultHardening());
    expect(await (c.chat.completions.create as any)(req)).toBe(body);
    await flush();
    expect(observations[0].status).toBe("success");
    expect(observations[0]).not.toHaveProperty("billing_account");
  });

  it("a streaming call is labelled when the stream ends, and the stream is the caller's", async () => {
    const { transport, observations } = capture();
    const stream = { [Symbol.asyncIterator]: async function* () {} };
    const c = client("https://api.openai.com/v1", () =>
      apiPromise(stream, new Headers({ "openai-organization": "org-oai" })),
    );
    openaiProvider.install(c, transport, null, null, null, defaultHardening());
    const got = await (c.chat.completions.create as any)({ ...req, stream: true });
    expect(got).toBe(stream);
    for await (const _ of got) void _;
    await flush();
    expect(observations[0].billing_account).toBe("org-oai");
  });

  it("a call the proxy made carries no field (the proxy labels it)", async () => {
    const { transport, observations } = capture();
    const direct = () => apiPromise(body, new Headers({ "openai-organization": "org-oai" }));
    const c = client("https://api.openai.com/v1", direct, { apiKey: "sk-test" });
    openaiProvider.install(
      c,
      transport,
      null,
      null,
      null,
      defaultHardening({
        maxLayer: "route",
        appliedMode: true,
        proxyUrl: "https://proxy.local",
        retry: { maxAttempts: 1, baseDelayMs: 1, maxDelayMs: 2 },
      }),
    );
    (transport as any).getToken = async () => "jwt";
    const saved = globalThis.fetch;
    globalThis.fetch = (async () =>
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json", "openai-organization": "org-oai" },
      })) as typeof fetch;
    try {
      await (c.chat.completions.create as any)(req);
    } finally {
      globalThis.fetch = saved;
    }
    await flush();
    expect(observations).toHaveLength(1);
    expect(observations[0]).not.toHaveProperty("billing_account");
  });
});

describe("azure_openai", () => {
  const body = { id: "chatcmpl-az", usage: { prompt_tokens: 1, completion_tokens: 2 } };

  it("names the resource from the endpoint host, lower-cased, with or without headers", async () => {
    const { transport, observations } = capture();
    const c = {
      baseURL: "https://Contoso-EastUS.openai.azure.com/openai",
      chat: { completions: { create: () => Promise.resolve(body) } },
    };
    openaiProvider.install(c, transport, null, null, null, defaultHardening());
    expect(await (c.chat.completions.create as any)({ model: "gpt-4o", messages: [] })).toBe(body);
    await flush();
    expect(observations[0].provider).toBe("azure_openai");
    expect(observations[0].billing_account).toBe("contoso-eastus");
  });

  it("an Azure client whose endpoint isn't an Azure host sends no field", async () => {
    const { transport, observations } = capture();
    const c = {
      azure: true,
      baseURL: "https://apim.corp/openai",
      chat: { completions: { create: () => Promise.resolve(body) } },
    };
    openaiProvider.install(c, transport, null, null, null, defaultHardening());
    await (c.chat.completions.create as any)({ model: "gpt-4o", messages: [] });
    await flush();
    expect(observations[0]).not.toHaveProperty("billing_account");
  });

  it("a baseURL getter that throws costs the field, not the call", async () => {
    const { transport, observations } = capture();
    const c = {
      get baseURL(): string {
        throw new Error("no");
      },
      azure: true,
      chat: { completions: { create: () => Promise.resolve(body) } },
    };
    openaiProvider.install(c, transport, null, null, null, defaultHardening());
    expect(await (c.chat.completions.create as any)({ model: "gpt-4o", messages: [] })).toBe(body);
    await flush();
    expect(observations[0]).not.toHaveProperty("billing_account");
  });
});

describe("vertex", () => {
  const resp = { usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 2 } };

  /** The fields of a @google/genai GoogleGenAI client this reads. */
  function genai(clientOptions: Record<string, unknown>, getRequestUrl?: () => string) {
    const apiClient = {
      clientOptions,
      getRequestUrl:
        getRequestUrl ??
        (() => `${(clientOptions.httpOptions as { baseUrl: string }).baseUrl.replace(/\/$/, "")}/v1beta1`),
    };
    return { apiClient, models: { generateContent: async () => resp, generateContentStream: async () => ({}) } };
  }

  it("sends the project from the request URL", async () => {
    const { transport, observations } = capture();
    const c = genai({
      vertexai: true,
      project: "my-proj",
      location: "us-central1",
      httpOptions: { baseUrl: "https://us-central1-aiplatform.googleapis.com/" },
    });
    expect(vertexCallUrl(c)).toBe(
      "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-proj/locations/us-central1/publishers/google/models",
    );
    vertexProvider.install(c, transport, null, null, null);
    expect(await c.models.generateContent()).toBe(resp);
    await c.models.generateContentStream();
    await flush();
    expect(observations.map((o) => o.billing_account)).toEqual(["my-proj", "my-proj"]);
  });

  it("express mode (an API key, no project in the path) and the Gemini API send no field", async () => {
    for (const opts of [
      { vertexai: true, apiKey: "k", httpOptions: { baseUrl: "https://aiplatform.googleapis.com/" } },
      { vertexai: false, apiKey: "k", httpOptions: { baseUrl: "https://generativelanguage.googleapis.com/" } },
    ]) {
      const { transport, observations } = capture();
      const c = genai(opts);
      vertexProvider.install(c, transport, null, null, null);
      await c.models.generateContent();
      await flush();
      expect(observations[0]).not.toHaveProperty("billing_account");
    }
  });

  it("a URL read that throws costs the field, not the call", async () => {
    const { transport, observations } = capture();
    const c = genai({ vertexai: true, project: "p", location: "global" }, () => {
      throw new Error("Base URL is not set.");
    });
    vertexProvider.install(c, transport, null, null, null);
    expect(await c.models.generateContent()).toBe(resp);
    await flush();
    expect(observations[0].status).toBe("success");
    expect(observations[0]).not.toHaveProperty("billing_account");
  });
});

describe("bedrock", () => {
  /** A Smithy client whose stack runs our middleware with the context the auth-scheme middleware filled. */
  function smithy(context: Record<string, unknown>, terminal: () => Promise<any>) {
    let mw: any;
    return {
      config: { serviceId: "Bedrock Runtime" },
      middlewareStack: {
        add(fn: unknown) {
          mw = fn;
        },
        identify: () => [],
      },
      send(command: { input: Record<string, unknown> }) {
        // Copy descriptors, not values, so a throwing getter stays a getter.
        const ctx = Object.defineProperties({ commandName: "ConverseCommand" }, Object.getOwnPropertyDescriptors(context));
        return mw(terminal, ctx)({ input: command.input });
      },
    };
  }
  const out = { output: { usage: { inputTokens: 1, outputTokens: 2 } } };
  const identity = (accountId: unknown) => ({
    __smithy_context: { selectedHttpAuthScheme: { identity: { accessKeyId: "AKIA", accountId } } },
  });

  it("sends the account the signing credentials carry", async () => {
    const { transport, observations } = capture();
    const c = smithy(identity("123456789012"), async () => out);
    bedrockProvider.install(c, transport, null, null, null);
    expect(await c.send({ input: { modelId: "anthropic.claude-3" } })).toBe(out);
    await flush();
    expect(observations[0].billing_account).toBe("123456789012");
  });

  it("sends no field when the credentials don't carry an account, or it isn't 12 digits", async () => {
    for (const ctx of [identity(undefined), identity("1234"), {}]) {
      const { transport, observations } = capture();
      const c = smithy(ctx, async () => out);
      bedrockProvider.install(c, transport, null, null, null);
      await c.send({ input: { modelId: "m" } });
      await flush();
      expect(observations[0]).not.toHaveProperty("billing_account");
    }
  });

  it("an identity read that throws costs the field, not the call", async () => {
    const { transport, observations } = capture();
    const ctx = {
      get __smithy_context(): never {
        throw new Error("no");
      },
    };
    const c = smithy(ctx, async () => out);
    bedrockProvider.install(c, transport, null, null, null);
    expect(await c.send({ input: { modelId: "m" } })).toBe(out);
    await flush();
    expect(observations[0].status).toBe("success");
    expect(observations[0]).not.toHaveProperty("billing_account");
  });
});
