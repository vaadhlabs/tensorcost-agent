import { describe, expect, it } from "vitest";
import { wrap } from "../wrap.js";
import { MissingConfigError, TensorCostConfigError } from "../config.js";
import { ObservationTransport } from "../transport.js";
import * as bedrockProvider from "../providers/bedrock.js";

function fakeOpenAIClient() {
  return {
    chat: {
      completions: {
        create: async (_args: unknown) => ({
          id: "chatcmpl-1",
          usage: { prompt_tokens: 1, completion_tokens: 2 },
        }),
      },
    },
  };
}

// Minimal BedrockRuntimeClient fake (mirrors makeFakeBedrockClient in bedrock.test.ts).
function fakeBedrockClient() {
  type MW = {
    fn: (next: (a: any) => Promise<any>, ctx: { commandName?: string }) => (a: any) => Promise<any>;
    name: string;
  };
  const middlewares: MW[] = [];

  return {
    config: { serviceId: "Bedrock Runtime" },
    middlewareStack: {
      add(fn: MW["fn"], opts: { name: string; step: string }) {
        middlewares.push({ fn, name: opts.name });
      },
      identify() {
        return middlewares.map((m) => [m.name, m.fn] as [string, unknown]);
      },
    },
    middlewares,
    async send(command: { constructor: { name: string }; input: Record<string, unknown> }) {
      const context = { commandName: command.constructor.name };
      let chain: (a: any) => Promise<any> = async () => ({ output: {} });
      for (let i = middlewares.length - 1; i >= 0; i--) {
        const current = chain;
        chain = middlewares[i].fn(current, context);
      }
      return chain({ input: command.input });
    },
  };
}

describe("wrap()", () => {
  it("returns the same client reference (mutates in place)", () => {
    const c = fakeOpenAIClient();
    const wrapped = wrap(c, {
      apiKey: "k",
      baseUrl: "http://localhost:1",
    });
    expect(wrapped).toBe(c);
  });

  it("replaces the create method with a wrapped one", () => {
    const c = fakeOpenAIClient();
    const original = c.chat.completions.create;
    wrap(c, { apiKey: "k", baseUrl: "http://localhost:1" });
    expect(c.chat.completions.create).not.toBe(original);
  });

  it("propagates MissingConfigError when apiKey is unset", () => {
    const saved = process.env.TENSORCOST_API_KEY;
    delete process.env.TENSORCOST_API_KEY;
    try {
      expect(() => wrap(fakeOpenAIClient())).toThrow(MissingConfigError);
    } finally {
      if (saved !== undefined) process.env.TENSORCOST_API_KEY = saved;
    }
  });

  it("does not double-wrap when called twice", () => {
    const c = fakeOpenAIClient();
    wrap(c, { apiKey: "k", baseUrl: "http://localhost:1" });
    const after1 = c.chat.completions.create;
    wrap(c, { apiKey: "k", baseUrl: "http://localhost:1" });
    expect(c.chat.completions.create).toBe(after1);
  });

  // ── Bedrock ───────────────────────────────────────────────────────────────

  it("returns the same Bedrock client reference (observe-only)", () => {
    const c = fakeBedrockClient();
    const wrapped = wrap(c, { apiKey: "k", baseUrl: "http://localhost:1" });
    expect(wrapped).toBe(c);
  });

  it("registers the Bedrock middleware when wrapping a BedrockRuntimeClient", () => {
    const c = fakeBedrockClient();
    wrap(c, { apiKey: "k", baseUrl: "http://localhost:1" });
    expect(c.middlewareStack.identify().length).toBe(1);
    expect(c.middlewareStack.identify()[0][0]).toBe(
      bedrockProvider.__test__.MIDDLEWARE_NAME,
    );
  });

  it("throws TensorCostConfigError when maxLayer route is used with a Bedrock client", () => {
    const c = fakeBedrockClient();
    expect(() =>
      wrap(c, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
        maxLayer: "route",
        proxyUrl: "http://proxy.local",
      }),
    ).toThrow(TensorCostConfigError);
  });

  it("maxLayer govern on Bedrock succeeds (observe-only telemetry)", () => {
    const c = fakeBedrockClient();
    const wrapped = wrap(c, {
      apiKey: "k",
      baseUrl: "http://localhost:1",
      maxLayer: "govern",
    });
    expect(wrapped).toBe(c);
    expect(c.middlewareStack.identify().length).toBe(1);
  });

  it("maxLayer off returns client unchanged with no middleware", () => {
    const c = fakeOpenAIClient();
    const original = c.chat.completions.create;
    const wrapped = wrap(c, {
      apiKey: "k",
      baseUrl: "http://localhost:1",
      maxLayer: "off",
    });
    expect(wrapped).toBe(c);
    expect(c.chat.completions.create).toBe(original);
  });
});
