/**
 * Tests for the Bedrock observe-only provider wrapper.
 *
 * We don't install @aws-sdk/client-bedrock-runtime as a dev dep — the
 * middleware interception only needs the shape of the Smithy client, so
 * we build minimal fakes that match the parts our code actually touches:
 *   - client.send(command) → the method customers call
 *   - client.middlewareStack.add(middleware, opts) → where we register
 *   - client.middlewareStack.identify() → double-wrap guard
 *   - command.constructor.name → commandName passed into the middleware context
 *
 * Each test drives a middleware chain end-to-end: the registered middleware
 * wraps a "next" that returns a fake resolved output (or throws), and we
 * verify the observation posted to the transport has the right shape.
 */

import { describe, expect, it, vi, beforeEach } from "vitest";
import { ObservationTransport } from "../transport.js";
import * as bedrockProvider from "../providers/bedrock.js";
import { TensorCostConfigError } from "../config.js";
import { wrap } from "../wrap.js";

// ── Transport stub ────────────────────────────────────────────────────────────

function fakeTransport(): { transport: ObservationTransport; observations: any[] } {
  const observations: any[] = [];
  const log = { observations };

  const fetchImpl: typeof fetch = async (input, init = {}) => {
    const url = typeof input === "string" ? input : (input as URL).toString();
    if (url.includes("sdk-token/exchange")) {
      return new Response(
        JSON.stringify({ token: "tok", expires_in: 900 }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }
    if (url.includes("/api/inference-proxy/observation")) {
      log.observations.push(JSON.parse(init?.body as string));
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
  for (let i = 0; i < times; i++) await Promise.resolve();
}

// ── Smithy client fake ────────────────────────────────────────────────────────

/**
 * Builds a fake BedrockRuntimeClient that satisfies our detection heuristic
 * (send + config.serviceId === "Bedrock Runtime") AND a minimal middlewareStack.
 *
 * We simulate the Smithy middleware execution: when send() is called, it
 * iterates through registered middlewares in reverse-add order, each wrapping
 * the next. The innermost "next" is `terminalHandler`, which we swap per-test.
 */
function makeFakeBedrockClient(opts: {
  commandName?: string;
  terminalHandler?: (args: { input: Record<string, unknown> }) => Promise<{
    output: Record<string, unknown>;
    response?: { headers?: Record<string, string> };
  }>;
} = {}) {
  // Middlewares registered via add() are stored here.
  type Middleware = {
    fn: (
      next: (args: any) => Promise<any>,
      context: { commandName?: string },
    ) => (args: any) => Promise<any>;
    name: string;
  };
  const middlewares: Middleware[] = [];

  const client = {
    config: { serviceId: "Bedrock Runtime" },
    middlewareStack: {
      add(
        middleware: (
          next: (args: any) => Promise<any>,
          context: { commandName?: string },
        ) => (args: any) => Promise<any>,
        options: { step: string; name: string },
      ) {
        middlewares.push({ fn: middleware, name: options.name });
      },
      identify() {
        return middlewares.map((m) => [m.name, m.fn] as [string, unknown]);
      },
    },

    async send(command: { constructor: { name: string }; input: Record<string, unknown> }) {
      const commandName = opts.commandName ?? command.constructor.name;
      const context = { commandName };

      // Build the chain: outer-most middleware wraps the next, which wraps
      // the next, ... which wraps the terminal handler. We add them in
      // registration order; the outermost (last-registered) runs first.
      const terminal = opts.terminalHandler ?? (() => Promise.resolve({ output: {} }));
      let chain: (args: any) => Promise<any> = (args) => terminal(args);

      // Apply middlewares in reverse order (last-added is outermost, per Smithy).
      for (let i = middlewares.length - 1; i >= 0; i--) {
        const current = chain;
        chain = middlewares[i].fn(current, context);
      }

      return chain({ input: command.input });
    },
  };

  return { client, middlewares };
}

// ── Detection tests ───────────────────────────────────────────────────────────
// (Full detection test matrix lives in detect.test.ts; these confirm the
// Bedrock detection path is wired correctly when going through install().)

describe("bedrockProvider.install()", () => {
  it("registers a middleware on a valid BedrockRuntimeClient", () => {
    const { client } = makeFakeBedrockClient();
    const { transport } = fakeTransport();
    bedrockProvider.install(client, transport, null, null, null);
    expect(client.middlewareStack.identify().length).toBe(1);
    expect(client.middlewareStack.identify()[0][0]).toBe(
      bedrockProvider.__test__.MIDDLEWARE_NAME,
    );
  });

  it("does not double-register when install() is called twice", () => {
    const { client } = makeFakeBedrockClient();
    const { transport } = fakeTransport();
    bedrockProvider.install(client, transport, null, null, null);
    bedrockProvider.install(client, transport, null, null, null);
    expect(client.middlewareStack.identify().length).toBe(1);
  });

  it("warns and bails when the client has no middlewareStack", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { transport } = fakeTransport();
    const clientWithNoStack = { config: { serviceId: "Bedrock Runtime" } };
    bedrockProvider.install(clientWithNoStack, transport, null, null, null);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("middlewareStack"));
    warn.mockRestore();
  });
});

// ── InvokeModel: Anthropic-on-Bedrock response shape ─────────────────────────

describe("bedrock InvokeModelCommand — token counts for any model family", () => {
  async function invoke(handler: () => Promise<{
    output: Record<string, unknown>;
    response?: { headers?: Record<string, string> };
  }>) {
    const { transport, observations } = fakeTransport();
    const { client } = makeFakeBedrockClient({
      commandName: "InvokeModelCommand",
      terminalHandler: handler,
    });
    bedrockProvider.install(client, transport, "tenant-z", null, null);
    await client.send({
      constructor: { name: "InvokeModelCommand" },
      input: { modelId: "us.meta.llama3-1-8b-instruct-v1:0", body: Buffer.from("{}") },
    });
    await flush();
    return observations[0]!;
  }

  it("reads Meta Llama's prompt_token_count / generation_token_count", async () => {
    const obs = await invoke(async () => ({
      output: {
        body: Buffer.from(
          JSON.stringify({ generation: "hi", prompt_token_count: 31, generation_token_count: 12 }),
        ),
      },
    }));
    expect(obs.input_tokens).toBe(31);
    expect(obs.output_tokens).toBe(12);
  });

  it("prefers the x-amzn-bedrock token headers, which every model family sets", async () => {
    const obs = await invoke(async () => ({
      output: { body: Buffer.from(JSON.stringify({ outputs: [{ text: "hi" }] })) },
      response: {
        headers: {
          "x-amzn-bedrock-input-token-count": "55",
          "x-amzn-bedrock-output-token-count": "9",
        },
      },
    }));
    expect(obs.input_tokens).toBe(55);
    expect(obs.output_tokens).toBe(9);
  });
});

describe("bedrock InvokeModelCommand — Anthropic-on-Bedrock token shape", () => {
  it("emits an observation with the right model + tokens on success", async () => {
    const { transport, observations } = fakeTransport();

    const responseBody = JSON.stringify({
      id: "msg_01",
      usage: { input_tokens: 42, output_tokens: 17 },
    });

    const { client } = makeFakeBedrockClient({
      commandName: "InvokeModelCommand",
      terminalHandler: async () => ({
        output: {
          body: Buffer.from(responseBody),
        },
      }),
    });

    bedrockProvider.install(client, transport, "tenant-z", null, null);

    const command = {
      constructor: { name: "InvokeModelCommand" },
      input: {
        modelId: "anthropic.claude-3-haiku-20240307-v1:0",
        body: Buffer.from(JSON.stringify({ messages: [] })),
      },
    };

    await client.send(command);
    await flush();

    expect(observations).toHaveLength(1);
    const obs = observations[0];
    expect(obs.provider).toBe("bedrock");
    expect(obs.model).toBe("anthropic.claude-3-haiku-20240307-v1:0");
    expect(obs.operation).toBe("bedrock.invoke_model");
    expect(obs.input_tokens).toBe(42);
    expect(obs.output_tokens).toBe(17);
    expect(obs.status).toBe("success");
    expect(obs.error_message).toBeNull();
    expect(obs.tenant_id).toBeUndefined();
    expect(obs.sdk_added_ms).toBeTypeOf("number");
    expect(obs.sdk_added_ms).toBeGreaterThanOrEqual(0);
  });

  it("sdk_added_ms excludes slow provider RTT", async () => {
    const { transport, observations } = fakeTransport();
    const { client } = makeFakeBedrockClient({
      commandName: "InvokeModelCommand",
      terminalHandler: async () => {
        await new Promise((r) => setTimeout(r, 40));
        return {
          output: {
            body: Buffer.from(JSON.stringify({ usage: { input_tokens: 1, output_tokens: 1 } })),
          },
        };
      },
    });

    bedrockProvider.install(client, transport, null, null, null);
    await client.send({
      constructor: { name: "InvokeModelCommand" },
      input: { modelId: "anthropic.claude-3-haiku-20240307-v1:0", body: Buffer.from("{}") },
    });
    await flush();

    expect(observations[0].sdk_added_ms).toBeLessThan(20);
  });
});

// ── InvokeModel: Amazon Nova / Titan response shape ───────────────────────────

describe("bedrock InvokeModelCommand — Nova/Titan token shape", () => {
  it("reads inputTokens/outputTokens for Nova Micro", async () => {
    const { transport, observations } = fakeTransport();

    const responseBody = JSON.stringify({
      output: { message: { content: [] } },
      usage: { inputTokens: 100, outputTokens: 55 },
    });

    const { client } = makeFakeBedrockClient({
      commandName: "InvokeModelCommand",
      terminalHandler: async () => ({
        output: { body: Buffer.from(responseBody) },
      }),
    });

    bedrockProvider.install(client, transport, null, null, null);

    await client.send({
      constructor: { name: "InvokeModelCommand" },
      input: { modelId: "amazon.nova-micro-v1:0" },
    });
    await flush();

    expect(observations).toHaveLength(1);
    expect(observations[0].input_tokens).toBe(100);
    expect(observations[0].output_tokens).toBe(55);
    expect(observations[0].model).toBe("amazon.nova-micro-v1:0");
    expect(observations[0].status).toBe("success");
  });
});

// ── ConverseCommand ───────────────────────────────────────────────────────────

describe("bedrock ConverseCommand", () => {
  it("emits observation with correct token counts", async () => {
    const { transport, observations } = fakeTransport();

    const { client } = makeFakeBedrockClient({
      commandName: "ConverseCommand",
      terminalHandler: async () => ({
        output: {
          output: { message: {} },
          usage: { inputTokens: 30, outputTokens: 10 },
        },
      }),
    });

    bedrockProvider.install(client, transport, null, "staging", null);

    await client.send({
      constructor: { name: "ConverseCommand" },
      input: { modelId: "anthropic.claude-3-5-sonnet-20241022-v2:0" },
    });
    await flush();

    expect(observations).toHaveLength(1);
    const obs = observations[0];
    expect(obs.operation).toBe("bedrock.converse");
    expect(obs.input_tokens).toBe(30);
    expect(obs.output_tokens).toBe(10);
    expect(obs.status).toBe("success");
    expect(obs.environment).toBe("staging");
  });
});

// ── Error path ───────────────────────────────────────────────────────────────

describe("bedrock command error handling", () => {
  it("re-throws the original error AND emits an error observation", async () => {
    const { transport, observations } = fakeTransport();

    const boom = new Error("ThrottlingException: too many requests");

    const { client } = makeFakeBedrockClient({
      commandName: "InvokeModelCommand",
      terminalHandler: async () => {
        throw boom;
      },
    });

    bedrockProvider.install(client, transport, null, null, null);

    await expect(
      client.send({
        constructor: { name: "InvokeModelCommand" },
        input: { modelId: "amazon.nova-micro-v1:0" },
      }),
    ).rejects.toThrow(boom);

    await flush();

    expect(observations).toHaveLength(1);
    expect(observations[0].status).toBe("error");
    expect(observations[0].error_message).toContain("ThrottlingException");
    expect(observations[0].input_tokens).toBeNull();
    expect(observations[0].output_tokens).toBeNull();
  });
});

// ── Streaming commands — placeholder observation ──────────────────────────────

describe("bedrock streaming commands", () => {
  it("InvokeModelWithResponseStreamCommand emits a placeholder observation", async () => {
    const { transport, observations } = fakeTransport();

    const { client } = makeFakeBedrockClient({
      commandName: "InvokeModelWithResponseStreamCommand",
      terminalHandler: async () => ({
        output: { body: {} }, // the actual stream object; we don't parse it
      }),
    });

    bedrockProvider.install(client, transport, null, null, "conn-456");

    await client.send({
      constructor: { name: "InvokeModelWithResponseStreamCommand" },
      input: { modelId: "amazon.nova-pro-v1:0" },
    });
    await flush();

    expect(observations).toHaveLength(1);
    const obs = observations[0];
    expect(obs.operation).toBe("bedrock.invoke_model_stream");
    expect(obs.input_tokens).toBeNull();
    expect(obs.output_tokens).toBeNull();
    expect(obs.status).toBe("success");
    // Operators should see this in the dashboard and know it's a streaming gap.
    expect(obs.error_message).toContain("streaming");
    expect(obs.connection_id).toBe("conn-456");
  });

  it("ConverseStreamCommand emits a placeholder observation", async () => {
    const { transport, observations } = fakeTransport();

    const { client } = makeFakeBedrockClient({
      commandName: "ConverseStreamCommand",
      terminalHandler: async () => ({
        output: { stream: {} },
      }),
    });

    bedrockProvider.install(client, transport, null, null, null);

    await client.send({
      constructor: { name: "ConverseStreamCommand" },
      input: { modelId: "anthropic.claude-3-5-sonnet-20241022-v2:0" },
    });
    await flush();

    expect(observations).toHaveLength(1);
    expect(observations[0].operation).toBe("bedrock.converse_stream");
    expect(observations[0].input_tokens).toBeNull();
  });
});

// ── Streaming error ───────────────────────────────────────────────────────────

describe("bedrock streaming command error", () => {
  it("re-throws the error AND emits an error observation for streaming commands", async () => {
    const { transport, observations } = fakeTransport();

    const boom = new Error("ServiceUnavailableException");

    const { client } = makeFakeBedrockClient({
      commandName: "ConverseStreamCommand",
      terminalHandler: async () => {
        throw boom;
      },
    });

    bedrockProvider.install(client, transport, null, null, null);

    await expect(
      client.send({
        constructor: { name: "ConverseStreamCommand" },
        input: { modelId: "amazon.nova-micro-v1:0" },
      }),
    ).rejects.toThrow(boom);

    await flush();
    expect(observations).toHaveLength(1);
    expect(observations[0].status).toBe("error");
    expect(observations[0].error_message).toContain("ServiceUnavailableException");
  });
});

// ── Applied-mode guard via wrap() ─────────────────────────────────────────────

describe("wrap() + Bedrock applied mode guard", () => {
  it("throws TensorCostConfigError when maxLayer route is passed with a Bedrock client", () => {
    const { client } = makeFakeBedrockClient();

    expect(() =>
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
        maxLayer: "route",
        proxyUrl: "http://proxy.local",
      }),
    ).toThrow(TensorCostConfigError);
  });

  it("wrap() succeeds with maxLayer govern", () => {
    const { client } = makeFakeBedrockClient();
    expect(() =>
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
        maxLayer: "govern",
      }),
    ).not.toThrow();
  });

  it("wrap() succeeds (observe-only) without appliedMode", () => {
    const { client } = makeFakeBedrockClient();
    expect(() =>
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
      }),
    ).not.toThrow();
    // The middleware was registered.
    expect(client.middlewareStack.identify().length).toBe(1);
  });
});

// ── connection_id stamping ────────────────────────────────────────────────────

describe("bedrock observation envelope fields", () => {
  it("stamps connection_id when configured", async () => {
    const { transport, observations } = fakeTransport();

    const { client } = makeFakeBedrockClient({
      commandName: "InvokeModelCommand",
      terminalHandler: async () => ({
        output: { body: Buffer.from(JSON.stringify({ usage: { input_tokens: 1, output_tokens: 1 } })) },
      }),
    });

    bedrockProvider.install(client, transport, "tid", "prod", "conn-789");

    await client.send({
      constructor: { name: "InvokeModelCommand" },
      input: { modelId: "anthropic.claude-3-haiku-20240307-v1:0" },
    });
    await flush();

    expect(observations[0].connection_id).toBe("conn-789");
    expect(observations[0].environment).toBe("prod");
    expect(observations[0].tenant_id).toBeUndefined();
  });
});
