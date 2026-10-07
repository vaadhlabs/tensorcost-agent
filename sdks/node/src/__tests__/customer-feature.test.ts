/**
 * Tests for A-04 chargeback attribution — customer + feature.
 *
 * Mirrors agent-run.test.ts: dedicated top-level Observation fields (not the
 * metadata bag), wrap-time defaults, withMeta() per-call overrides, env var
 * resolution, backward compatibility when unset, and applied-mode headers.
 */

import { describe, expect, it } from "vitest";
import { wrap } from "../wrap.js";
import { ObservationTransport } from "../transport.js";
import type { Observation } from "../types.js";

function fakeOpenAIClient() {
  return {
    baseURL: "https://api.openai.com/v1",
    apiKey: "sk-fake",
    defaultHeaders: {} as Record<string, string>,
    chat: {
      completions: {
        create: async (_args: unknown) => ({
          id: "chatcmpl-1",
          usage: { prompt_tokens: 10, completion_tokens: 20 },
        }),
      },
    },
  };
}

function fakeAnthropicClient() {
  return {
    baseURL: "https://api.anthropic.com",
    apiKey: "anth-fake",
    defaultHeaders: {} as Record<string, string>,
    messages: {
      create: async (_args: unknown) => ({
        id: "msg_1",
        usage: { input_tokens: 15, output_tokens: 25 },
      }),
    },
  };
}

function fakeBedrockClient() {
  type Middleware = (
    next: (args: any) => Promise<any>,
    context: { commandName?: string },
  ) => (args: any) => Promise<any>;
  const middlewares: Middleware[] = [];

  return {
    config: { serviceId: "Bedrock Runtime" },
    middlewareStack: {
      add(mw: Middleware) {
        middlewares.push(mw);
      },
      identify() {
        return middlewares.map((m, i) => [`mw-${i}`, m] as [string, unknown]);
      },
    },
    async send(command: {
      constructor: { name: string };
      input: Record<string, unknown>;
    }) {
      const context = { commandName: command.constructor.name };
      const terminal = () =>
        Promise.resolve({ output: { usage: { inputTokens: 5, outputTokens: 7 } } });
      let chain: (args: any) => Promise<any> = (args) => terminal();
      for (let i = middlewares.length - 1; i >= 0; i--) {
        chain = middlewares[i]!(chain, context);
      }
      return chain({ input: command.input });
    },
  };
}

class InvokeModelCommand {
  constructor(public input: Record<string, unknown>) {}
}

function spyOnTransport(): {
  posted: Observation[];
  restore: () => void;
} {
  const posted: Observation[] = [];
  const original = ObservationTransport.prototype.post;
  ObservationTransport.prototype.post = function (obs: Observation) {
    posted.push(obs);
  };
  return {
    posted,
    restore: () => {
      ObservationTransport.prototype.post = original;
    },
  };
}

describe("wrap() customer/feature — wrap-time default", () => {
  it("OpenAI observation carries top-level customer and feature", async () => {
    const { posted, restore } = spyOnTransport();
    try {
      const client = fakeOpenAIClient();
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
        customer: "acme-corp",
        feature: "summarizer",
      });
      await (client as any).chat.completions.create({ model: "gpt-4o", messages: [] });

      expect(posted).toHaveLength(1);
      expect(posted[0]!.customer).toBe("acme-corp");
      expect(posted[0]!.feature).toBe("summarizer");
      expect(posted[0]!.metadata).toBeUndefined();
    } finally {
      restore();
    }
  });

  it("Anthropic observation carries top-level customer and feature", async () => {
    const { posted, restore } = spyOnTransport();
    try {
      const client = fakeAnthropicClient();
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
        customer: "globex",
        feature: "support-bot",
      });
      await (client as any).messages.create({
        model: "claude-3-5-sonnet-20241022",
        max_tokens: 10,
        messages: [],
      });

      expect(posted).toHaveLength(1);
      expect(posted[0]!.customer).toBe("globex");
      expect(posted[0]!.feature).toBe("support-bot");
    } finally {
      restore();
    }
  });
});

describe("withMeta() customer/feature — per-call override", () => {
  it("observation carries customer/feature from withMeta() on OpenAI", async () => {
    const { posted, restore } = spyOnTransport();
    try {
      const client = fakeOpenAIClient();
      wrap(client, { apiKey: "k", baseUrl: "http://localhost:1" });

      const proxied = (client as any).withMeta({
        customer: "per-call-customer",
        feature: "per-call-feature",
      });
      await proxied.chat.completions.create({ model: "gpt-4o", messages: [] });

      expect(posted).toHaveLength(1);
      expect(posted[0]!.customer).toBe("per-call-customer");
      expect(posted[0]!.feature).toBe("per-call-feature");
    } finally {
      restore();
    }
  });

  it("withMeta() customer/feature do not pollute the metadata bag", async () => {
    const { posted, restore } = spyOnTransport();
    try {
      const client = fakeOpenAIClient();
      wrap(client, { apiKey: "k", baseUrl: "http://localhost:1" });

      const proxied = (client as any).withMeta({
        customer: "c-1",
        feature: "f-1",
        promptTemplateId: "tmpl-1",
      });
      await proxied.chat.completions.create({ model: "gpt-4o", messages: [] });

      expect(posted[0]!.customer).toBe("c-1");
      expect(posted[0]!.feature).toBe("f-1");
      expect(posted[0]!.metadata?.prompt_template_id).toBe("tmpl-1");
      expect((posted[0]!.metadata as any)?.customer).toBeUndefined();
      expect((posted[0]!.metadata as any)?.feature).toBeUndefined();
    } finally {
      restore();
    }
  });

  it("per-call customer/feature override wrap-time defaults", async () => {
    const { posted, restore } = spyOnTransport();
    try {
      const client = fakeOpenAIClient();
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
        customer: "default-customer",
        feature: "default-feature",
      });

      const proxied = (client as any).withMeta({
        customer: "override-customer",
        feature: "override-feature",
      });
      await proxied.chat.completions.create({ model: "gpt-4o", messages: [] });

      expect(posted[0]!.customer).toBe("override-customer");
      expect(posted[0]!.feature).toBe("override-feature");
    } finally {
      restore();
    }
  });
});

describe("absent customer/feature (backward compatibility)", () => {
  it("neither key is present on the wire object when unconfigured", async () => {
    const { posted, restore } = spyOnTransport();
    try {
      const client = fakeOpenAIClient();
      wrap(client, { apiKey: "k", baseUrl: "http://localhost:1" });
      await (client as any).chat.completions.create({ model: "gpt-4o", messages: [] });

      expect(posted).toHaveLength(1);
      expect(Object.prototype.hasOwnProperty.call(posted[0], "customer")).toBe(false);
      expect(Object.prototype.hasOwnProperty.call(posted[0], "feature")).toBe(false);
    } finally {
      restore();
    }
  });

  it("TENSORCOST_CUSTOMER / TENSORCOST_FEATURE env vars resolve when set", async () => {
    const savedCustomer = process.env.TENSORCOST_CUSTOMER;
    const savedFeature = process.env.TENSORCOST_FEATURE;
    process.env.TENSORCOST_CUSTOMER = "env-customer";
    process.env.TENSORCOST_FEATURE = "env-feature";
    const { posted, restore } = spyOnTransport();
    try {
      const client = fakeOpenAIClient();
      wrap(client, { apiKey: "k", baseUrl: "http://localhost:1" });
      await (client as any).chat.completions.create({ model: "gpt-4o", messages: [] });

      expect(posted[0]!.customer).toBe("env-customer");
      expect(posted[0]!.feature).toBe("env-feature");
    } finally {
      restore();
      if (savedCustomer === undefined) delete process.env.TENSORCOST_CUSTOMER;
      else process.env.TENSORCOST_CUSTOMER = savedCustomer;
      if (savedFeature === undefined) delete process.env.TENSORCOST_FEATURE;
      else process.env.TENSORCOST_FEATURE = savedFeature;
    }
  });
});

describe("Bedrock — wrap-time only", () => {
  it("InvokeModel observation carries wrap-time customer and feature", async () => {
    const { posted, restore } = spyOnTransport();
    try {
      const client = fakeBedrockClient();
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
        customer: "bedrock-customer",
        feature: "bedrock-feature",
      });

      await (client as any).send(
        new InvokeModelCommand({ modelId: "anthropic.claude-3-sonnet" }),
      );

      expect(posted).toHaveLength(1);
      expect(posted[0]!.customer).toBe("bedrock-customer");
      expect(posted[0]!.feature).toBe("bedrock-feature");
    } finally {
      restore();
    }
  });
});

describe("applied mode — x-tc-customer / x-tc-feature headers", () => {
  it("installs chargeback headers on defaultHeaders when appliedMode=true", () => {
    const client = fakeOpenAIClient();
    wrap(client, {
      apiKey: "k",
      baseUrl: "http://localhost:1",
      appliedMode: true,
      proxyUrl: "https://proxy.tensorcost.com",
      customer: "header-customer",
      feature: "header-feature",
    });

    const headers = client.defaultHeaders as Record<string, string>;
    expect(headers["x-tc-customer"]).toBe("header-customer");
    expect(headers["x-tc-feature"]).toBe("header-feature");
  });
});
