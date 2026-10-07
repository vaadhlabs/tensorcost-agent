/**
 * Tests for per-agent-run cost attribution — agent_id + workflow_id.
 *
 * Companion to metadata.test.ts (deployment / promptTemplateId), which
 * established the wrap-time-default + withMeta()-per-call-override pattern
 * this reuses. The key structural difference under test: agent_id /
 * workflow_id are dedicated top-level Observation fields, NOT members of
 * the `metadata` JSONB bag — see the Observation.agent_id doc in types.ts
 * for why (the metadata bag never reaches ai_spend_events.agent_id).
 *
 * Scenarios:
 *   1. wrap({ agentId }) -> observation.agent_id set, no metadata pollution.
 *   2. withMeta({ workflowId }) per-call -> observation.workflow_id set.
 *   3. wrap-time agentId + per-call workflowId together.
 *   4. per-call agentId overrides wrap-time default.
 *   5. absent-identity case: neither field configured -> both keys absent
 *      from the wire object entirely (backward compatibility).
 *   6. Bedrock: wrap-time only (no withMeta hook exists for Bedrock).
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

/**
 * Minimal fake satisfying the SDK's Bedrock detection heuristic
 * (send + config.serviceId === "Bedrock Runtime") plus a middlewareStack,
 * mirroring bedrock.test.ts's makeFakeBedrockClient.
 */
function fakeBedrockClient() {
  type Middleware = (next: (args: any) => Promise<any>, context: { commandName?: string }) => (args: any) => Promise<any>;
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
    async send(command: { constructor: { name: string }; input: Record<string, unknown> }) {
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

describe("wrap() agent_id — wrap-time default", () => {
  it("OpenAI observation carries top-level agent_id, not metadata.agent_id", async () => {
    const { posted, restore } = spyOnTransport();
    try {
      const client = fakeOpenAIClient();
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
        agentId: "code-review-agent",
      });
      await (client as any).chat.completions.create({ model: "gpt-4o", messages: [] });

      expect(posted).toHaveLength(1);
      expect(posted[0]!.agent_id).toBe("code-review-agent");
      // Must NOT leak into the metadata bag — the mapper reads the
      // top-level field only.
      expect(posted[0]!.metadata).toBeUndefined();
    } finally {
      restore();
    }
  });

  it("Anthropic observation carries top-level agent_id", async () => {
    const { posted, restore } = spyOnTransport();
    try {
      const client = fakeAnthropicClient();
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
        agentId: "support-triage-bot",
      });
      await (client as any).messages.create({
        model: "claude-3-5-sonnet-20241022",
        max_tokens: 10,
        messages: [],
      });

      expect(posted).toHaveLength(1);
      expect(posted[0]!.agent_id).toBe("support-triage-bot");
    } finally {
      restore();
    }
  });
});

describe("withMeta() workflow_id — per-call override", () => {
  it("observation carries workflow_id from withMeta() on OpenAI", async () => {
    const { posted, restore } = spyOnTransport();
    try {
      const client = fakeOpenAIClient();
      wrap(client, { apiKey: "k", baseUrl: "http://localhost:1" });

      const proxied = (client as any).withMeta({ workflowId: "run-8f21" });
      await proxied.chat.completions.create({ model: "gpt-4o", messages: [] });

      expect(posted).toHaveLength(1);
      expect(posted[0]!.workflow_id).toBe("run-8f21");
      expect(posted[0]!.agent_id).toBeUndefined();
    } finally {
      restore();
    }
  });

  it("observation carries workflow_id from withMeta() on Anthropic", async () => {
    const { posted, restore } = spyOnTransport();
    try {
      const client = fakeAnthropicClient();
      wrap(client, { apiKey: "k", baseUrl: "http://localhost:1" });

      const proxied = (client as any).withMeta({ workflowId: "run-cafe01" });
      await proxied.messages.create({
        model: "claude-3-5-sonnet-20241022",
        max_tokens: 10,
        messages: [],
      });

      expect(posted).toHaveLength(1);
      expect(posted[0]!.workflow_id).toBe("run-cafe01");
    } finally {
      restore();
    }
  });

  it("withMeta() workflowId does not pollute the metadata bag", async () => {
    const { posted, restore } = spyOnTransport();
    try {
      const client = fakeOpenAIClient();
      wrap(client, { apiKey: "k", baseUrl: "http://localhost:1" });

      const proxied = (client as any).withMeta({
        workflowId: "run-1",
        agentId: "agent-1",
        promptTemplateId: "tmpl-1",
      });
      await proxied.chat.completions.create({ model: "gpt-4o", messages: [] });

      expect(posted[0]!.workflow_id).toBe("run-1");
      expect(posted[0]!.agent_id).toBe("agent-1");
      // promptTemplateId still goes through the metadata bag, untouched.
      expect(posted[0]!.metadata?.prompt_template_id).toBe("tmpl-1");
      // agent_id / workflow_id must NOT also appear inside metadata.
      expect((posted[0]!.metadata as any)?.agentId).toBeUndefined();
      expect((posted[0]!.metadata as any)?.workflowId).toBeUndefined();
    } finally {
      restore();
    }
  });
});

describe("wrap-time agentId + per-call workflowId together", () => {
  it("both land on the same observation", async () => {
    const { posted, restore } = spyOnTransport();
    try {
      const client = fakeOpenAIClient();
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
        agentId: "code-review-agent",
      });

      const proxied = (client as any).withMeta({ workflowId: "run-42" });
      await proxied.chat.completions.create({ model: "gpt-4o", messages: [] });

      expect(posted).toHaveLength(1);
      expect(posted[0]!.agent_id).toBe("code-review-agent");
      expect(posted[0]!.workflow_id).toBe("run-42");
    } finally {
      restore();
    }
  });

  it("per-call agentId overrides the wrap-time default", async () => {
    const { posted, restore } = spyOnTransport();
    try {
      const client = fakeOpenAIClient();
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
        agentId: "default-agent",
      });

      const proxied = (client as any).withMeta({ agentId: "override-agent" });
      await proxied.chat.completions.create({ model: "gpt-4o", messages: [] });

      expect(posted[0]!.agent_id).toBe("override-agent");
    } finally {
      restore();
    }
  });

  it("plain call (no withMeta) uses the wrap-time agentId default only", async () => {
    const { posted, restore } = spyOnTransport();
    try {
      const client = fakeOpenAIClient();
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
        agentId: "default-agent",
      });

      await (client as any).chat.completions.create({ model: "gpt-4o", messages: [] });

      expect(posted[0]!.agent_id).toBe("default-agent");
      expect(posted[0]!.workflow_id).toBeUndefined();
    } finally {
      restore();
    }
  });
});

describe("absent-identity case (backward compatibility)", () => {
  it("neither agent_id nor workflow_id keys are present on the wire object when unconfigured", async () => {
    const { posted, restore } = spyOnTransport();
    try {
      const client = fakeOpenAIClient();
      wrap(client, { apiKey: "k", baseUrl: "http://localhost:1" });

      await (client as any).chat.completions.create({ model: "gpt-4o", messages: [] });

      expect(posted).toHaveLength(1);
      expect(Object.prototype.hasOwnProperty.call(posted[0], "agent_id")).toBe(false);
      expect(Object.prototype.hasOwnProperty.call(posted[0], "workflow_id")).toBe(false);
    } finally {
      restore();
    }
  });

  it("TENSORCOST_AGENT_ID / TENSORCOST_WORKFLOW_ID env vars resolve when set", async () => {
    const savedAgent = process.env.TENSORCOST_AGENT_ID;
    const savedWorkflow = process.env.TENSORCOST_WORKFLOW_ID;
    process.env.TENSORCOST_AGENT_ID = "env-agent";
    process.env.TENSORCOST_WORKFLOW_ID = "env-run";
    const { posted, restore } = spyOnTransport();
    try {
      const client = fakeOpenAIClient();
      wrap(client, { apiKey: "k", baseUrl: "http://localhost:1" });
      await (client as any).chat.completions.create({ model: "gpt-4o", messages: [] });

      expect(posted[0]!.agent_id).toBe("env-agent");
      expect(posted[0]!.workflow_id).toBe("env-run");
    } finally {
      restore();
      if (savedAgent === undefined) delete process.env.TENSORCOST_AGENT_ID;
      else process.env.TENSORCOST_AGENT_ID = savedAgent;
      if (savedWorkflow === undefined) delete process.env.TENSORCOST_WORKFLOW_ID;
      else process.env.TENSORCOST_WORKFLOW_ID = savedWorkflow;
    }
  });
});

describe("Bedrock — wrap-time only (no per-call hook)", () => {
  it("InvokeModel observation carries the wrap-time agent_id and workflow_id", async () => {
    const { posted, restore } = spyOnTransport();
    try {
      const client = fakeBedrockClient();
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
        agentId: "fleet-summarizer",
        workflowId: "batch-2026-07-19",
      });

      await (client as any).send(
        new InvokeModelCommand({ modelId: "anthropic.claude-3-sonnet" }),
      );

      expect(posted).toHaveLength(1);
      expect(posted[0]!.agent_id).toBe("fleet-summarizer");
      expect(posted[0]!.workflow_id).toBe("batch-2026-07-19");
    } finally {
      restore();
    }
  });

  it("Bedrock client has no withMeta() — workflow_id is constant across calls", () => {
    const client = fakeBedrockClient();
    wrap(client, {
      apiKey: "k",
      baseUrl: "http://localhost:1",
      workflowId: "one-run-per-client",
    });
    expect((client as any).withMeta).toBeUndefined();
  });
});
