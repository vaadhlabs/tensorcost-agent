/**
 * Tests for the SDK metadata story — deployment + promptTemplateId +
 * withMeta() per-call override.
 *
 * Four scenarios per the task spec:
 *   1. wrap() with deployment → observation carries metadata.deployment
 *   2. withMeta() per-call → observation carries metadata.prompt_template_id
 *   3. both at the same time → both fields land
 *   4. no metadata → metadata field absent (existing user metadata not broken)
 */

import { describe, expect, it, vi } from "vitest";
import { wrap } from "../wrap.js";
import { ObservationTransport } from "../transport.js";
import type { Observation } from "../types.js";

// ---------------------------------------------------------------------------
// Fake clients
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Test-harness helpers
// ---------------------------------------------------------------------------

/**
 * Intercepts `ObservationTransport.post` so we can inspect observations
 * without any real network calls. Returns the collected array and a
 * restorer so the spy can be torn down after the test.
 */
function spyOnTransport(): { posted: Observation[] } {
  const posted: Observation[] = [];
  const original = ObservationTransport.prototype.post;
  ObservationTransport.prototype.post = function (obs: Observation) {
    posted.push(obs);
  };
  return { posted };
}

function restoreTransport(original: typeof ObservationTransport.prototype.post): void {
  ObservationTransport.prototype.post = original;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("wrap() metadata — deployment", () => {
  it("observation carries metadata.deployment when set at wrap time", async () => {
    const posted: Observation[] = [];
    const original = ObservationTransport.prototype.post;
    ObservationTransport.prototype.post = function (obs: Observation) { posted.push(obs); };

    try {
      const client = fakeOpenAIClient();
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
        deployment: "sha-abc123",
      });
      await (client as any).chat.completions.create({ model: "gpt-4o", messages: [] });
      expect(posted).toHaveLength(1);
      expect(posted[0]!.metadata).toBeDefined();
      expect(posted[0]!.metadata!.deployment).toBe("sha-abc123");
    } finally {
      ObservationTransport.prototype.post = original;
    }
  });

  it("Anthropic client carries metadata.deployment", async () => {
    const posted: Observation[] = [];
    const original = ObservationTransport.prototype.post;
    ObservationTransport.prototype.post = function (obs: Observation) { posted.push(obs); };

    try {
      const client = fakeAnthropicClient();
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
        deployment: "v2.1-hotfix",
      });
      await (client as any).messages.create({ model: "claude-3-5-sonnet-20241022", max_tokens: 10, messages: [] });
      expect(posted).toHaveLength(1);
      expect(posted[0]!.metadata?.deployment).toBe("v2.1-hotfix");
    } finally {
      ObservationTransport.prototype.post = original;
    }
  });
});

describe("withMeta() per-call override", () => {
  it("observation carries metadata.prompt_template_id from withMeta()", async () => {
    const posted: Observation[] = [];
    const original = ObservationTransport.prototype.post;
    ObservationTransport.prototype.post = function (obs: Observation) { posted.push(obs); };

    try {
      const client = fakeOpenAIClient();
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
      });

      const proxied = (client as any).withMeta({ promptTemplateId: "personalization-rerank-v3" });
      await proxied.chat.completions.create({ model: "gpt-4o", messages: [] });

      expect(posted).toHaveLength(1);
      expect(posted[0]!.metadata).toBeDefined();
      expect(posted[0]!.metadata!.prompt_template_id).toBe("personalization-rerank-v3");
    } finally {
      ObservationTransport.prototype.post = original;
    }
  });

  it("withMeta() on Anthropic client carries prompt_template_id", async () => {
    const posted: Observation[] = [];
    const original = ObservationTransport.prototype.post;
    ObservationTransport.prototype.post = function (obs: Observation) { posted.push(obs); };

    try {
      const client = fakeAnthropicClient();
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
      });

      const proxied = (client as any).withMeta({ promptTemplateId: "summarizer-v1" });
      await proxied.messages.create({ model: "claude-3-5-sonnet-20241022", max_tokens: 10, messages: [] });

      expect(posted).toHaveLength(1);
      expect(posted[0]!.metadata?.prompt_template_id).toBe("summarizer-v1");
    } finally {
      ObservationTransport.prototype.post = original;
    }
  });
});

describe("wrap() + withMeta() combined", () => {
  it("both deployment (wrap-time) and prompt_template_id (per-call) land together", async () => {
    const posted: Observation[] = [];
    const original = ObservationTransport.prototype.post;
    ObservationTransport.prototype.post = function (obs: Observation) { posted.push(obs); };

    try {
      const client = fakeOpenAIClient();
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
        deployment: "sha-deadbeef",
        promptTemplateId: "default-chat-v2",
      });

      // Per-call override beats wrap-time promptTemplateId.
      const proxied = (client as any).withMeta({ promptTemplateId: "research-agent-v5" });
      await proxied.chat.completions.create({ model: "gpt-4o", messages: [] });

      expect(posted).toHaveLength(1);
      const meta = posted[0]!.metadata;
      expect(meta).toBeDefined();
      // wrap-time deployment survives (not overridden by withMeta)
      expect(meta!.deployment).toBe("sha-deadbeef");
      // per-call promptTemplateId overrides the wrap-time default
      expect(meta!.prompt_template_id).toBe("research-agent-v5");
    } finally {
      ObservationTransport.prototype.post = original;
    }
  });

  it("direct call (no withMeta) uses wrap-time promptTemplateId", async () => {
    const posted: Observation[] = [];
    const original = ObservationTransport.prototype.post;
    ObservationTransport.prototype.post = function (obs: Observation) { posted.push(obs); };

    try {
      const client = fakeOpenAIClient();
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
        deployment: "sha-cafebabe",
        promptTemplateId: "default-v1",
      });

      // Plain call — no withMeta override.
      await (client as any).chat.completions.create({ model: "gpt-4o", messages: [] });

      expect(posted).toHaveLength(1);
      const meta = posted[0]!.metadata;
      expect(meta?.deployment).toBe("sha-cafebabe");
      expect(meta?.prompt_template_id).toBe("default-v1");
    } finally {
      ObservationTransport.prototype.post = original;
    }
  });
});

describe("no metadata configured", () => {
  it("metadata field is absent from the observation when nothing is set", async () => {
    const posted: Observation[] = [];
    const original = ObservationTransport.prototype.post;
    ObservationTransport.prototype.post = function (obs: Observation) { posted.push(obs); };

    try {
      const client = fakeOpenAIClient();
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
        // neither deployment nor promptTemplateId set
      });

      await (client as any).chat.completions.create({ model: "gpt-4o", messages: [] });

      expect(posted).toHaveLength(1);
      // metadata key must be absent (not just undefined) so the wire payload
      // stays clean for existing tenants whose ingest code doesn't expect it.
      expect(Object.prototype.hasOwnProperty.call(posted[0], "metadata")).toBe(false);
    } finally {
      ObservationTransport.prototype.post = original;
    }
  });

  it("withMeta() with empty object still emits a valid observation", async () => {
    const posted: Observation[] = [];
    const original = ObservationTransport.prototype.post;
    ObservationTransport.prototype.post = function (obs: Observation) { posted.push(obs); };

    try {
      const client = fakeOpenAIClient();
      wrap(client, {
        apiKey: "k",
        baseUrl: "http://localhost:1",
      });

      // withMeta with no recognized keys — metadata bag is still empty.
      const proxied = (client as any).withMeta({});
      await proxied.chat.completions.create({ model: "gpt-4o", messages: [] });

      expect(posted).toHaveLength(1);
      expect(posted[0]!.status).toBe("success");
    } finally {
      ObservationTransport.prototype.post = original;
    }
  });
});
