/**
 * The runaway example, driven in-process through the real wrap() with a
 * stubbed fetch. Nothing here talks to a network: the stub plays the
 * TensorCost backend (token exchange, published layer, observation intake)
 * and the inference proxy (200 twice, then a refusal).
 */

import { describe, it, expect, afterEach } from "vitest";
import { main, runRunaway, type RunawayOptions, type RunawayDeps } from "./runaway-agent.js";

const PROXY_PATH = "/api/inference-proxy/v1/";

const STOPPED_BODY = {
  error: {
    code: "RUN_STOPPED",
    message:
      "Agent run run-x was stopped by ops@example.test at 2026-10-25T10:00:00.000Z: too much context",
    workflow_id: "run-x",
    stopped_by: "ops@example.test",
    stopped_at: "2026-10-25T10:00:00.000Z",
    reason: "too much context",
  },
};

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

interface Recorded {
  url: string;
  headers: Record<string, string>;
  body: Record<string, unknown> | null;
}

/** Backend + proxy stub. `proxyReplies` is consumed one per proxy call. */
function stubBackend(proxyReplies: Array<() => Response>) {
  const proxyCalls: Recorded[] = [];
  const otherUrls: string[] = [];
  const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const headers = (init?.headers ?? {}) as Record<string, string>;
    const body = init?.body ? (JSON.parse(String(init.body)) as Record<string, unknown>) : null;
    if (url.includes("/sdk-token/exchange")) return json(200, { token: "jwt", expires_in: 3600 });
    if (url.includes("/v1/sdk-layer")) return json(200, { published_layer: "route" });
    if (url.includes(PROXY_PATH)) {
      proxyCalls.push({ url, headers, body });
      const reply = proxyReplies.shift();
      if (!reply) throw new Error("stub ran out of proxy replies");
      return reply();
    }
    otherUrls.push(url);
    return json(200, {});
  }) as typeof fetch;
  return { fetchImpl, proxyCalls, otherUrls };
}

/** A 200 whose prompt_tokens track the request body, so growth is visible. */
function okFor(calls: Recorded[]): () => Response {
  return () => {
    const last = calls[calls.length - 1]!;
    const chars = JSON.stringify(last.body).length;
    return json(200, {
      id: "chatcmpl-1",
      model: "m",
      choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
      usage: { prompt_tokens: Math.ceil(chars / 4), completion_tokens: 3, total_tokens: 0 },
    });
  };
}

function makeDeps(fetchImpl: typeof fetch) {
  const out: string[] = [];
  const err: string[] = [];
  const deps: RunawayDeps = {
    fetch: fetchImpl,
    sleep: async () => undefined,
    out: (l) => out.push(l),
    err: (l) => err.push(l),
  };
  return { deps, out, err };
}

let seq = 0;
function opts(over: Partial<RunawayOptions> = {}): RunawayOptions {
  seq += 1;
  return {
    // Distinct key per test: the SDK shares one runtime (token cache) per key.
    apiKey: `test-key-${seq}`,
    baseUrl: "https://tc.test",
    proxyUrl: "https://tc.test",
    tenantId: "tenant-for-test",
    provider: "openai",
    providerUrl: "https://sandbox.inference.test/v1/chat/completions",
    model: "some-model",
    workflowId: "run-x",
    agentId: "research-agent",
    growth: 1.5,
    delayMs: 0,
    maxSteps: 10,
    ...over,
  };
}

const savedEnv = { ...process.env };
afterEach(() => {
  process.env = { ...savedEnv };
});

describe("runRunaway", () => {
  it("grows the context, prints 403 RUN_STOPPED and exits 3", async () => {
    const replies: Array<() => Response> = [];
    const stub = stubBackend(replies);
    // okFor reads the recorded calls of the stub that serves it.
    replies.push(okFor(stub.proxyCalls), okFor(stub.proxyCalls), () => json(403, STOPPED_BODY));
    const { deps, out } = makeDeps(stub.fetchImpl);

    const code = await runRunaway(opts(), deps);

    expect(code).toBe(3);
    expect(stub.proxyCalls).toHaveLength(3);

    const sizes = stub.proxyCalls.map((c) => JSON.stringify(c.body).length);
    expect(sizes[1]!).toBeGreaterThan(sizes[0]!);
    expect(sizes[2]!).toBeGreaterThan(sizes[1]!);

    const stepLines = out.filter((l) => /^step\s+\d+/.test(l));
    expect(stepLines).toHaveLength(3);
    expect(stepLines[0]).toMatch(/status=200/);
    expect(stepLines[2]).toMatch(/status=403/);
    const tokens = stepLines.map((l) => Number(/input_tokens=~?(\d+)/.exec(l)![1]));
    expect(tokens[1]!).toBeGreaterThan(tokens[0]!);

    expect(out).toContain(`403 RUN_STOPPED: ${STOPPED_BODY.error.message}`);
  });

  it("sends the run and agent ids and the provider url through the proxy", async () => {
    const stub = stubBackend([() => json(403, STOPPED_BODY)]);
    const { deps } = makeDeps(stub.fetchImpl);
    await runRunaway(opts({ workflowId: "run-abc", agentId: "an-agent" }), deps);

    const h = stub.proxyCalls[0]!.headers;
    expect(h["x-tc-workflow-id"]).toBe("run-abc");
    expect(h["x-tc-agent-id"]).toBe("an-agent");
    expect(h["x-tc-provider-url"]).toBe("https://sandbox.inference.test/v1/chat/completions");
    expect(stub.proxyCalls[0]!.url).toBe(`https://tc.test${PROXY_PATH}chat/completions`);
  });

  it("never falls back to calling the provider directly", async () => {
    // A proxy outage must end the run, not send the call around TensorCost.
    const stub = stubBackend(
      Array.from({ length: 10 }, () => () => json(503, { error: "down" })),
    );
    const { deps, err } = makeDeps(stub.fetchImpl);
    const code = await runRunaway(opts({ maxSteps: 1 }), deps);
    expect(code).toBe(1);
    expect(err.join("\n")).toMatch(/\S/);
    expect(stub.otherUrls.some((u) => u.includes("sandbox.inference.test"))).toBe(false);
  });

  it("hints at the sandbox flag when the proxy answers 400", async () => {
    const stub = stubBackend([
      () => json(400, { error: { code: "sandbox_not_enabled", message: "sandbox is not enabled" } }),
    ]);
    const { deps, err } = makeDeps(stub.fetchImpl);
    expect(await runRunaway(opts(), deps)).toBe(1);
    const text = err.join("\n");
    expect(text).toMatch(/400/);
    expect(text).toMatch(/inference-proxy\.sandbox\.enabled/);
  });

  it("uses the messages path for the anthropic shape", async () => {
    const stub = stubBackend([() => json(403, STOPPED_BODY)]);
    const { deps } = makeDeps(stub.fetchImpl);
    await runRunaway(
      opts({ provider: "anthropic", providerUrl: "https://sandbox.inference.test/v1/messages" }),
      deps,
    );
    expect(stub.proxyCalls[0]!.url).toBe(`https://tc.test${PROXY_PATH}messages`);
    expect(stub.proxyCalls[0]!.body).toHaveProperty("max_tokens");
  });

  it("exits 3 for a run-cap refusal too", async () => {
    const stub = stubBackend([
      () =>
        json(403, {
          error: { code: "RUN_BUDGET_EXCEEDED", message: "Agent run run-x reached its cap", workflow_id: "run-x" },
        }),
    ]);
    const { deps, out } = makeDeps(stub.fetchImpl);
    expect(await runRunaway(opts(), deps)).toBe(3);
    expect(out).toContain("403 RUN_BUDGET_EXCEEDED: Agent run run-x reached its cap");
  });

  it("exits 0 when max-steps is reached without a refusal", async () => {
    const replies: Array<() => Response> = [];
    const stub = stubBackend(replies);
    replies.push(okFor(stub.proxyCalls), okFor(stub.proxyCalls));
    const { deps, out } = makeDeps(stub.fetchImpl);
    expect(await runRunaway(opts({ maxSteps: 2 }), deps)).toBe(0);
    expect(out.filter((l) => /^step/.test(l))).toHaveLength(2);
  });
});

describe("main (CLI wrapper)", () => {
  it("is a usage error without --model", async () => {
    const stub = stubBackend([]);
    const { deps, err } = makeDeps(stub.fetchImpl);
    const code = await main(["--api-key", "k", "--provider-url", "https://x.test/v1/chat/completions"], {}, deps);
    expect(code).toBe(1);
    expect(err.join("\n")).toMatch(/--model is required/);
    expect(stub.proxyCalls).toHaveLength(0);
  });

  it("reads the api key, base url and provider url from the environment", async () => {
    const stub = stubBackend([() => json(403, STOPPED_BODY)]);
    const { deps, out } = makeDeps(stub.fetchImpl);
    const code = await main(
      ["--model", "m", "--delay-ms", "0"],
      {
        TENSORCOST_API_KEY: "env-key-main",
        TENSORCOST_BASE_URL: "https://env.tc.test",
        TENSORCOST_PROVIDER_URL: "https://sandbox.inference.test/v1/chat/completions",
        TENSORCOST_TENANT_ID: "tenant-for-test",
      },
      deps,
    );
    expect(code).toBe(3);
    expect(stub.proxyCalls[0]!.url.startsWith("https://env.tc.test")).toBe(true);
    expect(out.some((l) => l.startsWith("403 RUN_STOPPED:"))).toBe(true);
  });

  it("defaults the workflow id to runaway-<uuid> and the agent id to research-agent", async () => {
    const stub = stubBackend([() => json(403, STOPPED_BODY)]);
    const { deps } = makeDeps(stub.fetchImpl);
    await main(
      ["--api-key", "k-defaults", "--model", "m", "--provider-url", "https://x.test/v1/chat/completions", "--base-url", "https://tc.test"],
      { TENSORCOST_TENANT_ID: "tenant-for-test" },
      deps,
    );
    const h = stub.proxyCalls[0]!.headers;
    expect(h["x-tc-workflow-id"]).toMatch(/^runaway-[0-9a-f-]{36}$/);
    expect(h["x-tc-agent-id"]).toBe("research-agent");
  });
});
