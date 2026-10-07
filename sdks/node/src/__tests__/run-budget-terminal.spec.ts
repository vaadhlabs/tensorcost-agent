/**
 * The SDK must treat a run-budget refusal as terminal.
 *
 * This is the test that guards the whole feature. Applied mode is built to
 * fail open: when the proxy is unreachable or errors, the wrapper calls the
 * provider directly so TensorCost being down never takes customer inference
 * down. Before this change, a 403 budget refusal went down exactly that
 * path — the proxy would correctly refuse, the SDK would catch the throw,
 * mark `proxyFailed = true`, and then call Anthropic/OpenAI directly. The
 * cap would be enforced and then immediately bypassed, and the customer
 * would be billed anyway.
 *
 * `fetchImpl` is injected rather than stubbing `globalThis.fetch`: the
 * proxy client resolves its fetch at call time from the option, and stubbing
 * the global would be a silent no-op for any module that captured `fetch`
 * at module-eval. Injecting also means the assertions are on the real
 * request/response wire shapes, so a change to the error envelope the proxy
 * emits breaks this test rather than passing vacuously.
 */

import { describe, it, expect, vi } from "vitest";
import { proxyRequest, shouldFallBackToProvider } from "../proxy-client.js";
import {
  TensorCostRunBudgetExceededError,
  TensorCostModelGovernanceDeniedError,
  TensorCostGuardrailHardStopError,
  TensorCostProviderError,
  TensorCostProxyError,
} from "../errors.js";
import { CircuitBreaker } from "../circuit.js";

/** The exact envelope inference-proxy's RunBudgetExceededError serializes. */
const BUDGET_403_BODY = {
  error: {
    code: "RUN_BUDGET_EXCEEDED",
    message: "Agent run run-abc has reached its spend cap (150 of 100 cents used)",
    workflow_id: "run-abc",
    cap_cents: 100,
    spent_cents: 150,
  },
};

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function baseOpts(fetchImpl: typeof fetch) {
  return {
    proxyUrl: "https://proxy.test",
    bearerToken: "tok",
    providerUrl: "https://api.anthropic.com",
    providerAuth: "Bearer sk-test",
    body: { model: "claude-3-5-sonnet", messages: [] },
    model: "claude-3-5-sonnet",
    provider: "anthropic" as const,
    operation: "messages",
    correlationId: "corr-1",
    environment: null,
    circuit: new CircuitBreaker(),
    fetchImpl,
    // Retries must not apply to a budget refusal — if they did, a capped
    // run would hammer the proxy three times per call.
    retry: { maxAttempts: 3, baseDelayMs: 1, maxDelayMs: 2, jitter: false },
  };
}

describe("run-budget refusal is terminal in the SDK", () => {
  it("throws TensorCostRunBudgetExceededError on a 403 with our code", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse(403, BUDGET_403_BODY));
    await expect(
      proxyRequest(baseOpts(fetchImpl as unknown as typeof fetch)),
    ).rejects.toBeInstanceOf(TensorCostRunBudgetExceededError);
  });

  it("carries the cap detail through to the caller", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse(403, BUDGET_403_BODY));
    try {
      await proxyRequest(baseOpts(fetchImpl as unknown as typeof fetch));
      expect.unreachable("should have thrown");
    } catch (err) {
      const e = err as TensorCostRunBudgetExceededError;
      expect(e.workflowId).toBe("run-abc");
      expect(e.capCents).toBe(100);
      expect(e.spentCents).toBe(150);
    }
  });

  it("does NOT retry a budget refusal", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse(403, BUDGET_403_BODY));
    await expect(
      proxyRequest(baseOpts(fetchImpl as unknown as typeof fetch)),
    ).rejects.toBeInstanceOf(TensorCostRunBudgetExceededError);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("does NOT count a budget refusal against the circuit breaker", async () => {
    // A refusal that tripped the breaker would eventually route the
    // customer around their own cap — the failure mode this guards.
    const fetchImpl = vi.fn(async () => jsonResponse(403, BUDGET_403_BODY));
    const opts = baseOpts(fetchImpl as unknown as typeof fetch);
    for (let i = 0; i < 10; i++) {
      await proxyRequest(opts).catch(() => undefined);
    }
    expect(opts.circuit.shouldBypass()).toBe(false);
  });

  it("blocks the direct-provider fallback", () => {
    const err = new TensorCostRunBudgetExceededError("capped", "run-abc", 100, 150);
    expect(shouldFallBackToProvider(err)).toBe(false);
  });

  it("does not fall back for model governance or hard_stop denials", () => {
    expect(
      shouldFallBackToProvider(
        new TensorCostModelGovernanceDeniedError("denied", "openai", "gpt-4o", "gap"),
      ),
    ).toBe(false);
    expect(
      shouldFallBackToProvider(
        new TensorCostGuardrailHardStopError("blocked", "33333333-3333-3333-3333-333333333333"),
      ),
    ).toBe(false);
  });

  it("still permits the fallback for genuine proxy failures", () => {
    // The fail-open contract must survive intact for everything that is
    // actually TensorCost being unhealthy.
    expect(shouldFallBackToProvider(new TensorCostProxyError("proxy 503"))).toBe(true);
    expect(shouldFallBackToProvider(new TensorCostProviderError("provider 400"))).toBe(true);
    expect(shouldFallBackToProvider(new Error("socket hang up"))).toBe(true);
  });

  it("treats an unrelated 403 as a provider error, not a budget refusal", async () => {
    // An auth 403 from the proxy is a real failure and must keep its
    // existing fail-open behaviour.
    const fetchImpl = vi.fn(async () =>
      jsonResponse(403, { error: { code: "FORBIDDEN", message: "bad token" } }),
    );
    await expect(
      proxyRequest(baseOpts(fetchImpl as unknown as typeof fetch)),
    ).rejects.toBeInstanceOf(TensorCostProviderError);
  });

  it("treats a non-JSON 403 as a provider error", async () => {
    const fetchImpl = vi.fn(
      async () => new Response("nope", { status: 403 }),
    );
    await expect(
      proxyRequest(baseOpts(fetchImpl as unknown as typeof fetch)),
    ).rejects.toBeInstanceOf(TensorCostProviderError);
  });
});

describe("run attribution reaches the proxy", () => {
  it("sends x-tc-workflow-id and x-tc-agent-id when set", async () => {
    // Without these headers the proxy has no run to scope a cap to, so
    // this assertion is load-bearing for the whole feature.
    const fetchImpl = vi.fn(async () => jsonResponse(200, { ok: true }));
    await proxyRequest({
      ...baseOpts(fetchImpl as unknown as typeof fetch),
      agentId: "agent-7",
      workflowId: "run-abc",
    });

    const headers = fetchImpl.mock.calls[0]![1]!.headers as Record<string, string>;
    expect(headers["x-tc-workflow-id"]).toBe("run-abc");
    expect(headers["x-tc-agent-id"]).toBe("agent-7");
  });

  it("omits the headers entirely when no run ids are set", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse(200, { ok: true }));
    await proxyRequest(baseOpts(fetchImpl as unknown as typeof fetch));

    const headers = fetchImpl.mock.calls[0]![1]!.headers as Record<string, string>;
    expect(headers).not.toHaveProperty("x-tc-workflow-id");
    expect(headers).not.toHaveProperty("x-tc-agent-id");
  });
});
