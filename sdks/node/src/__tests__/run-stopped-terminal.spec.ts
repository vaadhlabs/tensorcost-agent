/**
 * An operator stop is terminal in the SDK, same as a run-budget refusal.
 *
 * The proxy answers a stopped run with 403 `RUN_STOPPED`. Before this was
 * recognised the SDK mapped it to a generic provider error, which
 * shouldFallBackToProvider() allows, so applied mode's fail-open path would
 * call the provider directly and the stop would be bypassed by the very
 * client it was meant to halt. It also dropped the body, so the caller never
 * learned who stopped the run or why.
 */

import { describe, it, expect, vi } from "vitest";
import {
  proxyRequest,
  proxyRequestStream,
  shouldFallBackToProvider,
} from "../proxy-client.js";
import { TensorCostRunStoppedError } from "../errors.js";
import { CircuitBreaker } from "../circuit.js";

/** The envelope inference-proxy's RunStoppedError serializes. */
const STOPPED_403_BODY = {
  error: {
    code: "RUN_STOPPED",
    message:
      "Agent run run-abc was stopped by ops@example.test at 2026-10-25T10:00:00.000Z: runaway context growth",
    workflow_id: "run-abc",
    stopped_by: "ops@example.test",
    stopped_at: "2026-10-25T10:00:00.000Z",
    reason: "runaway context growth",
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
    providerUrl: "https://api.openai.com/v1",
    providerAuth: "Bearer sk-test",
    body: { model: "gpt-4o", messages: [] },
    model: "gpt-4o",
    provider: "openai" as const,
    operation: "chat.completions",
    correlationId: "corr-1",
    environment: null,
    circuit: new CircuitBreaker(),
    fetchImpl,
    retry: { maxAttempts: 3, baseDelayMs: 1, maxDelayMs: 2, jitter: false },
  };
}

describe("run stop is terminal in the SDK", () => {
  it("throws TensorCostRunStoppedError carrying who, when and why", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse(403, STOPPED_403_BODY));
    try {
      await proxyRequest(baseOpts(fetchImpl as unknown as typeof fetch));
      expect.unreachable("should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(TensorCostRunStoppedError);
      const e = err as TensorCostRunStoppedError;
      expect(e.code).toBe("RUN_STOPPED");
      expect(e.workflowId).toBe("run-abc");
      expect(e.stoppedBy).toBe("ops@example.test");
      expect(e.stoppedAt).toBe("2026-10-25T10:00:00.000Z");
      expect(e.reason).toBe("runaway context growth");
      expect(e.message).toContain("was stopped by ops@example.test");
      expect(e.status).toBe(403);
    }
  });

  it("does not retry and does not trip the circuit breaker", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse(403, STOPPED_403_BODY));
    const opts = baseOpts(fetchImpl as unknown as typeof fetch);
    for (let i = 0; i < 10; i++) {
      await proxyRequest(opts).catch(() => undefined);
    }
    expect(fetchImpl).toHaveBeenCalledTimes(10);
    expect(opts.circuit.shouldBypass()).toBe(false);
  });

  it("blocks the direct-provider fallback", () => {
    const err = new TensorCostRunStoppedError("stopped", {
      workflowId: "run-abc",
      stoppedBy: "ops",
      stoppedAt: "2026-10-25T10:00:00.000Z",
      reason: "r",
    });
    expect(shouldFallBackToProvider(err)).toBe(false);
  });

  it("is terminal on the streaming path too", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse(403, STOPPED_403_BODY));
    await expect(
      proxyRequestStream({
        ...baseOpts(fetchImpl as unknown as typeof fetch),
        body: { model: "gpt-4o", messages: [], stream: true },
      }),
    ).rejects.toBeInstanceOf(TensorCostRunStoppedError);
  });
});
