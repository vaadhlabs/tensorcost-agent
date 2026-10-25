/**
 * A stopped run is terminal on the govern layer too.
 *
 * The govern layer asks POST /v1/admit before it calls the provider. The
 * proxy now answers a stopped run there with 403 RUN_STOPPED. The admit
 * client already treated any 403 as a refusal, but the wrapper turned it into
 * a generic "refused at govern layer" governance error, so the caller lost
 * who stopped the run and why, and couldn't tell a stop from a model denial.
 * These tests pin that a RUN_STOPPED 403 surfaces as TensorCostRunStoppedError
 * with its detail, that the provider is never called, and that other 403s and
 * transport failures keep their old behaviour.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { admitRequest } from "../admit-client.js";
import {
  TensorCostModelGovernanceDeniedError,
  TensorCostRunStoppedError,
} from "../errors.js";
import { shouldFallBackToProvider } from "../proxy-client.js";
import { ObservationTransport } from "../transport.js";
import * as openaiProvider from "../providers/openai.js";
import * as anthropicProvider from "../providers/anthropic.js";
import { defaultHardening } from "./test-harness.js";

const STOPPED_403_BODY = {
  error: {
    code: "RUN_STOPPED",
    message: "Agent run run-abc was stopped by ops@example.test at 2026-10-25T10:00:00.000Z: runaway",
    workflow_id: "run-abc",
    stopped_by: "ops@example.test",
    stopped_at: "2026-10-25T10:00:00.000Z",
    reason: "runaway",
  },
};

function jsonResponse(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

const META = {
  provider: "openai" as const,
  model: "gpt-4o-mini",
  operation: "chat.completions" as const,
  correlation_id: "c-1",
  workflow_id: "run-abc",
};

describe("admitRequest", () => {
  it("returns a TensorCostRunStoppedError refusal for a RUN_STOPPED 403", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse(403, STOPPED_403_BODY, {
        "x-tc-decision": "layer=govern; action=refuse; reason=run_stopped",
        "x-tc-request-id": "req-1",
      }),
    );
    const res = await admitRequest({
      baseUrl: "http://proxy.local",
      bearerToken: "tok",
      maxLayer: "govern",
      metadata: META,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(res.allowed).toBe(false);
    expect(res.decisionHeader).toBe("layer=govern; action=refuse; reason=run_stopped");
    expect(res.refusal).toBeInstanceOf(TensorCostRunStoppedError);
    const err = res.refusal as TensorCostRunStoppedError;
    expect(err).toMatchObject({
      workflowId: "run-abc",
      stoppedBy: "ops@example.test",
      stoppedAt: "2026-10-25T10:00:00.000Z",
      reason: "runaway",
      status: 403,
      requestId: "req-1",
    });
    expect(shouldFallBackToProvider(err)).toBe(false);
  });

  it("leaves other 403s as a plain refusal with no modelled error", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse(403, { error: { code: "MODEL_GOVERNANCE_DENIED", message: "no" } }),
    );
    const res = await admitRequest({
      baseUrl: "http://proxy.local",
      bearerToken: "tok",
      maxLayer: "govern",
      metadata: META,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(res).toEqual({ allowed: false, decisionHeader: null });
  });

  it("a 403 with a body that isn't JSON is still a refusal", async () => {
    const fetchImpl = vi.fn(async () => new Response("<html>403</html>", { status: 403 }));
    const res = await admitRequest({
      baseUrl: "http://proxy.local",
      bearerToken: "tok",
      maxLayer: "govern",
      metadata: META,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(res.allowed).toBe(false);
    expect(res.refusal).toBeUndefined();
  });

  it("still fails open on a transport failure", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new Error("ECONNREFUSED");
    });
    const res = await admitRequest({
      baseUrl: "http://proxy.local",
      bearerToken: "tok",
      maxLayer: "govern",
      metadata: META,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(res).toEqual({ allowed: true, decisionHeader: null });
  });
});

/** Routes the token exchange and observation posts; /v1/admit answers `admit`. */
function stubFetch(admit: () => Response) {
  const calls: string[] = [];
  const impl = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    calls.push(url);
    if (url.includes("sdk-token/exchange")) {
      return jsonResponse(200, { token: "tok", expires_in: 900 });
    }
    if (url.includes("/v1/admit")) return admit();
    return new Response(null, { status: 202 });
  });
  return { impl, calls };
}

describe("govern-layer wrappers on a stopped run", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const hardening = () =>
    defaultHardening({
      maxLayer: "govern",
      sdkLayer: {
        effectiveLayer: async () => "govern",
        getComplianceSnapshot: async () => null,
      } as never,
    });

  it("openai: throws TensorCostRunStoppedError and never calls the provider", async () => {
    const { impl } = stubFetch(() => jsonResponse(403, STOPPED_403_BODY));
    vi.stubGlobal("fetch", impl);
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: impl as unknown as typeof fetch,
    });
    const create = vi.fn(async () => ({ id: "x", usage: { prompt_tokens: 1, completion_tokens: 1 } }));
    const client = { chat: { completions: { create } } };
    openaiProvider.install(client, transport, "t-1", null, null, hardening(), null, null, "agent-a", "run-abc");

    const err = await client.chat.completions
      .create({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] })
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(TensorCostRunStoppedError);
    expect(err).not.toBeInstanceOf(TensorCostModelGovernanceDeniedError);
    expect((err as TensorCostRunStoppedError).stoppedBy).toBe("ops@example.test");
    expect(create).not.toHaveBeenCalled();
  });

  it("anthropic: throws TensorCostRunStoppedError and never calls the provider", async () => {
    const { impl } = stubFetch(() => jsonResponse(403, STOPPED_403_BODY));
    vi.stubGlobal("fetch", impl);
    const transport = new ObservationTransport({
      tenantId: "t-1",
      baseUrl: "http://api.local",
      apiKey: "k",
      fetchImpl: impl as unknown as typeof fetch,
    });
    const create = vi.fn(async () => ({ id: "x", usage: { input_tokens: 1, output_tokens: 1 } }));
    const client = { messages: { create } };
    anthropicProvider.install(client, transport, "t-1", null, null, hardening(), null, null, "agent-a", "run-abc");

    const err = await client.messages
      .create({ model: "claude-3-5-haiku-latest", max_tokens: 10, messages: [{ role: "user", content: "hi" }] })
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(TensorCostRunStoppedError);
    expect(create).not.toHaveBeenCalled();
  });
});
