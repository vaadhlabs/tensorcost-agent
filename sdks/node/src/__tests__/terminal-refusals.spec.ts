/**
 * 1.4.1: the fail-open rule is an allow-list of outages.
 *
 * 1.4.0 decided "may this call go around TensorCost?" with a deny-list of
 * refusal classes. A money budget's hard cap (BUDGET_HARD_CAP) wasn't on
 * it, so the proxy refused, the SDK called the provider anyway, and the cap
 * was a log line. These tests drive the real OpenAI wrapper (proxy path,
 * streaming proxy path, govern/admit path) against stubbed TensorCost
 * responses and pin, for every refusal shape, that TensorCost is asked
 * exactly once, the provider is never called, and the caller gets the
 * right typed error with the server's code and reason intact. Then the
 * other side of the contract: genuine outages still fail open when
 * failOpenEnabled is on, and don't when it's off.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import * as openaiProvider from "../providers/openai.js";
import { ObservationTransport } from "../transport.js";
import { CircuitBreaker } from "../circuit.js";
import { shouldFallBackToProvider } from "../proxy-client.js";
import {
  TensorCostBudgetHardCapError,
  TensorCostComplianceDeniedError,
  TensorCostGuardrailHardStopError,
  TensorCostModelGovernanceDeniedError,
  TensorCostPeriodBudgetExceededError,
  TensorCostProviderError,
  TensorCostProxyError,
  TensorCostRefusedError,
  TensorCostRunBudgetExceededError,
  TensorCostRunStoppedError,
} from "../errors.js";
import type { ControlLayer } from "../layer.js";
import { defaultHardening } from "./test-harness.js";

type Ctor = abstract new (...args: never[]) => TensorCostRefusedError;

/**
 * The money-budget bodies the proxy sends, as pinned by the server's
 * contract test (inference-proxy-service
 * src/__tests__/money-budget-wire-contract.spec.ts, monorepo 7046d775):
 * MoneyBudgetHardCapError("b-1", 1.02, "Prod AI", undefined, {monthly,
 * 10000, 10200}) and ("b-2", 0, "New budget", "spend_not_known_yet",
 * {weekly, 5000}). The second goes out with `Retry-After: 5`.
 */
const MONEY_BUDGET_REACHED = {
  error: {
    code: "PERIOD_BUDGET_EXCEEDED",
    budget_code: "BUDGET_HARD_CAP",
    message: 'Budget hard cap reached for budget "Prod AI" (b-1)',
    scope: "budget:b-1",
    period: "monthly",
    cap_cents: 10_000,
    spent_cents: 10_200,
    budget_id: "b-1",
    budget_name: "Prod AI",
    burn_pct: 1.02,
    retryable: false,
  },
};
const MONEY_BUDGET_NOT_KNOWN_YET = {
  error: {
    code: "PERIOD_BUDGET_EXCEEDED",
    budget_code: "BUDGET_HARD_CAP",
    message:
      'The spend for budget "New budget" (b-2) hasn\'t been counted yet, so its hard cap holds ' +
      "requests until it is (usually seconds after the budget is created or its scope changes)",
    scope: "budget:b-2",
    period: "weekly",
    cap_cents: 5_000,
    budget_id: "b-2",
    budget_name: "New budget",
    burn_pct: 0,
    reason: "spend_not_known_yet",
    retryable: true,
    retry_after_seconds: 5,
  },
};

interface Case {
  name: string;
  status?: number;
  body: unknown;
  headers?: Record<string, string>;
  cls: Ctor;
  code: string;
  reason: string | null;
  extra?: Record<string, unknown>;
  message?: RegExp;
}

const CASES: Case[] = [
  {
    name: "RUN_STOPPED",
    body: {
      error: {
        code: "RUN_STOPPED",
        message: "Agent run run-abc was stopped by ops@example.test: runaway",
        workflow_id: "run-abc",
        stopped_by: "ops@example.test",
        stopped_at: "2026-10-25T10:00:00.000Z",
        reason: "runaway",
      },
    },
    cls: TensorCostRunStoppedError,
    code: "RUN_STOPPED",
    reason: "runaway",
    extra: { workflowId: "run-abc", stoppedBy: "ops@example.test" },
  },
  {
    name: "RUN_BUDGET_EXCEEDED",
    body: {
      error: {
        code: "RUN_BUDGET_EXCEEDED",
        message: "Agent run run-abc has reached its spend cap (150 of 100 cents used)",
        workflow_id: "run-abc",
        cap_cents: 100,
        spent_cents: 150,
      },
    },
    cls: TensorCostRunBudgetExceededError,
    code: "RUN_BUDGET_EXCEEDED",
    reason: null,
    extra: { workflowId: "run-abc", capCents: 100, spentCents: 150 },
  },
  {
    name: "PERIOD_BUDGET_EXCEEDED",
    body: {
      error: {
        code: "PERIOD_BUDGET_EXCEEDED",
        message: "Agent period budget exceeded for agent:a1 (day): 600 of 500 cents used",
        scope: "agent:a1",
        period: "day",
        cap_cents: 500,
        spent_cents: 600,
      },
    },
    cls: TensorCostPeriodBudgetExceededError,
    code: "PERIOD_BUDGET_EXCEEDED",
    reason: null,
    extra: { scope: "agent:a1", period: "day", capCents: 500 },
  },
  {
    name: "BUDGET_HARD_CAP (older servers: error.code BUDGET_HARD_CAP, cap reached)",
    body: {
      error: {
        code: "BUDGET_HARD_CAP",
        message: "Budget hard cap reached for budget b1",
        budget_id: "b1",
        burn_pct: 104,
      },
    },
    cls: TensorCostBudgetHardCapError,
    code: "BUDGET_HARD_CAP",
    reason: null,
    extra: { budgetId: "b1", burnPct: 104, retryable: false, retryAfterSeconds: null },
    message: /the budget's hard cap is reached \(budget b1\)/,
  },
  {
    name: "BUDGET_HARD_CAP (older servers: error.code BUDGET_HARD_CAP, spend_not_known_yet)",
    body: {
      error: {
        code: "BUDGET_HARD_CAP",
        message: "The spend for budget \"Prod\" (b1) hasn't been counted yet...",
        budget_id: "b1",
        budget_name: "Prod",
        burn_pct: 0,
        reason: "spend_not_known_yet",
      },
    },
    headers: { "retry-after": "5" },
    cls: TensorCostBudgetHardCapError,
    code: "BUDGET_HARD_CAP",
    reason: "spend_not_known_yet",
    extra: { budgetId: "b1", budgetName: "Prod", retryable: true, retryAfterSeconds: 5 },
    message: /the budget's spend hasn't been counted yet; retry in 5 s/,
  },
  {
    name: "BUDGET_HARD_CAP (server shape, cap reached: contract fixture)",
    body: MONEY_BUDGET_REACHED,
    cls: TensorCostBudgetHardCapError,
    code: "PERIOD_BUDGET_EXCEEDED",
    reason: null,
    extra: {
      budgetCode: "BUDGET_HARD_CAP",
      budgetId: "b-1",
      budgetName: "Prod AI",
      scope: "budget:b-1",
      period: "monthly",
      capCents: 10_000,
      spentCents: 10_200,
      retryable: false,
      retryAfterSeconds: null,
    },
    message: /the budget's hard cap is reached \(budget "Prod AI", b-1\)/,
  },
  {
    name: "BUDGET_HARD_CAP (server shape, spend not known yet: contract fixture)",
    body: MONEY_BUDGET_NOT_KNOWN_YET,
    headers: { "retry-after": "5" },
    cls: TensorCostBudgetHardCapError,
    code: "PERIOD_BUDGET_EXCEEDED",
    reason: "spend_not_known_yet",
    extra: {
      budgetCode: "BUDGET_HARD_CAP",
      budgetId: "b-2",
      period: "weekly",
      capCents: 5_000,
      spentCents: null,
      retryable: true,
      retryAfterSeconds: 5,
    },
    message: /the budget's spend hasn't been counted yet; retry in 5 s/,
  },
  {
    name: "GUARDRAIL_HARD_STOP",
    body: {
      error: {
        code: "GUARDRAIL_HARD_STOP",
        message: "Request refused by guardrail hard_stop policy",
        policy_id: "33333333-3333-3333-3333-333333333333",
        reason: "pii",
      },
    },
    cls: TensorCostGuardrailHardStopError,
    code: "GUARDRAIL_HARD_STOP",
    reason: "pii",
    extra: { policyId: "33333333-3333-3333-3333-333333333333" },
  },
  {
    name: "MODEL_GOVERNANCE_DENIED",
    body: {
      error: {
        code: "MODEL_GOVERNANCE_DENIED",
        message: "gpt-4o is not approved for this tenant",
        provider: "openai",
        model: "gpt-4o",
        governance_status: "denied",
      },
    },
    cls: TensorCostModelGovernanceDeniedError,
    code: "MODEL_GOVERNANCE_DENIED",
    reason: null,
    extra: { governanceStatus: "denied" },
  },
  {
    name: "COMPLIANCE_DENIED",
    body: {
      error: {
        code: "COMPLIANCE_DENIED",
        message: "Request refused by compliance policy: region",
        reason: "region",
      },
    },
    cls: TensorCostComplianceDeniedError,
    code: "COMPLIANCE_DENIED",
    reason: "region",
  },
  {
    name: "SOMETHING_NEW (a code this SDK has never heard of)",
    body: {
      error: { code: "SOMETHING_NEW", message: "a future refusal", reason: "future_reason" },
    },
    cls: TensorCostRefusedError,
    code: "SOMETHING_NEW",
    reason: "future_reason",
    message: /tensorcost: a future refusal/,
  },
  {
    name: "SOMETHING_NEW in the flat shape",
    body: { code: "SOMETHING_NEW", message: "flat refusal", reason: "flat" },
    cls: TensorCostRefusedError,
    code: "SOMETHING_NEW",
    reason: "flat",
  },
  {
    name: "SOMETHING_NEW carried on a 402 (a refusal body makes a 402 terminal)",
    status: 402,
    body: { error: { code: "SOMETHING_NEW", message: "plan says no", reason: "plan" } },
    cls: TensorCostRefusedError,
    code: "SOMETHING_NEW",
    reason: "plan",
  },
  {
    name: "SOMETHING_NEW carried on a 429 (refusal wins over quota: not retried)",
    status: 429,
    body: { error: { code: "SOMETHING_NEW", message: "slow down, for policy reasons" } },
    cls: TensorCostRefusedError,
    code: "SOMETHING_NEW",
    reason: null,
  },
];

const PROXY_URL = "https://proxy.test";
const TC_BASE_URL = "https://tc.test";

function json(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

/** Observations go through the transport's own fetch; only TensorCost API calls are counted. */
function makeTransport(): ObservationTransport {
  const t = new ObservationTransport({
    tenantId: "t-1",
    baseUrl: TC_BASE_URL,
    apiKey: "k",
    fetchImpl: (async (input: RequestInfo | URL) =>
      String(input).includes("sdk-token/exchange")
        ? json(200, { token: "jwt", expires_in: 900 })
        : new Response(null, { status: 202 })) as unknown as typeof fetch,
  });
  (t as unknown as { getToken: () => Promise<string> }).getToken = async () => "jwt";
  return t;
}

function makeClient() {
  const direct = vi.fn(async () => ({
    id: "direct",
    usage: { prompt_tokens: 1, completion_tokens: 1 },
  }));
  const client = {
    baseURL: "https://api.openai.com/v1",
    apiKey: "sk-test",
    defaultHeaders: {
      "x-tc-provider-url": "https://api.openai.com/v1",
      "x-tc-provider-auth": "Bearer sk-test",
    } as Record<string, string>,
    chat: { completions: { create: direct } },
  };
  return { client, direct };
}

type PathName = "proxy" | "proxy-stream" | "admit";
const PATHS: { name: PathName; layer: ControlLayer; stream: boolean }[] = [
  { name: "proxy", layer: "route", stream: false },
  { name: "proxy-stream", layer: "route", stream: true },
  { name: "admit", layer: "govern", stream: false },
];

function install(layer: ControlLayer, failOpenEnabled = true, timeoutMs = 5_000) {
  const { client, direct } = makeClient();
  openaiProvider.install(
    client,
    makeTransport(),
    "t-1",
    null,
    null,
    defaultHardening({
      maxLayer: layer,
      appliedMode: layer === "route",
      proxyUrl: PROXY_URL,
      baseUrl: TC_BASE_URL,
      failOpenEnabled,
      timeoutMs,
      headersTimeoutMs: timeoutMs,
      // Retries on: a refusal must still be asked exactly once.
      retry: { maxAttempts: 3, baseDelayMs: 1, maxDelayMs: 2, jitter: false },
      circuit: new CircuitBreaker(),
    }),
  );
  return { client, direct };
}

const savedFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = savedFetch;
});

function stubTensorCost(respond: (url: string, init?: RequestInit) => Promise<Response>) {
  const calls: string[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push(url);
    return respond(url, init);
  }) as typeof fetch;
  return calls;
}

describe.each(PATHS)("terminal refusals on the $name path", ({ name, layer, stream }) => {
  it.each(CASES)("$name: one TensorCost call, no provider call, typed error", async (c) => {
    const { client, direct } = install(layer);
    const calls = stubTensorCost(async () => json(c.status ?? 403, c.body, c.headers));

    const err = await client.chat.completions
      .create({ model: "gpt-4o", messages: [], ...(stream ? { stream: true } : {}) })
      .catch((e: unknown) => e);

    expect(calls).toHaveLength(1);
    expect(calls[0]).toContain(name === "admit" ? "/v1/admit" : "/v1/chat/completions");
    expect(direct).not.toHaveBeenCalled();
    expect(err).toBeInstanceOf(c.cls);
    expect(err).toBeInstanceOf(TensorCostRefusedError);
    if (c.cls === TensorCostRefusedError || c.cls === TensorCostPeriodBudgetExceededError) {
      // Exact class: a plain PERIOD_BUDGET_EXCEEDED (no budget_code) keeps
      // 1.4.0's period-budget error and is not mistaken for a hard cap.
      expect((err as object).constructor).toBe(c.cls);
    }
    if (c.cls === TensorCostBudgetHardCapError) {
      // 1.4.0 raised the period-budget error for this wire shape; a catch
      // written against it still matches.
      expect(err).toBeInstanceOf(TensorCostPeriodBudgetExceededError);
    }
    const e = err as TensorCostRefusedError;
    expect(e.code).toBe(c.code);
    expect(e.reason).toBe(c.reason);
    expect(e.status).toBe(c.status ?? 403);
    if (c.extra) expect(e).toMatchObject(c.extra);
    if (c.message) expect(e.message).toMatch(c.message);
    expect(shouldFallBackToProvider(e)).toBe(false);
  });

  it("a hard-cap refusal is still terminal with failOpenEnabled=false", async () => {
    const { client, direct } = install(layer, false);
    const calls = stubTensorCost(async () => json(403, MONEY_BUDGET_REACHED));
    const err = await client.chat.completions
      .create({ model: "gpt-4o", messages: [], ...(stream ? { stream: true } : {}) })
      .catch((e: unknown) => e);
    expect(calls).toHaveLength(1);
    expect(direct).not.toHaveBeenCalled();
    expect(err).toBeInstanceOf(TensorCostBudgetHardCapError);
  });
});

describe("the admit path no longer mislabels a refusal as model governance", () => {
  it("a 403 with no readable code is a plain TensorCostRefusedError", async () => {
    const { client, direct } = install("govern");
    stubTensorCost(async () => new Response("<html>403</html>", { status: 403 }));
    const err = await client.chat.completions.create({ model: "gpt-4o" }).catch((e: unknown) => e);
    expect(direct).not.toHaveBeenCalled();
    expect(err).toBeInstanceOf(TensorCostRefusedError);
    expect(err).not.toBeInstanceOf(TensorCostModelGovernanceDeniedError);
    expect((err as TensorCostRefusedError).code).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Outages: the allow-list.

const abortable = (_url: string, init?: RequestInit): Promise<Response> =>
  new Promise((_resolve, reject) => {
    init?.signal?.addEventListener("abort", () => {
      reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
    });
  });

const OUTAGES: { name: string; respond: (url: string, init?: RequestInit) => Promise<Response> }[] = [
  {
    name: "network error",
    respond: async () => {
      throw new TypeError("fetch failed: ECONNREFUSED");
    },
  },
  { name: "timeout", respond: abortable },
  { name: "proxy 503", respond: async () => json(503, { error: "unavailable" }) },
  {
    name: "proxy 402 (plan request ceiling, no refusal body)",
    respond: async () =>
      json(402, {
        kind: "scale-cogs-exceeded",
        error: "Payment Required",
        plan: "scale",
        limit: "proxy_requests_per_month",
        cap: 100000,
        current: 100001,
        upgrade_hint: "Scale monthly proxy request ceiling reached. Contact us for Enterprise.",
      }),
  },
  {
    name: "proxy 429",
    respond: async () =>
      json(429, { statusCode: 429, message: "ThrottlerException: Too Many Requests" }),
  },
];

describe.each(PATHS)("outages on the $name path", ({ layer, stream }) => {
  it.each(OUTAGES)("$name with failOpenEnabled=true falls back to the provider", async (o) => {
    const { client, direct } = install(layer, true, 30);
    stubTensorCost(o.respond);
    const resp = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [],
      ...(stream ? { stream: true } : {}),
    });
    expect((resp as { id: string }).id).toBe("direct");
    expect(direct).toHaveBeenCalledTimes(1);
  });

  it.each(OUTAGES)("$name with failOpenEnabled=false does not", async (o) => {
    const { client, direct } = install(layer, false, 30);
    stubTensorCost(o.respond);
    const err = await client.chat.completions
      .create({ model: "gpt-4o", messages: [], ...(stream ? { stream: true } : {}) })
      .catch((e: unknown) => e);
    expect(direct).not.toHaveBeenCalled();
    expect(shouldFallBackToProvider(err) || err instanceof TensorCostProxyError).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// A provider's own 4xx, relayed by the proxy.

describe("a relayed provider 4xx", () => {
  const PROVIDER_400 = {
    error: {
      message: "Invalid value for 'messages': empty array",
      type: "invalid_request_error",
      param: "messages",
      code: "invalid_value",
    },
  };

  it.each([true, false])(
    "surfaces as TensorCostProviderError, one call, no retry, no fallback (failOpenEnabled=%s)",
    async (failOpenEnabled) => {
      const { client, direct } = install("route", failOpenEnabled);
      const calls = stubTensorCost(async () => json(400, PROVIDER_400));
      const err = await client.chat.completions
        .create({ model: "gpt-4o", messages: [] })
        .catch((e: unknown) => e);
      expect(calls).toHaveLength(1);
      expect(direct).not.toHaveBeenCalled();
      expect(err).toBeInstanceOf(TensorCostProviderError);
      expect(err).not.toBeInstanceOf(TensorCostRefusedError);
      const e = err as TensorCostProviderError;
      expect(e.status).toBe(400);
      expect(e.message).toBe(
        "tensorcost: upstream provider returned 400: Invalid value for 'messages': empty array",
      );
      expect(e.body).toEqual(PROVIDER_400);
    },
  );

  it("a lowercase provider code in a relayed 403 is not mistaken for a refusal", async () => {
    const { client, direct } = install("route");
    stubTensorCost(async () =>
      json(403, {
        error: { message: "Country not supported", code: "unsupported_country_region_territory" },
      }),
    );
    const err = await client.chat.completions.create({ model: "gpt-4o" }).catch((e: unknown) => e);
    expect(direct).not.toHaveBeenCalled();
    expect(err).toBeInstanceOf(TensorCostProviderError);
    expect((err as TensorCostProviderError).status).toBe(403);
  });
});
