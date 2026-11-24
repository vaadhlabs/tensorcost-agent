import { describe, expect, it } from "vitest";
import {
  ERROR_MESSAGE_MAX,
  boundObservation,
  truncateFreeText,
} from "../observation-limits.js";
import { ObservationTransport } from "../transport.js";
import type { Observation } from "../types.js";

// What an Azure content-filter refusal looks like by the time it is
// `${err.name}: ${err.message}`. Proxies before the fix 400ed it, and the
// SDK treats a 4xx as final, so the call's spend row was lost.
const AZURE_FILTER =
  "BadRequestError: 400 The response was filtered due to the prompt triggering Azure OpenAI's content management policy. " +
  "Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: " +
  "https://go.microsoft.com/fwlink/?linkid=2198766";

function errorObservation(error_message: string | null): Observation {
  return {
    sdk_version: "tensorcost-node/0.1.0",
    provider: "azure_openai",
    model: "support-bot",
    operation: "chat.completions",
    request_at: new Date().toISOString(),
    response_at: new Date().toISOString(),
    input_tokens: 412,
    output_tokens: null,
    cost_usd_cents: null,
    status: "error",
    error_message,
    correlation_id: "00000000-0000-0000-0000-000000000000",
  };
}

describe("truncateFreeText", () => {
  it("leaves a string at the cap alone", () => {
    expect(truncateFreeText("x".repeat(200), 200)).toBe("x".repeat(200));
  });

  it("cuts to exactly the cap, ending in an ellipsis", () => {
    const out = truncateFreeText(AZURE_FILTER, 200);
    expect(AZURE_FILTER.length).toBeGreaterThan(200);
    expect(out).toHaveLength(200);
    expect(out).toBe(AZURE_FILTER.slice(0, 199) + "…");
  });

  it("never leaves half an emoji at the cut", () => {
    const out = truncateFreeText("x".repeat(198) + "😀" + "y".repeat(10), 200);
    expect(out).toBe("x".repeat(198) + "…");
  });
});

describe("boundObservation", () => {
  it("returns the same object when nothing needs cutting", () => {
    const ok = errorObservation("RateLimitError: slow down");
    expect(boundObservation(ok)).toBe(ok);
    const none = errorObservation(null);
    expect(boundObservation(none)).toBe(none);
  });

  it("cuts error_message and leaves the caller's object and its keys alone", () => {
    const obs = { ...errorObservation(AZURE_FILTER), agent_id: "a".repeat(300) };
    const out = boundObservation(obs);
    expect(out.error_message).toHaveLength(ERROR_MESSAGE_MAX);
    expect(out.agent_id).toBe("a".repeat(300));
    expect(obs.error_message).toBe(AZURE_FILTER);
  });
});

describe("ObservationTransport with a long error_message", () => {
  it("posts the error cut to 200 characters", async () => {
    const bodies: Record<string, unknown>[] = [];
    const fetchImpl: typeof fetch = async (input, init = {}) => {
      const url = typeof input === "string" ? input : (input as URL).toString();
      if (url.includes("sdk-token/exchange")) {
        return new Response(JSON.stringify({ token: "t", expires_in: 900 }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      bodies.push(JSON.parse(init.body as string));
      return new Response(null, { status: 202 });
    };
    const t = new ObservationTransport({
      baseUrl: "http://api.local",
      apiKey: "k",
      tenantId: "e82770de-06d6-415c-911f-c0d65cf45190",
      fetchImpl,
    });
    t.post(errorObservation(AZURE_FILTER));
    for (let i = 0; i < 20; i += 1) await Promise.resolve();
    expect(bodies).toHaveLength(1);
    expect(bodies[0]!.error_message).toBe(AZURE_FILTER.slice(0, 199) + "…");
    expect(bodies[0]!.input_tokens).toBe(412);
  });
});
