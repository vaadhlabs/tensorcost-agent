import { AsyncLocalStorage } from "node:async_hooks";
import { describe, expect, it } from "vitest";
import { resolveModelCallLink, sdkModelCallSpanId } from "../trace-link.js";

// agent-sdk publishes the link on this registered symbol; reproduce that
// here rather than depend on the other package.
const KEY = Symbol.for("tensorcost.modelCallLink.v1");
function agentStore(): AsyncLocalStorage<Record<string, unknown>> {
  const g = globalThis as Record<symbol, unknown>;
  if (!g[KEY]) g[KEY] = new AsyncLocalStorage();
  return g[KEY] as AsyncLocalStorage<Record<string, unknown>>;
}

const CORR = "11111111-1111-4111-8111-111111111111";
const TRACE = "22222222-2222-4222-8222-222222222222";
const PARENT = "33333333-3333-4333-8333-333333333333";

describe("resolveModelCallLink", () => {
  it("mints a fresh correlation id and no link outside any scope", () => {
    const r = resolveModelCallLink({});
    expect(r.correlationId).toMatch(/^[0-9a-f-]{36}$/);
    expect(r.traceLink).toEqual({});
  });

  it("takes the agent-sdk link in scope, once", () => {
    agentStore().run(
      { correlationId: CORR, traceId: TRACE, parentSpanId: PARENT },
      () => {
        const first = resolveModelCallLink({});
        expect(first.correlationId).toBe(CORR);
        expect(first.traceLink).toEqual({ trace_id: TRACE, parent_span_id: PARENT });

        const second = resolveModelCallLink({});
        expect(second.correlationId).not.toBe(CORR);
        expect(second.traceLink).toEqual({});
      },
    );
  });

  it("prefers explicit withMeta values over the ambient link", () => {
    const explicit = "44444444-4444-4444-8444-444444444444";
    agentStore().run({ correlationId: CORR, traceId: TRACE }, () => {
      const r = resolveModelCallLink({ correlationId: explicit, traceId: TRACE });
      expect(r.correlationId).toBe(explicit);
      expect(r.traceLink).toEqual({ trace_id: TRACE });
    });
  });

  it("drops ids that are not UUIDs instead of sending them", () => {
    const r = resolveModelCallLink({
      traceId: "4bf92f3577b34da6a3ce929d0e0e4736",
      parentSpanId: "00f067aa0ba902b7",
    });
    expect(r.traceLink).toEqual({});
  });

  it("links a root-level call with a trace and no parent", () => {
    agentStore().run({ correlationId: CORR, traceId: TRACE, parentSpanId: null }, () => {
      expect(resolveModelCallLink({}).traceLink).toEqual({ trace_id: TRACE });
    });
  });

  it("derives the same span id agent-sdk records", () => {
    expect(sdkModelCallSpanId(CORR)).toBe(sdkModelCallSpanId(CORR));
    expect(sdkModelCallSpanId(CORR)).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
    );
  });
});
