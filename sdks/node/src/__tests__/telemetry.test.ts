/**
 * Tests for telemetry hook dispatch.
 */

import { describe, expect, it, vi } from "vitest";
import { emitEvent } from "../telemetry.js";
import type { LifecycleEvent } from "../telemetry.js";

function makeEvent(kind: LifecycleEvent["kind"] = "before_request"): LifecycleEvent {
  return {
    kind,
    provider: "openai",
    model: "gpt-4o",
    operation: "chat.completions",
    attemptNumber: 1,
    elapsedMs: 0,
  } as LifecycleEvent;
}

describe("emitEvent", () => {
  it("calls the callback with the event", () => {
    const cb = vi.fn();
    const event = makeEvent("before_request");
    emitEvent(cb, event);
    expect(cb).toHaveBeenCalledOnce();
    expect(cb).toHaveBeenCalledWith(event);
  });

  it("does nothing when callback is undefined", () => {
    // Should not throw.
    expect(() => emitEvent(undefined, makeEvent())).not.toThrow();
  });

  it("swallows errors thrown by the callback", () => {
    const cb = vi.fn(() => {
      throw new Error("boom");
    });
    // The SDK must never crash because a customer hook throws.
    expect(() => emitEvent(cb, makeEvent())).not.toThrow();
    expect(cb).toHaveBeenCalledOnce();
  });

  it("fires after_response event with status and requestId", () => {
    const events: LifecycleEvent[] = [];
    const cb = (e: LifecycleEvent) => events.push(e);
    const event: LifecycleEvent = {
      kind: "after_response",
      provider: "anthropic",
      model: "claude-3-5-sonnet-20241022",
      operation: "messages",
      attemptNumber: 1,
      elapsedMs: 150,
      status: 200,
      requestId: "req-abc",
    };
    emitEvent(cb, event);
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({ kind: "after_response", status: 200, requestId: "req-abc" });
  });

  it("fires on_retry event with delay and reason", () => {
    const events: LifecycleEvent[] = [];
    const cb = (e: LifecycleEvent) => events.push(e);
    const event: LifecycleEvent = {
      kind: "on_retry",
      provider: "openai",
      model: "gpt-4o",
      operation: "chat.completions",
      attemptNumber: 2,
      elapsedMs: 500,
      reason: "5xx",
      status: 503,
      delayMs: 1000,
    };
    emitEvent(cb, event);
    expect(events[0]).toMatchObject({ kind: "on_retry", reason: "5xx", delayMs: 1000 });
  });

  it("fires on_fallback event with consecutiveFailures", () => {
    const events: LifecycleEvent[] = [];
    const cb = (e: LifecycleEvent) => events.push(e);
    const event: LifecycleEvent = {
      kind: "on_fallback",
      provider: "openai",
      model: "gpt-4o-mini",
      operation: "chat.completions",
      attemptNumber: 1,
      elapsedMs: 0,
      consecutiveFailures: 3,
    };
    emitEvent(cb, event);
    expect(events[0]).toMatchObject({ kind: "on_fallback", consecutiveFailures: 3 });
  });
});
