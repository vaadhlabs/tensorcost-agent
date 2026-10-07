/**
 * Tests for retry utilities.
 */

import { describe, expect, it } from "vitest";
import {
  computeDelay,
  isRetryableStatus,
  retryAfterMs,
  sleep,
  DEFAULT_RETRY_CONFIG,
} from "../retry.js";

describe("computeDelay", () => {
  it("returns a value in the expected range for attempt 0", () => {
    const config = { ...DEFAULT_RETRY_CONFIG };
    // attempt 0: delay = min(30000, 500 * 2^0) * jitter = 500 * [0.5, 1.0]
    for (let i = 0; i < 20; i++) {
      const delay = computeDelay(0, config);
      expect(delay).toBeGreaterThanOrEqual(250); // 50% of 500
      expect(delay).toBeLessThanOrEqual(500);
    }
  });

  it("caps at maxDelayMs", () => {
    const config = { maxAttempts: 10, baseDelayMs: 1000, maxDelayMs: 2000 };
    // attempt 10: 1000 * 2^10 = 1_024_000, capped at 2000, jittered to [1000, 2000]
    for (let i = 0; i < 20; i++) {
      const delay = computeDelay(10, config);
      expect(delay).toBeLessThanOrEqual(2000);
      expect(delay).toBeGreaterThanOrEqual(1000);
    }
  });

  it("uses Retry-After override when provided", () => {
    const delay = computeDelay(0, DEFAULT_RETRY_CONFIG, 8000);
    expect(delay).toBe(8000);
  });

  it("caps Retry-After at maxDelayMs", () => {
    const delay = computeDelay(0, { ...DEFAULT_RETRY_CONFIG, maxDelayMs: 5000 }, 99_000);
    expect(delay).toBe(5000);
  });
});

describe("isRetryableStatus", () => {
  it("returns true for 5xx", () => {
    expect(isRetryableStatus(500)).toBe(true);
    expect(isRetryableStatus(503)).toBe(true);
    expect(isRetryableStatus(599)).toBe(true);
  });

  it("returns true for 429", () => {
    expect(isRetryableStatus(429)).toBe(true);
  });

  it("returns false for 4xx (except 429)", () => {
    expect(isRetryableStatus(400)).toBe(false);
    expect(isRetryableStatus(401)).toBe(false);
    expect(isRetryableStatus(404)).toBe(false);
    expect(isRetryableStatus(422)).toBe(false);
  });

  it("returns false for 2xx and 3xx", () => {
    expect(isRetryableStatus(200)).toBe(false);
    expect(isRetryableStatus(302)).toBe(false);
  });
});

describe("retryAfterMs", () => {
  it("parses a numeric Retry-After in seconds", () => {
    const resp = new Response(null, {
      status: 429,
      headers: { "retry-after": "5" },
    });
    expect(retryAfterMs(resp)).toBe(5000);
  });

  it("returns undefined when Retry-After header is absent", () => {
    const resp = new Response(null, { status: 429 });
    expect(retryAfterMs(resp)).toBeUndefined();
  });

  it("parses a fractional seconds Retry-After", () => {
    const resp = new Response(null, {
      status: 429,
      headers: { "retry-after": "1.5" },
    });
    // parseFloat("1.5") * 1000 = 1500; Math.ceil(1500) = 1500
    expect(retryAfterMs(resp)).toBe(1500);
  });
});

describe("sleep", () => {
  it("resolves after approximately the given delay", async () => {
    const t0 = Date.now();
    await sleep(10);
    expect(Date.now() - t0).toBeGreaterThanOrEqual(8);
  });

  it("rejects when AbortController fires before delay", async () => {
    const controller = new AbortController();
    const p = sleep(5000, controller.signal);
    controller.abort();
    await expect(p).rejects.toThrow();
  });

  it("rejects immediately when signal is already aborted", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(sleep(5000, controller.signal)).rejects.toThrow();
  });
});
