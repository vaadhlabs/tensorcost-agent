/**
 * providerNativeError: asking the client to build its own error type.
 *
 * Two header contracts are in the wild. openai@4 reads a plain object
 * (`headers["x-request-id"]`); openai v5/v6 and current @anthropic-ai/sdk call
 * `headers.get(...)` and throw on a plain object. The stub below mimics the
 * second; real-clients.spec.ts runs the real openai@4 and Anthropic factories.
 */

import { describe, expect, it } from "vitest";
import { providerNativeError } from "../provider-errors.js";
import { TensorCostProviderError } from "../errors.js";

const relayed = () =>
  new TensorCostProviderError("tensorcost: upstream provider returned 400: bad", {
    status: 400,
    body: { error: { message: "bad", type: "invalid_request_error", code: "invalid_value" } },
    headers: { "x-request-id": "req_123", "content-type": "application/json" },
  });

/** Same shape as Stainless v5+ APIError: headers.get in the constructor. */
class StubAPIError extends Error {
  readonly requestID: string | null;
  constructor(
    readonly status: number,
    readonly error: unknown,
    message: string | undefined,
    readonly headers: { get(name: string): string | null },
  ) {
    super(message ?? `${status} ${(error as { message?: string })?.message ?? ""}`);
    this.requestID = headers?.get("x-request-id") ?? null;
  }
}
class StubBadRequestError extends StubAPIError {}

const v6Client = {
  makeStatusError(status: number, body: any, message: string | undefined, headers: any) {
    return status === 400
      ? new StubBadRequestError(status, body?.error, message, headers)
      : new StubAPIError(status, body?.error, message, headers);
  },
};

describe("providerNativeError", () => {
  it("v5/v6 headers.get contract: retried with a Headers instance, request id kept", () => {
    const err = relayed();
    const native = providerNativeError(v6Client, err);
    expect(native).toBeInstanceOf(StubBadRequestError);
    expect((native as StubAPIError).requestID).toBe("req_123");
    expect((native as StubAPIError).headers).toBeInstanceOf(Headers);
    expect((native as { tensorcost?: unknown }).tensorcost).toBe(err);
  });

  it("a v4-style factory gets the plain object first", () => {
    const seen: unknown[] = [];
    const v4 = {
      makeStatusError(_s: number, _b: unknown, _m: unknown, headers: Record<string, string>) {
        seen.push(headers);
        return Object.assign(new Error("v4"), { request_id: headers["x-request-id"] });
      },
    };
    const native = providerNativeError(v4, relayed());
    expect((native as { request_id?: string }).request_id).toBe("req_123");
    expect(seen).toHaveLength(1);
    expect(seen[0]).not.toBeInstanceOf(Headers);
  });

  it("a factory that rejects both shapes leaves the TensorCostProviderError", () => {
    const err = relayed();
    const broken = {
      makeStatusError() {
        throw new TypeError("nope");
      },
    };
    expect(providerNativeError(broken, err)).toBe(err);
  });

  it("a client without a factory leaves the TensorCostProviderError", () => {
    const err = relayed();
    expect(providerNativeError({}, err)).toBe(err);
    expect(providerNativeError(null, err)).toBe(err);
  });

  it("only TensorCostProviderError is converted", () => {
    const other = new Error("x");
    expect(providerNativeError(v6Client, other)).toBe(other);
  });
});
