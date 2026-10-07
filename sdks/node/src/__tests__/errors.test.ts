/**
 * Tests for the typed error hierarchy.
 */

import { describe, expect, it } from "vitest";
import {
  TensorCostError,
  TensorCostNetworkError,
  TensorCostTimeoutError,
  TensorCostProxyError,
  TensorCostQuotaError,
  TensorCostProviderError,
} from "../errors.js";

describe("TensorCostError hierarchy", () => {
  it("TensorCostNetworkError is instanceof TensorCostError and TensorCostNetworkError", () => {
    const err = new TensorCostNetworkError("net fail");
    expect(err).toBeInstanceOf(TensorCostError);
    expect(err).toBeInstanceOf(TensorCostNetworkError);
    expect(err.message).toBe("net fail");
    expect(err.status).toBeNull();
    expect(err.requestId).toBeNull();
    expect(err.attempt).toBe(1);
    expect(err.name).toBe("TensorCostNetworkError");
  });

  it("TensorCostTimeoutError carries timeoutKind", () => {
    const err = new TensorCostTimeoutError("timed out", "total");
    expect(err).toBeInstanceOf(TensorCostError);
    expect(err).toBeInstanceOf(TensorCostTimeoutError);
    expect(err.timeoutKind).toBe("total");

    const idleErr = new TensorCostTimeoutError("idle timeout", "idle", { attempt: 3 });
    expect(idleErr.timeoutKind).toBe("idle");
    expect(idleErr.attempt).toBe(3);
  });

  it("TensorCostProxyError carries status and requestId", () => {
    const cause = new Error("underlying");
    const err = new TensorCostProxyError("proxy 503", {
      status: 503,
      requestId: "req-xyz",
      attempt: 2,
      cause,
    });
    expect(err).toBeInstanceOf(TensorCostProxyError);
    expect(err.status).toBe(503);
    expect(err.requestId).toBe("req-xyz");
    expect(err.attempt).toBe(2);
    expect(err.rootCause).toBe(cause);
    expect(err.name).toBe("TensorCostProxyError");
  });

  it("TensorCostQuotaError defaults status to 429 and carries retryAfterMs", () => {
    const err = new TensorCostQuotaError("rate limited", 5000);
    expect(err).toBeInstanceOf(TensorCostQuotaError);
    expect(err).toBeInstanceOf(TensorCostError);
    expect(err.status).toBe(429);
    expect(err.retryAfterMs).toBe(5000);

    const errNull = new TensorCostQuotaError("rate limited", null);
    expect(errNull.retryAfterMs).toBeNull();
  });

  it("TensorCostProviderError is instanceof TensorCostError", () => {
    const err = new TensorCostProviderError("upstream 401", { status: 401 });
    expect(err).toBeInstanceOf(TensorCostError);
    expect(err).toBeInstanceOf(TensorCostProviderError);
    expect(err.status).toBe(401);
    expect(err.name).toBe("TensorCostProviderError");
  });

  it("subclasses do not interfere with instanceof checks between them", () => {
    const network = new TensorCostNetworkError("net");
    const proxy = new TensorCostProxyError("proxy");
    expect(network).not.toBeInstanceOf(TensorCostProxyError);
    expect(proxy).not.toBeInstanceOf(TensorCostNetworkError);
  });
});
