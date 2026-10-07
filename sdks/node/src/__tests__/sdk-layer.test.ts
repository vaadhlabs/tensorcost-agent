import { describe, expect, it, vi } from "vitest";

import { SDK_LAYER_DECAY_MS, SDK_LAYER_FETCH_TIMEOUT_MS } from "../layer.js";
import { SdkLayerClient } from "../sdk-layer.js";

describe("SdkLayerClient cold start", () => {
  it("decays to govern on first fetch failure without using code ceiling", async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: false,
      status: 503,
    })) as unknown as typeof fetch;

    const client = new SdkLayerClient({
      baseUrl: "https://api.tensorcost.com",
      getToken: async () => "jwt",
      fetchImpl,
    });

    const layer = await client.effectiveLayer("route");
    expect(layer).toBe("govern");
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("aborts a hung cold fetch and decays to govern", async () => {
    const fetchImpl = vi.fn((_url: RequestInfo | URL, init?: RequestInit) => {
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          const err = new Error("aborted");
          err.name = "AbortError";
          reject(err);
        });
      });
    }) as unknown as typeof fetch;

    const client = new SdkLayerClient({
      baseUrl: "https://api.tensorcost.com",
      getToken: async () => "jwt",
      fetchImpl,
    });

    const started = Date.now();
    const layer = await client.effectiveLayer("route");
    expect(layer).toBe("govern");
    expect(Date.now() - started).toBeLessThan(SDK_LAYER_FETCH_TIMEOUT_MS + 200);
  });

  it("uses last-known-good within decay window after a prior success", async () => {
    let call = 0;
    const fetchImpl = vi.fn(async () => {
      call += 1;
      if (call === 1) {
        return {
          ok: true,
          status: 200,
          json: async () => ({ published_layer: "observe" }),
        };
      }
      return { ok: false, status: 503 };
    }) as unknown as typeof fetch;

    const client = new SdkLayerClient({
      baseUrl: "https://api.tensorcost.com",
      getToken: async () => "jwt",
      fetchImpl,
    });

    expect(await client.effectiveLayer("route")).toBe("observe");

    // Force cache expiry without waiting 30s.
    (client as unknown as { cache: { fetchedAt: number } | null }).cache = {
      layer: "observe",
      dataGrants: [],
      fetchedAt: Date.now() - SDK_LAYER_DECAY_MS,
    };

    expect(await client.effectiveLayer("route")).toBe("observe");
  });

  it("returns stale cache immediately while a hung refresh runs in the background", async () => {
    let call = 0;
    const fetchImpl = vi.fn((_url: RequestInfo | URL, init?: RequestInit) => {
      call += 1;
      if (call === 1) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({ published_layer: "observe" }),
        } as Response);
      }
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          const err = new Error("aborted");
          err.name = "AbortError";
          reject(err);
        });
      });
    }) as unknown as typeof fetch;

    const client = new SdkLayerClient({
      baseUrl: "https://api.tensorcost.com",
      getToken: async () => "jwt",
      fetchImpl,
    });

    expect(await client.effectiveLayer("route")).toBe("observe");
    (client as unknown as { cache: { fetchedAt: number } | null }).cache = {
      layer: "observe",
      dataGrants: [],
      fetchedAt: Date.now() - 31_000,
    };

    const started = Date.now();
    expect(await client.effectiveLayer("route")).toBe("observe");
    expect(Date.now() - started).toBeLessThan(50);
  });
});
