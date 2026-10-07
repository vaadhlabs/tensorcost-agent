import { describe, expect, it, vi } from "vitest";
import {
  intersectGrants,
  layerRank,
  minLayer,
  needsProxyUrl,
  proxyPathForOperation,
  resolveMaxLayer,
} from "../layer.js";

describe("layer helpers", () => {
  it("layerRank orders off < observe < govern < steer < route", () => {
    expect(layerRank("off")).toBeLessThan(layerRank("observe"));
    expect(layerRank("govern")).toBeLessThan(layerRank("steer"));
  });

  it("minLayer returns lower rung", () => {
    expect(minLayer("route", "govern")).toBe("govern");
  });

  it("needsProxyUrl is true only for steer and route", () => {
    expect(needsProxyUrl("observe")).toBe(false);
    expect(needsProxyUrl("govern")).toBe(false);
    expect(needsProxyUrl("steer")).toBe(true);
    expect(needsProxyUrl("route")).toBe(true);
  });

  it("resolveMaxLayer prefers explicit maxLayer", () => {
    expect(resolveMaxLayer({ maxLayer: "govern" })).toBe("govern");
  });

  it("resolveMaxLayer maps appliedMode to route with warning", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(resolveMaxLayer({ appliedMode: true })).toBe("route");
    warn.mockRestore();
  });

  it("proxyPathForOperation resolves chat.completions", () => {
    expect(proxyPathForOperation("chat.completions")).toContain(
      "chat/completions",
    );
  });

  it("intersectGrants treats explicit empty codeMax as no grants", () => {
    expect(intersectGrants([], ["telemetry"])).toEqual([]);
  });
});
