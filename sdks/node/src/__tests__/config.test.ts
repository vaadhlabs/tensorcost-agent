import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  DEFAULT_BASE_URL,
  MissingConfigError,
  TensorCostConfigError,
  resolveConfig,
} from "../config.js";

const ENV_KEYS = [
  "TENSORCOST_API_KEY",
  "TENSORCOST_BASE_URL",
  "TENSORCOST_TENANT_ID",
] as const;

describe("resolveConfig", () => {
  let saved: Record<string, string | undefined>;

  beforeEach(() => {
    saved = {};
    for (const k of ENV_KEYS) {
      saved[k] = process.env[k];
      delete process.env[k];
    }
  });

  afterEach(() => {
    for (const k of ENV_KEYS) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
  });

  it("throws MissingConfigError when apiKey is unset", () => {
    expect(() => resolveConfig()).toThrow(MissingConfigError);
  });

  it("uses env vars when no explicit options given", () => {
    process.env.TENSORCOST_API_KEY = "k-env";
    process.env.TENSORCOST_BASE_URL = "https://staging.tensorcost.com/";
    process.env.TENSORCOST_TENANT_ID = "tenant-env";
    const cfg = resolveConfig();
    expect(cfg.apiKey).toBe("k-env");
    expect(cfg.baseUrl).toBe("https://staging.tensorcost.com");
    expect(cfg.tenantId).toBe("tenant-env");
    expect(cfg.failOpen).toBe(true);
  });

  it("explicit options override env", () => {
    process.env.TENSORCOST_API_KEY = "k-env";
    const cfg = resolveConfig({
      apiKey: "k-explicit",
      baseUrl: "http://localhost:4115",
      tenantId: "t-explicit",
      failOpen: false,
    });
    expect(cfg.apiKey).toBe("k-explicit");
    expect(cfg.baseUrl).toBe("http://localhost:4115");
    expect(cfg.tenantId).toBe("t-explicit");
    expect(cfg.failOpen).toBe(false);
  });

  it("falls back to default base URL when nothing else set", () => {
    const cfg = resolveConfig({ apiKey: "k" });
    expect(cfg.baseUrl).toBe(DEFAULT_BASE_URL);
    expect(cfg.tenantId).toBeNull();
  });

  it("strips trailing slashes from baseUrl", () => {
    const cfg = resolveConfig({
      apiKey: "k",
      baseUrl: "https://api.tensorcost.com///",
    });
    expect(cfg.baseUrl).toBe("https://api.tensorcost.com");
  });

  it("defaults maxLayer to observe", () => {
    const cfg = resolveConfig({ apiKey: "k" });
    expect(cfg.maxLayer).toBe("observe");
    expect(cfg.appliedMode).toBe(false);
  });

  it("appliedMode true resolves maxLayer to route with deprecation path", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const cfg = resolveConfig({
      apiKey: "k",
      appliedMode: true,
      proxyUrl: "https://proxy.local",
    });
    expect(cfg.maxLayer).toBe("route");
    expect(cfg.appliedMode).toBe(true);
    warn.mockRestore();
  });

  it("maxLayer steer requires proxyUrl", () => {
    expect(() =>
      resolveConfig({ apiKey: "k", maxLayer: "steer" }),
    ).toThrow(TensorCostConfigError);
  });

  it("maxLayer govern does not require proxyUrl", () => {
    const cfg = resolveConfig({ apiKey: "k", maxLayer: "govern" });
    expect(cfg.maxLayer).toBe("govern");
    expect(cfg.proxyUrl).toBeNull();
  });
});
