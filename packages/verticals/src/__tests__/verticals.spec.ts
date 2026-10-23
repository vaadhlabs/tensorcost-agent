import { describe, expect, it } from "vitest";
import { installedVerticals, isServiceDeployed, isVerticalEnabled } from "../index.js";

describe("verticals", () => {
  it("defaults to all enabled when env unset", () => {
    expect(isVerticalEnabled("fleet", {})).toBe(true);
    expect(isServiceDeployed("gpu-service", {})).toBe(true);
  });

  it("respects TC_VERTICAL_FLEET=false", () => {
    const env = { TC_VERTICAL_FLEET: "false", TC_VERTICAL_AI: "true" };
    expect(isVerticalEnabled("fleet", env)).toBe(false);
    expect(isServiceDeployed("gpu-service", env)).toBe(false);
    expect(isServiceDeployed("cost-service", env)).toBe(true);
  });

  it("aiMl requires ai flag", () => {
    const env = { TC_VERTICAL_AI: "false", TC_VERTICAL_AI_ML: "true" };
    expect(isServiceDeployed("ml-service", env)).toBe(false);
  });

  it("installedVerticals snapshot", () => {
    expect(installedVerticals({ TC_VERTICAL_AI: "false" }).ai).toBe(false);
  });
});
