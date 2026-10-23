import { describe, expect, it } from "vitest";
import { isVerticalEnabled } from "./index.js";

describe("isVerticalEnabled", () => {
  it("aiMl requires AI when aiMl flag is on", () => {
    const env = {
      TC_VERTICAL_AI: "false",
      TC_VERTICAL_AI_ML: "true",
    };
    expect(isVerticalEnabled("aiMl", env)).toBe(false);
  });

  it("aiMl is on only when both flags are on", () => {
    const env = {
      TC_VERTICAL_AI: "true",
      TC_VERTICAL_AI_ML: "true",
    };
    expect(isVerticalEnabled("aiMl", env)).toBe(true);
  });
});
