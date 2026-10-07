import { describe, expect, it } from "vitest";
import { scanTextForCompliance } from "../dlp.js";
import type { CompiledComplianceSnapshot } from "../layer.js";

const hipaaSnap: CompiledComplianceSnapshot = {
  frameworks: ["hipaa"],
  mode: "enforce",
  fail_mode: "closed",
  max_grants: ["telemetry"],
  dlp: { enabled: true, detectors: ["phi_ssn"], action: "refuse" },
  audit: { stamp: true },
};

describe("scanTextForCompliance", () => {
  it("refuses SSN in enforce mode", () => {
    const result = scanTextForCompliance("patient ssn 123-45-6789", hipaaSnap);
    expect(result.outcome).toBe("refused");
    expect(result.matchedDetectors).toContain("phi_ssn");
  });

  it("returns none when no detectors match", () => {
    const result = scanTextForCompliance("hello world", hipaaSnap);
    expect(result.outcome).toBe("none");
  });

  it("observe mode does not refuse", () => {
    const observe = { ...hipaaSnap, mode: "observe" as const };
    const result = scanTextForCompliance("123-45-6789", observe);
    expect(result.outcome).toBe("none");
  });
});
