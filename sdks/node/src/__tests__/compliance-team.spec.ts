import { describe, expect, it } from "vitest";
import { resolveEffectiveTeam } from "../compliance-team.js";

describe("resolveEffectiveTeam", () => {
  it("returns mismatch when client team differs from token team", () => {
    const r = resolveEffectiveTeam({
      tokenTeamId: "team-a",
      clientTeamId: "team-b",
    });
    expect(r.error).toBe("compliance_team_mismatch");
  });

  it("allows unbound token with client team (narrow)", () => {
    const r = resolveEffectiveTeam({ tokenTeamId: null, clientTeamId: "team-b" });
    expect(r.error).toBeUndefined();
    expect(r.teamId).toBe("team-b");
  });
});
