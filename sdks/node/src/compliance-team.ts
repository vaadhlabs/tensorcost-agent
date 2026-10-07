/** Narrow-only team resolution — mirrors @tensorcost/contracts (no runtime dep). */

export type TeamResolutionError = "compliance_team_mismatch";

export interface ResolvedTeamContext {
  teamId: string | null;
  error?: TeamResolutionError;
}

export function resolveEffectiveTeam(args: {
  tokenTeamId?: string | null;
  clientTeamId?: string | null;
}): ResolvedTeamContext {
  const token = args.tokenTeamId?.trim() || null;
  const client = args.clientTeamId?.trim() || null;

  if (token && client && token !== client) {
    return { teamId: token, error: "compliance_team_mismatch" };
  }
  return { teamId: token ?? client };
}
