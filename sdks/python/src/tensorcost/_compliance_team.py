"""Narrow-only team resolution — mirrors @tensorcost/contracts."""

from __future__ import annotations

from typing import Literal, Optional, TypedDict


class ResolvedTeamContext(TypedDict, total=False):
    teamId: Optional[str]
    error: Literal["compliance_team_mismatch"]


def resolve_effective_team(
    *,
    token_team_id: Optional[str] = None,
    client_team_id: Optional[str] = None,
) -> ResolvedTeamContext:
    token = token_team_id.strip() if token_team_id and token_team_id.strip() else None
    client = client_team_id.strip() if client_team_id and client_team_id.strip() else None
    if token and client and token != client:
        return {"teamId": token, "error": "compliance_team_mismatch"}
    return {"teamId": token or client}
