"""Client-side compliance — team narrow-check + in-process DLP."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ._compliance_team import resolve_effective_team
from ._dlp import extract_openai_messages_text, scan_text_for_compliance
from ._errors import TensorCostComplianceDeniedError, TensorCostComplianceTeamMismatchError


def assert_team_scope(
    *,
    token_team_id: Optional[str] = None,
    client_team_id: Optional[str] = None,
) -> None:
    resolved = resolve_effective_team(
        token_team_id=token_team_id,
        client_team_id=client_team_id,
    )
    if resolved.get("error") == "compliance_team_mismatch":
        raise TensorCostComplianceTeamMismatchError(
            "tensorcost: SDK team scope does not match token binding"
        )


def enforce_in_process_dlp(
    text: str,
    snapshot: Optional[Mapping[str, Any]],
    *,
    args: Optional[tuple[Any, ...]] = None,
) -> dict[str, Any]:
    result = scan_text_for_compliance(text, snapshot)
    if result.outcome == "refused" and snapshot and snapshot.get("mode") == "enforce":
        raise TensorCostComplianceDeniedError(
            "tensorcost: in-process DLP refused this call"
        )
    if result.outcome == "redacted" and args and result.matched_detectors:
        _apply_redaction_to_openai_args(args, result.matched_detectors)
    return {
        "outcome": result.outcome,
        "matched_detectors": result.matched_detectors,
    }


def compliance_stamp_from(
    snapshot: Optional[Mapping[str, Any]],
    dlp: Mapping[str, Any],
) -> dict[str, Any]:
    stamp: dict[str, Any] = {}
    if snapshot and snapshot.get("audit", {}).get("stamp") and snapshot.get("frameworks"):
        stamp["compliance_frameworks"] = list(snapshot["frameworks"])
    if dlp.get("outcome") and dlp["outcome"] != "none":
        stamp["dlp_action"] = dlp["outcome"]
    return stamp


def _apply_redaction_to_openai_args(
    args: tuple[Any, ...],
    detectors: tuple[str, ...],
) -> None:
    if not args or not isinstance(args[0], dict):
        return
    req = args[0]
    messages = req.get("messages")
    if not isinstance(messages, list):
        return
    from ._dlp import redact_text_for_compliance

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = redact_text_for_compliance(content, detectors)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    block["text"] = redact_text_for_compliance(block["text"], detectors)
