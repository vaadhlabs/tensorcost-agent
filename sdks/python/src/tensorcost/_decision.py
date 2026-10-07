"""Parse x-tc-decision response header for observation metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ParsedDecision:
    layer: str
    action: str
    reason: Optional[str] = None


def parse_decision_header(header: Optional[str]) -> Optional[ParsedDecision]:
    if not header or not header.strip():
        return None
    layer: Optional[str] = None
    action: Optional[str] = None
    reason: Optional[str] = None
    for segment in header.split(";"):
        trimmed = segment.strip()
        if "=" not in trimmed:
            continue
        key, value = trimmed.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "layer":
            layer = value
        elif key == "action":
            action = value
        elif key == "reason":
            reason = value
    if not layer or not action:
        return None
    return ParsedDecision(layer=layer, action=action, reason=reason)


def decision_metadata(header: Optional[str]) -> Optional[dict[str, Any]]:
    parsed = parse_decision_header(header)
    if parsed is None:
        return None
    meta: dict[str, Any] = {"decision_action": parsed.action}
    if parsed.reason:
        meta["decision_reason"] = parsed.reason
    return meta
