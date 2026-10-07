"""In-process DLP — prompt text never leaves the customer process."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

DETECTOR_PATTERNS: dict[str, re.Pattern[str]] = {
    "phi_ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phi_mrn": re.compile(r"\bMRN[:\s#-]*\d{6,12}\b", re.I),
    "phi_dob": re.compile(
        r"\b(DOB|date of birth)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        re.I,
    ),
    "pci_pan": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "pci_cvv": re.compile(r"\b(CVV|CVC)[:\s#-]*\d{3,4}\b", re.I),
}


@dataclass(frozen=True)
class DlpScanResult:
    outcome: str  # none | redacted | refused
    matched_detectors: tuple[str, ...]


def extract_anthropic_messages_text(first_arg: Any) -> str:
    if not isinstance(first_arg, dict):
        return ""
    messages = first_arg.get("messages")
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
    return "\n".join(parts)


def extract_openai_messages_text(args: tuple[Any, ...]) -> str:
    if not args or not isinstance(args[0], dict):
        return ""
    messages = args[0].get("messages")
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
    return "\n".join(parts)


def redact_text_for_compliance(text: str, detectors: tuple[str, ...]) -> str:
    out = text
    for detector in detectors:
        pattern = DETECTOR_PATTERNS.get(detector)
        if pattern:
            out = pattern.sub("[REDACTED]", out)
    return out


def scan_text_for_compliance(
    text: str,
    snapshot: Optional[Mapping[str, Any]],
) -> DlpScanResult:
    if not snapshot:
        return DlpScanResult("none", ())
    dlp = snapshot.get("dlp") if isinstance(snapshot.get("dlp"), dict) else {}
    if not dlp.get("enabled"):
        return DlpScanResult("none", ())
    detectors = dlp.get("detectors") or []
    matched: list[str] = []
    for detector in detectors:
        if not isinstance(detector, str):
            continue
        pattern = DETECTOR_PATTERNS.get(detector)
        if pattern and pattern.search(text):
            matched.append(detector)
    if not matched:
        return DlpScanResult("none", ())
    mode = snapshot.get("mode", "observe")
    action = dlp.get("action", "refuse")
    if mode == "enforce" and action == "refuse":
        return DlpScanResult("refused", tuple(matched))
    if action == "redact":
        return DlpScanResult("redacted", tuple(matched))
    return DlpScanResult("none", tuple(matched))
