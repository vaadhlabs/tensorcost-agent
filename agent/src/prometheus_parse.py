"""Shared Prometheus text exposition parser."""

from __future__ import annotations

import re
from typing import Dict


_LABEL_RE = re.compile(r'\{([^}]*)\}')
_METRIC_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)"
)


def _label_key(labels: str) -> str:
    if not labels.strip():
        return ""
    parts = []
    for part in labels.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        parts.append(f"{key.strip()}={value.strip().strip(chr(34))}")
    return ",".join(sorted(parts))


def parse_prometheus_metrics(text: str) -> Dict[str, float]:
    """Parse Prometheus-format metrics into name[{labels}] → value."""
    metrics: Dict[str, float] = {}
    for line in text.split("\n"):
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE_RE.match(line)
        if not match:
            continue
        name = match.group(1)
        label_match = _LABEL_RE.search(line.split()[0] if " " in line else line)
        label_suffix = ""
        if label_match:
            label_suffix = "{" + _label_key(label_match.group(1)) + "}"
        key = f"{name}{label_suffix}"
        try:
            metrics[key] = float(match.group(2))
        except ValueError:
            continue
    return metrics
