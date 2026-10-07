"""Append GPU usage rows to a local FOCUS-shaped CSV (usage only, no cost)."""

from __future__ import annotations

import csv
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

FOCUS_HEADERS = (
    "BillingPeriodStart",
    "BillingPeriodEnd",
    "ChargeCategory",
    "ChargeDescription",
    "ConsumedQuantity",
    "ConsumedUnit",
    "ProviderName",
    "RegionId",
    "ResourceId",
    "ResourceType",
    "ServiceName",
)


class FocusUsageExporter:
    """Hourly GPU-hour style rows; no dollar attribution."""

    def __init__(self, path: str, interval_seconds: int = 3600) -> None:
        self._path = path
        self._interval = max(60, interval_seconds)
        self._lock = threading.Lock()
        self._last_flush: Optional[float] = None
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if not os.path.isfile(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(FOCUS_HEADERS)

    def record(self, points: List[Dict[str, Any]], region: str, provider: str) -> None:
        if not points:
            return
        now = datetime.now(timezone.utc)
        period_start = now.replace(minute=0, second=0, microsecond=0)
        period_end = period_start.replace(hour=period_start.hour + 1) if period_start.hour < 23 else period_start

        rows: List[List[str]] = []
        seen = set()
        for p in points:
            resource = p.get("cloud_id") or p.get("instance_id") or "unknown"
            gpu_idx = p.get("gpu_index", 0)
            key = (resource, gpu_idx)
            if key in seen:
                continue
            seen.add(key)
            util = float(p.get("util_pct") or 0.0)
            quantity = max(0.0, min(1.0, util / 100.0)) * (self._interval / 3600.0)
            rows.append(
                [
                    period_start.isoformat(),
                    period_end.isoformat(),
                    "Usage",
                    "GPU compute utilization sample",
                    f"{quantity:.6f}",
                    "GPU-Hours",
                    provider or "unknown",
                    region or "",
                    f"{resource}:gpu{gpu_idx}",
                    "Compute Instance",
                    "GPU Telemetry",
                ]
            )
        if not rows:
            return
        with self._lock:
            with open(self._path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)
        logger.debug("FOCUS usage appended %s rows to %s", len(rows), self._path)
