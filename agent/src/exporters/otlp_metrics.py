"""Optional OTLP HTTP metrics export."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OtlpMetricsExporter:
    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint.rstrip("/")
        if not self._endpoint.endswith("/v1/metrics"):
            self._endpoint = f"{self._endpoint}/v1/metrics"

    def export(self, points: List[Dict[str, Any]]) -> None:
        if not points:
            return
        try:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource
        except ImportError:
            logger.warning(
                "OTLP metrics export skipped: install opentelemetry-sdk and "
                "opentelemetry-exporter-otlp-proto-http"
            )
            return

        resource = Resource.create({"service.name": "tensorcost-gpu-agent"})
        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=self._endpoint),
            export_interval_millis=60000,
        )
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        meter = provider.get_meter("tensorcost.gpu")
        for p in points:
            attrs = {
                k: str(v)
                for k, v in (
                    ("cloud_provider", p.get("cloud_provider")),
                    ("cloud_id", p.get("cloud_id")),
                    ("instance_id", p.get("instance_id")),
                    ("gpu_index", p.get("gpu_index")),
                )
                if v is not None
            }
            util = p.get("util_pct")
            if util is not None:
                gauge = meter.create_gauge(
                    "gpu.utilization.percent",
                    description="GPU utilization percent",
                )
                gauge.set(float(util), attrs)
        try:
            provider.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.debug("OTLP provider shutdown: %s", exc)


def otlp_enabled() -> bool:
    return os.getenv("OTEL_METRICS_ENABLED", "").strip().lower() in ("1", "true", "yes")
